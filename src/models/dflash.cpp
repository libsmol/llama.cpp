#include "models.h"

#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"

void llama_model_dflash::load_arch_hparams(llama_model_loader & ml) {

    ml.get_key(LLM_KV_EMBEDDING_SCALE, hparams.f_embedding_scale, false);
    ml.get_key(LLM_KV_ATTENTION_SCALE, hparams.f_attention_scale, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_SCALE, hparams.f_attn_value_scale, false);

    hparams.llm_ffn_op = LLM_FFN_SILU;
    std::string hidden_act;
    if (ml.get_key(LLM_KV_HIDDEN_ACT, hidden_act, false)) {
        if (hidden_act == "gelu" || hidden_act == "gelu_pytorch_tanh") {
            hparams.llm_ffn_op = LLM_FFN_GELU;
        } else if (hidden_act == "swigluoai") {
            hparams.llm_ffn_op = LLM_FFN_SWIGLU_OAI_MOE;
        } else if (hidden_act != "silu") {
            throw std::runtime_error("unsupported DFlash hidden activation: " + hidden_act);
        }
    }

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LOGIT_SCALE,                 hparams.f_logit_scale, false);
    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);

    // drafts for M-RoPE targets carry degenerate sections [n_rot/2, 0, 0, 0]
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, false);

    ml.get_key(LLM_KV_DFLASH_BLOCK_SIZE,       hparams.dflash_block_size,       false);
    ml.get_key(LLM_KV_DFLASH_CONV_KERNEL_SIZE, hparams.dflash_conv_kernel_size, false);
    ml.get_key(LLM_KV_DFLASH_CONV_GROUP_SIZE,  hparams.dflash_conv_group_size,  false);
    ml.get_key(LLM_KV_DFLASH_SELECTOR_RANK,    hparams.dflash_selector_rank,    false);
    ml.get_key(LLM_KV_DFLASH_SELECTOR_TOP_K,   hparams.dflash_selector_top_k,   false);

    const bool has_markov   = ml.get_tensor_meta("markov_w1.weight")       != nullptr;
    const bool has_selector = ml.get_tensor_meta("selector_hidden.weight") != nullptr;
    const bool has_lilicorr = ml.get_key(LLM_KV_LILICORR_HIDDEN_SIZE, hparams.lilicorr_hidden_size, false);
    const bool has_dspine   = ml.get_key(LLM_KV_DSPINE_HEAD_TOP_K, hparams.dspine_head_top_k, false);
    if (has_dspine) {
        ml.get_key(LLM_KV_DSPINE_BRIDGE_EPS, hparams.dspine_bridge_eps);
        const uint32_t k = hparams.dspine_head_top_k;
        if (k < 2 || k > 64 || (k & (k - 1)) != 0 || hparams.dspine_bridge_eps <= 0.0f || hparams.dflash_block_size < 2 ||
                hparams.dflash_selector_top_k > 0 || (uint64_t) k * (k + 1) > hparams.n_embd ||
                has_markov || has_selector || has_lilicorr || hparams.dflash_conv_kernel_size > 0) {
            throw std::runtime_error("invalid DSpine metadata, or DSpine mixed with another draft head");
        }
        variant = VARIANT_DSPINE;
    } else if (has_markov) {
        variant = VARIANT_DSPARK;
        hparams.dspark_f_rnn = ml.get_tensor_meta("markov_ctx_norm.weight") != nullptr;
    } else if (has_selector) {
        variant = VARIANT_DFLASH2;
    } else if (has_lilicorr) {
        variant = VARIANT_LILICORR;
        load_hparams_lilicorr(ml);
    }

    if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false)) {
        throw std::runtime_error("DFlash model requires 'target_layers' in GGUF metadata");
    }

    hparams.n_embd_inp_enc_impl = (uint32_t) target_layer_ids.size() * hparams.n_embd;

    std::string layers;
    const char * sep = "";
    for (const auto id : target_layer_ids) {
        layers += sep;
        layers += std::to_string(id);
        sep = ", ";
    }
    LLAMA_LOG_INFO("%s: DFlash extract_layers = [%s]\n", __func__, layers.c_str());

    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT, hparams.dsv4_hc_mult, false);
    if (hparams.dsv4_hc_mult > 0) {
        load_hparams_dsv4(ml);
        backbone = hparams.dsv4_hc_single_pass ? BACKBONE_DSV41 : BACKBONE_DSV4;
    } else if (ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK, hparams.n_lora_kv, false)) {
        load_hparams_mistral_large4(ml);
        backbone = BACKBONE_MISTRAL_LARGE4;
    } else if (ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false) && hparams.n_swa > 0) {
        // optional interleaved sliding-window attention with per-layer pattern array.
        // DFlash has a single rope, so the SWA rope == main rope.
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);
        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    }

    ml.get_key_or_arr(LLM_KV_ATTENTION_ROPE_PATTERN, hparams.rope_pattern, hparams.n_layer(), false);

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_dflash::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_inp = hparams.n_embd_inp_enc();

    tok_embd        = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,       "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // reduced draft vocab (optional): d2t maps draft rows to target token ids
    int64_t n_vocab_draft = n_vocab;
    const struct ggml_tensor * d2t_meta = ml->get_tensor_meta("d2t");
    if (d2t_meta) {
        n_vocab_draft = d2t_meta->ne[0];
        d2t = create_tensor(tn(LLM_TENSOR_D2T), { n_vocab_draft }, 0);
        LLAMA_LOG_INFO("%s: DFlash using d2t mapping (draft_vocab_size = %lld)\n", __func__, (long long) n_vocab_draft);
    }

    switch (variant) {
        case VARIANT_DSPARK:   load_tensors_dspark(n_vocab_draft); break;
        case VARIANT_DFLASH2:  load_tensors_dflash2();             break;
        case VARIANT_LILICORR: load_tensors_lilicorr();            break;
        case VARIANT_DSPINE:   load_tensors_dspine();              break;
        case VARIANT_DFLASH:                                       break;
    }

    // drafts with per-layer context K/V (ctx_mix) have no shared fc
    const int fc_flags = ml->get_tensor_meta("blk.0.ctx_mix.weight") ? TENSOR_NOT_REQUIRED : 0;

    fc              = create_tensor(tn(LLM_TENSOR_FC,              "weight"), { n_embd_inp, n_embd }, fc_flags);
    fc_s            = create_tensor(tn(LLM_TENSOR_FC,              "scale"),  { 1 }, TENSOR_NOT_REQUIRED);
    fc_in_s         = create_tensor(tn(LLM_TENSOR_FC,              "input_scale"), { 1 }, TENSOR_NOT_REQUIRED);
    output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), { n_embd }, fc_flags); // encoder hidden_norm (after fc)
    output_norm     = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,    "weight"), { n_embd }, 0); // decoder final norm

    // optional: norm of the anchor embedding
    tok_norm        = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), { n_embd }, TENSOR_NOT_REQUIRED);

    // optional: reduced-vocab drafts ship their own lm head, full-vocab drafts can share the target's via ctx_other
    // a draft with its own embeddings + head references no target tensors and can run on devices the target does not use (e.g. -devd with a tensor-split target)
    output   = create_tensor(tn(LLM_TENSOR_OUTPUT,     "weight"), { n_embd, n_vocab_draft }, TENSOR_NOT_REQUIRED);

    switch (backbone) {
        case BACKBONE_BASE:           load_tensors_layers_base();           break;
        case BACKBONE_DSV4:
        case BACKBONE_DSV41:          load_tensors_layers_dsv4();           break;
        case BACKBONE_MISTRAL_LARGE4: load_tensors_layers_mistral_large4(); break;
    }
}

uint32_t llama_model_dflash::graph_nodes_per_tensor() const {
    if (hparams.dsv4_hc_mult > 0) {
        return 32;
    }
    if (variant == VARIANT_DSPINE) {
        // DSpine's per-layer code bridge and refinement are shape work rather than matmuls
        return 12u*std::max<uint32_t>(1u, hparams.dflash_block_size/8u);
    }
    if (hparams.dflash_selector_rank > 0) {
        // DFlash2's convolutions and selector are shape work rather than matmuls,
        // so they cost ~8.6 nodes per tensor against ~5.9 for a plain DFlash draft.
        // The selector lattice scales with the draft block size; 12u was calibrated
        // on block-8 drafters, so scale the budget for larger blocks (e.g. Ripple's 16).
        return 12u*std::max<uint32_t>(1u, hparams.dflash_block_size/8u);
    }
    return llama_model_base::graph_nodes_per_tensor();
}

// DSpine code bridge: a write after every layer, the last one refined over the top-k candidates
void llama_model_dflash::load_tensors_dspine() {
    LLAMA_LOAD_LOCALS;

    const struct ggml_tensor * cb_meta = ml->get_tensor_meta("dspine.codebook.weight");
    const struct ggml_tensor * gt_meta = ml->get_tensor_meta("dspine.gate.weight");
    if (!cb_meta || !gt_meta) {
        throw std::runtime_error("DSpine draft is missing its code bridge");
    }
    const int64_t code = cb_meta->ne[0];
    const int64_t rank = gt_meta->ne[1];

    dspine_codebook  = create_tensor(tn(LLM_TENSOR_DSPINE_CODEBOOK,     "weight"), { code, n_vocab }, 0);
    dspine_gate      = create_tensor(tn(LLM_TENSOR_DSPINE_BRIDGE_GATE,  "weight"), { code, rank }, 0);
    dspine_gate_b    = create_tensor(tn(LLM_TENSOR_DSPINE_BRIDGE_GATE,  "bias"),   { rank }, 0);
    dspine_pred_gate = create_tensor(tn(LLM_TENSOR_DSPINE_BRIDGE_PGATE, "weight"), { code, rank }, 0);
    dspine_prev      = create_tensor(tn(LLM_TENSOR_DSPINE_BRIDGE_PREV,  "weight"), { code, rank }, 0);
    for (int i = 0; i < n_layer; ++i) {
        layers[i].dspine_read  = create_tensor(tn(LLM_TENSOR_DSPINE_BRIDGE_READ,  "weight", i), { n_embd, code }, 0);
        layers[i].dspine_write = create_tensor(tn(LLM_TENSOR_DSPINE_BRIDGE_WRITE, "weight", i), { rank, n_embd }, 0);
    }

    LLAMA_LOG_INFO("%s: DSpine code bridge: code = %lld, rank = %lld, head top-k = %u, eps = %g\n", __func__,
            (long long) code, (long long) rank, hparams.dspine_head_top_k, hparams.dspine_bridge_eps);
}

void llama_model_dflash::load_tensors_layers_base() {
    LLAMA_LOAD_LOCALS;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), { n_embd, n_embd_k_gqa }, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), { n_embd, n_embd_v_gqa }, TENSOR_NOT_REQUIRED);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, 0);

        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), { n_embd }, TENSOR_NOT_REQUIRED);
        layer.ffn_post_norm  = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM,  "weight", i), { n_embd }, TENSOR_NOT_REQUIRED);
        layer.out_scale      = create_tensor(tn(LLM_TENSOR_LAYER_OUT_SCALE, "weight", i), { 1 }, TENSOR_NOT_REQUIRED);
        layer.rope_freqs     = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), { n_embd_head_k/2 }, TENSOR_NOT_REQUIRED | (i > 0 ? TENSOR_DUPLICATED : 0));

        // optional per-head attention sinks (e.g. Nemotron DSpark)
        layer.attn_sinks = create_tensor(tn(LLM_TENSOR_ATTN_SINKS, "weight", i), { n_head }, TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, 0);

        // optional: context K/V from this layer's own mix of the target features
        layer.dflash_ctx_mix = create_tensor(tn(LLM_TENSOR_DFLASH_CTX_MIX, "weight", i), { (int64_t) target_layer_ids.size() }, TENSOR_NOT_REQUIRED);
        if (layer.dflash_ctx_mix) {
            layer.dflash_ctx_norm        = create_tensor(tn(LLM_TENSOR_DFLASH_CTX_NORM,        "weight", i), { n_embd }, 0);
            layer.dflash_wk_ctx          = create_tensor(tn(LLM_TENSOR_DFLASH_ATTN_K_CTX,      "weight", i), { n_embd, n_embd_k_gqa }, 0);
            layer.dflash_wv_ctx          = create_tensor(tn(LLM_TENSOR_DFLASH_ATTN_V_CTX,      "weight", i), { n_embd, n_embd_v_gqa }, 0);
            layer.dflash_attn_k_ctx_norm = create_tensor(tn(LLM_TENSOR_DFLASH_ATTN_K_CTX_NORM, "weight", i), { n_embd_head_k }, 0);
        }
    }
}

template <>
ggml_tensor * llama_model_dflash::graph<true>::build_inp_embd_enc() const {
    const int64_t n_embd_inp = hparams.n_embd_inp_enc();
    auto inp_target = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, n_tokens);
    ggml_set_input(inp_target->embd);

    ggml_tensor * cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));

    return cur;
}

// DFlash Encoder: processes target model features through feature fusion layer
template <>
llama_model_dflash::graph<true>::graph(const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inp_embd_enc();

    cur = build_lora_mm(model.fc, cur, model.fc_s);
    cb(cur, "fc_out", -1);

    cur = build_norm(cur, model.output_norm_enc, NULL, LLM_NORM_RMS, -1);
    cb(cur, "enc_norm_out", -1);

    ggml_set_output(cur);
    res->t_h_nextn = cur;

    ggml_build_forward_expand(gf, cur);
}

// target features of an injection batch: host embd rows, or rows of the target's layer input device buffer
static ggml_tensor * build_inp_target_features(llm_graph_context & g, llm_graph_input_embd & inp, int64_t n_embd_inp) {
    if (g.ubatch.embd_row) {
        GGML_ASSERT(g.layer_inp_src != nullptr && g.layer_inp_src->ne[0] == n_embd_inp);

        inp.embd_rows = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, g.n_tokens);
        ggml_set_input(inp.embd_rows);

        return ggml_get_rows(g.ctx0, g.layer_inp_src, inp.embd_rows);
    }

    inp.embd = ggml_new_tensor_2d(g.ctx0, GGML_TYPE_F32, n_embd_inp, g.n_tokens);
    ggml_set_input(inp.embd);

    return inp.embd;
}

// injection batch: the target features fused through the encoder
ggml_tensor * llama_model_dflash::build_inject_inp(llm_graph_context & g, const llama_model & model) {
    const int64_t n_embd_inp = g.hparams.n_embd_inp_enc();

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    ggml_tensor * inp_target = build_inp_target_features(g, *inp, n_embd_inp);
    g.cb(inp_target, "inp_target_features", -1);

    g.res->add_input(std::move(inp));

    ggml_tensor * inp_g = g.build_lora_mm(model.fc, inp_target, model.fc_s);
    return g.build_norm(inp_g, model.output_norm_enc, nullptr, LLM_NORM_RMS, -1);
}

// tok_embd from the target model (shared via ctx_other) when the draft has none
ggml_tensor * llama_model_dflash::get_tok_embd(const llm_graph_context & g, const llama_model & model) {
    if (model.tok_embd) {
        return model.tok_embd;
    }
    GGML_ASSERT(g.cparams.ctx_other != nullptr);
    const auto * model_other = llama_get_model(g.cparams.ctx_other);

    GGML_ASSERT(model_other->tok_embd != nullptr && "DFlash draft requires the target model's token embeddings");
    return model_other->tok_embd;
}

// lm_head and its scale from the target model (shared via ctx_other) when the draft has none
std::pair<ggml_tensor *, ggml_tensor *> llama_model_dflash::get_lm_head(const llm_graph_context & g, const llama_model & model) {
    if (model.output) {
        return { model.output, model.output_s };
    }
    GGML_ASSERT(g.cparams.ctx_other != nullptr);
    const auto * model_other = llama_get_model(g.cparams.ctx_other);

    GGML_ASSERT(model_other->output != nullptr && "DFlash draft requires the target model's output projection");
    return { model_other->output, model_other->output_s };
}

bool llama_model_dflash::build_approx_candidates(llm_graph_context & g, const llama_model & model, ggml_tensor * cur, int64_t n_cand,
        ggml_tensor *& cand, ggml_tensor *& exact) {
    const auto [output, output_s] = get_lm_head(g, model);
    if (!model.dflash_output_approx || model.d2t || output_s != nullptr || model.dflash_output_approx->ne[1] != output->ne[1] ||
            n_cand >= output->ne[1]) {
        return false;
    }

    ggml_context * ctx0     = g.ctx0;
    const int64_t  n_embd   = cur->ne[0];
    const int64_t  n_tokens = cur->ne[1];
    const auto   & hparams  = g.hparams;

    ggml_tensor * approx = ggml_mul_mat(ctx0, model.dflash_output_approx, cur); // [n_vocab, n_tokens]
    g.cb(approx, "result_output_approx", -1);

    cand = ggml_top_k(ctx0, approx, n_cand);

    ggml_tensor * rows = ggml_get_rows(ctx0, output, ggml_reshape_1d(ctx0, cand, n_cand*n_tokens));
    rows  = ggml_reshape_3d(ctx0, rows, n_embd, n_cand, n_tokens);
    exact = ggml_mul_mat(ctx0, rows, ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens));
    exact = ggml_reshape_2d(ctx0, exact, n_cand, n_tokens);
    if (hparams.f_logit_scale != 0.0f) {
        exact = ggml_scale(ctx0, exact, hparams.f_logit_scale);
    }
    if (hparams.f_final_logit_softcapping > 0.0f) {
        exact = ggml_scale(ctx0, exact, 1.0f / hparams.f_final_logit_softcapping);
        exact = ggml_tanh(ctx0, exact);
        exact = ggml_scale(ctx0, exact, hparams.f_final_logit_softcapping);
    }
    g.cb(exact, "result_output_cand", -1);

    return true;
}

// DFlash decoder, dual-mode by batch type:
//   * embd batch  -> fused target features: project + inject K/V into the cache.
//   * token batch -> noise-block diffusion: attend over [committed, MASK...] to generate draft tokens
llama_model_dflash::graph_base::graph_base(const llm_graph_params & params) : llm_graph_context(params) {
    n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    inp_pos = build_inp_pos();

    // optional iSWA: pick the matching attention input
    use_iswa = hparams.swa_type != LLAMA_SWA_TYPE_NONE;
    if (use_iswa) {
        inp_attn_iswa = build_attn_inp_kv_iswa();
    } else {
        inp_attn = build_attn_inp_kv();
    }

    // drafts for M-RoPE targets use degenerate sections (temporal dim only)
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);
}

ggml_tensor * llama_model_dflash::graph_base::build_rope(ggml_tensor * cur, ggml_tensor * pos, int il) {
    if (!hparams.has_rope(il)) {
        return cur;
    }

    return rope_type == GGML_ROPE_TYPE_MROPE
        ? ggml_rope_multi(ctx0, cur, pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow)
        : ggml_rope_ext(ctx0, cur, pos, model.layers[0].rope_freqs,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
}

// write one layer's injected K/V to its (sub-)cache
void llama_model_dflash::graph_base::store_inject_kv(ggml_tensor * Kcur, ggml_tensor * Vcur, int il) {
    if (use_iswa) {
        // route each layer's K/V to its sub-cache: SWA layers -> sliding cache, full -> dense
        const bool    is_swa = hparams.is_swa(il);
        const auto  * kv     = is_swa ? inp_attn_iswa->mctx->get_swa() : inp_attn_iswa->mctx->get_base();
        ggml_tensor * k_idxs = is_swa ? inp_attn_iswa->get_k_idxs_swa() : inp_attn_iswa->get_k_idxs();
        ggml_tensor * v_idxs = is_swa ? inp_attn_iswa->get_v_idxs_swa() : inp_attn_iswa->get_v_idxs();
        // rotate K/V into the cache's rotated space
        ggml_tensor * k_rot  = is_swa ? inp_attn_iswa->self_k_rot_swa[il] : inp_attn_iswa->self_k_rot[il];
        ggml_tensor * v_rot  = is_swa ? inp_attn_iswa->self_v_rot_swa[il] : inp_attn_iswa->self_v_rot[il];
        if (k_rot) {
            Kcur = llama_mul_mat_hadamard(ctx0, Kcur, k_rot);
        }
        if (v_rot) {
            Vcur = llama_mul_mat_hadamard(ctx0, Vcur, v_rot);
        }
        ggml_build_forward_expand(gf, kv->cpy_k(ctx0, Kcur, k_idxs, il));
        ggml_build_forward_expand(gf, kv->cpy_v(ctx0, Vcur, v_idxs, il));
    } else {
        // rotate K/V into the cache's rotated space
        if (inp_attn->self_k_rot[il]) {
            Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp_attn->self_k_rot[il]);
        }
        if (inp_attn->self_v_rot[il]) {
            Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp_attn->self_v_rot[il]);
        }
        ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
        ggml_build_forward_expand(gf, inp_attn->mctx->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));
    }
}

// KV cache injection
ggml_tensor * llama_model_dflash::graph_base::build_inject() {
    // drafts without fc: each layer mixes the raw target features itself
    if (!model.fc) {
        build_inject_ctx_mix();
        return nullptr;
    }

    ggml_tensor * inp_g = build_inject_inp(*this, model);
    cb(inp_g, "inp_g_embeddings", -1);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * Kcur = build_lora_mm(layer.wk, inp_g, layer.wk_s);
        const bool shared_kv = layer.wv == nullptr;
        ggml_tensor * Vcur = shared_kv ? Kcur : build_lora_mm(layer.wv, inp_g, layer.wv_s);

        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
        if (shared_kv) {
            Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);
        }
        Kcur = build_rope(Kcur, inp_pos, il);
        cb(Kcur, "Kcur_injected", il);
        cb(Vcur, "Vcur_injected", il);

        store_inject_kv(Kcur, Vcur, il);
    }

    res->t_embd = inp_g;

    ggml_build_forward_expand(gf, inp_g);

    return inp_g;
}

// KV cache injection with per-layer context K/V (Inkling2 DSpark): rms_norm(sum_i ctx_mix[i] * target_i) * ctx_norm -> k_ctx / v_ctx
// also outputs f_rnn (the same mix for the markov head) as h_nextn
void llama_model_dflash::graph_base::build_inject_ctx_mix() {
    const int64_t n_embd_inp = hparams.n_embd_inp_enc();
    const int64_t n_target   = n_embd_inp / n_embd;

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);
    ggml_tensor * feats = build_inp_target_features(*this, *inp, n_embd_inp);
    res->add_input(std::move(inp));

    // [n_target, n_embd, n_tokens]: the mix is a matmul over the target taps
    feats = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, feats, n_embd, n_target, n_tokens), 1, 0, 2, 3));
    cb(feats, "inp_target_features", -1);

    const auto build_mix = [&](ggml_tensor * mix, ggml_tensor * norm, int il) {
        ggml_tensor * cur = ggml_reshape_2d(ctx0, ggml_mul_mat(ctx0, mix, feats), n_embd, n_tokens);
        return build_norm(cur, norm, nullptr, LLM_NORM_RMS, il);
    };

    // the driver keeps the f_rnn rows of the context positions
    if (model.dspark_markov_ctx_mix) {
        res->t_h_nextn = build_mix(model.dspark_markov_ctx_mix, model.dspark_markov_ctx_norm, -1);
        cb(res->t_h_nextn, "f_rnn", -1);
        ggml_build_forward_expand(gf, res->t_h_nextn);
    }

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        GGML_ASSERT(layer.dflash_ctx_mix && "DFlash draft without fc requires per-layer ctx_mix");

        ggml_tensor * cur = build_mix(layer.dflash_ctx_mix, layer.dflash_ctx_norm, il);

        ggml_tensor * Kcur = ggml_reshape_3d(ctx0, build_lora_mm(layer.dflash_wk_ctx, cur), n_embd_head, n_head_kv, n_tokens);
        ggml_tensor * Vcur = ggml_reshape_3d(ctx0, build_lora_mm(layer.dflash_wv_ctx, cur), n_embd_head, n_head_kv, n_tokens);

        Kcur = build_norm(Kcur, layer.dflash_attn_k_ctx_norm, NULL, LLM_NORM_RMS, il);
        Kcur = build_rope(Kcur, inp_pos, il);
        cb(Kcur, "Kcur_injected", il);
        cb(Vcur, "Vcur_injected", il);

        store_inject_kv(Kcur, Vcur, il);
    }
}

ggml_tensor * llama_model_dflash::graph_base::build_inp_tokens() {
    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_tokens);
    res->t_inp_tokens = inp_tokens;

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);
    inp->tokens = inp_tokens;
    res->add_input(std::move(inp));

    return inp_tokens;
}

// DSpark block [anchor, mask_1, ..., mask_k] per sequence: slot k > 0 uses mask row k - 1, the anchor uses its (normed) token embedding
ggml_tensor * llama_model_dflash::graph_base::build_inp_block(ggml_tensor * inp_tokens) {
    ggml_tensor * tok_embd = get_tok_embd(*this, model);

    const int64_t n_blocks     = ubatch.n_seqs_unq;
    const int64_t block_drafts = n_blocks > 0 ? n_tokens / n_blocks : 0;

    if (block_drafts == 0 || n_tokens % n_blocks != 0 || block_drafts > (int64_t) hparams.dflash_block_size) {
        // not a draft block (e.g. graph reserve)
        return ggml_get_rows(ctx0, tok_embd, inp_tokens);
    }

    ggml_tensor * anchors = ggml_view_2d(ctx0, inp_tokens, 1, n_blocks, (size_t) block_drafts * inp_tokens->nb[0], 0);
    anchors = ggml_get_rows(ctx0, tok_embd, ggml_cont_1d(ctx0, anchors, n_blocks));
    if (model.tok_norm) {
        anchors = build_norm(anchors, model.tok_norm, nullptr, LLM_NORM_RMS, -1);
    }

    ggml_tensor * cur = ggml_reshape_3d(ctx0, anchors, n_embd, 1, n_blocks);
    if (block_drafts > 1) {
        ggml_tensor * masks = ggml_view_3d(ctx0, model.dspark_mask_embd, n_embd, block_drafts - 1, 1,
                model.dspark_mask_embd->nb[1], model.dspark_mask_embd->nb[2], 0);
        masks = ggml_repeat_4d(ctx0, ggml_cast(ctx0, masks, GGML_TYPE_F32), n_embd, block_drafts - 1, n_blocks, 1);
        cur = ggml_concat(ctx0, cur, masks, 1);
    }

    return ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
}

// ---------------------------------------------------------------------------------------------------------
// DSpine (tencent/DSpine, arXiv:2609.36173): DFlash with causal block attention, a code bridge write after
// every layer and the serial refinement of the last write. Follows dspine_mlx/bridge.py (DSpineCodeBridge)
// and dspine_mlx/propose.py (RefinementTail) in f32; the reference rounds to bf16 at its rounding points.
// ---------------------------------------------------------------------------------------------------------

// pre-write state of the last layer's write, read again by the refinement (the reference "stash")
struct dspine_stash {
    ggml_tensor * raw = nullptr; // [n_embd, n_tokens] residual stream before the write
    ggml_tensor * rms = nullptr; // [1, n_tokens]      its rms (no eps)
    ggml_tensor * q   = nullptr; // [rank, n_tokens]   own-code gate projection Gq c_t (no bias)
};

// the anchor (row 0) of every block, [n_blocks]
static ggml_tensor * dspine_anchor_ids(ggml_context * ctx0, ggml_tensor * tokens, int64_t block, int64_t n_blocks) {
    return ggml_cont_1d(ctx0, ggml_view_2d(ctx0, tokens, 1, n_blocks, block * tokens->nb[0], 0), n_blocks);
}

// rows 1..block-1 of every block of t [ne0, block * n_blocks], as a strided view [ne0, 1, block - 1, n_blocks]
// that broadcasts over a candidate dim
static ggml_tensor * dspine_rows(ggml_context * ctx0, ggml_tensor * t, int64_t block, int64_t n_blocks) {
    return ggml_view_4d(ctx0, t, t->ne[0], 1, block - 1, n_blocks,
            t->nb[1], t->nb[1], block * t->nb[1], t->nb[1]);
}

// DSpineCodeBridge.__call__ after layer il, with hidden = the layer output (residual = None):
//   rms      = sqrt(mean(raw^2))
//   codes    = sqrt(C) * l2norm(R_il (raw / rms))         row 0: sqrt(C) * codebook[anchor]
//   q, s, v  = Gq c, Gp c, A c
//   message  = sigmoid(q_t + s_(t-1) + b) * v_(t-1)       row 0: none
//   out      = raw + rms * W_il message
// The read input is scale invariant (its codes are L2-normalized), so rms_norm stands in for raw / (rms + eps).
static ggml_tensor * build_dspine_write(llm_graph_context & g, const llama_model & model, ggml_tensor * raw,
        ggml_tensor * anchor_codes, int il, int64_t block, int64_t n_blocks, dspine_stash * stash) {
    ggml_context * ctx0 = g.ctx0;
    const auto & layer = model.layers[il];

    const int64_t n_embd   = raw->ne[0];
    const int64_t n_tokens = raw->ne[1];
    const int64_t code     = model.dspine_codebook->ne[0];
    const int64_t rank     = model.dspine_gate->ne[1];
    const float   eps      = g.hparams.dspine_bridge_eps;

    ggml_tensor * rms = ggml_sqrt(ctx0, ggml_scale(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, raw)), 1.0f/n_embd));
    g.cb(rms, "dspine_rms", il);

    ggml_tensor * codes = g.build_lora_mm(layer.dspine_read, ggml_rms_norm(ctx0, raw, eps));
    codes = ggml_scale(ctx0, ggml_l2_norm(ctx0, codes, eps), sqrtf((float) code));
    codes = ggml_reshape_3d(ctx0, codes, code, block, n_blocks);
    codes = ggml_concat(ctx0, anchor_codes,
            ggml_cont(ctx0, ggml_view_3d(ctx0, codes, code, block - 1, n_blocks, codes->nb[1], codes->nb[2], codes->nb[1])), 1);
    codes = ggml_reshape_2d(ctx0, codes, code, n_tokens);
    g.cb(codes, "dspine_codes", il);

    ggml_tensor * q = g.build_lora_mm(model.dspine_gate,      codes); // [rank, n_tokens]
    ggml_tensor * s = g.build_lora_mm(model.dspine_pred_gate, codes);
    ggml_tensor * v = g.build_lora_mm(model.dspine_prev,      codes);

    // rows 1..block-1 take the predecessor terms of rows 0..block-2
    const auto succ = [&](ggml_tensor * t) {
        return ggml_view_3d(ctx0, t, rank, block - 1, n_blocks, t->nb[1], block * t->nb[1], t->nb[1]);
    };
    const auto pred = [&](ggml_tensor * t) {
        return ggml_view_3d(ctx0, t, rank, block - 1, n_blocks, t->nb[1], block * t->nb[1], 0);
    };
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_add(ctx0, ggml_add(ctx0, succ(q), pred(s)), model.dspine_gate_b));
    ggml_tensor * msg  = ggml_mul(ctx0, gate, pred(v)); // [rank, block - 1, n_blocks]
    g.cb(msg, "dspine_message", il);

    ggml_tensor * delta = g.build_lora_mm(layer.dspine_write, ggml_reshape_2d(ctx0, msg, rank, (block - 1) * n_blocks));
    // the anchor row gets no write
    delta = ggml_pad_ext(ctx0, ggml_reshape_3d(ctx0, delta, n_embd, block - 1, n_blocks), 0, 0, 1, 0, 0, 0, 0, 0);
    delta = ggml_mul(ctx0, ggml_reshape_2d(ctx0, delta, n_embd, n_tokens), rms);

    if (stash) {
        stash->raw = raw;
        stash->rms = rms;
        stash->q   = q;
    }

    return ggml_add(ctx0, raw, delta);
}

// RefinementTail.sampled_table: for each row t >= 1 and each of its top-k candidates c (target head on the
// final hidden), redo the last write once per exact predecessor j -- the anchor for row 1, the candidates of
// row t-1 after that -- and score c:
//   e_j      = sqrt(C) * codebook[j]
//   refined  = raw_t + rms_t * W_last (sigmoid(q_t + Gp e_j + b) * (A e_j))
//   table    = head_c . (rms_norm(refined) * norm_w)
// Packed like the DFlash2 lattice so the drafter walks it the same way: per row
// [candidate ids (k), table (k x k, predecessor-major)], row 0 of a block is zero.
static void build_dspine_lattice(llm_graph_context & g, const llama_model & model, ggml_tensor * tokens,
        ggml_tensor * output, const dspine_stash & stash, int64_t block, int64_t n_blocks,
        ggml_tensor * approx_cand = nullptr, ggml_tensor * approx_logits = nullptr) {
    ggml_context * ctx0 = g.ctx0;
    auto         & res  = g.res;

    const int64_t n_embd = g.n_embd;
    const int64_t top_k  = g.hparams.dspine_head_top_k;
    const int64_t code   = model.dspine_codebook->ne[0];
    const int64_t rank   = model.dspine_gate->ne[1];
    const int64_t n_pos  = block - 1;

    // candidates of rows 1..block-1: the top k of the target head, or (approximate head) the top k of the exact
    // logits of the best n_cand by a Q2_K copy of it. ggml_top_k does not order its output; both are sets.
    ggml_tensor * cand_all = nullptr;
    if (approx_cand) {
        const int64_t n_cand = approx_cand->ne[0];
        ggml_tensor * pick = ggml_top_k(ctx0, approx_logits, top_k);   // positions within the n_cand candidates
        cand_all = ggml_reshape_2d(ctx0,
                ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, approx_cand, 1, n_cand, block * n_blocks), pick), top_k, block * n_blocks);
    } else {
        cand_all = ggml_top_k(ctx0, res->t_logits, top_k);
    }
    ggml_tensor * cand = ggml_reshape_3d(ctx0, cand_all, top_k, block, n_blocks);
    cand = ggml_cont(ctx0, ggml_view_3d(ctx0, cand, top_k, n_pos, n_blocks, cand->nb[1], cand->nb[2], cand->nb[1]));
    g.cb(cand, "dspine_candidates", -1);

    // predecessor codes [code, k, n_pos, n_blocks]
    ggml_tensor * e = ggml_get_rows(ctx0, model.dspine_codebook, dspine_anchor_ids(ctx0, tokens, block, n_blocks));
    e = ggml_repeat_4d(ctx0, ggml_reshape_4d(ctx0, e, code, 1, 1, n_blocks), code, top_k, 1, n_blocks);
    if (n_pos > 1) {
        ggml_tensor * prev_ids = ggml_cont(ctx0, ggml_view_3d(ctx0, cand, top_k, n_pos - 1, n_blocks, cand->nb[1], cand->nb[2], 0));
        ggml_tensor * prev = ggml_get_rows(ctx0, model.dspine_codebook, ggml_reshape_1d(ctx0, prev_ids, top_k * (n_pos - 1) * n_blocks));
        e = ggml_concat(ctx0, e, ggml_reshape_4d(ctx0, prev, code, top_k, n_pos - 1, n_blocks), 2);
    }
    e = ggml_scale(ctx0, ggml_reshape_2d(ctx0, e, code, top_k * n_pos * n_blocks), sqrtf((float) code));

    ggml_tensor * gp = ggml_reshape_4d(ctx0, g.build_lora_mm(model.dspine_pred_gate, e), rank, top_k, n_pos, n_blocks);
    ggml_tensor * pv = ggml_reshape_4d(ctx0, g.build_lora_mm(model.dspine_prev,      e), rank, top_k, n_pos, n_blocks);

    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_add(ctx0, ggml_add(ctx0, gp, dspine_rows(ctx0, stash.q, block, n_blocks)), model.dspine_gate_b));
    ggml_tensor * msg  = ggml_mul(ctx0, gate, pv);

    const auto & last = model.layers[g.n_layer - 1];
    ggml_tensor * delta = g.build_lora_mm(last.dspine_write, ggml_reshape_2d(ctx0, msg, rank, top_k * n_pos * n_blocks));
    delta = ggml_reshape_4d(ctx0, delta, n_embd, top_k, n_pos, n_blocks);

    ggml_tensor * refined = ggml_add(ctx0,
            ggml_mul(ctx0, delta, dspine_rows(ctx0, stash.rms, block, n_blocks)),
            dspine_rows(ctx0, stash.raw, block, n_blocks));
    refined = g.build_norm(refined, model.output_norm, NULL, LLM_NORM_RMS, -1);
    g.cb(refined, "dspine_refined", -1);

    // table[j][c] = head_c . refined_j, predecessor-major
    ggml_tensor * head = ggml_get_rows(ctx0, output, ggml_reshape_1d(ctx0, cand, top_k * n_pos * n_blocks));
    head = ggml_reshape_4d(ctx0, head, n_embd, top_k, n_pos, n_blocks);
    ggml_tensor * table = ggml_mul_mat(ctx0, head, refined); // [k (c), k (j), n_pos, n_blocks]
    g.cb(table, "dspine_table", -1);

    ggml_tensor * row = ggml_concat(ctx0,
            ggml_cast(ctx0, cand, GGML_TYPE_F32),
            ggml_reshape_3d(ctx0, table, top_k * top_k, n_pos, n_blocks), 0);
    // pad each row to n_embd and put the (unused) anchor row in front
    row = ggml_pad_ext(ctx0, row, 0, n_embd - top_k - top_k * top_k, 1, 0, 0, 0, 0, 0);
    ggml_tensor * packed = ggml_reshape_2d(ctx0, row, n_embd, block * n_blocks);
    g.cb(packed, "dspine_lattice", -1);

    res->t_h_nextn = packed;
    ggml_build_forward_expand(g.gf, packed);

    // only the lattice is read back; the full-vocabulary logits stay on the device
    res->t_logits = nullptr;
}

ggml_tensor * llama_model_dflash::graph_base::build_layers(ggml_tensor * inp_tokens) {
    const float kq_scale = hparams.f_attention_scale != 0.0f ? hparams.f_attention_scale : 1.0f/sqrtf(float(n_embd_head));

    ggml_tensor * inpL = model.dspark_mask_embd
        ? build_inp_block(inp_tokens)
        : ggml_get_rows(ctx0, get_tok_embd(*this, model), inp_tokens);
    if (hparams.f_embedding_scale != 0.0f) {
        inpL = ggml_scale(ctx0, inpL, hparams.f_embedding_scale);
    }
    cb(inpL, "inp_noise_embd", -1);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * noise_norm = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(noise_norm, "noise_norm", il);

        ggml_tensor * attn_dynamic = nullptr;
        if (layer.dflash_attn_conv_proj) {
            attn_dynamic = build_lora_mm(layer.dflash_attn_conv_proj, noise_norm);
            noise_norm = build_dflash2_conv(*this, noise_norm, attn_dynamic, layer.dflash_attn_conv_base, 0);
            cb(noise_norm, "attn_conv_in", il);
        }

        ggml_tensor * Qcur = build_lora_mm(layer.wq, noise_norm, layer.wq_s);
        ggml_tensor * Kcur = build_lora_mm(layer.wk, noise_norm, layer.wk_s);
        const bool shared_kv = layer.wv == nullptr;
        ggml_tensor * Vcur = shared_kv ? Kcur : build_lora_mm(layer.wv, noise_norm, layer.wv_s);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
        if (shared_kv) {
            Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);
        }

        Qcur = build_rope(Qcur, inp_pos, il);
        Kcur = build_rope(Kcur, inp_pos, il);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        // cache-aware, non-causal attention
        ggml_tensor * cur = use_iswa
            ? build_attn(inp_attn_iswa, layer.wo, NULL, layer.wo_s, Qcur, Kcur, Vcur, nullptr, layer.attn_sinks, nullptr, kq_scale, il)
            : build_attn(inp_attn,      layer.wo, NULL, layer.wo_s, Qcur, Kcur, Vcur, nullptr, layer.attn_sinks, nullptr, kq_scale, il);

        if (hparams.f_attn_value_scale != 0.0f) {
            cur = ggml_scale(ctx0, cur, hparams.f_attn_value_scale);
            cb(cur, "attn_out_scaled", il);
        }

        if (attn_dynamic) {
            cur = build_dflash2_conv(*this, cur, attn_dynamic, layer.dflash_attn_conv_base, 1);
            cb(cur, "attn_conv_out", il);
        }

        if (layer.attn_post_norm) {
            cur = build_norm(cur, layer.attn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_post_norm", il);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * ffn_dynamic = nullptr;
        if (layer.dflash_ffn_conv_proj) {
            ffn_dynamic = build_lora_mm(layer.dflash_ffn_conv_proj, cur);
            cur = build_dflash2_conv(*this, cur, ffn_dynamic, layer.dflash_ffn_conv_base, 0);
            cb(cur, "ffn_conv_in", il);
        }

        cur = build_ffn(cur,
                layer.ffn_up,   NULL, layer.ffn_up_s,
                layer.ffn_gate, NULL, layer.ffn_gate_s,
                layer.ffn_down, NULL, layer.ffn_down_s,
                NULL,
                hparams.llm_ffn_op, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        if (ffn_dynamic) {
            cur = build_dflash2_conv(*this, cur, ffn_dynamic, layer.dflash_ffn_conv_base, 1);
            cb(cur, "ffn_conv_out", il);
        }

        if (layer.ffn_post_norm) {
            cur = build_norm(cur, layer.ffn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "ffn_post_norm", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        if (layer.out_scale) {
            cur = ggml_mul(ctx0, cur, layer.out_scale);
        }
        if (post_layer) {
            cur = post_layer(cur, il);
        }
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    res->t_embd = cur;

    return cur;
}

ggml_tensor * llama_model_dflash::graph_base::build_logits(ggml_tensor * cur) {
    const auto [output, output_s] = get_lm_head(*this, model);
    cur = build_lora_mm(output, cur, output_s);

    // a padded lm head (e.g. shared from the target): drop the rows past the vocabulary
    if (!model.d2t && cur->ne[0] > (int64_t) model.vocab.n_tokens()) {
        cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, model.vocab.n_tokens(), cur->ne[1], cur->nb[1], 0));
    }

    // DFlash2 feeds these logits to the selector, so they need the target's output
    // transforms; DFlash1 and DSpark read them through the sampler instead
    if (model.dflash_selector_hidden) {
        if (hparams.f_logit_scale != 0.0f) {
            cur = ggml_scale(ctx0, cur, hparams.f_logit_scale);
        }
        if (hparams.f_final_logit_softcapping > 0.0f) {
            cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
            cur = ggml_tanh(ctx0, cur);
            cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
        }
    }

    // reduced-draft-vocab exports: scatter the draft logits to the target vocabulary via d2t
    if (model.d2t) {
        const int64_t n_draft_vocab = cur->ne[0];
        const int64_t n_outputs     = cur->ne[1];
        const int64_t n_vocab       = (int64_t) model.vocab.n_tokens();

        GGML_ASSERT(model.d2t->type == GGML_TYPE_I64);
        GGML_ASSERT(model.d2t->ne[0] == n_draft_vocab);

        ggml_tensor * logits = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab, n_outputs), -INFINITY);
        cur = ggml_set_rows(ctx0, logits,
                ggml_reshape_3d(ctx0, cur,       1,             n_draft_vocab, n_outputs),
                ggml_reshape_3d(ctx0, model.d2t, n_draft_vocab, 1,             1));
        cur = ggml_reshape_2d(ctx0, cur, n_vocab, n_outputs);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    return cur;
}

llama_model_dflash::graph_dflash::graph_dflash(const llm_graph_params & params) : graph_base(params) {
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.embd || ubatch.embd_row) {
        build_inject();
        return;
    }

    ggml_tensor * inp_tokens = build_inp_tokens();
    ggml_tensor * cur        = build_layers(inp_tokens);

    build_logits(cur);
}

// DSpine decoder: blocks of [anchor, mask...], one per sequence, a code bridge write after every layer, and the
// refinement lattice of the last write over the top-k candidates (packed like DFlash2's)
llama_model_dflash::graph_dspine::graph_dspine(const llm_graph_params & params) : graph_base(params) {
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.embd || ubatch.embd_row) {
        build_inject();
        return;
    }

    ggml_tensor * inp_tokens = build_inp_tokens();

    const int64_t n_blocks = ubatch.n_seqs_unq;
    const int64_t block    = n_blocks > 0 && n_tokens % n_blocks == 0 ? n_tokens / n_blocks : 0;

    // the anchor codes are shared by every layer's write
    dspine_stash stash;
    if (block >= 2) {
        const int64_t code = model.dspine_codebook->ne[0];
        ggml_tensor * anchor_codes = ggml_get_rows(ctx0, model.dspine_codebook, dspine_anchor_ids(ctx0, inp_tokens, block, n_blocks));
        anchor_codes = ggml_scale(ctx0, ggml_reshape_3d(ctx0, anchor_codes, code, 1, n_blocks), sqrtf((float) code));
        cb(anchor_codes, "dspine_anchor_codes", -1);

        post_layer = [&, anchor_codes](ggml_tensor * cur, int il) {
            return build_dspine_write(*this, model, cur, anchor_codes, il, block, n_blocks, il == n_layer - 1 ? &stash : nullptr);
        };
    }

    ggml_tensor * cur = build_layers(inp_tokens);
    post_layer = nullptr;

    // the refinement needs a whole block per sequence, within the trained block size
    const bool lattice = stash.raw && block <= (int64_t) hparams.dflash_block_size;

    ggml_tensor * output = get_lm_head(*this, model).first;

    // the approximate head picks the candidates of the lattice (the walk only reads the lattice). 128: on Qwen3.8-27B
    // the exact top 16 fall outside the approximate top 128 for 0.05% of the candidates (only exact ranks 9..16;
    // 64: 1%, ranks 5..16), at nearly the cost of 64
    const int64_t n_cand = 128;
    ggml_tensor * cand  = nullptr;
    ggml_tensor * exact = nullptr;
    if (lattice && n_cand >= (int64_t) hparams.dspine_head_top_k && hparams.f_logit_scale == 0.0f &&
            hparams.f_final_logit_softcapping <= 0.0f && build_approx_candidates(*this, model, cur, n_cand, cand, exact)) {
        build_dspine_lattice(*this, model, inp_tokens, output, stash, block, n_blocks, cand, exact);
        return;
    }

    build_logits(cur);

    if (lattice) {
        build_dspine_lattice(*this, model, inp_tokens, output, stash, block, n_blocks);
    }
}

std::unique_ptr<llm_graph_context> llama_model_dflash::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            switch (backbone) {
                case BACKBONE_BASE:           break;
                case BACKBONE_DSV4:           return std::make_unique<graph_dsv4>(params);
                case BACKBONE_DSV41:          return std::make_unique<graph_dsv41>(params);
                case BACKBONE_MISTRAL_LARGE4: return std::make_unique<graph_mistral_large4>(params);
            }
            switch (variant) {
                case VARIANT_DFLASH:   return std::make_unique<graph_dflash>(params);
                case VARIANT_DFLASH2:  return std::make_unique<graph_dflash2>(params);
                case VARIANT_DSPARK:   return std::make_unique<graph_dspark>(params);
                case VARIANT_LILICORR: return std::make_unique<graph_lilicorr>(params);
                case VARIANT_DSPINE:   return std::make_unique<graph_dspine>(params);
            }
            GGML_ABORT("invalid DFlash variant");
        default:
            GGML_ABORT("invalid graph type");
    };
}
