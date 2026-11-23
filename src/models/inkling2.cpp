#include "models.h"
#include "../llama-memory-hybrid-iswa.h"

#include <algorithm>
#include <vector>

// the fused-mask attention: about the max elements of its per-head F16 mask per call (2 GiB; chunks are whole multiples
// of 64 rows), and the max query chunks per layer
static constexpr int64_t n_kq_mask_max  = int64_t(1) << 30;
static constexpr int64_t n_kq_chunk_max = 64;

static bool is_f32(const ggml_tensor * t) {
    return t == nullptr || t->type == GGML_TYPE_F32;
}

void llama_model_inkling2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_QK_CLIP,           hparams.f_qk_clip);
    ml.get_key(LLM_KV_ATTN_RES_BLOCK_SIZE,         hparams.attn_res_block_size);
    ml.get_key(LLM_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);

    // relative-position bias, and the log-scaled positions of the full-attention layers
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_LENGTH, hparams.n_relative);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_EXTENT, hparams.n_relative_extent);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_CLIP,   hparams.f_relative_clip);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_FLOOR, hparams.n_log_scale_floor);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_ALPHA, hparams.f_log_scale_alpha);

    // the embedding and output tensors are padded past the vocabulary
    ml.get_key(LLM_KV_VOCAB_SIZE,          hparams.n_vocab_padded);
    ml.get_key(LLM_KV_VOCAB_SIZE_UNPADDED, hparams.n_vocab_unpadded);

    // MoE
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);

    ml.get_key(LLM_KV_ACTIVATION_SITU_BETA,        hparams.situ_beta);
    ml.get_key(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, hparams.situ_linear_beta);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);

    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);

    // sliding window
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,         hparams.n_swa);
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);

    uint32_t n_pattern = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern);

    if (n_pattern != hparams.n_layer() || !hparams.n_relative || !hparams.n_relative_extent || !hparams.n_swa ||
        !hparams.attn_res_block_size || !hparams.n_log_scale_floor || hparams.n_shortconv_l_cache < 2 ||
        hparams.n_shortconv_l_cache > 64 || hparams.n_relative_extent > INT32_MAX || hparams.n_swa > INT32_MAX ||
        !hparams.n_vocab_unpadded || hparams.n_vocab_unpadded > hparams.n_vocab_padded) {
        throw std::runtime_error("inkling2: invalid dimensions, vocabulary, or layer pattern");
    }

    for (float value : {hparams.f_norm_rms_eps, hparams.f_qk_clip, hparams.f_relative_clip, hparams.situ_beta, hparams.situ_linear_beta}) {
        if (!std::isfinite(value) || value <= 0.0f) {
            throw std::runtime_error("inkling2: normalization, clipping and scaling parameters must be finite and positive");
        }
    }

    if (!std::isfinite(hparams.f_log_scale_alpha) || hparams.f_log_scale_alpha < 0.0f) {
        throw std::runtime_error("inkling2: invalid logarithmic scale");
    }

    hparams.swa_type                = LLAMA_SWA_TYPE_STANDARD;
    hparams.expert_router_f32       = true;
    hparams.expert_weights_norm_eps = 1e-20f;

    // no RoPE, the relative-position bias carries the positions
    hparams.rope_pattern.fill(false);

    // the convolution state keeps K, V, the attention output and the FFN output of the widest layer
    uint64_t n_channels_max = 0;

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (!hparams.n_head_kv(il) || hparams.n_head(il) % hparams.n_head_kv(il) ||
            hparams.n_embd_head_k(il) != hparams.n_embd_head_v(il) || !hparams.n_expert_used(il) ||
            hparams.n_expert_used(il) > hparams.n_expert || !hparams.n_ff_exp(il)) {
            throw std::runtime_error("inkling2: invalid attention or expert dimensions");
        }

        n_channels_max = std::max<uint64_t>(n_channels_max, 2ull * hparams.n_embd_k_gqa(il) + 2ull * hparams.n_embd);
    }

    if (n_channels_max * (hparams.n_shortconv_l_cache - 1) > UINT32_MAX) {
        throw std::runtime_error("inkling2: convolution state is too large");
    }

    hparams.n_sconv_channels = n_channels_max;

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_inkling2::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    if (hparams.n_vocab_unpadded != vocab.n_tokens()) {
        throw std::runtime_error("inkling2: tokenizer size must equal the unpadded vocabulary size");
    }

    const int64_t n_vocab_padded = hparams.n_vocab_padded;
    const int64_t n_rel          = hparams.n_relative;
    const int64_t n_conv         = hparams.n_shortconv_l_cache;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"),    {n_embd, n_vocab_padded}, 0);
    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);

    // output
    output           = create_tensor(tn(LLM_TENSOR_OUTPUT,           "weight"), {n_embd, n_vocab_padded}, 0);
    prec_policy.set_acc(output, GGML_PREC_F32);
    output_norm      = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,      "weight"), {n_embd}, 0);
    output_res_query = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_QUERY, "weight"), {n_embd}, 0);

    for (int i = 0; i < n_layer_phys(); ++i) {
        auto & layer = layers[i];

        const int64_t n_head      = hparams.n_head(i);
        const int64_t n_head_kv   = hparams.n_head_kv(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);
        const int64_t n_ff_exp    = hparams.n_ff_exp(i);
        const int64_t n_extent    = hparams.is_swa(i) ? hparams.n_swa : hparams.n_relative_extent;

        // attention; the r9700 GGUF keeps the rows of q, gate, rel, k and v in one weight, and of the k and v convolutions in another
        const bool merged = ml.get_tensor_meta(tn(LLM_TENSOR_ATTN_QGRKV, "weight", i).str().c_str()) != nullptr;
        if (merged) {
            layer.attn_qgrkv = create_tensor(tn(LLM_TENSOR_ATTN_QGRKV, "weight", i),
                    {n_embd, 2*n_embd_head*n_head + n_rel*n_head + 2*n_embd_head*n_head_kv}, 0);
        } else {
            layer.wq        = create_tensor(tn(LLM_TENSOR_ATTN_Q,    "weight", i), {n_embd, n_embd_head*n_head}, 0);
            layer.wk        = create_tensor(tn(LLM_TENSOR_ATTN_K,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
            layer.wv        = create_tensor(tn(LLM_TENSOR_ATTN_V,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
            layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_embd_head*n_head}, 0);
        }
        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", i), {n_embd_head*n_head, n_embd}, 0);

        layer.attn_q_clip    = create_tensor(tn(LLM_TENSOR_ATTN_Q_CLIP,    "weight", i), {n_embd_head, n_head}, 0);
        layer.attn_sink_proj = create_tensor(tn(LLM_TENSOR_ATTN_SINK_PROJ, "weight", i), {n_embd, n_head}, 0);

        // relative-position bias
        if (!merged) {
            layer.attn_rel  = create_tensor(tn(LLM_TENSOR_ATTN_REL,      "weight", i), {n_embd, n_rel*n_head}, 0);
        }
        layer.attn_rel_bias = create_tensor(tn(LLM_TENSOR_ATTN_REL_BIAS, "weight", i), {n_rel, n_extent}, 0);
        layer.attn_rel_clip = create_tensor(tn(LLM_TENSOR_ATTN_REL_CLIP, "weight", i), {n_rel, n_head}, 0);

        // short convolutions
        if (merged) {
            layer.attn_kv_conv = create_tensor(tn(LLM_TENSOR_ATTN_KV_CONV, "weight", i), {n_conv, 2*n_embd_head*n_head_kv}, 0);
        } else {
            layer.attn_k_conv  = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV,  "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
            layer.attn_v_conv  = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV,  "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        }
        layer.attn_post_conv = create_tensor(tn(LLM_TENSOR_ATTN_POST_CONV, "weight", i), {n_conv, n_embd}, 0);
        layer.ffn_post_conv  = create_tensor(tn(LLM_TENSOR_FFN_POST_CONV,  "weight", i), {n_conv, n_embd}, 0);

        if (hparams.is_swa(i)) {
            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        } else {
            // full-attention layers: separate Q, K and V input norms, and input-dependent taps for the K and V convolutions
            layer.attn_q_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_K_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_v_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_V_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
            layer.attn_v_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
        }

        // block attention residuals
        if (i > 0) {
            layer.attn_res_query = create_tensor(tn(LLM_TENSOR_ATTN_RES_QUERY, "weight", i), {n_embd, hparams.is_swa(i) ? 1 : 3}, 0);
        }
        layer.ffn_res_query = create_tensor(tn(LLM_TENSOR_FFN_RES_QUERY, "weight", i), {n_embd}, 0);

        // MoE
        layer.ffn_norm         = create_tensor(tn(LLM_TENSOR_FFN_NORM,         "weight", i), {n_embd}, 0);
        layer.ffn_gate_inp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,     "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b  = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,  "bias",   i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, 2*n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,    "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_inkling2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

// scale each row down to an L2 norm of at most threshold, after the optional per-channel scale; post: optional per-row multiplier
ggml_tensor * llama_model_inkling2::graph::clip(ggml_tensor * cur, ggml_tensor * scale, float threshold, ggml_tensor * post) {
    if (is_f32(scale) && is_f32(post)) {
        return ggml_clip_norm(ctx0, cur, scale, post, threshold, 1e-6f);
    }

    if (scale) {
        cur = ggml_mul(ctx0, cur, scale);
    }

    ggml_tensor * norm = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, cur)));
    norm = ggml_scale_bias(ctx0, norm, 1.0f, 1e-6f);

    ggml_tensor * factor = ggml_div(ctx0, ggml_fill(ctx0, norm, threshold), norm);

    cur = ggml_mul(ctx0, cur, ggml_clamp(ctx0, factor, 0.0f, 1.0f));

    return post ? ggml_mul(ctx0, cur, post) : cur;
}

// attention over the depth: mix the earlier block outputs and cur with learned queries
ggml_tensor * llama_model_inkling2::graph::mix(const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur, ggml_tensor * query) {
    if (blocks.size() + 1 <= 9 && query->ne[1] <= 4) {
        std::vector<ggml_tensor *> states;
        states.reserve(blocks.size() + 1);
        for (ggml_tensor * block : blocks) {
            states.push_back(ggml_reshape_2d(ctx0, block, n_embd, n_tokens));
        }
        states.push_back(ggml_reshape_2d(ctx0, cur, n_embd, n_tokens));

        ggml_tensor * q = ggml_n_dims(query) == 1 ? ggml_reshape_2d(ctx0, query, n_embd, 1) : query;
        return ggml_res_mix(ctx0, q, states.data(), (int) states.size(), 1e-6f, 1.0f/sqrtf(n_embd));
    }

    ggml_tensor * states = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
    for (ggml_tensor * block : blocks) {
        states = ggml_concat(ctx0, ggml_reshape_3d(ctx0, block, n_embd, 1, n_tokens), states, 1);
    }
    states = ggml_rms_norm(ctx0, states, 1e-6f);

    ggml_tensor * queries = ggml_repeat(ctx0, query, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, query->ne[1], n_tokens));

    ggml_tensor * scores = ggml_mul_mat(ctx0, states, queries);
    scores = ggml_soft_max_ext(ctx0, scores, nullptr, 1.0f/sqrtf(n_embd), 0.0f);

    return ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, states)), scores);
}

ggml_tensor * llama_model_inkling2::graph::mix_row(ggml_tensor * mixed, int64_t j) {
    if (mixed->op == GGML_OP_RES_MIX) {
        return ggml_view_2d(ctx0, mixed, n_embd, n_tokens, mixed->nb[1], j*mixed->nb[2]);
    }
    return ggml_cont(ctx0, ggml_view_2d(ctx0, mixed, n_embd, n_tokens, mixed->nb[2], j*mixed->nb[1]));
}

// causal short convolution with a residual; dynamic adds input-dependent taps per head
// the last n_conv - 1 inputs of each sequence live in the recurrent state at channel offset `offset`
ggml_tensor * llama_model_inkling2::graph::convolve(
        ggml_tensor * cur,
        ggml_tensor * weight,
        ggml_tensor * dynamic,
        ggml_tensor * state,
        ggml_tensor * history,
        llm_graph_input_rs * inp,
        int64_t offset,
        ggml_tensor * residual) {
    const int64_t n_conv       = hparams.n_shortconv_l_cache;
    const int64_t n_channels   = cur->ne[0];
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_seqs       = ubatch.n_seqs;
    const float   scale        = 1.0f/sqrtf((dynamic ? 2 : 1)*n_conv + 1);

    ggml_tensor * past = ggml_view_3d(ctx0, history, n_conv - 1, n_channels, n_seqs, history->nb[1], history->nb[2], offset*history->nb[1]);

    // the fused op stores the new inputs through a view of the state that is not its result, so it has to run where the
    // state is: the scheduler places it by its weight, and hands it a copy of a state elsewhere (-nkvo, -ot); weight
    // may be a view of a weight, which has no buffer yet
    const ggml_tensor * w_root = weight->view_src ? weight->view_src : weight;
    const ggml_tensor * s_root = state->view_src  ? state->view_src  : state;
    const bool state_with_op = s_root->buffer && w_root->buffer &&
        (ggml_backend_buffer_is_host(s_root->buffer) ? ggml_backend_buffer_is_host(w_root->buffer) :
                                                       ggml_backend_buffer_get_type(s_root->buffer) == ggml_backend_buffer_get_type(w_root->buffer));
    if (cparams.n_rs_seq == 0 && state_with_op) {
        // reads the past inputs from the history and stores the new ones in the cell, as build_conv_state_store()
        const size_t row_size = state->nb[1];
        ggml_tensor * cell = ggml_view_2d(ctx0, state, (n_conv - 1)*n_channels, n_seqs, row_size,
                inp->mctx->get_head()*row_size + offset*(n_conv - 1)*ggml_element_size(state));
        ggml_tensor * out = ggml_short_conv_rs_add(ctx0, ggml_is_matrix(cur) ? cur : ggml_reshape_2d(ctx0, cur, n_channels, n_tokens), past, weight, dynamic, cell,
                residual, scale);
        // (on a backend without the op the scheduler would run it on a copy of the state)
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(s_root->buffer));
        if (dev == nullptr || ggml_backend_dev_supports_op(dev, out)) {
            return ggml_reshape_2d(ctx0, out, n_channels, n_tokens);
        }
    }

    ggml_tensor * x = ggml_transpose(ctx0, ggml_view_3d(ctx0, cur, n_channels, n_seq_tokens, n_seqs, cur->nb[1], n_seq_tokens*cur->nb[1], 0));
    x = ggml_concat(ctx0, past, x, 0);

    build_conv_state_store(inp, state, x, n_conv - 1, offset*(n_conv - 1)*ggml_element_size(state));

    ggml_tensor * out = ggml_reshape_2d(ctx0, ggml_short_conv(ctx0, x, weight, dynamic, scale), n_channels, n_tokens);
    return residual ? ggml_add(ctx0, residual, out) : out;
}

// mix a cross-pass message into a looped layer input: (1 - forget)*cur + gain*message, both gated on cur and message
ggml_tensor * llama_model_inkling2::graph::control(ggml_tensor * cur, ggml_tensor * message, const loop_layer & loop, int ctrl) {
    // matrix-vector form for short batches (decode, speculative verification)
    if (n_tokens <= 16 && loras->empty() &&
        is_f32(loop.ctrl_down[ctrl]) && is_f32(loop.ctrl_gain[ctrl]) && is_f32(loop.ctrl_forget[ctrl])) {
        ggml_tensor * h = ggml_loop_down(ctx0, cur, message, loop.ctrl_down[ctrl], 1e-6f);
        return ggml_loop_up(ctx0, cur, message, h, loop.ctrl_gain[ctrl], loop.ctrl_forget[ctrl]);
    }

    ggml_tensor * gate = ggml_concat(ctx0, ggml_rms_norm(ctx0, cur, 1e-6f), ggml_rms_norm(ctx0, message, 1e-6f), 0);
    gate = build_lora_mm(loop.ctrl_down[ctrl], gate);

    return ggml_loop_gate(ctx0, cur, message, build_lora_mm(loop.ctrl_gain[ctrl], gate), build_lora_mm(loop.ctrl_forget[ctrl], gate));
}

ggml_tensor * llama_model_inkling2::graph::build_inputs() {
    inp  = build_inp_mem_hybrid_iswa_rel(hparams.n_relative_extent, hparams.n_swa);
    attn = inp->get_attn();
    recr = inp->get_recr();

    inp_pos     = build_inp_pos();
    inp_out_ids = build_inp_out_ids();

    // log-scaled positions: 1 + alpha*log(max(1, (pos + 1)/floor)), multiplies q and r of the full-attention layers
    tau = ggml_scale_bias(ctx0, ggml_cast(ctx0, inp_pos, GGML_TYPE_F32), 1.0f, 1.0f);
    tau = ggml_log(ctx0, ggml_clamp(ctx0, ggml_scale(ctx0, tau, 1.0f/hparams.n_log_scale_floor), 1.0f, INFINITY));
    tau = ggml_reshape_3d(ctx0, ggml_scale_bias(ctx0, tau, hparams.f_log_scale_alpha, 1.0f), 1, 1, n_tokens);

    ggml_tensor * cur = build_inp_embd(model.tok_embd);

    // important: do not normalize raw embeddings input (i.e. encoded image embeddings)
    if (ubatch.token) {
        cur = build_norm(cur, model.tok_norm, nullptr, LLM_NORM_RMS, -1);
    }

    return cur;
}

ggml_tensor * llama_model_inkling2::graph::build_layer(int il, int ip, ggml_tensor * cur, std::vector<ggml_tensor *> & blocks,
        ggml_tensor * messages, const loop_layer * loop) {
    GGML_ASSERT(!messages == !loop);

    const auto & layer = model.layers[il];

    // [ne0*ne1, n_tokens] as [ne0, ne1, n_tokens]; t may be a row view of a wider matmul result
    auto heads = [&](ggml_tensor * t, int64_t ne0, int64_t ne1) {
        return ggml_is_contiguous(t) ? ggml_reshape_3d(ctx0, t, ne0, ne1, n_tokens) :
            ggml_view_3d(ctx0, t, ne0, ne1, n_tokens, ne0*ggml_element_size(t), t->nb[1], 0);
    };
    const int64_t n_conv_k = hparams.n_shortconv_l_cache;

    const bool    is_swa       = hparams.is_swa(il);
    const int64_t n_head       = hparams.n_head(il);
    const int64_t n_head_kv    = hparams.n_head_kv(il);
    const int64_t n_embd_head  = hparams.n_embd_head_k(il);
    const int64_t n_embd_k_gqa = n_embd_head*n_head_kv;

    // the message of controller ctrl, [n_embd, n_tokens]
    auto message = [&](int ctrl) {
        return mix_row(messages, ctrl);
    };

    ggml_tensor * state   = recr->mctx->get_r_l(il);
    ggml_tensor * history = build_rs(recr, state, hparams.n_embd_r(), ubatch.n_seqs);
    history = ggml_reshape_3d(ctx0, history, hparams.n_shortconv_l_cache - 1, hparams.n_sconv_channels, ubatch.n_seqs);

    ggml_tensor * inpSA = cur;

    // attention inputs: SWA layers mix one, full-attention layers one each for Q, K and V
    ggml_tensor * inputs[3] = {cur, cur, cur};
    if (layer.attn_res_query) {
        ggml_tensor * mixed = mix(blocks, cur, layer.attn_res_query);
        for (int j = 0; j < 3; ++j) {
            inputs[j] = mix_row(mixed, is_swa ? 0 : j);
        }
    }

    if (messages) {
        // the messages make the Q, K and V inputs of a SWA layer differ
        for (int j = 0; j < 3; ++j) {
            inputs[j] = control(inputs[j], message(LOOP_CTRL_Q + j), *loop, LOOP_CTRL_Q + j);
        }
    }

    if (is_swa && messages) {
        for (int j = 0; j < 3; ++j) {
            inputs[j] = build_norm(inputs[j], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        }
    } else if (is_swa) {
        inputs[0] = build_norm(inputs[0], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = inputs[2] = inputs[0];
    } else {
        inputs[0] = build_norm(inputs[0], layer.attn_q_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = build_norm(inputs[1], layer.attn_k_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[2] = build_norm(inputs[2], layer.attn_v_inp_norm, nullptr, LLM_NORM_RMS, il);
    }

    // self-attention
    {
        // the projections; with the r9700 GGUF the weights of one input are one matmul
        const int64_t n_q_rows = n_embd_head*n_head;
        const int64_t n_r_rows = hparams.n_relative*n_head;

        ggml_tensor * Qcur = nullptr;
        ggml_tensor * Gcur = nullptr; // the output gate
        ggml_tensor * Rcur = nullptr;
        ggml_tensor * Kcur = nullptr;
        ggml_tensor * Vcur = nullptr;
        ggml_tensor * KVcur = nullptr; // K and V rows next to each other, when one matmul computes both

        // rows [r0, r0 + nr) of a matmul result, as [nr, n_tokens]
        auto out_rows = [&](ggml_tensor * t, int64_t r0, int64_t nr) {
            return ggml_view_2d(ctx0, t, nr, n_tokens, t->nb[1], r0*ggml_element_size(t));
        };
        if (layer.attn_qgrkv) {
            ggml_tensor * w = layer.attn_qgrkv;
            auto w_rows = [&](int64_t r0, int64_t nr) {
                return ggml_view_2d(ctx0, w, w->ne[0], nr, w->nb[1], r0*w->nb[1]);
            };
            const int64_t o_k = 2*n_q_rows + n_r_rows;
            const int64_t o_v = o_k + n_embd_k_gqa;
            ggml_tensor * qgr = nullptr;
            if (inputs[1] == inputs[0] && inputs[2] == inputs[0]) {
                qgr   = build_lora_mm(w, inputs[0]);
                Kcur  = out_rows(qgr, o_k, n_embd_k_gqa);
                Vcur  = out_rows(qgr, o_v, n_embd_k_gqa);
                KVcur = out_rows(qgr, o_k, 2*n_embd_k_gqa);
            } else {
                qgr  = build_lora_mm(w_rows(0, o_k), inputs[0]);
                Kcur = build_lora_mm(w_rows(o_k, n_embd_k_gqa), inputs[1]);
                Vcur = build_lora_mm(w_rows(o_v, n_embd_k_gqa), inputs[2]);
            }
            Qcur = out_rows(qgr, 0,          n_q_rows);
            Gcur = out_rows(qgr, n_q_rows,   n_q_rows);
            Rcur = out_rows(qgr, 2*n_q_rows, n_r_rows);
        } else {
            Qcur = build_lora_mm(layer.wq,        inputs[0]);
            Gcur = build_lora_mm(layer.wqkv_gate, inputs[0]);
            Rcur = build_lora_mm(layer.attn_rel,  inputs[0]);
            Kcur = build_lora_mm(layer.wk,        inputs[1]);
            Vcur = build_lora_mm(layer.wv,        inputs[2]);
        }

        // the attention sinks read the attention input only: put them next to the projections in the graph, so that the
        // backends compute them together (the mask that reads them then needs no extra sync)
        ggml_tensor * sinks = build_lora_mm(layer.attn_sink_proj, inputs[0]);
        // (only with one attention input: the separate inputs of the full-attention layers have their own chains, which
        // the backends interleave better in the default order)
        const bool order = inputs[1] == inputs[0] && inputs[2] == inputs[0];
        if (order) {
            ggml_build_forward_expand(gf, Qcur);
            ggml_build_forward_expand(gf, sinks);
        }

        // the short convolutions of K and V, in one call over both when their rows are next to each other (no dynamic taps)
        ggml_tensor * k_conv = layer.attn_kv_conv ? ggml_view_2d(ctx0, layer.attn_kv_conv, n_conv_k, n_embd_k_gqa, layer.attn_kv_conv->nb[1], 0) : layer.attn_k_conv;
        ggml_tensor * v_conv = layer.attn_kv_conv ? ggml_view_2d(ctx0, layer.attn_kv_conv, n_conv_k, n_embd_k_gqa, layer.attn_kv_conv->nb[1],
                n_embd_k_gqa*layer.attn_kv_conv->nb[1]) : layer.attn_v_conv;
        if (KVcur && is_swa) {
            ggml_tensor * kv = convolve(KVcur, layer.attn_kv_conv, nullptr, state, history, recr, 0);
            Kcur = out_rows(kv, 0,            n_embd_k_gqa);
            Vcur = out_rows(kv, n_embd_k_gqa, n_embd_k_gqa);
        } else {
            Kcur = convolve(Kcur, k_conv, is_swa ? nullptr : build_lora_mm(layer.attn_k_conv_proj, inputs[1]), state, history, recr, 0);
            Vcur = convolve(Vcur, v_conv, is_swa ? nullptr : build_lora_mm(layer.attn_v_conv_proj, inputs[2]), state, history, recr, n_embd_k_gqa);
        }

        const int64_t n_stream = (is_swa ? attn->self_attn_mask_swa.raw : attn->self_attn_mask.raw)->ne[3];

        // the fused-mask attention: one stream, flash attention (F16 mask)
        const bool fused_attn = n_stream == 1 && cparams.flash_attn &&
                                is_f32(layer.attn_rel_bias) && is_f32(layer.attn_rel_clip) && is_f32(layer.attn_q_clip);

        // the fused-mask attention takes Q, K and V already rotated for the quantized KV cache and returns the output rotated:
        // the norms apply the rotations, the output gate undoes the V rotation
        ggml_tensor * k_rot = is_swa ? attn->self_k_rot_swa[il] : attn->self_k_rot[il];
        ggml_tensor * v_rot = is_swa ? attn->self_v_rot_swa[il] : attn->self_v_rot[il];
        const int n_rot_k = fused_attn && k_rot ? (int) k_rot->ne[0] : 0;
        const int n_rot_v = fused_attn && v_rot ? (int) v_rot->ne[0] : 0;

        ggml_tensor * tau_l = is_swa ? nullptr : tau;

        if (fused_attn) {
            Qcur = ggml_norm_rot(ctx0, heads(Qcur, n_embd_head, n_head), layer.attn_q_clip, tau_l,
                    false, hparams.f_qk_clip, 1e-6f, n_rot_k);
            Kcur = ggml_norm_rot(ctx0, heads(Kcur, n_embd_head, n_head_kv), nullptr, nullptr,
                    false, hparams.f_qk_clip, 1e-6f, n_rot_k);
            Vcur = ggml_norm_rot(ctx0, heads(Vcur, n_embd_head, n_head_kv), nullptr, nullptr,
                    true, 0.0f, hparams.f_norm_rms_eps, n_rot_v);
        } else {
            Qcur = clip(heads(Qcur, n_embd_head, n_head),    layer.attn_q_clip, hparams.f_qk_clip, tau_l);
            Kcur = clip(heads(Kcur, n_embd_head, n_head_kv), nullptr,           hparams.f_qk_clip);
            Vcur = ggml_rms_norm(ctx0, heads(Vcur, n_embd_head, n_head_kv), hparams.f_norm_rms_eps);
        }

        // the fused-mask attention clips R itself
        Rcur = heads(Rcur, hparams.n_relative, n_head);
        if (!fused_attn) {
            Rcur = clip(Rcur, layer.attn_rel_clip, hparams.f_relative_clip, tau_l);
        }

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
        cb(Rcur, "Rcur", il);

        ggml_tensor * rel_pos = is_swa ? inp->self_rel_pos_swa : inp->self_rel_pos;

        const int64_t n_kv = rel_pos->ne[0];

        if (fused_attn) {
            // one F16 mask per head: mask + attn_rel_bias[:, q_pos - kv_pos] . Rcur - sink, the bias in F32
            ggml_tensor * sinks_t = ggml_reshape_2d(ctx0, sinks, n_head, n_tokens);

            ggml_tensor *& zero_sinks = inp->zero_sinks;
            if (!zero_sinks) {
                zero_sinks = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_head), 0.0f);
            }

            ggml_tensor * mask_raw = is_swa ? attn->self_attn_mask_swa.raw : attn->self_attn_mask.raw;
            ggml_tensor * kv_pos   = is_swa ? inp->self_kv_pos_swa : inp->self_kv_pos;

            // the mask is n_kv x n_rows x n_head: query rows are chunked past the budget, the first chunk stores K and V
            const int64_t n_kv_max = std::max<int64_t>(n_kv, (is_swa ? attn->mctx->get_swa() : attn->mctx->get_base())->get_size());
            int64_t n_rows_max = std::max({ int64_t(1), n_kq_mask_max/(n_kv_max*n_head), (n_tokens + n_kq_chunk_max - 1)/n_kq_chunk_max });
            if (n_rows_max < n_tokens) {
                const int64_t n_chunks = (n_tokens + n_rows_max - 1)/n_rows_max;
                n_rows_max = std::min<int64_t>(n_tokens, GGML_PAD((n_tokens + n_chunks - 1)/n_chunks, 64));
            }

            std::vector<ggml_tensor *> outs;
            for (int64_t r0 = 0; r0 < n_tokens; r0 += n_rows_max) {
                const int64_t nr  = std::min<int64_t>(n_rows_max, n_tokens - r0);
                const bool    all = nr == n_tokens;

                ggml_tensor * q_c = all ? Qcur : ggml_view_3d(ctx0, Qcur, n_embd_head, n_head, nr, Qcur->nb[1], Qcur->nb[2], r0*Qcur->nb[2]);

                ggml_tensor * mask_c  = ggml_view_2d(ctx0, mask_raw, n_kv, nr, mask_raw->nb[1], r0*mask_raw->nb[1]);
                ggml_tensor * rcur_c  = ggml_view_3d(ctx0, Rcur, Rcur->ne[0], n_head, nr, Rcur->nb[1], Rcur->nb[2], r0*Rcur->nb[2]);
                ggml_tensor * sinks_c = ggml_view_2d(ctx0, sinks_t, n_head, nr, sinks_t->nb[1], r0*sinks_t->nb[1]);
                ggml_tensor * qpos_c  = ggml_view_1d(ctx0, inp_pos, nr, r0*ggml_element_size(inp_pos));

                ggml_tensor * kq_mask = ggml_rel_bias_mask_clip(ctx0, mask_c, layer.attn_rel_bias, rcur_c, layer.attn_rel_clip,
                        tau_l ? ggml_view_1d(ctx0, tau_l, nr, r0*ggml_element_size(tau_l)) : nullptr, hparams.f_relative_clip,
                        sinks_c, qpos_c, kv_pos);
                cb(kq_mask, "kq_mask_rel", il);

                // the mask before the K/V stores in the graph: it then runs with them, its output allocated before their
                // temporaries are freed (a reuse of those would need a sync)
                if (order) {
                    ggml_build_forward_expand(gf, kq_mask);
                }

                outs.push_back(build_attn_masked(inp, q_c, r0 == 0 ? Kcur : nullptr, r0 == 0 ? Vcur : nullptr,
                        kq_mask, zero_sinks, 1.0f/sqrtf(n_embd_head), il));
            }
            while (outs.size() > 1) {
                std::vector<ggml_tensor *> next;
                for (size_t c = 0; c + 1 < outs.size(); c += 2) {
                    next.push_back(ggml_concat(ctx0, outs[c], outs[c + 1], 1));
                }
                if (outs.size() % 2) {
                    next.push_back(outs.back());
                }
                outs = std::move(next);
            }
            cur = outs[0];
        } else {
            // project r on the bias table first, then gather by distance: only the gathered bias is n_kv x n_tokens x n_head
            ggml_tensor * table = ggml_mul_mat(ctx0, ggml_pad(ctx0, layer.attn_rel_bias, 0, 1, 0, 0), Rcur);

            // the sink logit t is per query: exp(s - t)/(1 + sum(exp(s - t))) = exp(s)/(exp(t) + sum(exp(s))), with a zero sink
            table = ggml_sub(ctx0, table, ggml_reshape_3d(ctx0, sinks, 1, n_head, n_tokens));
            table = ggml_cont(ctx0, ggml_permute(ctx0, table, 1, 0, 2, 3));

            ggml_tensor * kq_b = ggml_get_rows(ctx0, table, rel_pos);
            kq_b = ggml_reshape_4d(ctx0, kq_b, n_head, n_kv, n_tokens/n_stream, n_stream);
            kq_b = ggml_cont(ctx0, ggml_permute(ctx0, kq_b, 2, 0, 1, 3));
            cb(kq_b, "kq_b", il);

            sinks = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_head), 0.0f);

            cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, kq_b, sinks, 1.0f/sqrtf(n_embd_head), il);
        }

        // the output gate, with the inverse V rotation of the fused-mask attention
        cur = ggml_rot_gate(ctx0, ggml_reshape_2d(ctx0, cur, n_embd_head*n_head, n_tokens), Gcur, n_rot_v);

        // the first layer of each block starts a new residual stream, the others add the attention output to it
        const bool block_start = ip % hparams.attn_res_block_size == 0;
        cur = convolve(build_lora_mm(layer.wo, cur), layer.attn_post_conv, nullptr, state, history, recr, 2*n_embd_k_gqa,
                block_start ? nullptr : inpSA);
        // (upstream's attn_out is the convolution output before the residual)
        cb(cur, "attn_res", il);
    }

    // the first layer of each block starts a new residual stream, the earlier one joins the mixed blocks
    if (ip % hparams.attn_res_block_size == 0) {
        blocks.push_back(inpSA);
    }

    ggml_tensor * ffn_inp = cur;
    cb(ffn_inp, "ffn_inp", il);

    // MoE
    cur = ggml_reshape_2d(ctx0, mix(blocks, cur, layer.ffn_res_query), n_embd, n_tokens);
    if (messages) {
        cur = control(cur, message(LOOP_CTRL_FFN), *loop, LOOP_CTRL_FFN);
    }
    cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);

    cur = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            nullptr,
            nullptr,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            hparams.n_expert,
            hparams.n_expert_used(il),
            LLM_FFN_SITU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
            il,
            nullptr,
            layer.ffn_gate_up_exps);

    // the FFN output with the residual added
    cur = convolve(cur, layer.ffn_post_conv, nullptr, state, history, recr, 2*n_embd_k_gqa + n_embd, ffn_inp);
    cb(cur, "l_out", il);

    return cur;
}

void llama_model_inkling2::graph::build_output(ggml_tensor * cur, const std::vector<ggml_tensor *> & blocks) {
    cur = ggml_reshape_2d(ctx0, mix(blocks, cur, model.output_res_query), n_embd, n_tokens);
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    // DSpark target features: the last one is the final norm output of every token
    res->t_layer_inp[n_layer] = cur;

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    if (hparams.f_final_logit_softcapping != 0.0f) {
        const float cap = hparams.f_final_logit_softcapping;

        cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/cap)), cap);
    }

    // drop the padded vocabulary
    cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, hparams.n_vocab_unpadded, cur->ne[1], cur->nb[1], 0));

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

llama_model_inkling2::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inputs();

    std::vector<ggml_tensor *> blocks;
    for (int il = 0; il < n_layer; ++il) {
        // DSpark target features: the partial block at the layer input
        res->t_layer_inp[il] = cur;
        cur = build_layer(il, il, cur, blocks);
    }

    build_output(cur, blocks);
}
