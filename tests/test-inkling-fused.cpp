// fused inkling2 ops against the op chains they replace, on the same backend and the same inputs
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

static std::mt19937 rng(1234);

struct pair_out {
    ggml_tensor * fused;
    ggml_tensor * ref;
    double        tol;
};

struct test_ctx {
    ggml_context * ctx;
    std::vector<ggml_tensor *> inputs;
    std::vector<std::function<void(ggml_tensor *)>> init;
    std::vector<pair_out> outs;

    ggml_tensor * input(ggml_tensor * t, std::function<void(ggml_tensor *)> f = nullptr) {
        ggml_set_input(t);
        inputs.push_back(t);
        init.push_back(f);
        return t;
    }
};

static void fill_uniform(ggml_tensor * t, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(ggml_nelements(t));
    for (auto & x : v) x = d(rng);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(v.size());
        ggml_fp32_to_fp16_row(v.data(), h.data(), v.size());
        ggml_backend_tensor_set(t, h.data(), 0, ggml_nbytes(t));
    } else {
        GGML_ABORT("type");
    }
}

static std::vector<float> get_f32(ggml_tensor * t) {
    std::vector<float> out(ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(out.size());
        ggml_backend_tensor_get(t, h.data(), 0, ggml_nbytes(t));
        ggml_fp16_to_fp32_row(h.data(), out.data(), out.size());
    } else {
        GGML_ABORT("type");
    }
    return out;
}

// relative error: max |a - b| over the rms of b; the non-finite entries must match exactly
static double compare(const std::vector<float> & a, const std::vector<float> & b, bool & ok_nonfinite) {
    double err = 0.0, ss = 0.0;
    size_t n = 0;
    ok_nonfinite = true;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            if (!(a[i] == b[i])) ok_nonfinite = false;
            continue;
        }
        err = std::max(err, (double) std::fabs(a[i] - b[i]));
        ss += (double) b[i]*b[i];
        n++;
    }
    const double rms = n ? std::sqrt(ss/n) : 0.0;
    return err/std::max(rms, 1e-12);
}

static int n_fail = 0, n_pass = 0, n_skip = 0;

// reference outputs of the first (CPU) backend, by test name and output index
#include <map>
static std::map<std::string, std::vector<std::vector<float>>> cpu_refs;
static bool is_cpu_backend = true;

static void run(ggml_backend_t backend, const char * name, const std::function<void(test_ctx &)> & build) {
    rng.seed(std::hash<std::string>{}(name));
    ggml_init_params ip = { 256*ggml_tensor_overhead() + ggml_graph_overhead_custom(4096, false), nullptr, true };
    test_ctx tc;
    tc.ctx = ggml_init(ip);
    build(tc);

    ggml_cgraph * gf = ggml_new_graph_custom(tc.ctx, 4096, false);
    for (auto & o : tc.outs) {
        ggml_set_output(o.fused);
        ggml_set_output(o.ref);
        ggml_build_forward_expand(gf, o.ref);
        ggml_build_forward_expand(gf, o.fused);
    }
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        const bool is_view = node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE;
        if (node->op != GGML_OP_NONE && !is_view && !ggml_backend_supports_op(backend, node)) {
            printf("  %-60s SKIP (%s unsupported)\n", name, ggml_op_desc(node));
            n_skip++;
            ggml_free(tc.ctx);
            return;
        }
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(tc.ctx, backend);
    for (size_t i = 0; i < tc.inputs.size(); ++i) {
        if (tc.init[i]) tc.init[i](tc.inputs[i]); else fill_uniform(tc.inputs[i], -2.0f, 2.0f);
    }
    ggml_backend_graph_compute(backend, gf);

    bool all_ok = true;
    std::string detail;
    for (size_t oi = 0; oi < tc.outs.size(); ++oi) {
        auto & o = tc.outs[oi];
        const std::vector<float> vf = get_f32(o.fused);
        const std::vector<float> vr = get_f32(o.ref);
        bool nf;
        const double rel = compare(vf, vr, nf);
        char b[192];
        if (is_cpu_backend) {
            cpu_refs[name].push_back(vr);
            snprintf(b, sizeof(b), " [%zu] fused-ref=%.2e%s", oi, rel, nf ? "" : "(nonfinite!)");
            all_ok &= nf && rel <= o.tol;
        } else {
            // judge against the CPU reference chain: the fused op must be at least as close as the backend's own chain
            bool nf1, nf2;
            const auto & cr = cpu_refs[name][oi];
            const double rel_f = compare(vf, cr, nf1);
            const double rel_r = compare(vr, cr, nf2);
            snprintf(b, sizeof(b), " [%zu] fused-ref=%.2e fused-cpuref=%.2e ref-cpuref=%.2e%s", oi, rel, rel_f, rel_r, nf1 ? "" : "(nonfinite!)");
            all_ok &= nf1 && (rel_f <= o.tol || rel_f <= rel_r);
        }
        detail += b;
    }
    printf("  %-60s %s%s\n", name, all_ok ? "OK  " : "FAIL", detail.c_str());
    (all_ok ? n_pass : n_fail)++;
    ggml_backend_buffer_free(buf);
    ggml_free(tc.ctx);
}

// ---- reference chains, as in src/models/inkling2.cpp with LLAMA_INKLING_LEGACY=1 ----

static ggml_tensor * ref_clip(ggml_context * ctx0, ggml_tensor * cur, ggml_tensor * scale, float threshold, ggml_tensor * post) {
    if (scale) cur = ggml_mul(ctx0, cur, scale);
    ggml_tensor * norm = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, cur)));
    norm = ggml_scale_bias(ctx0, norm, 1.0f, 1e-6f);
    ggml_tensor * factor = ggml_div(ctx0, ggml_fill(ctx0, norm, threshold), norm);
    cur = ggml_mul(ctx0, cur, ggml_clamp(ctx0, factor, 0.0f, 1.0f));
    return post ? ggml_mul(ctx0, cur, post) : cur;
}

static ggml_tensor * ref_mix(ggml_context * ctx0, const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur, ggml_tensor * query, int64_t n_embd, int64_t n_tokens) {
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

// returns the output; *x_out the concatenated inputs (for the state reference)
static ggml_tensor * ref_conv(ggml_context * ctx0, ggml_tensor * cur, ggml_tensor * weight, ggml_tensor * dynamic, ggml_tensor * past,
                              int64_t n_seq_tokens, int64_t n_seqs, ggml_tensor ** x_out) {
    const int64_t n_conv     = weight->ne[0];
    const int64_t n_channels = cur->ne[0];
    const int64_t n_tokens   = cur->ne[1];
    ggml_tensor * x = ggml_transpose(ctx0, ggml_reshape_3d(ctx0, cur, n_channels, n_seq_tokens, n_seqs));
    x = ggml_concat(ctx0, past, x, 0);
    if (x_out) *x_out = x;
    ggml_tensor * out = ggml_ssm_conv(ctx0, x, weight);
    out = ggml_reshape_2d(ctx0, out, n_channels, n_tokens);
    if (dynamic) {
        const int64_t n_heads = dynamic->ne[0]/n_conv;
        dynamic = ggml_reshape_3d(ctx0, dynamic, n_conv, n_heads, n_tokens);
        for (int64_t lag = 0; lag < n_conv; ++lag) {
            ggml_tensor * tap    = ggml_view_3d(ctx0, dynamic, 1, n_heads, n_tokens, dynamic->nb[1], dynamic->nb[2], lag*dynamic->nb[0]);
            ggml_tensor * window = ggml_view_3d(ctx0, x, n_seq_tokens, n_channels, n_seqs, x->nb[1], x->nb[2], (n_conv - 1 - lag)*x->nb[0]);
            window = ggml_cont(ctx0, ggml_transpose(ctx0, window));
            window = ggml_reshape_3d(ctx0, window, n_channels/n_heads, n_heads, n_tokens);
            ggml_tensor * term = ggml_reshape_2d(ctx0, ggml_mul(ctx0, window, tap), n_channels, n_tokens);
            out = ggml_add(ctx0, out, term);
        }
    }
    return ggml_scale(ctx0, ggml_add(ctx0, cur, out), 1.0f/sqrtf((dynamic ? 2 : 1)*n_conv + 1));
}

static ggml_tensor * ref_control(ggml_context * ctx0, ggml_tensor * cur, ggml_tensor * message, ggml_tensor * wd, ggml_tensor * wg, ggml_tensor * wf, int64_t n_embd, int64_t n_tokens) {
    ggml_tensor * h = ggml_concat(ctx0, ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens), ggml_reshape_3d(ctx0, message, n_embd, 1, n_tokens), 1);
    h = ggml_reshape_2d(ctx0, ggml_rms_norm(ctx0, h, 1e-6f), 2*n_embd, n_tokens);
    h = ggml_mul_mat(ctx0, wd, h);
    ggml_tensor * gain   = ggml_tanh(ctx0, ggml_mul_mat(ctx0, wg, h));
    ggml_tensor * forget = ggml_tanh(ctx0, ggml_mul_mat(ctx0, wf, h));
    return ggml_add(ctx0, ggml_mul(ctx0, ggml_scale_bias(ctx0, forget, -1.0f, 1.0f), cur), ggml_mul(ctx0, gain, message));
}

int main() {
    ggml_backend_load_all();

    std::vector<ggml_backend_dev_t> devs;
    for (size_t di = 0; di < ggml_backend_dev_count(); ++di) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(di);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) devs.insert(devs.begin(), dev);
        else if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_ACCEL) devs.push_back(dev);
    }
    for (ggml_backend_dev_t dev : devs) {
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        printf("backend %s\n", ggml_backend_dev_name(dev));
        is_cpu_backend = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;

        const int64_t n_embd = 2304;

        // SITU
        for (int64_t n_t : {1, 37}) {
            run(backend, ("situ n_t=" + std::to_string(n_t)).c_str(), [&](test_ctx & tc) {
                ggml_tensor * gu = tc.input(ggml_new_tensor_3d(tc.ctx, GGML_TYPE_F32, 2*1152, 8, n_t), [](ggml_tensor * t) { fill_uniform(t, -30.0f, 30.0f); });
                ggml_tensor * g = ggml_view_3d(tc.ctx, gu, 1152, 8, n_t, gu->nb[1], gu->nb[2], 0);
                ggml_tensor * u = ggml_view_3d(tc.ctx, gu, 1152, 8, n_t, gu->nb[1], gu->nb[2], 1152*sizeof(float));
                const float beta = 4.0f, lb = 25.0f;
                ggml_tensor * act = ggml_scale(tc.ctx, ggml_tanh(tc.ctx, ggml_scale(tc.ctx, ggml_cont(tc.ctx, g), 1.0f/beta)), beta);
                act = ggml_mul(tc.ctx, act, ggml_sigmoid(tc.ctx, ggml_cont(tc.ctx, g)));
                ggml_tensor * up = ggml_scale(tc.ctx, ggml_tanh(tc.ctx, ggml_scale(tc.ctx, ggml_cont(tc.ctx, u), 1.0f/lb)), lb);
                tc.outs.push_back({ ggml_situ(tc.ctx, gu, nullptr, beta, lb), ggml_mul(tc.ctx, act, up), 1e-5 });
            });
        }

        // CLIP_NORM: Q (scale, tau), K, R (scale, tau)
        for (int64_t n_t : {1, 37}) {
            run(backend, ("clip_norm n_t=" + std::to_string(n_t)).c_str(), [&](test_ctx & tc) {
                ggml_tensor * q   = tc.input(ggml_new_tensor_3d(tc.ctx, GGML_TYPE_F32, 128, 24, n_t), [](ggml_tensor * t) { fill_uniform(t, -4.0f, 4.0f); });
                ggml_tensor * qs  = tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, 128, 24));
                ggml_tensor * tau = tc.input(ggml_new_tensor_3d(tc.ctx, GGML_TYPE_F32, 1, 1, n_t), [](ggml_tensor * t) { fill_uniform(t, 1.0f, 1.3f); });
                ggml_tensor * k   = tc.input(ggml_new_tensor_3d(tc.ctx, GGML_TYPE_F32, 128, 8, n_t), [](ggml_tensor * t) { fill_uniform(t, -0.5f, 0.5f); });
                ggml_tensor * r   = tc.input(ggml_new_tensor_3d(tc.ctx, GGML_TYPE_F32, 16, 24, n_t), [](ggml_tensor * t) { fill_uniform(t, -4.0f, 4.0f); });
                ggml_tensor * rs  = tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, 16, 24));
                tc.outs.push_back({ ggml_clip_norm(tc.ctx, q, qs, tau, 22.6f, 1e-6f), ref_clip(tc.ctx, q, qs, 22.6f, tau), 1e-5 });
                tc.outs.push_back({ ggml_clip_norm(tc.ctx, k, nullptr, nullptr, 22.6f, 1e-6f), ref_clip(tc.ctx, k, nullptr, 22.6f, nullptr), 1e-5 });
                tc.outs.push_back({ ggml_clip_norm(tc.ctx, r, rs, tau, 8.0f, 1e-6f), ref_clip(tc.ctx, r, rs, 8.0f, tau), 1e-5 });
            });
        }

        // NORM_ROT (CLIP_NORM + KV-cache Hadamard rotation) and ROT_GATE against llama_mul_mat_hadamard chains
        for (int64_t n_t : {1, 37}) {
            run(backend, ("norm_rot/rot_gate n_t=" + std::to_string(n_t)).c_str(), [&](test_ctx & tc) {
                auto hadamard = [](int n) {
                    std::vector<float> h(n*n);
                    h[0] = 1.0f/sqrtf((float) n);
                    for (int s = 1; s < n; s *= 2) for (int i = 0; i < s; i++) for (int j = 0; j < s; j++) {
                        const float v = h[i*n + j];
                        h[(i + s)*n + j] = v; h[i*n + j + s] = v; h[(i + s)*n + j + s] = -v;
                    }
                    return h;
                };
                std::vector<float> h128 = hadamard(128), h64 = hadamard(64);
                ggml_tensor * r128 = tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, 128, 128), [h128](ggml_tensor * t) { ggml_backend_tensor_set(t, h128.data(), 0, ggml_nbytes(t)); });
                ggml_tensor * r64  = tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, 64, 64),   [h64](ggml_tensor * t)  { ggml_backend_tensor_set(t, h64.data(),  0, ggml_nbytes(t)); });
                auto hmm = [&](ggml_tensor * cur, ggml_tensor * rot) {
                    const int64_t n = rot->ne[0];
                    ggml_tensor * res = ggml_mul_mat(tc.ctx, rot, ggml_reshape_2d(tc.ctx, cur, n, ggml_nelements(cur)/n));
                    return ggml_reshape_4d(tc.ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
                };
                ggml_tensor * q   = tc.input(ggml_new_tensor_3d(tc.ctx, GGML_TYPE_F32, 128, 24, n_t), [](ggml_tensor * t) { fill_uniform(t, -4.0f, 4.0f); });
                ggml_tensor * qs  = tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, 128, 24));
                ggml_tensor * tau = tc.input(ggml_new_tensor_3d(tc.ctx, GGML_TYPE_F32, 1, 1, n_t), [](ggml_tensor * t) { fill_uniform(t, 1.0f, 1.3f); });
                ggml_tensor * v   = tc.input(ggml_new_tensor_3d(tc.ctx, GGML_TYPE_F32, 128, 8, n_t));
                ggml_tensor * o   = tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, 3072, n_t));
                ggml_tensor * g   = tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, 3072, n_t), [](ggml_tensor * t) { fill_uniform(t, -6.0f, 6.0f); });
                tc.outs.push_back({ ggml_norm_rot(tc.ctx, q, qs, tau, false, 22.6f, 1e-6f, 128), hmm(ref_clip(tc.ctx, q, qs, 22.6f, tau), r128), 1e-5 });
                tc.outs.push_back({ ggml_norm_rot(tc.ctx, v, nullptr, nullptr, true, 0.0f, 1e-6f, 64), hmm(ggml_rms_norm(tc.ctx, v, 1e-6f), r64), 1e-5 });
                tc.outs.push_back({ ggml_rot_gate(tc.ctx, o, g, 64), ggml_mul(tc.ctx, hmm(o, r64), ggml_sigmoid(tc.ctx, g)), 1e-5 });
                tc.outs.push_back({ ggml_rot_gate(tc.ctx, o, g, 0), ggml_mul(tc.ctx, o, ggml_sigmoid(tc.ctx, g)), 1e-5 });
            });
        }

        // RES_MIX: 1..5 states, 1..3 queries, query as a row view of a table
        for (int64_t n_t : {1, 37}) {
            for (int n_blocks : {0, 1, 4}) {
                for (int64_t n_q : {1, 2, 3}) {
                    std::string nm = "res_mix n_t=" + std::to_string(n_t) + " blocks=" + std::to_string(n_blocks) + " n_q=" + std::to_string(n_q);
                    run(backend, nm.c_str(), [&](test_ctx & tc) {
                        std::vector<ggml_tensor *> blocks;
                        for (int b = 0; b < n_blocks; ++b) {
                            blocks.push_back(tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, n_embd, n_t), [](ggml_tensor * t) { fill_uniform(t, -3.0f, 3.0f); }));
                        }
                        ggml_tensor * cur = tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, n_embd, n_t));
                        ggml_tensor * qt  = tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, n_embd, n_q + 2), [](ggml_tensor * t) { fill_uniform(t, -0.3f, 0.3f); });
                        ggml_tensor * q   = ggml_view_2d(tc.ctx, qt, n_embd, n_q, qt->nb[1], 2*qt->nb[1]);
                        std::vector<ggml_tensor *> states = blocks;
                        states.push_back(cur);
                        ggml_tensor * fused = ggml_res_mix(tc.ctx, q, states.data(), (int) states.size(), 1e-6f, 1.0f/sqrtf(n_embd));
                        // fused is [n_embd, n_t, n_q], the reference [n_embd, n_q, n_t]
                        fused = ggml_cont(tc.ctx, ggml_permute(tc.ctx, fused, 0, 2, 1, 3));
                        // the reference views q as a contiguous copy, as the model's mix() does with ggml_repeat
                        tc.outs.push_back({ fused, ref_mix(tc.ctx, blocks, cur, ggml_cont(tc.ctx, q), n_embd, n_t), 1e-5 });
                    });
                }
            }
        }

        // SHORT_CONV and SHORT_CONV_RS (output and stored state), with and without dynamic taps
        for (int64_t n_t : {1, 2, 5, 37}) {
            for (int64_t n_s : {1, 2}) {
                for (bool dyn : {false, true}) {
                    const int64_t n_ch = dyn ? 1024 : 2304, n_head = 8, n_conv = 4;
                    std::string nm = "short_conv(+rs) n_t=" + std::to_string(n_t) + " n_s=" + std::to_string(n_s) + (dyn ? " dynamic" : "");
                    run(backend, nm.c_str(), [&](test_ctx & tc) {
                        ggml_tensor * cur  = tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, n_ch, n_t*n_s));
                        ggml_tensor * hist = tc.input(ggml_new_tensor_3d(tc.ctx, GGML_TYPE_F32, n_conv - 1, n_ch + 64, n_s));
                        ggml_tensor * past = ggml_view_3d(tc.ctx, hist, n_conv - 1, n_ch, n_s, hist->nb[1], hist->nb[2], 64*hist->nb[1]);
                        ggml_tensor * w    = tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, n_conv, n_ch), [](ggml_tensor * t) { fill_uniform(t, -0.5f, 0.5f); });
                        ggml_tensor * d    = dyn ? tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, n_conv*n_head, n_t*n_s), [](ggml_tensor * t) { fill_uniform(t, -0.5f, 0.5f); }) : nullptr;
                        ggml_tensor * st   = tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, (n_conv - 1)*n_ch, n_s));
                        ggml_tensor * x = nullptr;
                        ggml_tensor * ref = ref_conv(tc.ctx, cur, w, d, past, n_t, n_s, &x);
                        // x columns after the first n_t are the last n_conv - 1 inputs of each channel
                        ggml_tensor * ref_state = ggml_cont(tc.ctx, ggml_view_3d(tc.ctx, x, n_conv - 1, n_ch, n_s, x->nb[1], x->nb[2], n_t*ggml_element_size(x)));
                        // x of the plain SHORT_CONV: the same concat
                        ggml_tensor * xc = ggml_concat(tc.ctx, past, ggml_transpose(tc.ctx, ggml_reshape_3d(tc.ctx, cur, n_ch, n_t, n_s)), 0);
                        ggml_tensor * f1 = ggml_reshape_2d(tc.ctx, ggml_short_conv(tc.ctx, xc, w, d, 1.0f/sqrtf((dyn ? 2 : 1)*n_conv + 1)), n_ch, n_t*n_s);
                        ggml_tensor * f2 = ggml_reshape_2d(tc.ctx, ggml_short_conv_rs(tc.ctx, cur, past, w, d, st, 1.0f/sqrtf((dyn ? 2 : 1)*n_conv + 1)), n_ch, n_t*n_s);
                        ggml_tensor * res  = tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, n_ch, n_t*n_s));
                        ggml_tensor * st2  = tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, (n_conv - 1)*n_ch, n_s));
                        ggml_tensor * f3 = ggml_reshape_2d(tc.ctx, ggml_short_conv_rs_add(tc.ctx, cur, past, w, d, st2, res, 1.0f/sqrtf((dyn ? 2 : 1)*n_conv + 1)), n_ch, n_t*n_s);
                        ggml_set_name(f1, "conv");
                        ggml_set_name(f2, "conv_rs");
                        ggml_set_name(f3, "conv_rs_add");
                        tc.outs.push_back({ f1, ref, 1e-5 });
                        tc.outs.push_back({ f2, ref, 1e-5 });
                        tc.outs.push_back({ f3, ggml_add(tc.ctx, res, ref), 1e-5 });
                        // read the state after the op: a scale of it that depends on f2 through a zero-weighted add
                        ggml_tensor * st_after = ggml_add(tc.ctx, ggml_reshape_3d(tc.ctx, st, n_conv - 1, n_ch, n_s),
                                ggml_scale(tc.ctx, ggml_sum(tc.ctx, f2), 0.0f));
                        ggml_set_name(st_after, "state");
                        tc.outs.push_back({ st_after, ref_state, 0.0 });
                    });
                }
            }
        }

        // loop controller: LOOP_GATE (with the matmuls) and LOOP_DOWN + LOOP_UP
        for (int64_t n_t : {1, 4, 16}) {
            run(backend, ("loop control n_t=" + std::to_string(n_t)).c_str(), [&](test_ctx & tc) {
                ggml_tensor * cur = tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, n_embd, n_t));
                ggml_tensor * msg = tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, n_embd, n_t), [](ggml_tensor * t) { fill_uniform(t, -5.0f, 5.0f); });
                ggml_tensor * wd  = tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, 2*n_embd, 64), [](ggml_tensor * t) { fill_uniform(t, -0.05f, 0.05f); });
                ggml_tensor * wg  = tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, 64, n_embd), [](ggml_tensor * t) { fill_uniform(t, -0.3f, 0.3f); });
                ggml_tensor * wf  = tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, 64, n_embd), [](ggml_tensor * t) { fill_uniform(t, -0.3f, 0.3f); });
                ggml_tensor * ref = ref_control(tc.ctx, cur, msg, wd, wg, wf, n_embd, n_t);
                ggml_tensor * h = ggml_concat(tc.ctx, ggml_reshape_3d(tc.ctx, cur, n_embd, 1, n_t), ggml_reshape_3d(tc.ctx, msg, n_embd, 1, n_t), 1);
                h = ggml_mul_mat(tc.ctx, wd, ggml_reshape_2d(tc.ctx, ggml_rms_norm(tc.ctx, h, 1e-6f), 2*n_embd, n_t));
                ggml_tensor * f1 = ggml_loop_gate(tc.ctx, cur, msg, ggml_mul_mat(tc.ctx, wg, h), ggml_mul_mat(tc.ctx, wf, h));
                ggml_tensor * f2 = ggml_loop_up(tc.ctx, cur, msg, ggml_loop_down(tc.ctx, cur, msg, wd, 1e-6f), wg, wf);
                ggml_set_name(f1, "gate");
                ggml_set_name(f2, "down_up");
                tc.outs.push_back({ f1, ref, 1e-5 });
                tc.outs.push_back({ f2, ref, 1e-5 });
            });
        }

        // F32-weight matmuls (loop controller, rel-bias table, router) with and without GGML_PREC_F32
        for (int64_t n_t : {1, 16, 512}) {
            run(backend, ("f32 matmul prec n_t=" + std::to_string(n_t)).c_str(), [&](test_ctx & tc) {
                ggml_tensor * w = tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, 2*n_embd, 64), [](ggml_tensor * t) { fill_uniform(t, -0.05f, 0.05f); });
                ggml_tensor * x = tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, 2*n_embd, n_t));
                ggml_tensor * a = ggml_mul_mat(tc.ctx, w, x);
                ggml_tensor * b = ggml_mul_mat(tc.ctx, w, x);
                ggml_prec_set_acc(b, GGML_PREC_F32);
                ggml_set_name(a, "default");
                ggml_set_name(b, "prec_f32");
                // both against the same reference: the CPU result of the default op
                tc.outs.push_back({ a, a, 1.0 });
                tc.outs.push_back({ b, a, 1.0 });
            });
        }

        // REL_BIAS_MASK against the dense gather (rel_pos computed on the host, as llama_kv_cache::set_input_relative_pos)
        for (int64_t n_t : {1, 7}) {
            for (int64_t n_kv : {256, 1536}) {
                const int64_t extent = 1024, n_head = 24;
                std::string nm = "rel_bias_mask n_t=" + std::to_string(n_t) + " n_kv=" + std::to_string(n_kv);
                std::vector<int32_t> qpos(n_t), kvpos(n_kv);
                for (int64_t j = 0; j < n_kv; ++j) kvpos[j] = (j*7) % n_kv;           // a permuted cache
                kvpos[n_kv/3] = INT32_MIN;                                             // an empty cell
                for (int64_t i = 0; i < n_t; ++i) qpos[i] = (int32_t) (n_kv - n_t + i);
                std::vector<int32_t> relpos(n_kv*n_t);
                for (int64_t i = 0; i < n_t; ++i) {
                    for (int64_t j = 0; j < n_kv; ++j) {
                        const int64_t dd = kvpos[j] == INT32_MIN ? -1 : (int64_t) qpos[i] - kvpos[j];
                        relpos[i*n_kv + j] = dd >= 0 && dd < extent ? (int32_t) dd : (int32_t) extent;
                    }
                }
                run(backend, nm.c_str(), [&](test_ctx & tc) {
                    ggml_tensor * bias  = tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, 16, extent), [](ggml_tensor * t) { fill_uniform(t, -1.0f, 1.0f); });
                    ggml_tensor * R     = tc.input(ggml_new_tensor_3d(tc.ctx, GGML_TYPE_F32, 16, n_head, n_t));
                    ggml_tensor * sinks = tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, n_head, n_t), [](ggml_tensor * t) { fill_uniform(t, -4.0f, 4.0f); });
                    ggml_tensor * mask  = tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F16, n_kv, n_t), [&](ggml_tensor * t) {
                        std::vector<ggml_fp16_t> m(n_kv*n_t);
                        for (int64_t i = 0; i < n_t; ++i) for (int64_t j = 0; j < n_kv; ++j) {
                            const bool causal = kvpos[j] != INT32_MIN && kvpos[j] <= qpos[i];
                            m[i*n_kv + j] = ggml_fp32_to_fp16(causal ? 0.0f : -INFINITY);
                        }
                        ggml_backend_tensor_set(t, m.data(), 0, ggml_nbytes(t));
                    });
                    ggml_tensor * qp = tc.input(ggml_new_tensor_1d(tc.ctx, GGML_TYPE_I32, n_t), [&](ggml_tensor * t) { ggml_backend_tensor_set(t, qpos.data(), 0, ggml_nbytes(t)); });
                    ggml_tensor * kp = tc.input(ggml_new_tensor_1d(tc.ctx, GGML_TYPE_I32, n_kv), [&](ggml_tensor * t) { ggml_backend_tensor_set(t, kvpos.data(), 0, ggml_nbytes(t)); });
                    ggml_tensor * rp = tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_I32, n_kv, n_t), [&](ggml_tensor * t) { ggml_backend_tensor_set(t, relpos.data(), 0, ggml_nbytes(t)); });

                    // reference: inkling2.cpp dense path, n_stream = 1, one chunk
                    ggml_tensor * table = ggml_mul_mat(tc.ctx, ggml_pad(tc.ctx, bias, 0, 1, 0, 0), R);
                    table = ggml_sub(tc.ctx, table, ggml_reshape_3d(tc.ctx, sinks, 1, n_head, n_t));
                    table = ggml_cont(tc.ctx, ggml_permute(tc.ctx, table, 1, 0, 2, 3));
                    ggml_tensor * kq_b = ggml_get_rows(tc.ctx, table, rp);
                    kq_b = ggml_reshape_4d(tc.ctx, kq_b, n_head, n_kv, n_t, 1);
                    kq_b = ggml_cont(tc.ctx, ggml_permute(tc.ctx, kq_b, 2, 0, 1, 3));
                    ggml_tensor * ref = ggml_cast(tc.ctx, ggml_add(tc.ctx, kq_b, ggml_cast(tc.ctx, mask, GGML_TYPE_F32)), GGML_TYPE_F16);

                    ggml_tensor * fused = ggml_rel_bias_mask(tc.ctx, mask, bias, R, sinks, qp, kp);
                    // F16 outputs: one F16 ulp of the largest values
                    tc.outs.push_back({ fused, ref, 2e-3 });

                    // R clipped in the op, against the clip chain feeding the dense reference
                    ggml_tensor * R0  = tc.input(ggml_new_tensor_3d(tc.ctx, GGML_TYPE_F32, 16, n_head, n_t), [](ggml_tensor * t) { fill_uniform(t, -4.0f, 4.0f); });
                    ggml_tensor * rsc = tc.input(ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, 16, n_head));
                    ggml_tensor * tau = tc.input(ggml_new_tensor_3d(tc.ctx, GGML_TYPE_F32, 1, 1, n_t), [](ggml_tensor * t) { fill_uniform(t, 1.0f, 1.3f); });
                    ggml_tensor * Rc  = ref_clip(tc.ctx, R0, rsc, 8.0f, tau);
                    ggml_tensor * table2 = ggml_mul_mat(tc.ctx, ggml_pad(tc.ctx, bias, 0, 1, 0, 0), Rc);
                    table2 = ggml_sub(tc.ctx, table2, ggml_reshape_3d(tc.ctx, sinks, 1, n_head, n_t));
                    table2 = ggml_cont(tc.ctx, ggml_permute(tc.ctx, table2, 1, 0, 2, 3));
                    ggml_tensor * kq_b2 = ggml_get_rows(tc.ctx, table2, rp);
                    kq_b2 = ggml_cont(tc.ctx, ggml_permute(tc.ctx, ggml_reshape_4d(tc.ctx, kq_b2, n_head, n_kv, n_t, 1), 2, 0, 1, 3));
                    ggml_tensor * ref2 = ggml_cast(tc.ctx, ggml_add(tc.ctx, kq_b2, ggml_cast(tc.ctx, mask, GGML_TYPE_F32)), GGML_TYPE_F16);
                    tc.outs.push_back({ ggml_rel_bias_mask_clip(tc.ctx, mask, bias, R0, rsc, tau, 8.0f, sinks, qp, kp), ref2, 2e-3 });
                });
            }
        }

        ggml_backend_free(backend);
    }

    printf("%d passed, %d failed, %d skipped\n", n_pass, n_fail, n_skip);
    return n_fail ? 1 : 0;
}
