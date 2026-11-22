// bench-vk-chain: GPU cost of a chain of dependent tiny ops vs the same number of independent ones
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <vector>

static double run(ggml_backend_t be, int n_ops, bool chain, int64_t n) {
    ggml_init_params ip = { (size_t) (n_ops + 16)*ggml_tensor_overhead() + ggml_graph_overhead_custom(n_ops + 16, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_input(x);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, n_ops + 16, false);
    ggml_tensor * cur = x;
    for (int i = 0; i < n_ops; ++i) {
        ggml_tensor * y = ggml_scale(ctx, chain ? cur : x, 1.0001f);
        ggml_build_forward_expand(gf, y);
        cur = y;
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    std::vector<float> d(n, 1.0f);
    ggml_backend_tensor_set(x, d.data(), 0, n*sizeof(float));
    ggml_backend_graph_compute(be, gf);
    const int reps = 20;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < reps; ++r) {
        ggml_backend_graph_compute(be, gf);
    }
    ggml_backend_synchronize(be);
    double us = std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now() - t0).count()/reps;
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return us/n_ops;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) { fprintf(stderr, "no GPU\n"); return 1; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    for (int64_t n : {2304, 2304*8, 2304*64}) {
        printf("n=%-7lld dependent %.2f us/op | independent %.2f us/op\n", (long long) n, run(be, 1000, true, n), run(be, 1000, false, n));
    }
    ggml_backend_free(be);
    return 0;
}
