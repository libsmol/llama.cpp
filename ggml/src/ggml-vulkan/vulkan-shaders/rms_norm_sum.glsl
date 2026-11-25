// The sum over a workgroup of the per-invocation sums of squares of a row, in rms_norm.comp's order: subgroup sums
// where USE_SUBGROUP_ADD, else a shared-memory tree. A fused norm with rms_norm.comp's block size reduces in it gives
// the bits of the RMS_NORM it replaces, so a fusion that the host has to leave out (overlapping memory) changes nothing.
// The includer declares BLOCK_SIZE and shared float sumsh[BLOCK_SIZE], and with USE_SUBGROUP_ADD enables
// GL_KHR_shader_subgroup_basic and GL_KHR_shader_subgroup_arithmetic.

float rms_norm_sum(float sum) {
#if USE_SUBGROUP_ADD
    sum = subgroupAdd(sum);
    if (subgroupElect()) {
        sumsh[gl_SubgroupID] = sum;
    }
    barrier();
    sum = 0.0f;
    if (gl_NumSubgroups <= 4) {
        [[unroll]] for (uint i = 0; i < gl_NumSubgroups; ++i) {
            sum += sumsh[i];
        }
    } else {
        for (uint i = gl_SubgroupInvocationID; i < gl_NumSubgroups; i += gl_SubgroupSize) {
            sum += sumsh[i];
        }
        sum = subgroupAdd(sum);
    }
#else
    const uint tid = gl_LocalInvocationID.x;
    sumsh[tid] = sum;
    barrier();
    [[unroll]] for (int s = int(BLOCK_SIZE) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum += sumsh[tid + s];
            sumsh[tid] = sum;
        }
        barrier();
    }
    sum = sumsh[0];
#endif
    return sum;
}
