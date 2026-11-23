// The f16 B rows of mul_mm_rdna4 / mul_mmv_rdna4: a row whose largest magnitude would overflow f16 is stored times
// 2^-e and has the scale 2^e, every other row has the scale 1. e <= 126 keeps both powers of two normal; an infinite
// element stays infinite, as it would unscaled.
const float F16_OVERFLOW = 65520.0;

// e for the row maximum amax >= F16_OVERFLOW
uint f16_row_exp(const float amax) {
    uint e = 0;
    for (float m = amax; m >= F16_OVERFLOW && e < 126; m *= 0.5) {
        e++;
    }
    return e;
}

float f16_row_down(const uint e) {
    return uintBitsToFloat((127u - e) << 23);
}

float f16_row_scale(const uint e) {
    return uintBitsToFloat((127u + e) << 23);
}
