// overflow-safe tanh, as in unary.comp
float tanh_safe(const float x) {
    return 1.0 - 2.0/(exp(2.0*x) + 1.0);
}
