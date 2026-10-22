// Hand-written decode kernels (no torch headers: nvcc -> cubin, loaded by
// tinyinfer/native.py through the CUDA driver API).
//
// sinkhorn: Hyper-Connection mix matrix. hc_mult is 4 here, so the eager path issues
// 20 iterations x 4 ops x 2 calls x 40 layers of 16-element kernels -- ~10k launches per
// token. One thread per matrix. Reduction order matches torch (sequential), so the result
// differs only by 1 ulp of fp32 (measured 1.19e-7).

extern "C" __global__ void sinkhorn_hc(const float* __restrict__ c,
                                       float* __restrict__ o,
                                       int n, int hc, int iters, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int m = hc * hc;
    const float* src = c + (long)i * m;
    float v[16];
    for (int k = 0; k < m; ++k) v[k] = src[k];
    // torch.softmax(comb, -1) + eps
    for (int r = 0; r < hc; ++r) {
        float mx = v[r * hc];
        for (int j = 1; j < hc; ++j) mx = fmaxf(mx, v[r * hc + j]);
        for (int j = 0; j < hc; ++j) v[r * hc + j] = expf(v[r * hc + j] - mx);
        float s = 0.f;
        for (int j = 0; j < hc; ++j) s += v[r * hc + j];
        for (int j = 0; j < hc; ++j) v[r * hc + j] = v[r * hc + j] / s + eps;
    }
    // comb / (comb.sum(-2) + eps)
    for (int j = 0; j < hc; ++j) {
        float s = 0.f;
        for (int r = 0; r < hc; ++r) s += v[r * hc + j];
        s += eps;
        for (int r = 0; r < hc; ++r) v[r * hc + j] /= s;
    }
    for (int it = 1; it < iters; ++it) {
        for (int r = 0; r < hc; ++r) {
            float s = 0.f;
            for (int j = 0; j < hc; ++j) s += v[r * hc + j];
            s += eps;
            for (int j = 0; j < hc; ++j) v[r * hc + j] /= s;
        }
        for (int j = 0; j < hc; ++j) {
            float s = 0.f;
            for (int r = 0; r < hc; ++r) s += v[r * hc + j];
            s += eps;
            for (int r = 0; r < hc; ++r) v[r * hc + j] /= s;
        }
    }
    float* dst = o + (long)i * m;
    for (int k = 0; k < m; ++k) dst[k] = v[k];
}
