// NESSi-side kernel benchmark: full-contour Type-1 convolution
// C = A * B over the whole two-time grid (counterpart of
// examples/conv_bench.rs). Both operands are filled with deterministic
// pseudo-random data on all four components; Acc = A and Bcc = B, the
// most common DMFT calling convention.
//
// Usage: ./kernel_conv_bench [nt ntau(intervals) d k]
#include <cntr/cntr.hpp>
#include <chrono>
#include <cstdio>
#include <cmath>

using namespace cntr;
typedef std::complex<double> cplx;

// deterministic pseudo-random complex in (-0.5, 0.5)
static inline cplx prand(unsigned long long &s) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    double re = (double)((s >> 33) % 1000) / 1000.0 - 0.5;
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    double im = (double)((s >> 33) % 1000) / 1000.0 - 0.5;
    return cplx(re, im);
}

int main(int argc, char **argv) {
    int nt = 400, ntau = 200, d = 2, k = 5;
    if (argc > 4) {
        nt = atoi(argv[1]);
        ntau = atoi(argv[2]);
        d = atoi(argv[3]);
        k = atoi(argv[4]);
    }
    double beta = 10.0, h = 0.05;

    herm_matrix<double> A(nt, ntau, d, d, -1);
    herm_matrix<double> B(nt, ntau, d, d, -1);
    herm_matrix<double> C(nt, ntau, d, d, -1);
    unsigned long long seed = 0x9E3779B97F4A7C15ULL;

    Eigen::MatrixXcd M(d, d);
    for (int m = 0; m <= ntau; m++) {
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++) M(i, j) = prand(seed);
        A.set_mat(m, M);
        B.set_mat(m, M);
    }
    for (int n = 0; n <= nt; n++) {
        for (int m = 0; m <= n; m++) {
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++) M(i, j) = prand(seed);
            A.set_ret(n, m, M);
            B.set_ret(n, m, M);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++) M(i, j) = prand(seed);
            A.set_les(n, m, M);
            B.set_les(n, m, M);
        }
        for (int m = 0; m <= ntau; m++) {
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++) M(i, j) = prand(seed);
            A.set_tv(n, m, M);
            B.set_tv(n, m, M);
        }
    }

    integration::Integrator<double> I(k);
    // warm-up run (page faults / caches), timed run second
    convolution(C, A, A, B, B, I, beta, h);
    auto t0 = std::chrono::steady_clock::now();
    convolution(C, A, A, B, B, I, beta, h);
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    double gmax = std::abs(C.matptr(0)[0]);
    int fin = (std::isfinite(gmax) && gmax < 1e8) ? 1 : 0;
    printf("nessi conv nt=%d ntau=%d d=%d k=%d: %.4f s (finite: %d)\n", nt,
           ntau, d, k, dt, fin);
    return 0;
}
