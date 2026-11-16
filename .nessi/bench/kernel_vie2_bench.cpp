// NESSi-side kernel benchmark: VIE2 propagation [1+F] G = Q
// (counterpart of examples/vie2_bench.rs): vie2_mat (fixpoint) +
// vie2_start + the vie2_timestep loop, with deterministic
// pseudo-random F and Q filled on all components.
//
// Usage: ./kernel_vie2_bench [nt ntau(intervals) d k]
#include <cntr/cntr.hpp>
#include <chrono>
#include <cstdio>
#include <cmath>

using namespace cntr;
typedef std::complex<double> cplx;

static inline cplx prand(unsigned long long &s) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    double re = (double)((s >> 33) % 1000) / 1000.0 - 0.5;
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    double im = (double)((s >> 33) % 1000) / 1000.0 - 0.5;
    return cplx(re, im);
}

int main(int argc, char **argv) {
    int nt = 500, ntau = 200, d = 2, k = 5;
    if (argc > 4) {
        nt = atoi(argv[1]);
        ntau = atoi(argv[2]);
        d = atoi(argv[3]);
        k = atoi(argv[4]);
    }
    double beta = 10.0, h = 0.05;

    herm_matrix<double> G(nt, ntau, d, d, -1);
    herm_matrix<double> F(nt, ntau, d, d, -1);
    herm_matrix<double> Q(nt, ntau, d, d, -1);
    unsigned long long seed = 0xD1B54A32D192ED03ULL;

    // small pseudo-random kernel (|F| < 0.05) so the fixpoint converges
    Eigen::MatrixXcd M(d, d);
    for (int m = 0; m <= ntau; m++) {
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++) M(i, j) = 0.05 * prand(seed);
        F.set_mat(m, M);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++) M(i, j) = prand(seed);
        Q.set_mat(m, M);
    }
    for (int n = 0; n <= nt; n++) {
        for (int m = 0; m <= n; m++) {
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++) M(i, j) = 0.05 * prand(seed);
            F.set_ret(n, m, M);
            F.set_les(n, m, M);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++) M(i, j) = prand(seed);
            Q.set_ret(n, m, M);
            Q.set_les(n, m, M);
        }
        for (int m = 0; m <= ntau; m++) {
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++) M(i, j) = 0.05 * prand(seed);
            F.set_tv(n, m, M);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++) M(i, j) = prand(seed);
            Q.set_tv(n, m, M);
        }
    }

    integration::Integrator<double> I(k);
    auto t0 = std::chrono::steady_clock::now();
    vie2_mat(G, F, F, Q, beta, I, CNTR_MAT_FIXPOINT);
    vie2_start(G, F, F, Q, I, beta, h);
    for (int n = k + 1; n <= nt; n++) {
        vie2_timestep(n, G, F, F, Q, I, beta, h);
    }
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    double gmax = std::abs(G.matptr(0)[0]);
    int fin = (std::isfinite(gmax) && gmax < 1e8) ? 1 : 0;
    printf("nessi vie2 nt=%d ntau=%d d=%d k=%d: %.4f s (finite: %d)\n", nt,
           ntau, d, k, dt, fin);
    return 0;
}
