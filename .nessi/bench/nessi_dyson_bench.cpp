// NESSi counterpart of examples/dyson_bench.rs (same parameters, same
// timed region: dyson_start + the dyson_timestep loop; the Matsubara
// preparation is outside the timer in both programs).
#include <cntr/cntr.hpp>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <Eigen/Dense>

using namespace cntr;
typedef std::complex<double> cplx;

int main(int argc, char **argv) {
    int nt = 500;
    int ntau = 201;   // KB points (201) -> NESSi intervals (200, even)
    int d = 2;
    int k = 5;
    double beta = 10.0, h = 0.05, mu = 0.0;
    if (argc > 1) nt = atoi(argv[1]);

    herm_matrix<double> G(nt, ntau - 1, d, d, -1 /*fermion*/);
    herm_matrix<double> Sigma(nt, ntau - 1, d, d, -1);
    function<double> H(nt, d, d);

    // deterministic Hamiltonian and self-energy (same values as the
    // Rust example): H(t) = H0 + diag(0, 0.02)
    Eigen::MatrixXcd HM = Eigen::MatrixXcd::Zero(d, d);
    for (int i = 0; i < d; i++) HM(i, i) = cplx(0.3 - 0.4 * i + 0.02 * i, 0);
    for (int t = -1; t <= nt; t++) H.set_value(t, HM);

    Eigen::MatrixXcd SM = Eigen::MatrixXcd::Zero(d, d);
    Eigen::MatrixXcd SR = Eigen::MatrixXcd::Zero(d, d);
    Eigen::MatrixXcd SL = Eigen::MatrixXcd::Zero(d, d);
    Eigen::MatrixXcd SX = Eigen::MatrixXcd::Zero(d, d);
    for (int i = 0; i < d; i++) {
        SM(i, i) = cplx(-0.2 * (0.1 + 0.05 * i), 0);
        SR(i, i) = cplx(0.15 * (0.1 + 0.05 * i), 0);
        SL(i, i) = cplx(0.1 * (0.1 + 0.05 * i), 0);
        SX(i, i) = cplx(0.12 * (0.1 + 0.05 * i), 0);
    }
    for (int m = 0; m <= ntau - 1; m++) Sigma.set_mat(m, SM);
    for (int n = 0; n <= nt; n++) {
        for (int m = 0; m <= n; m++) {
            Sigma.set_ret(n, m, SR);
            Sigma.set_les(n, m, SL);
        }
        for (int m = 0; m <= ntau - 1; m++) Sigma.set_tv(n, m, SX);
    }

    // Matsubara preparation (outside the timer; the Rust example uses
    // from_ham outside its timer as well)
    integration::Integrator<double> I(k);
    dyson_mat(G, Sigma, mu, H, I, beta, CNTR_MAT_FIXPOINT, true);

    auto t0 = std::chrono::steady_clock::now();
    dyson_start(G, mu, H, Sigma, I, beta, h);
    for (int n = k + 1; n <= nt; n++) {
        dyson_timestep(n, G, mu, H, Sigma, I, beta, h);
    }
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // sanity through the Matsubara pointer: the propagator stays finite
    double gmax = std::abs(G.matptr(0)[0]);
    int fin = (std::isfinite(gmax) && gmax < 1e6) ? 1 : 0;
    printf("nessi dyson nt=%d ntau=%d d=%d k=%d: %.4f s (finite: %d)\n", nt,
           ntau - 1, d, k, dt, fin);
    return 0;
}
