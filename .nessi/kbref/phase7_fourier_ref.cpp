// kbref generator: FFT, corrected/adaptive Fourier integrals, Matsubara
// Fourier transforms, and the Fourier Matsubara VIE2 of NESSi.
//
// Usage: phase7_fourier_ref <output-dir>
// Output: <output-dir>/phase7_fourier.dat (kbref-v1, data_kind
// phase7_fourier, see reference_io.hpp for the index usage).
//
// Inputs (the same closed-form formulas are used in the Julia test
// test/nessi/t_phase7_reference.jl; nothing is read back):
//   * FFT: N = 2, 3, 4, 5, 6, 10, 12, 16, 25, 60, 97, 101,
//     x_n = cos(0.3 n + 0.1 N) + i sin(0.7 n^2 / N); FFTW forward
//     (e^{-2 pi i nk/N}) and backward (e^{+2 pi i nk/N}), unnormalized.
//   * dftcorr: theta = 0, 1e-8, -1e-8, 0.049999, -0.049999, 0.050001,
//     -0.050001, 0.7 (index 0..7), linear (j = 1) and cubic (j = 3).
//   * adaptive: f0 = 1 on [0, 1], f1 = e^{-x} on [0, 2], f2 = e^{-x^2} on
//     [-5, 5]; omega = -3, -0.7, 0, 0.5, 2.3, 10 (index 0..5);
//     dft_cplx with 32 intervals, adft_func with n = 20, limit = 40.
//   * Matsubara: size 2, ntau = 20, beta = 2.5, case 0 fermion, case 1
//     boson, G(tau) = [-0.6 e^{-0.4 tau}, 0.2 cos(1.3 tau) + 0.05 i tau;
//     0.2 cos(1.3 tau) - 0.05 i tau, -0.3 e^{0.25 tau}];
//     matsubara_ft with order 0, 1, 3 and m = -25, -3, -1, 0, 1, 2, 7, 30.
//   * VIE2 (vie2_mat_fourier_dispatch, order 3): size 2, ntau = 40,
//     beta = 3, case 0 fermion, case 1 boson,
//     F(tau) = 0.4 [e^{-0.5 tau}, 0.1 + 0.05 i tau; 0.1 - 0.05 i tau,
//     e^{-1.2 tau}], Q(tau) = [-e^{-0.3 tau}, 0.05; 0.05, -0.5 e^{0.2 tau}],
//     pcf = 20 (NESSi default) and pcf = 10 (counterexample).
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include <vector>

#include <fftw3.h>

// CNTR_NO_EXTERN_TEMPLATES is defined by the CMake target.
#include "cntr/cntr.hpp"
#include "cntr/fourier.hpp"
#include "reference_io.hpp"

using namespace cntr;
typedef std::complex<double> cplx;

static const cplx CI(0.0, 1.0);

struct f_const {
    cplx operator()(double) { return cplx(1.0, 0.0); }
};
struct f_exp {
    cplx operator()(double x) { return cplx(std::exp(-x), 0.0); }
};
struct f_gauss {
    cplx operator()(double x) { return cplx(std::exp(-x * x), 0.0); }
};

template <class F>
static void adaptive_rows(FILE *fp, const kbref::Phase7Schema &s, int id, F &f,
                          double lo, double hi) {
    const double omegas[] = {-3.0, -0.7, 0.0, 0.5, 2.3, 10.0};
    const int n = 32;
    std::vector<cplx> vals(n + 1);
    for (int k = 0; k <= n; k++) vals[k] = f(lo + (hi - lo) * k / n);
    for (int w = 0; w < 6; w++) {
        double om = omegas[w];
        cplx res, err;
        fourier::dft_cplx(om, n, lo, hi, vals.data(), res, err);
        fourier::adft_func adft;
        adft.sample(om, lo, hi, f, 20, 40);
        cplx res2, err2;
        adft.dft(om, res2, err2);
        kbref::write_phase7_row(fp, s, "adaptive", id, w, 0, 0, cplx(om, 0.0));
        kbref::write_phase7_row(fp, s, "adaptive", id, w, 1, 0, res);
        kbref::write_phase7_row(fp, s, "adaptive", id, w, 2, 0, res2);
    }
}

static cdmatrix mats_input(double tau) {
    cdmatrix M(2, 2);
    M(0, 0) = -0.6 * std::exp(-0.4 * tau);
    M(0, 1) = 0.2 * std::cos(1.3 * tau) + 0.05 * CI * tau;
    M(1, 0) = 0.2 * std::cos(1.3 * tau) - 0.05 * CI * tau;
    M(1, 1) = -0.3 * std::exp(0.25 * tau);
    return M;
}

static cdmatrix vie2_F(double tau) {
    cdmatrix M(2, 2);
    M(0, 0) = 0.4 * std::exp(-0.5 * tau);
    M(0, 1) = 0.4 * (0.1 + 0.05 * CI * tau);
    M(1, 0) = 0.4 * (0.1 - 0.05 * CI * tau);
    M(1, 1) = 0.4 * std::exp(-1.2 * tau);
    return M;
}

static cdmatrix vie2_Q(double tau) {
    cdmatrix M(2, 2);
    M(0, 0) = -std::exp(-0.3 * tau);
    M(0, 1) = 0.05;
    M(1, 0) = 0.05;
    M(1, 1) = -0.5 * std::exp(0.2 * tau);
    return M;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: phase7_fourier_ref <output-dir>\n");
        return 1;
    }
    const std::string outdir = argv[1];
    FILE *fp = kbref::open_output(outdir, "phase7_fourier.dat");
    kbref::Metadata meta;
    meta.push_back(std::make_pair("generator", std::string("phase7_fourier_ref")));
    meta.push_back(std::make_pair("case_set", std::string("fft_dftcorr_adaptive_matsubara_vie2")));
    meta.push_back(std::make_pair("size_max", std::string("2")));
    kbref::write_metadata(fp, "phase7_fourier", meta);
    kbref::Phase7Schema s = kbref::phase7_schema("phase7_fourier", meta);

    // FFT
    const int Ns[] = {2, 3, 4, 5, 6, 10, 12, 16, 25, 60, 97, 101};
    for (int N : Ns) {
        std::vector<cplx> x(N), yf(N), yb(N);
        for (int n = 0; n < N; n++)
            x[n] = cplx(std::cos(0.3 * n + 0.1 * N), std::sin(0.7 * n * n / N));
        fftw_plan pf = fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex *>(x.data()),
                                        reinterpret_cast<fftw_complex *>(yf.data()),
                                        FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_plan pb = fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex *>(x.data()),
                                        reinterpret_cast<fftw_complex *>(yb.data()),
                                        FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(pf);
        fftw_execute(pb);
        fftw_destroy_plan(pf);
        fftw_destroy_plan(pb);
        for (int n = 0; n < N; n++) kbref::write_phase7_row(fp, s, "fft_input", N, n, 0, 0, x[n]);
        for (int n = 0; n < N; n++) kbref::write_phase7_row(fp, s, "fft_forward", N, n, 0, 0, yf[n]);
        for (int n = 0; n < N; n++) kbref::write_phase7_row(fp, s, "fft_backward", N, n, 0, 0, yb[n]);
    }

    // Correction factors
    const double thetas[] = {0.0, 1e-8, -1e-8, 0.049999, -0.049999, 0.050001, -0.050001, 0.7};
    for (int i = 0; i < 8; i++) {
        double corfac;
        cplx endcor[4];
        fourier::get_dftcorr_linear(thetas[i], &corfac, endcor);
        kbref::write_phase7_row(fp, s, "dftcorr", i, 1, 0, 0, cplx(thetas[i], 0.0));
        kbref::write_phase7_row(fp, s, "dftcorr", i, 1, 1, 0, cplx(corfac, 0.0));
        for (int a = 0; a < 2; a++)
            kbref::write_phase7_row(fp, s, "dftcorr", i, 1, a + 2, 0, endcor[a]);
        fourier::get_dftcorr_cubic(thetas[i], &corfac, endcor);
        kbref::write_phase7_row(fp, s, "dftcorr", i, 3, 0, 0, cplx(thetas[i], 0.0));
        kbref::write_phase7_row(fp, s, "dftcorr", i, 3, 1, 0, cplx(corfac, 0.0));
        for (int a = 0; a < 4; a++)
            kbref::write_phase7_row(fp, s, "dftcorr", i, 3, a + 2, 0, endcor[a]);
    }

    // Corrected and adaptive Fourier integrals
    f_const f0;
    f_exp f1;
    f_gauss f2;
    adaptive_rows(fp, s, 0, f0, 0.0, 1.0);
    adaptive_rows(fp, s, 1, f1, 0.0, 2.0);
    adaptive_rows(fp, s, 2, f2, -5.0, 5.0);

    // Matsubara Fourier transforms
    {
        const int ntau = 20;
        const double beta = 2.5;
        const int ms[] = {-25, -3, -1, 0, 1, 2, 7, 30};
        const int orders[] = {0, 1, 3};
        for (int c = 0; c < 2; c++) {
            int sig = (c == 0) ? -1 : 1;
            herm_matrix<double> G(0, ntau, 2, sig);
            for (int r = 0; r <= ntau; r++) {
                cdmatrix M = mats_input(r * beta / ntau);
                G.set_mat(r, M);
            }
            std::vector<cplx> mdft((ntau + 1) * 4);
            matsubara_dft<double, herm_matrix<double>, 2>(mdft.data(), G, sig);
            for (int m = 0; m <= ntau; m++)
                for (int a = 0; a < 2; a++)
                    for (int b = 0; b < 2; b++)
                        kbref::write_phase7_row(fp, s, "matsubara_dft_result", c, m, a, b,
                                                mdft[m * 4 + a * 2 + b]);
            for (int order : orders)
                for (int m : ms) {
                    cplx res[4];
                    matsubara_ft<double, herm_matrix<double>, 2>(res, m, G, mdft.data(), sig,
                                                                 beta, order);
                    for (int a = 0; a < 2; a++)
                        for (int b = 0; b < 2; b++)
                            kbref::write_phase7_row(fp, s, "matsubara_ft_result", 10 * c + order,
                                                    m, a, b, res[a * 2 + b]);
                }
        }
    }

    // Fourier Matsubara VIE2, pcf = 20 and pcf = 10
    {
        const int ntau = 40;
        const double beta = 3.0;
        for (int c = 0; c < 2; c++) {
            int sig = (c == 0) ? -1 : 1;
            herm_matrix<double> F(0, ntau, 2, sig), Q(0, ntau, 2, sig);
            for (int r = 0; r <= ntau; r++) {
                double tau = r * beta / ntau;
                cdmatrix MF = vie2_F(tau), MQ = vie2_Q(tau);
                F.set_mat(r, MF);
                Q.set_mat(r, MQ);
            }
            const int pcfs[] = {20, 10};
            for (int pcf : pcfs) {
                herm_matrix<double> G(0, ntau, 2, sig);
                vie2_mat_fourier_dispatch<double, herm_matrix<double>, 2>(G, F, F, Q, beta, pcf, 3);
                const char *comp = (pcf == 20) ? "vie2_pcf20" : "vie2_pcf10";
                cdmatrix M;
                for (int r = 0; r <= ntau; r++) {
                    G.get_mat(r, M);
                    for (int a = 0; a < 2; a++)
                        for (int b = 0; b < 2; b++)
                            kbref::write_phase7_row(fp, s, comp, c, r, a, b, M(a, b));
                }
            }
        }
    }

    std::fclose(fp);
    return 0;
}
