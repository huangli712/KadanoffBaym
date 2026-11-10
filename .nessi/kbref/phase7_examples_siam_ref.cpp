// kbref generator: NESSi examples/programs/ness2_siam.cpp (lines 20-113)
// with small parameters, without the HDF5 output.
//
// Usage: phase7_examples_siam_ref <output-dir>
// Output: <output-dir>/phase7_examples_siam.dat (kbref-v1, data_kind
// phase7_example, case siam).
//
// Parameters (the same as the reference test in
// test/basic/t_ness_examples.jl): U = 1.5, beta = 10, V = 0.8,
// epsd = 0.3, eta = 0.1, Nft = 512, h = 0.05.
//
// Records: G^R/G^< of the interacting Green's function at the time and
// frequency indices n = 0, 1, 5, 20, 100, 255, 256, 300, 511; the density
// 2 Re tr rho; the current 4 Im tr [SL * G] (j = 0) and -4 Im tr [G * SL]
// (j = 1); Eint = Re tr [SU * G]. The example has no iterations, no
// iteration error is written. NESSi overwrites SLG with [G * SL]
// (ness2_siam.cpp:95-96); here [SL * G] and [G * SL] are computed
// separately, and the inputs of convolution_density_matrix are copied
// (NESSi transforms them in place).
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>

// CNTR_NO_EXTERN_TEMPLATES and CNTR_USE_NESS are defined by the target.
#include "cntr/cntr.hpp"
#include "cntr/ness2.hpp"
#include "reference_io.hpp"

using namespace ness2;
typedef std::complex<double> cplx;

class box_dos {
  public:
    double hi_, lo_, wc_, nu_;
    box_dos() : hi_(15.0), lo_(-15.0), wc_(10.0), nu_(3.0) {}
    double operator()(double w) {
        return 1.0 / ((1 + std::exp(nu_ * (w - wc_))) * (1 + std::exp(-nu_ * (wc_ + w)))) / M_PI;
    }
};

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: phase7_examples_siam_ref <output-dir>\n");
        return 1;
    }
    const std::string outdir = argv[1];
    FILE *fp = kbref::open_output(outdir, "phase7_examples_siam.dat");
    kbref::Metadata meta;
    meta.push_back(std::make_pair("generator", std::string("phase7_examples_siam_ref")));
    meta.push_back(std::make_pair("case_set", std::string("phase7_examples")));
    meta.push_back(std::make_pair("case", std::string("siam")));
    kbref::write_metadata(fp, "phase7_example", meta);
    kbref::Phase7Schema s = kbref::phase7_schema("phase7_example", meta);

    const double U = 1.5, beta = 10.0, V = 0.8, epsd = 0.3, eta = 0.1, h = 0.05;
    const int Nft = 512, size = 1;

    herm_matrix_ness G0(Nft, size), G(Nft, size), SL(Nft, size), SR(Nft, size);
    herm_matrix_ness Sbath(Nft, size), SU(Nft, size), S(Nft, size), W(Nft, size);

    box_dos dos;
    green_equilibrium_ness(FERMION, SL, dos, beta, +0.5 * V, h, FFT_TRAPEZ);
    green_equilibrium_ness(FERMION, SR, dos, beta, -0.5 * V, h, FFT_TRAPEZ);
    Sbath = SL;
    Sbath.incr(SR, 1.0, fft_domain::time);

    cdmatrix eps_matrix(1, 1);
    eps_matrix(0, 0) = epsd;
    double mu = 0.0;
    {
        herm_matrix_ness Sb(Nft, size);
        Sb = Sbath;
        dyson(G0, mu, eps_matrix, Sb, h, FFT_TRAPEZ, BATH_GAUSS, eta, beta, mu);
    }

    Bubble1_ness(W, G0, G0);
    W.smul(U * U, fft_domain::time);
    Bubble2_ness(SU, G0, W);
    SU.smul(-1.0, fft_domain::time);

    S = Sbath;
    S.incr(SU, 1.0, fft_domain::time);
    {
        herm_matrix_ness Sc(Nft, size);
        Sc = S;
        dyson(G, mu, eps_matrix, Sc, h, FFT_TRAPEZ, BATH_GAUSS, eta, beta, mu);
    }

    cdmatrix SLG, GSL, SUG, rho;
    {
        herm_matrix_ness A(Nft, size), B(Nft, size);
        A = SL;
        B = G;
        convolution_density_matrix(SLG, FERMION, A, B, h);
        A = G;
        B = SL;
        convolution_density_matrix(GSL, FERMION, A, B, h);
        A = SU;
        B = G;
        convolution_density_matrix(SUG, FERMION, A, B, h);
    }
    density_matrix(rho, FERMION, G);
    double current = 4.0 * SLG.trace().imag();
    double current1 = -4.0 * GSL.trace().imag();
    double eint = SUG.trace().real();
    double dens = 2 * rho.trace().real();

    G.integral_transform_to_freq(h, FFT_TRAPEZ);

    const int ns[] = {0, 1, 5, 20, 100, 255, 256, 300, 511};
    cdmatrix M;
    for (int n : ns) {
        G.get_ret(n, M, fft_domain::time);
        kbref::write_phase7_row(fp, s, "gf_ret_time", 0, n, 0, 0, M(0, 0));
        G.get_les(n, M, fft_domain::time);
        kbref::write_phase7_row(fp, s, "gf_les_time", 0, n, 0, 0, M(0, 0));
        G.get_ret(n, M, fft_domain::freq);
        kbref::write_phase7_row(fp, s, "gf_ret_freq", 0, n, 0, 0, M(0, 0));
        G.get_les(n, M, fft_domain::freq);
        kbref::write_phase7_row(fp, s, "gf_les_freq", 0, n, 0, 0, M(0, 0));
    }
    kbref::write_phase7_row(fp, s, "density", 0, 0, 0, 0, cplx(dens, 0.0));
    kbref::write_phase7_row(fp, s, "current", 0, 0, 0, 0, cplx(current, 0.0));
    kbref::write_phase7_row(fp, s, "current", 0, 1, 0, 0, cplx(current1, 0.0));
    kbref::write_phase7_row(fp, s, "eint", 0, 0, 0, 0, cplx(eint, 0.0));

    std::fclose(fp);
    return 0;
}
