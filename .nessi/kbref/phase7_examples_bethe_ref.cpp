// kbref generator: NESSi examples/programs/ness2_bethe.cpp (lines 37-164)
// with small parameters, without the HDF5 output.
//
// Usage: phase7_examples_bethe_ref <output-dir>
// Output: <output-dir>/phase7_examples_bethe.dat (kbref-v1, data_kind
// phase7_example, case bethe).
//
// Parameters (the same as the reference test in
// test/basic/t_ness_examples.jl): Nft = 512, h = 0.05, beta = 5,
// U = 1.5, mu = 0.2, Niter = 6, errmax = 1e-30 (all iterations are
// done), mixing = 0.6, ipt_flag = 1.
//
// Records: the error of every iteration (j = 1..Niter); the final G^R/G^<
// at the time and frequency indices n = 0, 1, 5, 20, 100, 255, 256, 300,
// 511; the density 2 Re tr rho and Eint = Re tr [SU * G]. The model has
// no current. The input of dyson (the self-energy argument) and of
// convolution_density_matrix is copied, NESSi transforms it in place.
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

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: phase7_examples_bethe_ref <output-dir>\n");
        return 1;
    }
    const std::string outdir = argv[1];
    FILE *fp = kbref::open_output(outdir, "phase7_examples_bethe.dat");
    kbref::Metadata meta;
    meta.push_back(std::make_pair("generator", std::string("phase7_examples_bethe_ref")));
    meta.push_back(std::make_pair("case_set", std::string("phase7_examples")));
    meta.push_back(std::make_pair("case", std::string("bethe")));
    kbref::write_metadata(fp, "phase7_example", meta);
    kbref::Phase7Schema s = kbref::phase7_schema("phase7_example", meta);

    const int size = 1, Nft = 512, N_it = 6, ipt_flag = 1;
    const double h = 0.05, beta = 5.0, U = 1.5, mu = 0.2, mix = 0.6, errmax = 1e-30;

    herm_matrix_ness G_ness(Nft, size), G_old(Nft, size), SU_ness(Nft, size);
    herm_matrix_ness W_ness(Nft, size), K_ness(Nft, size), Gweiss_ness(Nft, size);
    cdmatrix ham = cdmatrix::Zero(size, size);

    ness2::bethedos dos(-2, 2);
    green_equilibrium_ness(FERMION, G_ness, dos, beta, mu, h, FFT_TRAPEZ);

    for (int iter = 1; iter <= N_it; iter++) {
        G_old = G_ness;
        if (ipt_flag) {
            herm_matrix_ness D(Nft, size);
            D = G_ness;
            dyson(Gweiss_ness, mu, ham, D, h, FFT_TRAPEZ);
        } else {
            Gweiss_ness = G_ness;
        }
        Bubble1_ness(W_ness, Gweiss_ness, Gweiss_ness);
        W_ness.smul(U * U, fft_domain::time);
        Bubble2_ness(SU_ness, Gweiss_ness, W_ness);
        SU_ness.smul(-1.0, fft_domain::time);

        K_ness.set_zero(fft_domain::time);
        K_ness.incr(SU_ness, 1.0, fft_domain::time);
        K_ness.incr(G_ness, 1.0, fft_domain::time);
        {
            herm_matrix_ness Kc(Nft, size);
            Kc = K_ness;
            dyson(G_ness, mu, ham, Kc, h, FFT_TRAPEZ);
        }
        double err = distance_norm2(G_ness, G_old, fft_domain::time);
        G_ness.smul(mix, fft_domain::time);
        G_ness.incr(G_old, 1.0 - mix, fft_domain::time);
        kbref::write_phase7_row(fp, s, "iteration_error", 1, iter, 0, 0, cplx(err, 0.0));
        if (err < errmax && iter > 3) break;
    }

    cdmatrix SUG, rho;
    {
        herm_matrix_ness A(Nft, size), B(Nft, size);
        A = SU_ness;
        B = G_ness;
        convolution_density_matrix(SUG, FERMION, A, B, h);
    }
    density_matrix(rho, FERMION, G_ness);
    double eint = SUG.trace().real();
    double dens = 2 * rho.trace().real();

    G_ness.integral_transform_to_freq(h, FFT_TRAPEZ);

    const int ns[] = {0, 1, 5, 20, 100, 255, 256, 300, 511};
    cdmatrix M;
    for (int n : ns) {
        G_ness.get_ret(n, M, fft_domain::time);
        kbref::write_phase7_row(fp, s, "gf_ret_time", 1, n, 0, 0, M(0, 0));
        G_ness.get_les(n, M, fft_domain::time);
        kbref::write_phase7_row(fp, s, "gf_les_time", 1, n, 0, 0, M(0, 0));
        G_ness.get_ret(n, M, fft_domain::freq);
        kbref::write_phase7_row(fp, s, "gf_ret_freq", 1, n, 0, 0, M(0, 0));
        G_ness.get_les(n, M, fft_domain::freq);
        kbref::write_phase7_row(fp, s, "gf_les_freq", 1, n, 0, 0, M(0, 0));
    }
    kbref::write_phase7_row(fp, s, "density", 1, 0, 0, 0, cplx(dens, 0.0));
    kbref::write_phase7_row(fp, s, "eint", 1, 0, 0, 0, cplx(eint, 0.0));

    std::fclose(fp);
    return 0;
}
