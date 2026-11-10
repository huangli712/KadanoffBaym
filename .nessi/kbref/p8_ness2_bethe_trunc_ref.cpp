// kbref generator: NESSi examples/programs/ness2_bethe.cpp (steady state,
// lines 37-164) followed by ness2_bethe_trunc.cpp (lines 82-139) in one
// process, without HDF5.
//
// Usage: p8_ness2_bethe_trunc_ref <output-dir>
// Output: <output-dir>/p8_ness2_bethe_trunc.dat (kbref-v1, data_kind
// series, case_set ness2_bethe_trunc).
//
// Parameters (the same as the "NESS start" test in
// test/basic/t_trunc_bethe.jl): Nft = 256, h = 0.02, beta = 10, U = 2,
// mu = 0, Niter = 20, errmax = 1e-6, mixing = 0.6, ipt_flag = 1;
// tc = 20, tmax = 4, downsampling = 2, SolveOrder = 5, CorrectorSteps = 3.
//
// Records (step = NESS relative step, KB time step = step + 1):
//   density        step = 0..tmax, i sig G^<(t,t) of the leading row
//   g_ret, g_less  step = 0..tmax, k = deltj: leading row G(t,t-deltj)
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
typedef cntr::herm_matrix_moving<double> KTrunc;
typedef cntr::herm_matrix_timestep_moving<double> KTruncTstp;
typedef cntr::function_moving<double> KFuncMov;

static void sigma_moving(KTrunc &Sigma, KTrunc &G, KFuncMov &U) {
    KTruncTstp W(Sigma.tc(), Sigma.size1(), BOSON);
    cntr::Bubble1(W, G, G);
    W.left_multiply(U);
    W.right_multiply(U);
    cntr::Bubble2(Sigma, G, W);
    Sigma.smul(0, -1.0);
}

static void kernel_moving(KTrunc &K, KTrunc &Sigma, KTrunc &G) {
    K.set_timestep(0, Sigma, 0);
    K.incr_timestep(0, G, 0, 1.0);
}

static void write_row0(FILE *fp, const kbref::SeriesSchema &s, int step, KTrunc &G) {
    kbref::write_series_row(fp, s, "density", step, -1, -1, -1,
                            cplx(G.density_matrix(0).real(), 0.0));
    for (int j = 0; j <= G.tc(); j++) {
        cplx x;
        G.get_ret(0, j, x);
        kbref::write_series_row(fp, s, "g_ret", step, j, 0, 0, x);
        G.get_les(0, j, x);
        kbref::write_series_row(fp, s, "g_less", step, j, 0, 0, x);
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: p8_ness2_bethe_trunc_ref <output-dir>\n");
        return 1;
    }
    const int size = 1, Nft = 256, N_it = 20, ipt_flag = 1;
    const double h = 0.02, beta = 10.0, U = 2.0, mu = 0.0, mix = 0.6, errmax = 1e-6;
    const int tc = 20, tmax = 4, downsampling = 2, SolveOrder = 5, CorrectorSteps = 3;

    FILE *fp = kbref::open_output(argv[1], "p8_ness2_bethe_trunc.dat");
    kbref::Metadata meta;
    meta.push_back(std::make_pair("generator", std::string("p8_ness2_bethe_trunc_ref")));
    meta.push_back(std::make_pair("case_set", std::string("ness2_bethe_trunc")));
    meta.push_back(std::make_pair("components", std::string("density,g_ret,g_less")));
    meta.push_back(std::make_pair("nsteps", std::to_string(tmax + 1)));
    meta.push_back(std::make_pair("nk", std::to_string(tc + 1)));
    meta.push_back(std::make_pair("size", std::string("1")));
    meta.push_back(std::make_pair("source", std::string("ness2_bethe.cpp+ness2_bethe_trunc.cpp")));
    meta.push_back(std::make_pair("tc", std::to_string(tc)));
    meta.push_back(std::make_pair("index_map", std::string("kb_tstp=step+1")));
    kbref::write_metadata(fp, "series", meta);
    kbref::SeriesSchema s = kbref::series_schema(meta);

    // Steady state (ness2_bethe.cpp)
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
        if (err < errmax && iter > 3) break;
    }

    // Truncated propagation (ness2_bethe_trunc.cpp)
    const double h_cntr = h * downsampling;
    KTrunc Gcntr(tc, 1, FERMION), Scntr(tc, 1, FERMION), Kcntr(tc, 1, FERMION),
        Gweisscntr(tc, 1, FERMION);
    KFuncMov U_t(tc, 1), eloc_t(tc, 1);
    herm_matrix_ness Gness1;
    Gness1 = downsample(G_ness, downsampling);
    ness2cntr(Gcntr, Gness1);
    Gness1 = downsample(SU_ness, downsampling);
    ness2cntr(Scntr, Gness1);
    Gness1 = downsample(Gweiss_ness, downsampling);
    ness2cntr(Gweisscntr, Gness1);
    Kcntr = Gcntr;
    for (int t1 = 0; t1 <= tc; t1++) {
        cdmatrix m = cdmatrix::Constant(1, 1, U);
        U_t.set_value(t1, m);
        m = cdmatrix::Zero(1, 1);
        eloc_t.set_value(t1, m);
        Kcntr.incr_timestep(t1, Scntr, t1, 1.0);
    }
    write_row0(fp, s, 0, Gcntr);
    for (int tstp = 1; tstp <= tmax; tstp++) {
        KTruncTstp gtmp(tc, 1, FERMION);
        Scntr.forward();
        Kcntr.forward();
        Gcntr.forward();
        Gweisscntr.forward();
        cntr::extrapolate_timestep(Gcntr, SolveOrder);
        for (int iter = 0; iter <= CorrectorSteps; iter++) {
            gtmp.set_timestep(Gcntr, 0);
            if (ipt_flag) {
                cntr::dyson_timestep(Gweisscntr, Gcntr, eloc_t, mu, SolveOrder, h_cntr);
            } else {
                Gweisscntr.set_timestep(0, Gcntr, 0);
            }
            sigma_moving(Scntr, Gweisscntr, U_t);
            kernel_moving(Kcntr, Scntr, Gcntr);
            cntr::dyson_timestep(Gcntr, Kcntr, eloc_t, mu, SolveOrder, h_cntr);
        }
        write_row0(fp, s, tstp, Gcntr);
    }
    std::fclose(fp);
    return 0;
}
