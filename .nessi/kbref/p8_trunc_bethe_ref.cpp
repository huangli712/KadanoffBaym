// kbref generator: NESSi examples/programs/trunc_bethe_start.cpp (full
// start, lines 84-147) followed by trunc_bethe.cpp (moving propagation,
// lines 145-185) in one process, without HDF5.
//
// Usage: p8_trunc_bethe_ref <output-dir>
// Output: <output-dir>/p8_trunc_bethe.dat (kbref-v1, data_kind series,
// case_set trunc_bethe).
//
// Parameters (the same as TB_PARAMS in test/basic/t_trunc_bethe.jl):
// nt = 15, ntau = 50 (NESSi), h = 0.02, beta = 5, U = 1 (U = 0 on the
// Matsubara branch), mu = 0, SolveOrder = 5, BootstrapMaxIter = 40,
// BootstrapMaxErr = 1e-10, CorrectorSteps = 3, tc = 15, tmax = 25. The
// lattice functions G_k are not propagated (they do not enter G).
//
// Records (step = NESSi time step):
//   density  step = 0..tmax   i sig G^<(t,t) of the window rows / leading row
//   error    step = tc+1..tmax  corrector distance of the last iteration
//   ekin     step = tc+1..tmax  truncated kinetic integral (see below)
//   g_ret, g_less, sigma_ret, sigma_less  step = tc+1..tmax, k = deltj:
//            the leading rows G(t,t-deltj), Sigma(t,t-deltj)
// The kinetic integral uses a verbatim copy of get_kinetic_trunc_timestep
// (cntr_dyson_moving_impl.hpp:20-64, the debug output removed); this
// header is not part of the installed libcntr build.
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>

#include "cntr/cntr.hpp"
#include "reference_io.hpp"

typedef std::complex<double> cplx;
typedef cntr::herm_matrix<double> KGreen;
typedef cntr::herm_matrix_timestep<double> KGreenTstp;
typedef cntr::herm_matrix_moving<double> KTrunc;
typedef cntr::herm_matrix_timestep_moving<double> KTruncTstp;
typedef cntr::function<double> KFunc;
typedef cntr::function_moving<double> KFuncMov;

static double kinetic_trunc(KTrunc &G, KTrunc &D, integration::Integrator<double> &I,
                            double h) {
    int k = I.get_k();
    int tc = G.tc();
    cplx s(0.0, 0.0), g, gl, dl, dr;
    for (int l = 0; l <= tc; l++) {
        double w = (l <= k) ? I.gregory_omega(l) : (l >= tc - k ? I.gregory_omega(tc - l) : 1.0);
        G.get_ret(0, l, g);
        G.get_les(0, l, gl);
        D.get_les(0, l, dl);
        D.get_ret(0, l, dr);
        s += h * w * std::conj(g) * dl;
        s += h * w * (-std::conj(gl)) * dr;
    }
    return s.imag();
}

static void sigma_full(int tstp, KGreen &Sigma, KGreen &G, KFunc &U) {
    KGreenTstp W(tstp, Sigma.ntau(), Sigma.size1(), BOSON);
    cntr::Bubble1(tstp, W, G, G);
    W.left_multiply(tstp, U);
    W.right_multiply(tstp, U);
    cntr::Bubble2(tstp, Sigma, G, W);
    Sigma.smul(tstp, -1.0);
}

static void kernel_full(int tstp, KGreen &K, KGreen &Sigma, KGreen &G) {
    K.set_timestep(tstp, Sigma);
    K.incr_timestep(tstp, G, 1.0);
}

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

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: p8_trunc_bethe_ref <output-dir>\n");
        return 1;
    }
    const int nt = 15, ntau = 50, SolveOrder = 5, BootstrapMaxIter = 40, CorrectorSteps = 3;
    const int tc = 15, tmax = 25;
    const double beta = 5.0, h = 0.02, U1 = 1.0, BootstrapMaxErr = 1e-10, mu = 0.0;

    FILE *fp = kbref::open_output(argv[1], "p8_trunc_bethe.dat");
    kbref::Metadata meta;
    meta.push_back(std::make_pair("generator", std::string("p8_trunc_bethe_ref")));
    meta.push_back(std::make_pair("case_set", std::string("trunc_bethe")));
    meta.push_back(std::make_pair(
        "components", std::string("density,error,ekin,g_ret,g_less,sigma_ret,sigma_less")));
    meta.push_back(std::make_pair("nsteps", std::to_string(tmax + 1)));
    meta.push_back(std::make_pair("nk", std::to_string(tc + 1)));
    meta.push_back(std::make_pair("size", std::string("1")));
    meta.push_back(std::make_pair("source", std::string("trunc_bethe_start.cpp+trunc_bethe.cpp")));
    meta.push_back(std::make_pair("tc", std::to_string(tc)));
    meta.push_back(std::make_pair("nt", std::to_string(nt)));
    meta.push_back(std::make_pair("ntau", std::to_string(ntau)));
    kbref::write_metadata(fp, "series", meta);
    kbref::SeriesSchema s = kbref::series_schema(meta);

    // Full start (trunc_bethe_start.cpp)
    KGreen G(nt, ntau, 1, FERMION), Sigma(nt, ntau, 1, FERMION), K(nt, ntau, 1, FERMION);
    KFunc U(nt, 1), eloc(nt, 1);
    for (int t = 0; t <= nt; t++) {
        cdmatrix ut = cdmatrix::Constant(1, 1, U1);
        U.set_value(t, ut);
    }
    cntr::green_equilibrium_mat_bethe(G, beta);
    Sigma.set_timestep_zero(-1);
    kernel_full(-1, K, Sigma, G);
    {
        int tstp = SolveOrder, iter;
        KGreenTstp gtemp(tstp, ntau, 1);
        cntr::set_t0_from_mat(G);
        for (iter = 0; iter <= BootstrapMaxIter; iter++) {
            gtemp.set_timestep(tstp, G);
            for (int n = 0; n <= SolveOrder; n++) {
                sigma_full(n, Sigma, G, U);
                kernel_full(n, K, Sigma, G);
            }
            cntr::dyson_start(G, mu, eloc, K, beta, h, SolveOrder);
            double err = cntr::distance_norm2(tstp, gtemp, G);
            if (err < BootstrapMaxErr && iter > 3) break;
        }
        if (iter > BootstrapMaxIter) {
            std::fprintf(stderr, "bootstrapping not converged\n");
            return 1;
        }
    }
    for (int tstp = SolveOrder + 1; tstp <= nt; tstp++) {
        KGreenTstp gtemp(tstp, ntau, 1);
        cntr::extrapolate_timestep(tstp - 1, G, SolveOrder);
        for (int iter = 0; iter <= CorrectorSteps; iter++) {
            G.get_timestep(tstp, gtemp);
            sigma_full(tstp, Sigma, G, U);
            kernel_full(tstp, K, Sigma, G);
            cntr::dyson_timestep(tstp, G, mu, eloc, K, beta, h, SolveOrder);
        }
    }

    // Handoff (trunc_bethe.cpp)
    KTrunc G_t(tc, 1, FERMION), Sigma_t(tc, 1, FERMION), K_t(tc, 1, FERMION);
    KFuncMov U_t(tc, 1), eloc_t(tc, 1);
    G_t.set_from_G_backward(G, G, tc);
    Sigma_t.set_from_G_backward(Sigma, Sigma, tc);
    for (int t1 = 0; t1 <= tc; t1++) {
        cdmatrix m = cdmatrix::Constant(1, 1, U1);
        U_t.set_value(t1, m);
        cdmatrix z = cdmatrix::Zero(1, 1);
        eloc_t.set_value(t1, z);
    }
    K_t = G_t;
    for (int t1 = 0; t1 <= tc; t1++) K_t.incr_timestep(t1, Sigma_t, t1, 1.0);
    for (int t1 = 0; t1 <= tc; t1++)
        kbref::write_series_row(fp, s, "density", tc - t1, -1, -1, -1,
                                cplx(G_t.density_matrix(t1).real(), 0.0));

    integration::Integrator<double> I(SolveOrder);
    for (int tstp = tc + 1; tstp <= tmax; tstp++) {
        double err = 0.0;
        KTruncTstp gtmp(tc, 1, FERMION);
        Sigma_t.forward();
        K_t.forward();
        G_t.forward();
        cntr::extrapolate_timestep(G_t, SolveOrder);
        for (int iter = 0; iter <= CorrectorSteps; iter++) {
            gtmp.set_timestep(G_t, 0);
            sigma_moving(Sigma_t, G_t, U_t);
            kernel_moving(K_t, Sigma_t, G_t);
            cntr::dyson_timestep(G_t, K_t, eloc_t, mu, SolveOrder, h);
            err = cntr::distance_norm2(G_t, 0, gtmp);
        }
        kbref::write_series_row(fp, s, "density", tstp, -1, -1, -1,
                                cplx(G_t.density_matrix(0).real(), 0.0));
        kbref::write_series_row(fp, s, "error", tstp, -1, -1, -1, cplx(err, 0.0));
        kbref::write_series_row(fp, s, "ekin", tstp, -1, -1, -1,
                                cplx(kinetic_trunc(G_t, G_t, I, h), 0.0));
        for (int j = 0; j <= tc; j++) {
            cplx x;
            G_t.get_ret(0, j, x);
            kbref::write_series_row(fp, s, "g_ret", tstp, j, 0, 0, x);
            G_t.get_les(0, j, x);
            kbref::write_series_row(fp, s, "g_less", tstp, j, 0, 0, x);
            Sigma_t.get_ret(0, j, x);
            kbref::write_series_row(fp, s, "sigma_ret", tstp, j, 0, 0, x);
            Sigma_t.get_les(0, j, x);
            kbref::write_series_row(fp, s, "sigma_less", tstp, j, 0, 0, x);
        }
    }
    std::fclose(fp);
    return 0;
}
