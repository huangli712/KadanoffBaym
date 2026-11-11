// kbref generator: NESSi examples/programs/ness2_bethe_prop.cpp (lines
// 48-197) with small parameters, without the HDF5 output.
//
// Usage: phase7_examples_bethe_prop_ref <output-dir>
// Output: <output-dir>/phase7_examples_bethe_prop.dat (kbref-v1, data_kind
// phase7_example, case bethe_prop).
//
// Parameters (the same as the reference test in
// test/t_ness_examples.jl): nt = 8, ntau = 40 (KB: 41 points),
// h = 0.05, beta = 5, U = 1.5, mu = 0, SolveOrder = 3, ipt_flag = 1,
// Matsubara Dyson CNTR_MAT_FIXPOINT, BootstrapMaxIter = 20,
// BootstrapMaxErr = 1e-8, CorrectorSteps = 2.
//
// Records (time steps in the KB convention, NESSi tstp + 1):
//   * iteration_error: a = stage (0 Matsubara, 1 bootstrapping, 2 time
//     stepping), b = time step, j = iteration;
//   * density: j = time step 0..nt+1 (0 = Matsubara), 2 Re rho;
//   * gf_ret_time, gf_les_time: the last time slice viewed as a steady
//     state (ness2::cntr2ness with Nft = 2 (nt + 1)), j = t = 0..nt, i.e.
//     G^R(t_nt, t_nt - t) and G^<(t_nt - t, t_nt).
// The example computes neither a current nor Eint.
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>

// CNTR_NO_EXTERN_TEMPLATES and CNTR_USE_NESS are defined by the target.
#include "cntr/cntr.hpp"
#include "cntr/ness2.hpp"
#include "reference_io.hpp"

using namespace cntr;
typedef std::complex<double> cplx;

#define GREEN cntr::herm_matrix<double>
#define GREEN_TSTP cntr::herm_matrix_timestep<double>
#define CFUNC cntr::function<double>

void get_Sigma_timestep(int tstp, GREEN &Sigma, GREEN &Gweiss, CFUNC &U) {
    GREEN_TSTP W(tstp, Sigma.ntau(), Sigma.size1(), BOSON);
    cntr::Bubble1(tstp, W, Gweiss, Gweiss);
    W.left_multiply(tstp, U);
    W.right_multiply(tstp, U);
    cntr::Bubble2(tstp, Sigma, Gweiss, W);
    Sigma.smul(tstp, -1.0);
}

void get_K_timestep(int tstp, GREEN &K, GREEN &Sigma, GREEN &G) {
    K.set_timestep(tstp, Sigma);
    K.incr_timestep(tstp, G, 1.0);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: phase7_examples_bethe_prop_ref <output-dir>\n");
        return 1;
    }
    const std::string outdir = argv[1];
    FILE *fp = kbref::open_output(outdir, "phase7_examples_bethe_prop.dat");
    kbref::Metadata meta;
    meta.push_back(std::make_pair("generator", std::string("phase7_examples_bethe_prop_ref")));
    meta.push_back(std::make_pair("case_set", std::string("phase7_examples")));
    meta.push_back(std::make_pair("case", std::string("bethe_prop")));
    kbref::write_metadata(fp, "phase7_example", meta);
    kbref::Phase7Schema s = kbref::phase7_schema("phase7_example", meta);

    const int nt = 8, ntau = 40, SolveOrder = 3, ipt_flag = 1;
    const int BootstrapMaxIter = 20, CorrectorSteps = 2;
    const double h = 0.05, beta = 5.0, U = 1.5, mu = 0.0, BootstrapMaxErr = 1e-8;
    int iter, tstp;

    GREEN G(nt, ntau, 1, FERMION);
    GREEN Gweiss(nt, ntau, 1, FERMION);
    GREEN Sigma(nt, ntau, 1, FERMION);
    GREEN K(nt, ntau, 1, FERMION);
    CFUNC Uvec(nt, 1);
    CFUNC eloc(nt, 1);
    for (tstp = -1; tstp <= nt; tstp++) {
        cdmatrix ut = cdmatrix::Constant(1, 1, U);
        Uvec.set_value(tstp, ut);
        cdmatrix e0 = cdmatrix::Zero(1, 1);
        eloc.set_value(tstp, e0);
    }

    cntr::green_equilibrium_mat_bethe(G, beta);
    Gweiss.set_timestep(-1, G);
    Sigma.set_timestep_zero(-1);
    get_K_timestep(-1, K, Sigma, G);

    // Matsubara
    {
        tstp = -1;
        GREEN_TSTP gtemp(tstp, ntau, 1);
        for (iter = 0; iter <= BootstrapMaxIter; iter++) {
            gtemp.set_timestep(tstp, G);
            if (ipt_flag) {
                cntr::dyson_mat(Gweiss, mu, eloc, G, beta, SolveOrder, CNTR_MAT_FIXPOINT);
            } else {
                Gweiss.set_timestep(-1, G);
            }
            get_Sigma_timestep(tstp, Sigma, Gweiss, Uvec);
            get_K_timestep(tstp, K, Sigma, G);
            cntr::dyson_mat(G, mu, eloc, K, beta, SolveOrder, CNTR_MAT_FIXPOINT);
            double err = cntr::distance_norm2(tstp, gtemp, G);
            kbref::write_phase7_row(fp, s, "iteration_error", 2, iter, 0, 0, cplx(err, 0.0));
            if (err < BootstrapMaxErr && iter > 3) break;
        }
    }

    // Bootstrapping
    {
        tstp = SolveOrder;
        GREEN_TSTP gtemp(tstp, ntau, 1);
        cntr::set_t0_from_mat(G);
        for (iter = 0; iter <= BootstrapMaxIter; iter++) {
            gtemp.set_timestep(tstp, G);
            if (ipt_flag) {
                cntr::dyson_start(Gweiss, mu, eloc, G, beta, h, SolveOrder);
            } else {
                for (int n = 0; n <= SolveOrder; n++) Gweiss.set_timestep(n, G);
            }
            for (int n = 0; n <= SolveOrder; n++) {
                get_Sigma_timestep(n, Sigma, Gweiss, Uvec);
                get_K_timestep(n, K, Sigma, G);
            }
            cntr::dyson_start(G, mu, eloc, K, beta, h, SolveOrder);
            double err = cntr::distance_norm2(tstp, gtemp, G);
            kbref::write_phase7_row(fp, s, "iteration_error", 2, iter, 1, tstp + 1,
                                    cplx(err, 0.0));
            if (err < BootstrapMaxErr && iter > 3) break;
        }
    }

    // Time stepping
    for (tstp = SolveOrder + 1; tstp <= nt; tstp++) {
        GREEN_TSTP gtemp(tstp, ntau, 1);
        cntr::extrapolate_timestep(tstp - 1, G, SolveOrder);
        for (iter = 0; iter <= CorrectorSteps; iter++) {
            G.get_timestep(tstp, gtemp);
            if (ipt_flag) {
                cntr::dyson_timestep(tstp, Gweiss, mu, eloc, G, beta, h, SolveOrder);
            } else {
                Gweiss.set_timestep(tstp, G);
            }
            get_Sigma_timestep(tstp, Sigma, Gweiss, Uvec);
            get_K_timestep(tstp, K, Sigma, G);
            cntr::dyson_timestep(tstp, G, mu, eloc, K, beta, h, SolveOrder);
            double err = cntr::distance_norm2(tstp, gtemp, G);
            kbref::write_phase7_row(fp, s, "iteration_error", 2, iter, 2, tstp + 1,
                                    cplx(err, 0.0));
        }
    }

    // Densities, KB time step tstp + 1
    for (tstp = -1; tstp <= nt; tstp++) {
        cdmatrix rho;
        G.density_matrix(tstp, rho);
        kbref::write_phase7_row(fp, s, "density", 2, tstp + 1, 0, 0,
                                cplx(2.0 * rho(0, 0).real(), 0.0));
    }

    // The last time slice as a steady-state Green's function
    ness2::herm_matrix_ness Gn(2 * (nt + 1), 1);
    ness2::cntr2ness(Gn, G, nt);
    cdmatrix M;
    for (int t = 0; t <= nt; t++) {
        Gn.get_ret(t, M, ness2::fft_domain::time);
        kbref::write_phase7_row(fp, s, "gf_ret_time", 2, t, 0, 0, M(0, 0));
        Gn.get_les(-t, M, ness2::fft_domain::time);
        kbref::write_phase7_row(fp, s, "gf_les_time", 2, t, 0, 0, M(0, 0));
    }

    std::fclose(fp);
    return 0;
}
