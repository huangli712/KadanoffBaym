// kbref generator: moving (truncated) solvers of NESSi.
//
// Usage: p8_moving_solver_ref <output-dir>
// Output: <output-dir>/p8_moving_solver.dat (kbref-v1, data_kind series,
// case_set moving_solver).
//
// Inputs (rebuilt in test/nessi/t_moving_reference.jl): nt = 30,
// ntau = 40 (NESSi), beta = 1, h = 0.02, mu = -0.1, tc = 12, constant
// hermitian H0 (size 1: -0.3; size 2: [-0.3 0.2i; -0.2i 0.4]).
// Sigma = green_from_H(H0) times f(n - m) = 0.3 exp(-((n - m) h)^2) on
// the real-time components (tv and Matsubara set to zero). G is the full
// Dyson solution of order k. The windows are set_from_G_backward at the
// NESSi time step nt.
//
// Cases (step = case index): c = 0..5 is (size, k) = (1,1) (1,3) (1,5)
// (2,1) (2,3) (2,5). Components (k = deltj of the leading row):
//   dyson_ret, dyson_less       leading row after cntr::dyson_timestep
//   bubble1_ret, bubble1_less   Bubble1(C, a, a, G, G, a, a, S, S, b, b)
//   bubble2_ret, bubble2_less   Bubble2(C, a, a, G, G, a, a, S, S, b, b)
//                               (diagonal orbital indices, NESSi transposes
//                               the indices of the lesser in Bubble2)
//   kinetic                     truncated kinetic integral of G and S
//   vie2_ret, vie2_less         truncated VIE2 G + S * G = G (F = Fcc = S,
//                               Q = G) without the tv*vt term: NESSi has no
//                               moving VIE2, the full vie2_timestep is solved
//                               at n = tc for the window shifted to the time
//                               origin with tv = mat = 0 (source
//                               truncated_vie2_no_tv). The less row is
//                               stored as G^<(n, n-j) = -G^<(n-j, n)^dagger
//                               (j > 0), the diagonal block as solved.
// The bubbles store C(a,a) for a = b; for size 2 the element (a, b) of the
// matrix block is the bubble of the orbitals a (G) and b (S).
// The kinetic integral is a verbatim copy of get_kinetic_trunc_timestep
// (cntr_dyson_moving_impl.hpp:20-64, debug output removed), which is not
// part of the installed libcntr build.
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>

#include "cntr/cntr.hpp"
#include "reference_io.hpp"

typedef std::complex<double> cplx;
typedef cntr::herm_matrix<double> KGreen;
typedef cntr::herm_matrix_moving<double> KTrunc;
typedef cntr::herm_matrix_timestep_moving<double> KTruncTstp;
typedef cntr::function<double> KFunc;
typedef cntr::function_moving<double> KFuncMov;

static double kinetic_trunc(KTrunc &G, KTrunc &D, integration::Integrator<double> &I,
                            double h) {
    int k = I.get_k(), tc = G.tc(), n = G.size1();
    cdmatrix s = cdmatrix::Zero(n, n), g, gl, dl, dr;
    for (int l = 0; l <= tc; l++) {
        double w = (l <= k) ? I.gregory_omega(l) : (l >= tc - k ? I.gregory_omega(tc - l) : 1.0);
        G.get_ret(0, l, g);
        G.get_les(0, l, gl);
        D.get_les(0, l, dl);
        D.get_ret(0, l, dr);
        s += h * w * (g.adjoint() * dl);
        s -= h * w * (gl.adjoint() * dr);
    }
    return s.trace().imag();
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: p8_moving_solver_ref <output-dir>\n");
        return 1;
    }
    const int nt = 30, ntau = 40, tc = 12;
    const double beta = 1.0, h = 0.02, mu = -0.1;
    const int sizes[] = {1, 1, 1, 2, 2, 2}, orders[] = {1, 3, 5, 1, 3, 5};

    FILE *fp = kbref::open_output(argv[1], "p8_moving_solver.dat");
    kbref::Metadata meta;
    meta.push_back(std::make_pair("generator", std::string("p8_moving_solver_ref")));
    meta.push_back(std::make_pair("case_set", std::string("moving_solver")));
    meta.push_back(std::make_pair(
        "components", std::string("kinetic,dyson_ret,dyson_less,bubble1_ret,bubble1_less,"
                                  "bubble2_ret,bubble2_less,vie2_ret,vie2_less")));
    meta.push_back(std::make_pair("nsteps", std::string("6")));
    meta.push_back(std::make_pair("nk", std::to_string(tc + 1)));
    meta.push_back(std::make_pair("size", std::string("2")));
    meta.push_back(std::make_pair("source", std::string("dyson_moving+truncated_vie2_no_tv")));
    meta.push_back(std::make_pair("tc", std::to_string(tc)));
    meta.push_back(std::make_pair("nt", std::to_string(nt)));
    meta.push_back(std::make_pair("ntau", std::to_string(ntau)));
    kbref::write_metadata(fp, "series", meta);
    kbref::SeriesSchema s = kbref::series_schema(meta);

    for (int c = 0; c < 6; c++) {
        const int size = sizes[c], k = orders[c];
        cdmatrix H0(size, size);
        if (size == 1) {
            H0(0, 0) = -0.3;
        } else {
            H0(0, 0) = -0.3; H0(0, 1) = cplx(0.0, 0.2);
            H0(1, 0) = cplx(0.0, -0.2); H0(1, 1) = 0.4;
        }
        KGreen G(nt, ntau, size, -1), S(nt, ntau, size, -1);
        cntr::green_from_H(S, mu, H0, beta, h);
        for (int n = 0; n <= nt; n++) {
            for (int m = 0; m <= n; m++) {
                double f = 0.3 * std::exp(-((n - m) * h) * ((n - m) * h));
                cdmatrix x;
                S.get_ret(n, m, x); x *= f; S.set_ret(n, m, x);
                S.get_les(m, n, x); x *= f; S.set_les(m, n, x);
            }
            cdmatrix z = cdmatrix::Zero(size, size);
            for (int m = 0; m <= ntau; m++) S.set_tv(n, m, z);
        }
        S.set_timestep_zero(-1);
        KFunc eps(nt, size);
        for (int t = -1; t <= nt; t++) eps.set_value(t, H0);
        cntr::dyson(G, mu, eps, S, beta, h, k, CNTR_MAT_FIXPOINT);

        KTrunc Gm(tc, size, -1), Sm(tc, size, -1);
        KFuncMov em(tc, size);
        Gm.set_from_G_backward(G, G, nt);
        Sm.set_from_G_backward(S, S, nt);
        for (int i = 0; i <= tc; i++) em.set_value(i, H0);

        // Bubbles of the window functions (before the Dyson step)
        for (int a = 0; a < size; a++) {
            for (int b = 0; b < size; b++) {
                KTruncTstp B1(tc, 1, 1), B2(tc, 1, -1);
                cntr::Bubble1(B1, 0, 0, Gm, Gm, a, a, Sm, Sm, b, b);
                cntr::Bubble2(B2, 0, 0, Gm, Gm, a, a, Sm, Sm, b, b);
                for (int j = 0; j <= tc; j++) {
                    cplx x;
                    B1.get_ret(j, x); kbref::write_series_row(fp, s, "bubble1_ret", c, j, a, b, x);
                    B1.get_les(j, x); kbref::write_series_row(fp, s, "bubble1_less", c, j, a, b, x);
                    B2.get_ret(j, x); kbref::write_series_row(fp, s, "bubble2_ret", c, j, a, b, x);
                    B2.get_les(j, x); kbref::write_series_row(fp, s, "bubble2_less", c, j, a, b, x);
                }
            }
        }

        // Truncated VIE2 (see the header): shifted full objects, tv = mat = 0
        {
            KGreen Fs(tc, ntau, size, -1), Qs(tc, ntau, size, -1), Gs(tc, ntau, size, -1);
            const int o = nt - tc;
            for (int a = 0; a <= tc; a++) {
                for (int b = 0; b <= a; b++) {
                    cdmatrix x;
                    S.get_ret(o + a, o + b, x); Fs.set_ret(a, b, x);
                    S.get_les(o + b, o + a, x); Fs.set_les(b, a, x);
                    G.get_ret(o + a, o + b, x); Qs.set_ret(a, b, x);
                    G.get_les(o + b, o + a, x); Qs.set_les(b, a, x);
                }
            }
            integration::Integrator<double> Iv(k);
            cntr::vie2_timestep(tc, Gs, Fs, Fs, Qs, Iv, beta, h);
            for (int j = 0; j <= tc; j++) {
                cdmatrix R, L;
                Gs.get_ret(tc, tc - j, R);
                Gs.get_les(tc - j, tc, L);
                if (j > 0) L = (-L.adjoint()).eval();
                for (int a = 0; a < size; a++) {
                    for (int b = 0; b < size; b++) {
                        kbref::write_series_row(fp, s, "vie2_ret", c, j, a, b, R(a, b));
                        kbref::write_series_row(fp, s, "vie2_less", c, j, a, b, L(a, b));
                    }
                }
            }
        }

        integration::Integrator<double> I(k);
        kbref::write_series_row(fp, s, "kinetic", c, -1, -1, -1,
                                cplx(kinetic_trunc(Gm, Sm, I, h), 0.0));
        cntr::dyson_timestep(Gm, Sm, em, mu, I, h);
        for (int j = 0; j <= tc; j++) {
            cdmatrix R, L;
            Gm.get_ret(0, j, R);
            Gm.get_les(0, j, L);
            for (int a = 0; a < size; a++) {
                for (int b = 0; b < size; b++) {
                    kbref::write_series_row(fp, s, "dyson_ret", c, j, a, b, R(a, b));
                    kbref::write_series_row(fp, s, "dyson_less", c, j, a, b, L(a, b));
                }
            }
        }
    }
    std::fclose(fp);
    return 0;
}
