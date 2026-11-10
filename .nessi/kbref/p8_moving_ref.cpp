// kbref generator: moving (truncated) data structure and algebra of NESSi.
//
// Usage: p8_moving_ref <output-dir>
// Output: <output-dir>/p8_moving_case<c>.dat, c = 0..3 (kbref-v1, data_kind
// moving, one case per file, the whole window ret/less).
//
// Input (rebuilt in test/nessi/t_moving_reference.jl): size 2, nt = 30,
// ntau = 40 (NESSi), beta = 1, h = 0.02, mu = -0.1, tc = 12,
// H0 = [-0.3 0.2i; -0.2i 0.4]; Sigma = green_from_H(H0) times
// f(n - m) = 0.3 exp(-((n - m) h)^2) on the real-time components (tv and
// Matsubara zero); G = full Dyson solution of order 3. The function
// F(t_n) = [1 + 0.1 n h, 0.2i; -0.1, 0.8 - 0.05 n h] (NESSi step n).
//
// Cases:
//   0  Gm.set_from_G_backward(G, G, nt)
//   1  case 0, forward(), extrapolate_timestep(Gm, 3)
//   2  case 1, left_multiply(F_m), right_multiply(F_m) of the leading row
//      (F_m.set_value(i, F(nt + 1 - i)))
//   3  case 2, smul(0, 0.7), incr_timestep(0, Sm, 0, 0.3) where Sm is the
//      window of Sigma at nt forwarded once, its leading row set to the
//      row 1 (NESSi forward() keeps the stale row tc as the new leading
//      row, KB forward! sets it to zero; the explicit copy makes both agree)
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>

#include "cntr/cntr.hpp"
#include "reference_io.hpp"

typedef std::complex<double> cplx;
typedef cntr::herm_matrix<double> KGreen;
typedef cntr::herm_matrix_moving<double> KTrunc;
typedef cntr::function<double> KFunc;
typedef cntr::function_moving<double> KFuncMov;

static void write_case(const std::string &dir, int c, int tstp, KTrunc &G) {
    FILE *fp = kbref::open_output(dir, "p8_moving_case" + std::to_string(c) + ".dat");
    kbref::Metadata meta;
    meta.push_back(std::make_pair("generator", std::string("p8_moving_ref")));
    meta.push_back(std::make_pair("case_set", std::string("moving_algebra")));
    meta.push_back(std::make_pair("case_id", std::to_string(c)));
    meta.push_back(std::make_pair("source", std::string("herm_matrix_moving")));
    meta.push_back(std::make_pair("tc", std::to_string(G.tc())));
    meta.push_back(std::make_pair("size", std::to_string(G.size1())));
    meta.push_back(std::make_pair("sign", std::to_string(G.sig())));
    meta.push_back(std::make_pair("tstp", std::to_string(tstp)));
    kbref::write_metadata(fp, "moving", meta);
    for (int comp = 0; comp < 2; comp++) {
        for (int i = 0; i <= G.tc(); i++) {
            for (int j = 0; j <= G.tc(); j++) {
                cdmatrix M;
                if (comp == 0) G.get_ret(i, j, M);
                else G.get_les(i, j, M);
                for (int a = 0; a < G.size1(); a++)
                    for (int b = 0; b < G.size1(); b++)
                        kbref::write_moving_row(fp, meta, comp == 0 ? "ret" : "less", i, j,
                                                a, b, M(a, b));
            }
        }
    }
    std::fclose(fp);
}

static cdmatrix ffun(int n, double h) {
    cdmatrix F(2, 2);
    F(0, 0) = 1.0 + 0.1 * n * h;
    F(0, 1) = cplx(0.0, 0.2);
    F(1, 0) = -0.1;
    F(1, 1) = 0.8 - 0.05 * n * h;
    return F;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: p8_moving_ref <output-dir>\n");
        return 1;
    }
    const std::string dir = argv[1];
    const int nt = 30, ntau = 40, tc = 12, size = 2, k = 3;
    const double beta = 1.0, h = 0.02, mu = -0.1;
    cdmatrix H0(2, 2);
    H0(0, 0) = -0.3; H0(0, 1) = cplx(0.0, 0.2);
    H0(1, 0) = cplx(0.0, -0.2); H0(1, 1) = 0.4;
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
    Gm.set_from_G_backward(G, G, nt);
    Sm.set_from_G_backward(S, S, nt);
    write_case(dir, 0, nt, Gm);

    Gm.forward();
    cntr::extrapolate_timestep(Gm, k);
    write_case(dir, 1, nt + 1, Gm);

    KFuncMov Fm(tc, size);
    for (int i = 0; i <= tc; i++) {
        cdmatrix F = ffun(nt + 1 - i, h);
        Fm.set_value(i, F);
    }
    Gm.left_multiply(Fm);
    Gm.right_multiply(Fm);
    write_case(dir, 2, nt + 1, Gm);

    Sm.forward();
    Sm.set_timestep(0, Sm, 1);
    Gm.smul(0, 0.7);
    Gm.incr_timestep(0, Sm, 0, cplx(0.3, 0.0));
    write_case(dir, 3, nt + 1, Gm);
    return 0;
}
