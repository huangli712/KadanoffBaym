// kbref generator: NESSi Dyson equation (Matsubara, start-up, time
// stepping, and full solution).
//
// Usage: p3_dyson_ref <output-dir>
// Output (kbref-v1, data_kind herm_matrix):
//   dyson_mat_fixpoint_fermion.dat, dyson_mat_cg_fermion.dat,
//   dyson_mat_cg_boson.dat                  embedding model, Matsubara only
//   dyson_ncomm2x2_{start,timestep,full}_k{1..5}.dat
//                                           authoritative 2x2 case
//   dyson_embedding_1x1_{start,timestep,full}.dat
//                                           embedding model, k = 5
//
// Inputs (the same closed-form formulas are used in the Julia test
// test/nessi/t_phase3_dyson.jl; nothing is read back from the output):
//
//   * Authoritative case (size 2, fermions), t = n h, tau = q dtau:
//       H(t) = c1(t) A + c2(t) B, with the dense hermitian, non-commuting
//       A = [0.5, 0.3 - 0.2i; 0.3 + 0.2i, -0.4],
//       B = [0.1, 0.25i; -0.25i, 0.6],
//       c1(t) = 1 + 0.3 sin(t), c2(t) = 0.5 + 0.2 cos(1.3 t),
//       H(-1) = H(0).
//     The independent blocks of Sigma are non-hermitian (amplitude 0.1):
//       mat(tau)  = 0.1 [-0.5 e^{-tau}, 0.2 + 0.1 i tau / beta;
//                        -0.15 + 0.05 i, -0.3 e^{-tau/2}]
//       ret(t,t') = 0.1 [-i cos(t - t'), 0.2 e^{-i(t - t'/2)};
//                        -0.1 + 0.3 i sin(t + t'), -0.5 i e^{-0.3 (t - t')}]
//       tv(t,tau) = 0.1 [0.3 i e^{-tau} cos(t), 0.1 + 0.2 i t;
//                        -0.2 e^{-tau/2}, 0.4 i sin(t + tau)]
//       les(t',t) = 0.1 [0.2 i cos(t - t'), 0.1 + 0.1 i t';
//                        -0.3 + 0.05 i t, 0.25 i]
//   * Embedding model (size 1): H = eps1, Sigma = lam^2 g2, where g2 is the
//     free Green's function of eps2 (cntr::green_from_H).
#include <cmath>
#include <complex>
#include <cstdio>
#include <stdexcept>
#include <string>

// CNTR_NO_EXTERN_TEMPLATES is defined by the CMake target.
#include "cntr/cntr.hpp"
#include "reference_io.hpp"

using namespace cntr;
typedef std::complex<double> cplx;

static const cplx CI(0.0, 1.0);

//
// Authoritative 2x2 case
//
static const int NC_NT = 20;
static const int NC_NTAU = 40;
static const double NC_BETA = 2.0;
static const double NC_H = 0.05;
static const double NC_MU = 0.2;

static cdmatrix nc_A() {
    cdmatrix M(2, 2);
    M(0, 0) = 0.5;
    M(0, 1) = cplx(0.3, -0.2);
    M(1, 0) = cplx(0.3, 0.2);
    M(1, 1) = -0.4;
    return M;
}

static cdmatrix nc_B() {
    cdmatrix M(2, 2);
    M(0, 0) = 0.1;
    M(0, 1) = cplx(0.0, 0.25);
    M(1, 0) = cplx(0.0, -0.25);
    M(1, 1) = 0.6;
    return M;
}

static cdmatrix nc_ham(double t) {
    double c1 = 1.0 + 0.3 * std::sin(t);
    double c2 = 0.5 + 0.2 * std::cos(1.3 * t);
    return c1 * nc_A() + c2 * nc_B();
}

static cdmatrix nc_sigma_mat(double tau) {
    cdmatrix M(2, 2);
    M(0, 0) = -0.5 * std::exp(-tau);
    M(0, 1) = 0.2 + 0.1 * CI * tau / NC_BETA;
    M(1, 0) = cplx(-0.15, 0.05);
    M(1, 1) = -0.3 * std::exp(-0.5 * tau);
    return 0.1 * M;
}

static cdmatrix nc_sigma_ret(double t, double tp) {
    cdmatrix M(2, 2);
    M(0, 0) = -CI * std::cos(t - tp);
    M(0, 1) = 0.2 * std::exp(-CI * (t - 0.5 * tp));
    M(1, 0) = -0.1 + 0.3 * CI * std::sin(t + tp);
    M(1, 1) = -0.5 * CI * std::exp(-0.3 * (t - tp));
    return 0.1 * M;
}

static cdmatrix nc_sigma_tv(double t, double tau) {
    cdmatrix M(2, 2);
    M(0, 0) = 0.3 * CI * std::exp(-tau) * std::cos(t);
    M(0, 1) = 0.1 + 0.2 * CI * t;
    M(1, 0) = -0.2 * std::exp(-0.5 * tau);
    M(1, 1) = 0.4 * CI * std::sin(t + tau);
    return 0.1 * M;
}

static cdmatrix nc_sigma_les(double tp, double t) {
    cdmatrix M(2, 2);
    M(0, 0) = 0.2 * CI * std::cos(t - tp);
    M(0, 1) = 0.1 + 0.1 * CI * tp;
    M(1, 0) = -0.3 + 0.05 * CI * t;
    M(1, 1) = 0.25 * CI;
    return 0.1 * M;
}

static void require(bool cond, const char *msg) {
    if (!cond) throw std::runtime_error(std::string("p3_dyson_ref: ") + msg);
}

static bool commutes(const cdmatrix &X, const cdmatrix &Y) {
    return (X * Y - Y * X).norm() < 1e-8;
}

static bool hermitian(const cdmatrix &X) {
    return (X - X.adjoint()).norm() < 1e-14 * (1.0 + X.norm());
}

static bool all_nonzero(const cdmatrix &X) {
    for (int a = 0; a < X.rows(); a++)
        for (int b = 0; b < X.cols(); b++)
            if (std::abs(X(a, b)) == 0.0) return false;
    return true;
}

static void nc_build(function<double> &H, herm_matrix<double> &S) {
    const double dtau = NC_BETA / NC_NTAU;
    require(!commutes(nc_A(), nc_B()), "A and B commute");
    cdmatrix H0 = nc_ham(0.0);
    H.set_value(-1, H0);
    for (int n = 0; n <= NC_NT; n++) {
        cdmatrix Hn = nc_ham(n * NC_H);
        require(hermitian(Hn), "H(t) is not hermitian");
        H.set_value(n, Hn);
    }
    for (int q = 0; q <= NC_NTAU; q++) {
        cdmatrix M = nc_sigma_mat(q * dtau);
        require(!hermitian(M) && all_nonzero(M), "bad Sigma^M");
        require(!commutes(M, nc_ham(0.0)), "Sigma^M commutes with H");
        S.set_mat(q, M);
    }
    for (int i = 0; i <= NC_NT; i++) {
        cdmatrix Hi = nc_ham(i * NC_H);
        for (int j = 0; j <= i; j++) {
            cdmatrix M = nc_sigma_ret(i * NC_H, j * NC_H);
            require(!hermitian(M) && all_nonzero(M), "bad Sigma^R");
            require(!commutes(M, Hi), "Sigma^R commutes with H");
            S.set_ret(i, j, M);
        }
        for (int q = 0; q <= NC_NTAU; q++) {
            cdmatrix M = nc_sigma_tv(i * NC_H, q * dtau);
            require(M.norm() > 0.0 && !hermitian(M), "bad Sigma^tv");
            S.set_tv(i, q, M);
        }
        for (int j = i; j <= NC_NT; j++) {
            cdmatrix M = nc_sigma_les(i * NC_H, j * NC_H);
            require(!hermitian(M) && all_nonzero(M), "bad Sigma^<");
            require(!commutes(M, Hi), "Sigma^< commutes with H");
            S.set_les(i, j, M);
        }
    }
}

//
// Embedding model
//
static const double EM_EPS1 = -1.0;
static const double EM_EPS2 = 1.0;
static const double EM_LAM = 0.5;

static void em_build(int nt, int ntau, int sig, double eps1, double eps2,
                     double mu, double beta, double h, function<double> &H,
                     herm_matrix<double> &S) {
    cdmatrix h1(1, 1), h2(1, 1);
    h1(0, 0) = eps1;
    h2(0, 0) = eps2;
    for (int n = -1; n <= nt; n++) H.set_value(n, h1);
    green_from_H(S, mu, h2, beta, h);
    for (int n = -1; n <= nt; n++) S.smul(n, EM_LAM * EM_LAM);
}

//
// Output
//
static void write_case(const std::string &outdir, const std::string &name,
                       herm_matrix<double> &G, double beta, double h,
                       double mu, int k, const std::string &method,
                       const std::string &stage) {
    FILE *fp = kbref::open_output(outdir, name + ".dat");
    kbref::Metadata meta;
    meta.push_back(std::make_pair("case", name));
    meta.push_back(std::make_pair("generator", std::string("p3_dyson_ref")));
    meta.push_back(std::make_pair("nt", kbref::fmt_int(G.nt())));
    meta.push_back(std::make_pair("ntau", kbref::fmt_int(G.ntau())));
    meta.push_back(std::make_pair("size1", kbref::fmt_int(G.size1())));
    meta.push_back(std::make_pair("size2", kbref::fmt_int(G.size2())));
    meta.push_back(std::make_pair("beta", kbref::fmt_double(beta)));
    meta.push_back(std::make_pair("h", kbref::fmt_double(h)));
    meta.push_back(std::make_pair("mu", kbref::fmt_double(mu)));
    meta.push_back(std::make_pair("solve_order", kbref::fmt_int(k)));
    meta.push_back(std::make_pair("sig", kbref::fmt_int(G.sig())));
    meta.push_back(std::make_pair("method", method));
    meta.push_back(std::make_pair("stage", stage));
    kbref::write_metadata(fp, "herm_matrix", meta);
    cdmatrix M;
    kbref::write_herm_matrix(fp, G, M);
    std::fclose(fp);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <output-dir>\n", argv[0]);
        return 1;
    }
    const std::string outdir(argv[1]);

    //
    // Matsubara Dyson equation, embedding model
    //
    {
        const int ntau = 100, k = 5;
        const double beta = 5.0;
        // Fermions: FIXPOINT and CG
        for (int method = 0; method < 2; method++) {
            const double mu = 0.1;
            function<double> H(0, 1);
            herm_matrix<double> S(0, ntau, 1, -1), G(0, ntau, 1, -1);
            em_build(0, ntau, -1, EM_EPS1, EM_EPS2, mu, beta, 1.0, H, S);
            int m = (method == 0) ? CNTR_MAT_FIXPOINT : CNTR_MAT_CG;
            dyson_mat(G, mu, H, S, beta, k, m, true);
            std::string name = (method == 0) ? "dyson_mat_fixpoint_fermion"
                                             : "dyson_mat_cg_fermion";
            write_case(outdir, name, G, beta, 1.0, mu, k,
                       method == 0 ? "fixpoint" : "cg", "mat");
        }
        // Bosons without zero mode: CG
        {
            const double mu = -0.5;
            function<double> H(0, 1);
            herm_matrix<double> S(0, ntau, 1, 1), G(0, ntau, 1, 1);
            em_build(0, ntau, 1, 1.0, 2.0, mu, beta, 1.0, H, S);
            dyson_mat(G, mu, H, S, beta, k, CNTR_MAT_CG, true);
            write_case(outdir, "dyson_mat_cg_boson", G, beta, 1.0, mu, k, "cg", "mat");
        }
    }

    //
    // Authoritative 2x2 case
    //
    for (int k = 1; k <= 5; k++) {
        function<double> H(NC_NT, 2);
        herm_matrix<double> S(NC_NT, NC_NTAU, 2, -1);
        nc_build(H, S);
        const std::string kk = std::to_string(k);
        // Matsubara + start-up
        herm_matrix<double> G(NC_NT, NC_NTAU, 2, -1);
        dyson_mat(G, NC_MU, H, S, NC_BETA, k, CNTR_MAT_FIXPOINT, true);
        dyson_start(G, NC_MU, H, S, NC_BETA, NC_H, k);
        // The initial correlations do not vanish
        cdmatrix M;
        G.get_tv(1, NC_NTAU / 2, M);
        require(all_nonzero(M), "vanishing element of G^tv");
        write_case(outdir, "dyson_ncomm2x2_start_k" + kk, G, NC_BETA, NC_H, NC_MU,
                   k, "fixpoint", "start");
        // First BDF time step and two more
        for (int n = k + 1; n <= k + 3; n++)
            dyson_timestep(n, G, NC_MU, H, S, NC_BETA, NC_H, k);
        write_case(outdir, "dyson_ncomm2x2_timestep_k" + kk, G, NC_BETA, NC_H, NC_MU,
                   k, "fixpoint", "timestep");
        // Full solution
        herm_matrix<double> F(NC_NT, NC_NTAU, 2, -1);
        dyson(F, NC_MU, H, S, NC_BETA, NC_H, k, CNTR_MAT_FIXPOINT, true);
        F.get_les(NC_NT / 2, NC_NT, M);
        require(all_nonzero(M), "vanishing element of G^<");
        write_case(outdir, "dyson_ncomm2x2_full_k" + kk, F, NC_BETA, NC_H, NC_MU,
                   k, "fixpoint", "full");
    }

    //
    // Embedding model, real time, k = 5
    //
    {
        const int nt = 30, ntau = 50, k = 5;
        const double beta = 5.0, h = 0.05, mu = -0.1;
        function<double> H(nt, 1);
        herm_matrix<double> S(nt, ntau, 1, -1);
        em_build(nt, ntau, -1, EM_EPS1, EM_EPS2, mu, beta, h, H, S);
        herm_matrix<double> G(nt, ntau, 1, -1);
        dyson_mat(G, mu, H, S, beta, k, CNTR_MAT_FIXPOINT, true);
        dyson_start(G, mu, H, S, beta, h, k);
        write_case(outdir, "dyson_embedding_1x1_start", G, beta, h, mu, k,
                   "fixpoint", "start");
        for (int n = k + 1; n <= k + 3; n++)
            dyson_timestep(n, G, mu, H, S, beta, h, k);
        write_case(outdir, "dyson_embedding_1x1_timestep", G, beta, h, mu, k,
                   "fixpoint", "timestep");
        herm_matrix<double> F(nt, ntau, 1, -1);
        dyson(F, mu, H, S, beta, h, k, CNTR_MAT_FIXPOINT, true);
        write_case(outdir, "dyson_embedding_1x1_full", F, beta, h, mu, k,
                   "fixpoint", "full");
    }
    return 0;
}
