// kbref generator: NESSi VIE2 [1 + F] * G = Q.
//
// Usage: phase4_vie2 <output-dir>
// Output (kbref-v1, one case per file):
//   phase4_vie2_dyson_size1.dat          libcntr/test/dyson.cpp, integral form
//   phase4_vie2_dyson_new_size1.dat      libcntr/test/dyson_new.cpp, integral form
//   phase4_vie2_downfold_size3.dat       libcntr/test/downfold.cpp, interacting
//   phase4_vie2_ncomm_size2_k{1..5}.dat  non-commuting 2x2 case, vie2 (fermions)
//   phase4_vie2_ncomm_boson_size2_k5.dat non-commuting 2x2 case, vie2 (bosons)
//   phase4_vie2_route_{fermion,boson}_n0.dat
//                                        vie2_timestep for n = -1, 0 only
//   phase4_vie2_route_fermion_n2.dat     vie2_timestep for n = -1, 0, 1, 2
//   phase4_vie2_sin_size2_k{3,5}.dat     vie2_timestep_sin for n = -1..nt
//                                        (fermions), non-zero Fsin and Qsin
//   phase4_vie2_sin_boson_size2_k5.dat   the same for bosons
//   phase4_vie2_start_less_source_size2.dat
//                                        element: start-up lesser source
//                                        blocks of vie2_start (k = 5)
//   phase4_vie2_fixpoint_trajectory_size2.dat
//                                        iteration: vie2_mat_fixpoint,
//                                        nfixpoint = 0..6
//   phase4_vie2_cg_trajectory_size2.dat  iteration: vie2_mat_steep_dispatch
//
// Inputs of the non-commuting 2x2 case (the same closed-form formulas are
// used in test/t_phase4.jl; nothing is read back from the output),
// t = n h, t' = m h, tau = q dtau:
//   F:   mat(tau)  = 0.2 [-0.5 e^{-tau}, 0.2 + 0.1 i tau / beta;
//                         -0.15 + 0.05 i, -0.3 e^{-tau/2}]
//        ret(t,t') = 0.2 [-i cos(t - t'), 0.2 e^{-i(t - t'/2)};
//                         -0.1 + 0.3 i sin(t + t'), -0.5 i e^{-0.3 (t - t')}]
//        tv(t,tau) = 0.2 [0.3 i e^{-tau} cos(t), 0.1 + 0.2 i t;
//                         -0.2 e^{-tau/2}, 0.4 i sin(t + tau)]
//        les(t',t) = 0.2 [0.2 i cos(t - t'), 0.1 + 0.1 i t';
//                         -0.3 + 0.05 i t, 0.25 i]
//   Fcc: mat(tau)  = 0.2 [0.4 e^{-tau}, -0.1 + 0.2 i; 0.3 i tau / beta, -0.25]
//        ret(t,t') = 0.2 [0.5 i sin(t - t') - 0.3, 0.1 e^{i t'};
//                         0.2 i, -0.4 cos(t + t')]
//        tv(t,tau) = 0.2 [0.2 cos(t) e^{-tau}, 0.1 i tau;
//                         -0.3 + 0.1 i t, 0.2 e^{-tau/2}]
//        les(t',t) = 0.2 [0.1 i, 0.2 cos(t) + 0.1 i; -0.1 t', 0.3 i sin(t')]
//   Q:   green_from_H(Q, mu, HQ, beta, h), HQ = [0.5, 0.3 - 0.2i;
//        0.3 + 0.2i, -0.4] (fermions) or HQ + 1.5 (bosons), mu = 0.2.
//   Singular parts (n = -1 uses t = 0):
//        Fsin(t) = [0.3 + 0.1 sin(t), 0.1 i + 0.05 t; -0.1 i + 0.05 t, -0.2]
//        Qsin(t) = [0.5, 0.2 + 0.1 i t; -0.3 cos(t), -0.4 + 0.1 t]
#include <cmath>
#include <complex>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

// CNTR_NO_EXTERN_TEMPLATES is defined by the CMake target.
#include "cntr/cntr.hpp"
#include "reference_io.hpp"

using namespace cntr;
typedef std::complex<double> cplx;
typedef herm_matrix<double> GF;

static const cplx CI(0.0, 1.0);
static const std::string GEN = "phase4_vie2";

static void require(bool cond, const char *msg) {
    if (!cond) throw std::runtime_error(std::string("phase4_vie2: ") + msg);
}

//
// Non-commuting 2x2 case
//
static const int NC_NT = 20;
static const int NC_NTAU = 40;
static const double NC_BETA = 2.0;
static const double NC_H = 0.05;
static const double NC_MU = 0.2;
static const double NC_AMP = 0.2;

static cdmatrix m2(cplx a, cplx b, cplx c, cplx d) {
    cdmatrix M(2, 2);
    M(0, 0) = a;
    M(0, 1) = b;
    M(1, 0) = c;
    M(1, 1) = d;
    return M;
}

static cdmatrix f_mat(double tau) {
    return NC_AMP * m2(-0.5 * std::exp(-tau), 0.2 + 0.1 * CI * tau / NC_BETA,
                       cplx(-0.15, 0.05), -0.3 * std::exp(-0.5 * tau));
}
static cdmatrix f_ret(double t, double tp) {
    return NC_AMP * m2(-CI * std::cos(t - tp), 0.2 * std::exp(-CI * (t - 0.5 * tp)),
                       -0.1 + 0.3 * CI * std::sin(t + tp),
                       -0.5 * CI * std::exp(-0.3 * (t - tp)));
}
static cdmatrix f_tv(double t, double tau) {
    return NC_AMP * m2(0.3 * CI * std::exp(-tau) * std::cos(t), 0.1 + 0.2 * CI * t,
                       -0.2 * std::exp(-0.5 * tau), 0.4 * CI * std::sin(t + tau));
}
static cdmatrix f_les(double tp, double t) {
    return NC_AMP * m2(0.2 * CI * std::cos(t - tp), 0.1 + 0.1 * CI * tp,
                       -0.3 + 0.05 * CI * t, 0.25 * CI);
}
static cdmatrix fcc_mat(double tau) {
    return NC_AMP * m2(0.4 * std::exp(-tau), cplx(-0.1, 0.2), 0.3 * CI * tau / NC_BETA,
                       -0.25);
}
static cdmatrix fcc_ret(double t, double tp) {
    return NC_AMP * m2(0.5 * CI * std::sin(t - tp) - 0.3, 0.1 * std::exp(CI * tp),
                       0.2 * CI, -0.4 * std::cos(t + tp));
}
static cdmatrix fcc_tv(double t, double tau) {
    return NC_AMP * m2(0.2 * std::cos(t) * std::exp(-tau), 0.1 * CI * tau,
                       -0.3 + 0.1 * CI * t, 0.2 * std::exp(-0.5 * tau));
}
static cdmatrix fcc_les(double tp, double t) {
    return NC_AMP * m2(0.1 * CI, 0.2 * std::cos(t) + 0.1 * CI, -0.1 * tp,
                       0.3 * CI * std::sin(tp));
}

static void nc_kernel(GF &X, bool cc) {
    const int nt = X.nt(), ntau = X.ntau();
    const double dtau = NC_BETA / ntau;
    for (int q = 0; q <= ntau; q++) {
        cdmatrix M = cc ? fcc_mat(q * dtau) : f_mat(q * dtau);
        X.set_mat(q, M);
    }
    for (int i = 0; i <= nt; i++) {
        for (int j = 0; j <= i; j++) {
            cdmatrix M = cc ? fcc_ret(i * NC_H, j * NC_H) : f_ret(i * NC_H, j * NC_H);
            X.set_ret(i, j, M);
        }
        for (int q = 0; q <= ntau; q++) {
            cdmatrix M = cc ? fcc_tv(i * NC_H, q * dtau) : f_tv(i * NC_H, q * dtau);
            X.set_tv(i, q, M);
        }
        for (int j = i; j <= nt; j++) {
            cdmatrix M = cc ? fcc_les(i * NC_H, j * NC_H) : f_les(i * NC_H, j * NC_H);
            X.set_les(i, j, M);
        }
    }
}

static void nc_build(int sig, GF &F, GF &Fcc, GF &Q) {
    nc_kernel(F, false);
    nc_kernel(Fcc, true);
    cdmatrix HQ = m2(0.5, cplx(0.3, -0.2), cplx(0.3, 0.2), -0.4);
    if (sig == 1) HQ += 1.5 * cdmatrix::Identity(2, 2);
    green_from_H(Q, NC_MU, HQ, NC_BETA, NC_H);
}

//
// Output
//
static kbref::Metadata meta_base(const std::string &name, int sig) {
    kbref::Metadata meta;
    meta.push_back(std::make_pair("case", name));
    meta.push_back(std::make_pair("generator", GEN));
    meta.push_back(std::make_pair("case_set", std::string("phase4_vie2")));
    meta.push_back(std::make_pair("sig", kbref::fmt_int(sig)));
    return meta;
}

static void write_green(const std::string &outdir, const std::string &name, GF &G,
                        double beta, double h, double mu, int k,
                        const std::string &method) {
    FILE *fp = kbref::open_output(outdir, name + ".dat");
    kbref::Metadata meta = meta_base(name, G.sig());
    meta.push_back(std::make_pair("nt", kbref::fmt_int(G.nt())));
    meta.push_back(std::make_pair("ntau", kbref::fmt_int(G.ntau())));
    meta.push_back(std::make_pair("size1", kbref::fmt_int(G.size1())));
    meta.push_back(std::make_pair("size2", kbref::fmt_int(G.size2())));
    meta.push_back(std::make_pair("beta", kbref::fmt_double(beta)));
    meta.push_back(std::make_pair("h", kbref::fmt_double(h)));
    meta.push_back(std::make_pair("mu", kbref::fmt_double(mu)));
    meta.push_back(std::make_pair("solve_order", kbref::fmt_int(k)));
    meta.push_back(std::make_pair("method", method));
    kbref::write_metadata(fp, "herm_matrix", meta);
    cdmatrix M;
    kbref::write_herm_matrix(fp, G, M);
    std::fclose(fp);
}

static kbref::Metadata meta_iteration(const std::string &name, const std::string &alg,
                                      int niter, int ntau) {
    kbref::Metadata meta = meta_base(name, -1);
    meta.push_back(std::make_pair("algorithm", alg));
    meta.push_back(std::make_pair("niter", kbref::fmt_int(niter)));
    meta.push_back(std::make_pair("ntau", kbref::fmt_int(ntau)));
    meta.push_back(std::make_pair("size1", std::string("2")));
    meta.push_back(std::make_pair("size2", std::string("2")));
    meta.push_back(std::make_pair("beta", kbref::fmt_double(NC_BETA)));
    meta.push_back(std::make_pair("solve_order", std::string("5")));
    return meta;
}

static void write_g(FILE *fp, int iter, GF &G) {
    cdmatrix M;
    for (int q = 0; q <= G.ntau(); q++) {
        G.get_mat(q, M);
        for (int a = 0; a < 2; a++)
            for (int b = 0; b < 2; b++)
                kbref::write_iteration_row(fp, "g", iter, q, a, b, M(a, b));
    }
}

//
// Weighted inner product of the CG method, see vie2_mat_steep_dispatch:
// sum_r w_r sum_ab Re[(X_r^+ Y_r)_ab]
//
static double cg_inner(GF &X, GF &Y, int k) {
    const int ntau = X.ntau();
    integration::Integrator<double> I(k);
    double s = 0.0;
    cdmatrix A, B;
    for (int r = 0; r <= ntau; r++) {
        double w;
        if (r <= k)
            w = I.gregory_omega(r);
        else if (r >= ntau - k)
            w = I.gregory_omega(ntau - r);
        else
            w = 1.0;
        X.get_mat(r, A);
        Y.get_mat(r, B);
        cdmatrix P = A.adjoint() * B;
        for (int a = 0; a < P.rows(); a++)
            for (int b = 0; b < P.cols(); b++) s += w * P(a, b).real();
    }
    return s;
}

// A[X] = X + (F * X + X * Fcc) / 2
static void cg_apply(GF &AX, GF &X, GF &F, GF &Fcc, GF &C1, GF &C2,
                     integration::Integrator<double> &I) {
    convolution_matsubara(C1, F, X, I, NC_BETA);
    convolution_matsubara(C2, X, Fcc, I, NC_BETA);
    cdmatrix M1, M2, M3;
    for (int r = 0; r <= X.ntau(); r++) {
        C1.get_mat(r, M1);
        C2.get_mat(r, M2);
        X.get_mat(r, M3);
        cdmatrix M = 0.5 * (M1 + M2) + M3;
        AX.set_mat(r, M);
    }
}

// A copy of the loop of vie2_mat_steep_dispatch (cntr_vie2_impl.hpp) with
// maxiter = 40, maxerr = 1e-12, pcf = 3, order = 3, which records every
// iteration.
static void cg_trajectory(const std::string &outdir) {
    const int k = 5, maxiter = 40, ntau = NC_NTAU;
    const double maxerr = 1e-12;
    integration::Integrator<double> I(k);
    GF F(0, ntau, 2, -1), Fcc(0, ntau, 2, -1), Q(0, ntau, 2, -1), G(0, ntau, 2, -1);
    GF Ffull(NC_NT, ntau, 2, -1), Fccfull(NC_NT, ntau, 2, -1), Qfull(NC_NT, ntau, 2, -1);
    nc_build(-1, Ffull, Fccfull, Qfull);
    cdmatrix M;
    for (int q = 0; q <= ntau; q++) {
        Ffull.get_mat(q, M);
        F.set_mat(q, M);
        Fccfull.get_mat(q, M);
        Fcc.set_mat(q, M);
        Qfull.get_mat(q, M);
        Q.set_mat(q, M);
    }
    vie2_mat_fourier_dispatch<double, GF, LARGESIZE>(G, F, Fcc, Q, NC_BETA, 3, 3);

    GF R(0, ntau, 2, -1), P(0, ntau, 2, -1), AP(0, ntau, 2, -1);
    GF C1(0, ntau, 2, -1), C2(0, ntau, 2, -1);
    cg_apply(R, G, F, Fcc, C1, C2, I);
    cdmatrix MR, MQ, MP, MA, MG;
    for (int r = 0; r <= ntau; r++) {
        R.get_mat(r, MR);
        Q.get_mat(r, MQ);
        cdmatrix X = MQ - MR;
        R.set_mat(r, X);
        P.set_mat(r, X);
    }
    double rsold = cg_inner(R, R, k);
    require(rsold > maxerr, "CG does not iterate");

    struct Rec {
        double rsold, c2, alpha, rsnew, beta;
        std::vector<cdmatrix> g;
    };
    std::vector<Rec> recs;
    for (int iter = 0; iter < maxiter; iter++) {
        cg_apply(AP, P, F, Fcc, C1, C2, I);
        double c2 = cg_inner(P, AP, k);
        double alpha = rsold / c2;
        for (int r = 0; r <= ntau; r++) {
            G.get_mat(r, MG);
            P.get_mat(r, MP);
            R.get_mat(r, MR);
            AP.get_mat(r, MA);
            cdmatrix X = MG + alpha * MP;
            cdmatrix Y = MR - alpha * MA;
            G.set_mat(r, X);
            R.set_mat(r, Y);
        }
        double rsnew = cg_inner(R, R, k);
        for (int r = 0; r <= ntau; r++) {
            R.get_mat(r, MR);
            P.get_mat(r, MP);
            cdmatrix X = MR + (rsnew / rsold) * MP;
            P.set_mat(r, X);
        }
        Rec rec;
        rec.rsold = rsold;
        rec.c2 = c2;
        rec.alpha = alpha;
        rec.rsnew = rsnew;
        rec.beta = rsnew / rsold;
        for (int r = 0; r <= ntau; r++) {
            G.get_mat(r, MG);
            rec.g.push_back(MG);
        }
        require(std::isfinite(c2) && c2 > 0.0 && rsnew > 0.0, "bad CG scalars");
        recs.push_back(rec);
        rsold = rsnew;
        if (std::sqrt(rsold) < maxerr) break;
    }

    // The recorded loop must reproduce the library routine
    GF G2(0, ntau, 2, -1);
    vie2_mat_steep_dispatch<double, GF, LARGESIZE>(G2, F, Fcc, Q, NC_BETA, I, maxiter,
                                                   maxerr, 3, 3);
    for (int r = 0; r <= ntau; r++) {
        G.get_mat(r, MG);
        G2.get_mat(r, M);
        require((MG - M).norm() == 0.0, "CG copy differs from the library");
    }

    const int niter = (int)recs.size();
    std::string name = "phase4_vie2_cg_trajectory_size2";
    FILE *fp = kbref::open_output(outdir, name + ".dat");
    kbref::write_metadata(fp, "iteration", meta_iteration(name, "cg", niter, ntau));
    for (int it = 1; it <= niter; it++) {
        const Rec &rec = recs[it - 1];
        for (int q = 0; q <= ntau; q++)
            for (int a = 0; a < 2; a++)
                for (int b = 0; b < 2; b++)
                    kbref::write_iteration_row(fp, "g", it, q, a, b, rec.g[q](a, b));
        kbref::write_iteration_row(fp, "rsold", it, -1, -1, -1, rec.rsold);
        kbref::write_iteration_row(fp, "c2", it, -1, -1, -1, rec.c2);
        kbref::write_iteration_row(fp, "alpha", it, -1, -1, -1, rec.alpha);
        kbref::write_iteration_row(fp, "rsnew", it, -1, -1, -1, rec.rsnew);
        kbref::write_iteration_row(fp, "beta_cg", it, -1, -1, -1, rec.beta);
    }
    std::fclose(fp);
}

static void fixpoint_trajectory(const std::string &outdir) {
    const int k = 5, ntau = NC_NTAU, niter = 6;
    integration::Integrator<double> I(k);
    GF F(NC_NT, ntau, 2, -1), Fcc(NC_NT, ntau, 2, -1), Q(NC_NT, ntau, 2, -1);
    nc_build(-1, F, Fcc, Q);
    std::string name = "phase4_vie2_fixpoint_trajectory_size2";
    FILE *fp = kbref::open_output(outdir, name + ".dat");
    kbref::write_metadata(fp, "iteration", meta_iteration(name, "fixpoint", niter, ntau));
    for (int nf = 0; nf <= niter; nf++) {
        GF G(NC_NT, ntau, 2, -1);
        vie2_mat_fixpoint(G, F, Fcc, Q, NC_BETA, I, nf);
        write_g(fp, nf, G);
    }
    std::fclose(fp);
}

//
// Start-up lesser source blocks of vie2_start, NESSi n = 0..k-1, j = 0..k:
// startup_lmix_rmix = [F^tv * G^vt](j,n), startup_less_adv = [F^< * G^A](j,n)
//
static void start_less_source(const std::string &outdir) {
    const int k = 5;
    integration::Integrator<double> I(k);
    GF F(NC_NT, NC_NTAU, 2, -1), Fcc(NC_NT, NC_NTAU, 2, -1), Q(NC_NT, NC_NTAU, 2, -1);
    GF G(NC_NT, NC_NTAU, 2, -1);
    nc_build(-1, F, Fcc, Q);
    vie2_mat(G, F, Fcc, Q, NC_BETA, I, CNTR_MAT_FIXPOINT);
    vie2_start(G, F, Fcc, Q, I, NC_BETA, NC_H);

    std::string name = "phase4_vie2_start_less_source_size2";
    FILE *fp = kbref::open_output(outdir, name + ".dat");
    kbref::Metadata meta = meta_base(name, -1);
    meta.push_back(std::make_pair("nt", kbref::fmt_int(NC_NT)));
    meta.push_back(std::make_pair("size1", std::string("2")));
    meta.push_back(std::make_pair("size2", std::string("2")));
    meta.push_back(std::make_pair("beta", kbref::fmt_double(NC_BETA)));
    meta.push_back(std::make_pair("h", kbref::fmt_double(NC_H)));
    meta.push_back(std::make_pair("solve_order", kbref::fmt_int(k)));
    kbref::write_metadata(fp, "element", meta);
    for (int op = 0; op < 2; op++) {
        for (int n = 0; n < k; n++) {
            std::vector<cplx> buf((k + 1) * 4, cplx(0.0, 0.0));
            if (op == 0)
                convolution_timestep_les_tvvt<double, GF, LARGESIZE>(n, buf.data(), G, F, Fcc,
                                                                     G, G, I, NC_BETA, NC_H);
            else
                convolution_timestep_les_lesadv<double, GF, LARGESIZE>(n, buf.data(), G, F, Fcc,
                                                                       G, G, I, NC_BETA, NC_H);
            const char *comp = (op == 0) ? "startup_lmix_rmix" : "startup_less_adv";
            for (int j = 0; j <= k; j++)
                for (int a = 0; a < 2; a++)
                    for (int b = 0; b < 2; b++)
                        kbref::write_element_row(fp, comp, n, j, a, b, buf[j * 4 + a * 2 + b]);
        }
    }
    std::fclose(fp);
}

//
// VIE2 with singular kernel, vie2_timestep_sin for n = -1..nt
//
static cdmatrix fsin_value(double t) {
    return m2(0.3 + 0.1 * std::sin(t), 0.1 * CI + 0.05 * t, -0.1 * CI + 0.05 * t, -0.2);
}

static cdmatrix qsin_value(double t) {
    return m2(0.5, 0.2 + 0.1 * CI * t, -0.3 * std::cos(t), -0.4 + 0.1 * t);
}

static void vie2_sin(const std::string &outdir, int sig, int k) {
    GF F(NC_NT, NC_NTAU, 2, sig), Fcc(NC_NT, NC_NTAU, 2, sig), Q(NC_NT, NC_NTAU, 2, sig);
    GF G(NC_NT, NC_NTAU, 2, sig);
    nc_build(sig, F, Fcc, Q);
    function<double> Fsin(NC_NT, 2), Qsin(NC_NT, 2), Gsin(NC_NT, 2);
    for (int n = -1; n <= NC_NT; n++) {
        double t = (n < 0) ? 0.0 : n * NC_H;
        cdmatrix A = fsin_value(t), B = qsin_value(t);
        require((A - A.adjoint()).norm() == 0.0, "Fsin is not hermitian");
        Fsin.set_value(n, A);
        Qsin.set_value(n, B);
    }
    for (int n = -1; n <= NC_NT; n++)
        vie2_timestep_sin(n, G, Gsin, F, Fcc, Fsin, Q, Qsin, NC_BETA, NC_H, k);
    cdmatrix M;
    G.get_les(NC_NT / 2, NC_NT, M);
    require(M.norm() > 0.0, "vanishing G^<");
    std::string name = std::string("phase4_vie2_sin_") + (sig == 1 ? "boson_" : "") +
                       "size2_k" + std::to_string(k);
    write_green(outdir, name, G, NC_BETA, NC_H, NC_MU, k, "fixpoint");
}

//
// Embedding model of libcntr/test/dyson.cpp and dyson_new.cpp
//
static void embedding(const std::string &outdir, const std::string &name, double mu,
                      double beta, int method) {
    const int nt = 200, ntau = 400, k = 5;
    const double eps1 = -1.0, eps2 = 1.0, lam = 0.2, h = 0.05;
    integration::Integrator<double> I(k);
    cdmatrix h1(1, 1), h2(1, 1);
    h1(0, 0) = eps1;
    h2(0, 0) = eps2;
    function<double> H(nt, 1);
    for (int n = -1; n <= nt; n++) H.set_value(n, h1);
    GF S(nt, ntau, 1, -1), G0(nt, ntau, 1, -1), G(nt, ntau, 1, -1);
    green_from_H(S, mu, h2, beta, h);
    for (int n = -1; n <= nt; n++) S.smul(n, lam * lam);
    green_from_H(G0, mu, h1, beta, h);
    // Matsubara component from the Dyson equation, as in the Catch tests
    dyson_mat(G, S, mu, H, I, beta, method, true);
    GF F(nt, ntau, 1, -1), Fcc(nt, ntau, 1, -1);
    for (int n = -1; n <= nt; n++) {
        convolution_timestep(n, F, G0, S, I, beta, h);
        convolution_timestep(n, Fcc, S, G0, I, beta, h);
        F.smul(n, -1.0);
        Fcc.smul(n, -1.0);
    }
    vie2_start(G, F, Fcc, G0, I, beta, h);
    for (int n = k + 1; n <= nt; n++) vie2_timestep(n, G, F, Fcc, G0, I, beta, h);
    write_green(outdir, name, G, beta, h, mu, k, method == CNTR_MAT_CG ? "cg" : "fixpoint");
}

//
// Interacting section of libcntr/test/downfold.cpp: the full 3x3 problem
// G = G0 + G0 * Sigma * G solved by vie2, Sigma = G0 = free 3x3 GF.
//
static void downfold(const std::string &outdir) {
    const int nt = 20, ntau = 300, size = 3, k = 5;
    const double beta = 1.0, h = 0.01;
    const double pi = std::acos(-1.0);
    integration::Integrator<double> I(k);
    cdmatrix a(size, size);
    a(0, 0) = 1.0;
    a(1, 1) = 2.0;
    a(2, 2) = 3.0;
    a(0, 1) = std::exp(CI * pi * 0.0);
    a(0, 2) = std::exp(CI * pi * 0.0);
    a(1, 2) = std::exp(CI * pi * 0.0);
    a(1, 0) = std::exp(-CI * pi * 0.0);
    a(2, 0) = std::exp(-CI * pi * 0.0);
    a(2, 1) = std::exp(-CI * pi * 0.0);
    GF S(nt, ntau, size, -1), G0(nt, ntau, size, -1), G(nt, ntau, size, -1);
    green_from_H(S, 0.0, a, beta, h);
    green_from_H(G0, 0.0, a, beta, h);
    GF G0S(nt, ntau, size, -1), SG0(nt, ntau, size, -1);
    convolution(G0S, G0, G0, S, S, I, beta, h);
    convolution(SG0, S, S, G0, G0, I, beta, h);
    for (int n = -1; n <= nt; n++) {
        G0S.smul(n, -1.0);
        SG0.smul(n, -1.0);
    }
    vie2(G, G0S, SG0, G0, I, beta, h, CNTR_MAT_FIXPOINT);
    write_green(outdir, "phase4_vie2_downfold_size3", G, beta, h, 0.0, k, "fixpoint");
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <output-dir>\n", argv[0]);
        return 1;
    }
    const std::string outdir(argv[1]);

    embedding(outdir, "phase4_vie2_dyson_size1", -0.1, 10.0, CNTR_MAT_FIXPOINT);
    embedding(outdir, "phase4_vie2_dyson_new_size1", 0.0, 20.0, CNTR_MAT_CG);
    downfold(outdir);

    // Non-commuting 2x2 case, full solution
    for (int k = 1; k <= 5; k++) {
        integration::Integrator<double> I(k);
        GF F(NC_NT, NC_NTAU, 2, -1), Fcc(NC_NT, NC_NTAU, 2, -1), Q(NC_NT, NC_NTAU, 2, -1);
        GF G(NC_NT, NC_NTAU, 2, -1);
        nc_build(-1, F, Fcc, Q);
        vie2(G, F, Fcc, Q, I, NC_BETA, NC_H, CNTR_MAT_FIXPOINT);
        cdmatrix M;
        G.get_les(NC_NT / 2, NC_NT, M);
        require(M.norm() > 0.0, "vanishing G^<");
        write_green(outdir, "phase4_vie2_ncomm_size2_k" + std::to_string(k), G, NC_BETA,
                    NC_H, NC_MU, k, "fixpoint");
    }
    {
        const int k = 5;
        integration::Integrator<double> I(k);
        GF F(NC_NT, NC_NTAU, 2, 1), Fcc(NC_NT, NC_NTAU, 2, 1), Q(NC_NT, NC_NTAU, 2, 1);
        GF G(NC_NT, NC_NTAU, 2, 1);
        nc_build(1, F, Fcc, Q);
        vie2(G, F, Fcc, Q, I, NC_BETA, NC_H, CNTR_MAT_FIXPOINT);
        write_green(outdir, "phase4_vie2_ncomm_boson_size2_k5", G, NC_BETA, NC_H, NC_MU, k,
                    "fixpoint");
    }

    // Routing of vie2_timestep: reduced orders at the first time steps
    for (int sig = -1; sig <= 1; sig += 2) {
        for (int last = 0; last <= 2; last += 2) {
            if (sig == 1 && last == 2) continue;
            const int k = 5;
            integration::Integrator<double> I(k);
            GF F(NC_NT, NC_NTAU, 2, sig), Fcc(NC_NT, NC_NTAU, 2, sig);
            GF Q(NC_NT, NC_NTAU, 2, sig), G(NC_NT, NC_NTAU, 2, sig);
            nc_build(sig, F, Fcc, Q);
            for (int n = -1; n <= last; n++)
                vie2_timestep(n, G, F, Fcc, Q, I, NC_BETA, NC_H, CNTR_MAT_FIXPOINT);
            std::string name = std::string("phase4_vie2_route_") +
                               (sig == -1 ? "fermion" : "boson") + "_n" + std::to_string(last);
            write_green(outdir, name, G, NC_BETA, NC_H, NC_MU, k, "fixpoint");
        }
    }

    vie2_sin(outdir, -1, 3);
    vie2_sin(outdir, -1, 5);
    vie2_sin(outdir, 1, 5);

    start_less_source(outdir);
    fixpoint_trajectory(outdir);
    cg_trajectory(outdir);
    return 0;
}
