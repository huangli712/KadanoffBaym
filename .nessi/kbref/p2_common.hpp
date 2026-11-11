// Shared inputs of the P2 kbref generators.
//
// All inputs are closed-form; the same formulas are rebuilt in the Julia
// test test/t_phase2_utilities.jl (nothing is read back from the
// generated files).
//
//   * Free Green's functions for H_X(t) = H_X0 + s I + a_X(t) I, where
//     s = 0 for fermions (sig = -1) and s = 1.5 for bosons (sig = +1), so
//     that all bosonic energies are positive. With eps = H_X0 + s - mu =
//     V diag(e) V^+, n(e) = 1 / (exp(beta e) - sig), U(t) =
//     exp(-i eps t) exp(-i phi_X(t)):
//       mat(tau)  = -V diag(exp(-e tau) (1 + sig n(e))) V^+
//       ret(t,t') = -i U(t) U(t')^+
//       tv(t,tau) = -i sig U(t) V diag(n(e) exp(e tau)) V^+
//       les(t,t') = -i sig U(t) V diag(n(e)) V^+ U(t')^+
//     The phases are phi_A(t) = 0.3 (1 - cos t), phi_B(t) = 0.1 (1 - cos 2t).
//   * f(t) = F(t) for t = i h (i >= 0), and f(-1) = F(-0.5), where
//       F(t) = [1 + 0.2 t, 0.1 + 0.3 i t; -0.2 + 0.05 i, 0.8].
//   * size = 1 uses the top-left corner of all 2x2 matrices.
#ifndef KBREF_P2_COMMON_HPP
#define KBREF_P2_COMMON_HPP

#include <cmath>
#include <complex>
#include <string>
#include <vector>

#include "cntr/cntr.hpp"
#include "reference_io.hpp"

namespace p2 {

typedef std::complex<double> cplx;

static const int NT = 12;
static const int NTAU = 24;
static const double BETA = 2.0;
static const double H = 0.05;
static const double MU = 0.1;

inline cdmatrix hamiltonian(int which) {
    cdmatrix M(2, 2);
    if (which == 0) {
        M(0, 0) = -0.5;
        M(0, 1) = cplx(0.2, 0.1);
        M(1, 0) = std::conj(M(0, 1));
        M(1, 1) = 0.4;
    } else {
        M(0, 0) = 0.3;
        M(0, 1) = cplx(0.25, -0.05);
        M(1, 0) = std::conj(M(0, 1));
        M(1, 1) = -0.2;
    }
    return M;
}

inline double phase(int which, double t) {
    if (which == 0) return 0.3 * (1.0 - std::cos(t));
    return 0.1 * (1.0 - std::cos(2.0 * t));
}

inline cdmatrix middle(double t) {
    cdmatrix F(2, 2);
    F(0, 0) = 1.0 + 0.2 * t;
    F(0, 1) = cplx(0.1, 0.3 * t);
    F(1, 0) = cplx(-0.2, 0.05);
    F(1, 1) = 0.8;
    return F;
}

inline cdmatrix from_diag(const cdmatrix &V, const cdvector &g) {
    return V * g.asDiagonal() * V.adjoint();
}

// Free Green's function on the grid (nt, ntau) with the given beta and h.
inline void build_green(cntr::herm_matrix<double> &G, int which, int size, int sig,
                        double beta, double h) {
    const int nt = G.nt(), ntau = G.ntau();
    const double shift = (sig == 1) ? 1.5 : 0.0;
    cdmatrix eps = hamiltonian(which).topLeftCorner(size, size);
    eps += (shift - MU) * cdmatrix::Identity(size, size);
    Eigen::SelfAdjointEigenSolver<cdmatrix> es(eps);
    cdmatrix V = es.eigenvectors();
    Eigen::VectorXd e = es.eigenvalues();
    std::vector<double> n(size);
    for (int a = 0; a < size; a++) n[a] = 1.0 / (std::exp(beta * e(a)) - sig);

    const cplx I(0.0, 1.0);
    cdvector g(size);
    for (int q = 0; q <= ntau; q++) {
        double tau = q * beta / ntau;
        for (int a = 0; a < size; a++)
            g(a) = -std::exp(-e(a) * tau) * (1.0 + sig * n[a]);
        cdmatrix M = from_diag(V, g);
        G.set_mat(q, M);
    }
    std::vector<cdmatrix> U(nt + 1);
    for (int i = 0; i <= nt; i++) {
        double t = i * h;
        for (int a = 0; a < size; a++) g(a) = std::exp(cplx(0.0, -e(a) * t));
        U[i] = from_diag(V, g) * std::exp(cplx(0.0, -phase(which, t)));
    }
    for (int a = 0; a < size; a++) g(a) = n[a];
    cdmatrix rho = from_diag(V, g);
    for (int i = 0; i <= nt; i++) {
        for (int j = 0; j <= i; j++) {
            cdmatrix R = -I * U[i] * U[j].adjoint();
            G.set_ret(i, j, R);
        }
        for (int j = i; j <= nt; j++) {
            cdmatrix L = (-I * (double)sig) * U[i] * rho * U[j].adjoint();
            G.set_les(i, j, L);
        }
        for (int q = 0; q <= ntau; q++) {
            double tau = q * beta / ntau;
            for (int a = 0; a < size; a++) g(a) = n[a] * std::exp(e(a) * tau);
            cdmatrix T = (-I * (double)sig) * U[i] * from_diag(V, g);
            G.set_tv(i, q, T);
        }
    }
}

inline void build_function(cntr::function<double> &f, int size, double h) {
    for (int i = -1; i <= f.nt(); i++) {
        double t = (i < 0) ? -0.5 : i * h;
        cdmatrix F = middle(t).topLeftCorner(size, size);
        f.set_value(i, F);
    }
}

// Common metadata of all P2 files.
inline kbref::Metadata base_meta(const std::string &generator,
                                 const std::string &case_set) {
    kbref::Metadata meta;
    meta.push_back(std::make_pair("generator", generator));
    meta.push_back(std::make_pair("case_set", case_set));
    return meta;
}

// Write a herm_matrix file with a single case.
inline void write_gf(const std::string &outdir, const std::string &generator,
                     const std::string &name, int case_id,
                     cntr::herm_matrix<double> &G, double beta, double h,
                     const kbref::Metadata &extra = kbref::Metadata()) {
    FILE *fp = kbref::open_output(outdir, name + ".dat");
    kbref::Metadata meta = base_meta(generator, name);
    meta.push_back(std::make_pair("case", name));
    meta.push_back(std::make_pair("case_id", kbref::fmt_int(case_id)));
    meta.push_back(std::make_pair("nt", kbref::fmt_int(G.nt())));
    meta.push_back(std::make_pair("ntau", kbref::fmt_int(G.ntau())));
    meta.push_back(std::make_pair("size1", kbref::fmt_int(G.size1())));
    meta.push_back(std::make_pair("size2", kbref::fmt_int(G.size2())));
    meta.push_back(std::make_pair("beta", kbref::fmt_double(beta)));
    meta.push_back(std::make_pair("h", kbref::fmt_double(h)));
    meta.push_back(std::make_pair("sig", kbref::fmt_int(G.sig())));
    for (size_t n = 0; n < extra.size(); n++) meta.push_back(extra[n]);
    kbref::write_metadata(fp, "herm_matrix", meta);
    cdmatrix M;
    kbref::write_herm_matrix(fp, G, M);
    std::fclose(fp);
}

// A labels file collecting (case_id, order[, alpha]) entries.
struct Labels {
    std::vector<int> id;
    std::vector<int> order;
    std::vector<cplx> alpha;
    std::vector<bool> has_alpha;
    void add(int case_id, int k) {
        id.push_back(case_id); order.push_back(k);
        alpha.push_back(0.0); has_alpha.push_back(false);
    }
    void add(int case_id, int k, cplx a) {
        id.push_back(case_id); order.push_back(k);
        alpha.push_back(a); has_alpha.push_back(true);
    }
    void write(const std::string &outdir, const std::string &generator,
               const std::string &name) const {
        FILE *fp = kbref::open_output(outdir, name + ".dat");
        kbref::write_metadata(fp, "labels", base_meta(generator, name));
        for (size_t n = 0; n < id.size(); n++) {
            kbref::write_label_row(fp, "order", id[n], cplx(order[n], 0.0));
            if (has_alpha[n]) kbref::write_label_row(fp, "alpha", id[n], alpha[n]);
        }
        std::fclose(fp);
    }
};

} // namespace p2

#endif // KBREF_P2_COMMON_HPP
