// kbref generator: NESSi Type-1 (C = A*B) and Type-2 (C = A*f*B) full
// contour convolutions.
//
// Usage: convolution <output-dir>
// Output: <output-dir>/convolution_type{1,2}_size{1,2}_k{1..5}.dat
//         (kbref-v1, data_kind herm_matrix)
//
// Inputs (identical closed-form formulas are used in the Julia test
// test/nessi/t_convolution.jl; nothing is read back from the output):
//
//   * Fermionic free Green's functions A and B for the time-dependent
//     hermitian Hamiltonians H_X(t) = H_X0 + a_X(t) I, with a_X(0) = 0:
//       a_A(t) = 0.3 sin(t),   phi_A(t) = 0.3 (1 - cos(t)),
//       a_B(t) = 0.2 sin(2t),  phi_B(t) = 0.1 (1 - cos(2t)),
//     where phi_X(t) = int_0^t a_X. With eps = H_X0 - mu = V diag(e) V^+,
//     n(e) = 1 / (1 + exp(beta e)) and U(t) = exp(-i eps t) exp(-i phi(t)):
//       mat(tau)  = -V diag(exp(-e tau) / (1 + exp(-beta e))) V^+
//       ret(t,t') = -i U(t) U(t')^+
//       tv(t,tau) =  i U(t) V diag(n(e) exp(e tau)) V^+
//       les(t,t') =  i U(t) V diag(n(e)) V^+ U(t')^+
//   * Time-dependent, non-hermitian f(t) (Matsubara value f(-1) = f(t=0)),
//     which does not commute with A or B for size = 2.
//   * size = 1 uses the top-left corner of all 2x2 matrices.
//
// The hermitian NESSi shorthand convolution(C, A, A, [f,] B, B, ...) is
// used, i.e. Acc = A and Bcc = B.
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include <vector>

// CNTR_NO_EXTERN_TEMPLATES is defined by the CMake target.
#include "cntr/cntr.hpp"
#include "reference_io.hpp"

using namespace cntr;
typedef std::complex<double> cplx;

static const int NT = 30;
static const int NTAU = 60;
static const double BETA = 4.0;
static const double H = 0.05;
static const double MU = 0.1;
static const int SIG = -1;

static cdmatrix hamiltonian(int which) {
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

static double phase(int which, double t) {
    if (which == 0) return 0.3 * (1.0 - std::cos(t));
    return 0.1 * (1.0 - std::cos(2.0 * t));
}

static cdmatrix middle_function(double t) {
    cdmatrix F(2, 2);
    F(0, 0) = 1.0 + 0.2 * t;
    F(0, 1) = cplx(0.1, 0.3 * t);
    F(1, 0) = cplx(-0.2, 0.05);
    F(1, 1) = 0.8;
    return F;
}

// V diag(g) V^+
static cdmatrix from_diag(const cdmatrix &V, const cdvector &g) {
    return V * g.asDiagonal() * V.adjoint();
}

static void build_green(herm_matrix<double> &G, int which, int size) {
    cdmatrix eps = hamiltonian(which).topLeftCorner(size, size);
    eps -= MU * cdmatrix::Identity(size, size);
    Eigen::SelfAdjointEigenSolver<cdmatrix> es(eps);
    cdmatrix V = es.eigenvectors();
    Eigen::VectorXd e = es.eigenvalues();

    cdvector g(size);
    // Matsubara component
    for (int q = 0; q <= NTAU; q++) {
        double tau = q * BETA / NTAU;
        for (int a = 0; a < size; a++)
            g(a) = -std::exp(-e(a) * tau) / (1.0 + std::exp(-BETA * e(a)));
        cdmatrix M = from_diag(V, g);
        G.set_mat(q, M);
    }
    // Propagators U(t)
    std::vector<cdmatrix> U(NT + 1);
    for (int i = 0; i <= NT; i++) {
        double t = i * H;
        for (int a = 0; a < size; a++)
            g(a) = std::exp(cplx(0.0, -e(a) * t));
        U[i] = from_diag(V, g) * std::exp(cplx(0.0, -phase(which, t)));
    }
    // Occupation matrix
    for (int a = 0; a < size; a++)
        g(a) = 1.0 / (1.0 + std::exp(BETA * e(a)));
    cdmatrix rho = from_diag(V, g);
    const cplx I(0.0, 1.0);
    for (int i = 0; i <= NT; i++) {
        for (int j = 0; j <= i; j++) {
            cdmatrix R = -I * U[i] * U[j].adjoint();
            G.set_ret(i, j, R);
        }
        for (int j = i; j <= NT; j++) {
            cdmatrix L = I * U[i] * rho * U[j].adjoint();
            G.set_les(i, j, L);
        }
        for (int q = 0; q <= NTAU; q++) {
            double tau = q * BETA / NTAU;
            for (int a = 0; a < size; a++)
                g(a) = std::exp(e(a) * tau) / (1.0 + std::exp(BETA * e(a)));
            cdmatrix T = I * U[i] * from_diag(V, g);
            G.set_tv(i, q, T);
        }
    }
}

static void write_case(const std::string &outdir, int type, int size, int k,
                       herm_matrix<double> &C) {
    std::string name = "convolution_type" + std::to_string(type) + "_size" +
                       std::to_string(size) + "_k" + std::to_string(k);
    FILE *fp = kbref::open_output(outdir, name + ".dat");
    kbref::Metadata meta;
    meta.push_back(std::make_pair("case", name));
    meta.push_back(std::make_pair("nt", kbref::fmt_int(NT)));
    meta.push_back(std::make_pair("ntau", kbref::fmt_int(NTAU)));
    meta.push_back(std::make_pair("size1", kbref::fmt_int(size)));
    meta.push_back(std::make_pair("size2", kbref::fmt_int(size)));
    meta.push_back(std::make_pair("beta", kbref::fmt_double(BETA)));
    meta.push_back(std::make_pair("h", kbref::fmt_double(H)));
    meta.push_back(std::make_pair("solve_order", kbref::fmt_int(k)));
    meta.push_back(std::make_pair("sig", kbref::fmt_int(SIG)));
    kbref::write_metadata(fp, "herm_matrix", meta);
    cdmatrix M;
    kbref::write_herm_matrix(fp, C, M);
    std::fclose(fp);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <output-dir>\n", argv[0]);
        return 1;
    }
    const std::string outdir(argv[1]);

    for (int size = 1; size <= 2; size++) {
        herm_matrix<double> A(NT, NTAU, size, SIG), B(NT, NTAU, size, SIG);
        build_green(A, 0, size);
        build_green(B, 1, size);

        function<double> f(NT, size);
        for (int i = -1; i <= NT; i++) {
            double t = (i < 0) ? 0.0 : i * H;
            cdmatrix F = middle_function(t).topLeftCorner(size, size);
            f.set_value(i, F);
        }

        for (int k = 1; k <= 5; k++) {
            herm_matrix<double> C1(NT, NTAU, size, SIG), C2(NT, NTAU, size, SIG);
            convolution(C1, A, A, B, B, BETA, H, k);
            convolution(C2, A, A, f, B, B, BETA, H, k);
            write_case(outdir, 1, size, k, C1);
            write_case(outdir, 2, size, k, C2);
        }
    }
    return 0;
}
