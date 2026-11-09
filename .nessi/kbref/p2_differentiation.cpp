// kbref generator: P2 differentiation of herm_matrix.
//
// Usage: p2_differentiation <output-dir>
// Output (kbref-v1), all files start with "p2_differentiation_":
//   {fermi,bose}_k{1..5}_d{1,2}.dat   herm_matrix  deriv{1,2}_timestep for
//                                     tstp = -1..NT on the standard grid
//   min_k{1..5}_d{1,2}.dat            herm_matrix  the same on the minimal
//                                     legal Matsubara grid ntau = 2k + 1
//   labels.dat                        labels       order of every case
//
// The hermitian NESSi convention Acc = A is used. Inputs: see p2_common.hpp.
//
// NOTE: the differentiation routines of NESSi 2.0.0
// (cntr_differentiation_impl.hpp) read and write the matrix elements by
// the scalar accessors get_mat(i, cplx&), get_ret(i, j, cplx&), ... and
// set_mat(i, cplx), ..., i.e. only the element (0,0) is processed. They are
// therefore valid for size1 = 1 only, and all cases here use size = 1.
// The multi-orbital derivatives of KB are verified analytically in
// test/basic/t_differentiation.jl.
#include <cstdio>
#include <string>

// CNTR_NO_EXTERN_TEMPLATES is defined by the CMake target.
#include "p2_common.hpp"

using namespace cntr;
using namespace p2;

static const std::string GEN = "p2_differentiation";

static std::string fname(const std::string &s) { return GEN + "_" + s; }

static void run_case(const std::string &outdir, Labels &labels, int case_id,
                     const std::string &tag, int k, int sig, int ntau) {
    herm_matrix<double> A(NT, ntau, 1, sig);
    build_green(A, 0, 1, sig, BETA, H);
    herm_matrix<double> d1(NT, ntau, 1, sig), d2(NT, ntau, 1, sig);
    for (int tstp = -1; tstp <= NT; tstp++) {
        deriv1_timestep(tstp, d1, A, A, BETA, H, k);
        deriv2_timestep(tstp, d2, A, A, BETA, H, k);
    }
    write_gf(outdir, GEN, fname(tag + "_d1"), case_id, d1, BETA, H);
    write_gf(outdir, GEN, fname(tag + "_d2"), case_id + 1, d2, BETA, H);
    labels.add(case_id, k);
    labels.add(case_id + 1, k);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <output-dir>\n", argv[0]);
        return 1;
    }
    const std::string outdir(argv[1]);
    Labels labels;
    int id = 0;
    for (int k = 1; k <= 5; k++) {
        std::string ks = "_k" + std::to_string(k);
        run_case(outdir, labels, id, "fermi" + ks, k, -1, NTAU);
        id += 2;
        run_case(outdir, labels, id, "bose" + ks, k, 1, NTAU);
        id += 2;
        run_case(outdir, labels, id, "min" + ks, k, -1, 2 * k + 1);
        id += 2;
    }
    labels.write(outdir, GEN, fname("labels"));
    return 0;
}
