// kbref generator: NESSi Bubble1 and Bubble2.
//
// Usage: phase4_bubble <output-dir>
// Output (kbref-v1, data_kind herm_matrix, one case per file):
//   phase4_bubble{1,2}_{ff,fb,bf,bb}_size{1,2}.dat
//
// Inputs (the same closed-form formulas are used in test/nessi/t_phase4.jl;
// nothing is read back from the output): A and B are the free Green's
// functions green_from_H(X, mu, H, beta, h) with
//   HA = [0.4, 0.3 - 0.2i; 0.3 + 0.2i, -0.6],
//   HB = [1.2, 0.1i; -0.1i, 1.9],
//   mu = 0.1 (fermions) or -0.5 (bosons) for A,
//   mu = 0.2 (fermions) or -0.3 (bosons) for B,
// and size 1 uses the top-left corners. For size 2 the element (c1,c2) of
// C is the bubble of A(c1,c2) and B(c2,c1) (Bubble1), or of A(c1,c2) and
// B(c1,c2) (Bubble2), i.e., Bubble(tstp, C, c1, c2, A, c1, c2, B, b1, b2).
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
typedef herm_matrix<double> GF;

static const int NT = 10;
static const int NTAU = 20;
static const double BETA = 2.0;
static const double H = 0.1;

static cdmatrix ham(int which, int size) {
    cdmatrix M(2, 2);
    if (which == 0) {
        M(0, 0) = 0.4;
        M(0, 1) = cplx(0.3, -0.2);
        M(1, 0) = cplx(0.3, 0.2);
        M(1, 1) = -0.6;
    } else {
        M(0, 0) = 1.2;
        M(0, 1) = cplx(0.0, 0.1);
        M(1, 0) = cplx(0.0, -0.1);
        M(1, 1) = 1.9;
    }
    return M.topLeftCorner(size, size);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <output-dir>\n", argv[0]);
        return 1;
    }
    const std::string outdir(argv[1]);
    const char *tags[2] = {"f", "b"};

    for (int bubble = 1; bubble <= 2; bubble++)
        for (int size = 1; size <= 2; size++)
            for (int ia = 0; ia < 2; ia++)
                for (int ib = 0; ib < 2; ib++) {
                    const int siga = ia == 0 ? -1 : 1, sigb = ib == 0 ? -1 : 1;
                    GF A(NT, NTAU, size, siga), B(NT, NTAU, size, sigb);
                    GF C(NT, NTAU, size, siga * sigb);
                    cdmatrix HA = ham(0, size), HB = ham(1, size);
                    green_from_H(A, siga == -1 ? 0.1 : -0.5, HA, BETA, H);
                    green_from_H(B, sigb == -1 ? 0.2 : -0.3, HB, BETA, H);
                    for (int tstp = -1; tstp <= NT; tstp++)
                        for (int c1 = 0; c1 < size; c1++)
                            for (int c2 = 0; c2 < size; c2++) {
                                if (bubble == 1)
                                    Bubble1(tstp, C, c1, c2, A, c1, c2, B, c2, c1);
                                else
                                    Bubble2(tstp, C, c1, c2, A, c1, c2, B, c1, c2);
                            }
                    std::string name = "phase4_bubble" + std::to_string(bubble) + "_" +
                                       tags[ia] + tags[ib] + "_size" + std::to_string(size);
                    FILE *fp = kbref::open_output(outdir, name + ".dat");
                    kbref::Metadata meta;
                    meta.push_back(std::make_pair("case", name));
                    meta.push_back(std::make_pair("generator", std::string("phase4_bubble")));
                    meta.push_back(std::make_pair("case_set", std::string("phase4_bubble")));
                    meta.push_back(std::make_pair("sig", kbref::fmt_int(C.sig())));
                    meta.push_back(std::make_pair("bubble", std::to_string(bubble)));
                    meta.push_back(std::make_pair("signA", kbref::fmt_int(siga)));
                    meta.push_back(std::make_pair("signB", kbref::fmt_int(sigb)));
                    meta.push_back(std::make_pair("nt", kbref::fmt_int(NT)));
                    meta.push_back(std::make_pair("ntau", kbref::fmt_int(NTAU)));
                    meta.push_back(std::make_pair("size1", kbref::fmt_int(size)));
                    meta.push_back(std::make_pair("size2", kbref::fmt_int(size)));
                    meta.push_back(std::make_pair("beta", kbref::fmt_double(BETA)));
                    meta.push_back(std::make_pair("h", kbref::fmt_double(H)));
                    kbref::write_metadata(fp, "herm_matrix", meta);
                    cdmatrix M;
                    kbref::write_herm_matrix(fp, C, M);
                    std::fclose(fp);
                }
    return 0;
}
