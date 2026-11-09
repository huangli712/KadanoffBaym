// kbref generator: P2 response convolution.
//
// Usage: p2_response <output-dir>
// Output (kbref-v1):
//   p2_response_values.dat  scalar  response_convolution(tstp, c, W, a1, a2,
//                                   f, b1, b2, k, BETA, H) for the herm_matrix
//                                   W, tstp = -1 and k..NT (the NESSi domain)
//   p2_response_labels.dat  labels  order of every case
//
// case_id = (((size - 1) * 5 + (k - 1)) * 4 + orb) * 16 + (tstp + 1), where
// orb indexes the orbital choices (a1, a2, b1, b2):
//   0: (0,0,0,0)  1: (0,1,1,0)  2: (1,0,0,1)  3: (1,1,1,1)
// For size = 1 only orb = 0 is used. W is the free Green's function A of
// p2_common.hpp (hermitian, fermionic), f is the contour function there.
#include <cstdio>
#include <cstdlib>
#include <string>

// CNTR_NO_EXTERN_TEMPLATES is defined by the CMake target.
#include "p2_common.hpp"

using namespace cntr;
using namespace p2;

static const std::string GEN = "p2_response";

// NESSi 2.0.0 declares response_convolution(..., herm_matrix<T> &W, ...)
// in cntr_response_convolution_decl.hpp, but defines it in the impl header
// as a different template with `GREEN &W` (GREEN = herm_matrix<double>).
// Both are visible here, so every call is ambiguous and cannot be resolved
// by the caller. The body below is copied verbatim from the NESSi
// definition (cntr_response_convolution_impl.hpp, herm_matrix version,
// branches tstp == -1 and tstp >= SolveOrder) and uses the NESSi helper
// response_integrate and Integrator without any change.
static void response(int tstp, cplx &cc, herm_matrix<double> &W, int a1, int a2,
                     function<double> &f, int b1, int b2, int SolveOrder,
                     double beta, double h) {
    int ntau = W.ntau();
    double dtau = beta / ntau;
    int sizew = W.size1(), idxw = a1 * sizew + a2, sw = sizew * sizew;
    int sizef = f.size1(), idxf = b1 * sizef + b2, sf = sizef * sizef;
    cplx res;
    if (tstp == -1) {
        response_integrate<double>(ntau, dtau, res, W.matptr(0), sw, idxw,
                                   integration::I<double>(SolveOrder));
        cc = res * f.ptr(-1)[idxf];
    } else if (tstp >= SolveOrder) {
        response_integrate<double>(ntau, dtau, res, W.tvptr(tstp, 0), sw, idxw,
                                   integration::I<double>(SolveOrder));
        cc = res * (cplx(0, -1.0) * f.ptr(-1)[idxf]);
        response_integrate<double>(tstp, h, res, W.retptr(tstp, 0), sw, idxw, f.ptr(0), sf,
                                   idxf, integration::I<double>(SolveOrder));
        cc += res;
    } else {
        std::fprintf(stderr, "p2_response: tstp < SolveOrder is not supported\n");
        std::exit(1);
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <output-dir>\n", argv[0]);
        return 1;
    }
    const std::string outdir(argv[1]);
    const int orbs[4][4] = {{0, 0, 0, 0}, {0, 1, 1, 0}, {1, 0, 0, 1}, {1, 1, 1, 1}};
    Labels labels;

    std::string name = GEN + "_values";
    FILE *fp = kbref::open_output(outdir, name + ".dat");
    kbref::Metadata meta = base_meta(GEN, name);
    meta.push_back(std::make_pair("nt", kbref::fmt_int(NT)));
    meta.push_back(std::make_pair("ntau", kbref::fmt_int(NTAU)));
    meta.push_back(std::make_pair("beta", kbref::fmt_double(BETA)));
    meta.push_back(std::make_pair("h", kbref::fmt_double(H)));
    meta.push_back(std::make_pair("orbitals",
                                  "0:(0,0,0,0),1:(0,1,1,0),2:(1,0,0,1),3:(1,1,1,1)"));
    kbref::write_metadata(fp, "scalar", meta);

    for (int size = 1; size <= 2; size++) {
        herm_matrix<double> W(NT, NTAU, size, -1);
        build_green(W, 0, size, -1, BETA, H);
        function<double> f(NT, size);
        build_function(f, size, H);
        for (int k = 1; k <= 5; k++) {
            for (int orb = 0; orb < (size == 1 ? 1 : 4); orb++) {
                for (int tstp = -1; tstp <= NT; tstp++) {
                    if (tstp >= 0 && tstp < k) continue;
                    int id = (((size - 1) * 5 + (k - 1)) * 4 + orb) * 16 + (tstp + 1);
                    cplx c;
                    response(tstp, c, W, orbs[orb][0], orbs[orb][1], f,
                             orbs[orb][2], orbs[orb][3], k, BETA, H);
                    kbref::write_scalar_row(fp, "response", id, c);
                    labels.add(id, k);
                }
            }
        }
    }
    std::fclose(fp);
    labels.write(outdir, GEN, GEN + "_labels");
    return 0;
}
