// kbref generator: NESSi density matrix of herm_matrix.
//
// Usage: phase4_observables <output-dir>
// Output (kbref-v1, data_kind element, component density_matrix):
//   phase4_observables_density_equilibrium.dat  j = -1
//   phase4_observables_density_realtime.dat     j = 0..nt
//
// Input (the same formulas are used in test/t_phase4.jl): the free
// bosonic Green's function green_from_H(G, mu, H, beta, h) with
// H = [1.2, 0.3 - 0.2i; 0.3 + 0.2i, 1.8], mu = -0.4, and fermionic one
// with H - 1.5, mu = 0.1. The equilibrium file uses the bosons and the
// realtime file uses the fermions. The case id is 0.
#include <complex>
#include <cstdio>
#include <string>

// CNTR_NO_EXTERN_TEMPLATES is defined by the CMake target.
#include "cntr/cntr.hpp"
#include "reference_io.hpp"

using namespace cntr;
typedef std::complex<double> cplx;
typedef herm_matrix<double> GF;

static const int NT = 8;
static const int NTAU = 16;
static const double BETA = 3.0;
static const double H = 0.1;

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <output-dir>\n", argv[0]);
        return 1;
    }
    const std::string outdir(argv[1]);
    cdmatrix Hb(2, 2);
    Hb(0, 0) = 1.2;
    Hb(0, 1) = cplx(0.3, -0.2);
    Hb(1, 0) = cplx(0.3, 0.2);
    Hb(1, 1) = 1.8;

    for (int rt = 0; rt <= 1; rt++) {
        const int sig = rt ? -1 : 1;
        GF G(NT, NTAU, 2, sig);
        cdmatrix Hf = Hb - 1.5 * cdmatrix::Identity(2, 2);
        if (sig == 1)
            green_from_H(G, -0.4, Hb, BETA, H);
        else
            green_from_H(G, 0.1, Hf, BETA, H);
        std::string name = std::string("phase4_observables_density_") +
                           (rt ? "realtime" : "equilibrium");
        FILE *fp = kbref::open_output(outdir, name + ".dat");
        kbref::Metadata meta;
        meta.push_back(std::make_pair("case", name));
        meta.push_back(std::make_pair("generator", std::string("phase4_observables")));
        meta.push_back(std::make_pair("case_set", std::string("phase4_observables")));
        meta.push_back(std::make_pair("sig", kbref::fmt_int(sig)));
        meta.push_back(std::make_pair("nt", kbref::fmt_int(NT)));
        meta.push_back(std::make_pair("ntau", kbref::fmt_int(NTAU)));
        meta.push_back(std::make_pair("size1", std::string("2")));
        meta.push_back(std::make_pair("size2", std::string("2")));
        meta.push_back(std::make_pair("beta", kbref::fmt_double(BETA)));
        meta.push_back(std::make_pair("h", kbref::fmt_double(H)));
        kbref::write_metadata(fp, "element", meta);
        cdmatrix M;
        for (int tstp = (rt ? 0 : -1); tstp <= (rt ? NT : -1); tstp++) {
            G.density_matrix(tstp, M);
            for (int a = 0; a < 2; a++)
                for (int b = 0; b < 2; b++)
                    kbref::write_element_row(fp, "density_matrix", 0, tstp, a, b, M(a, b));
        }
        std::fclose(fp);
    }
    return 0;
}
