// kbref-v1: plain-text reference data writer shared by all KB generators.
//
// Format (see .omo/plans/01-phase1-foundation.md, section 3.4):
//   * metadata lines start with '#', the first one is "# kbref 1";
//   * data lines have exactly 7 columns: component i j a b re im;
//   * all floating point numbers are written with %.17e;
//   * non-finite values are rejected.
#ifndef KBREF_REFERENCE_IO_HPP
#define KBREF_REFERENCE_IO_HPP

#include <cmath>
#include <complex>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kbref {

typedef std::vector<std::pair<std::string, std::string> > Metadata;

inline std::string fmt_double(double x) {
    if (!std::isfinite(x)) {
        throw std::runtime_error("kbref: non-finite value");
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17e", x);
    return std::string(buf);
}

inline std::string fmt_int(int x) {
    return std::to_string(x);
}

// Open an output file inside the given output directory.
inline FILE *open_output(const std::string &dir, const std::string &name) {
    std::string path = dir + "/" + name;
    FILE *fp = std::fopen(path.c_str(), "w");
    if (fp == nullptr) {
        throw std::runtime_error("kbref: cannot open " + path);
    }
    return fp;
}

// Write the metadata block. data_kind must be "weights" or "herm_matrix".
inline void write_metadata(FILE *fp, const std::string &data_kind,
                           const Metadata &meta) {
    if (data_kind != "weights" && data_kind != "herm_matrix") {
        throw std::runtime_error("kbref: unknown data_kind " + data_kind);
    }
    std::fprintf(fp, "# kbref 1\n");
    std::fprintf(fp, "# data_kind %s\n", data_kind.c_str());
    for (size_t n = 0; n < meta.size(); n++) {
        std::fprintf(fp, "# %s %s\n", meta[n].first.c_str(),
                     meta[n].second.c_str());
    }
    std::fprintf(fp, "# columns component i j a b re im\n");
}

inline void write_row(FILE *fp, const char *comp, int i, int j, int a, int b,
                      double re, double im) {
    std::string sre = fmt_double(re), sim = fmt_double(im);
    std::fprintf(fp, "%s %d %d %d %d %s %s\n", comp, i, j, a, b,
                 sre.c_str(), sim.c_str());
}

// Weight table entry: the value is stored in `re`, `im` is zero.
inline void write_weight_row(FILE *fp, const char *comp, int i, int j, int a,
                             int b, double value) {
    write_row(fp, comp, i, j, a, b, value, 0.0);
}

// Write the independent storage domain of a herm_matrix in the fixed order
// mat, ret, tv, les; each component is ordered lexicographically by
// (i, j, a, b). Only the public get_* accessors are used; M is a work
// matrix (e.g. cdmatrix).
template <class GF, class Matrix>
void write_herm_matrix(FILE *fp, const GF &G, Matrix &M) {
    const int nt = G.nt(), ntau = G.ntau();
    const int s1 = G.size1(), s2 = G.size2();
    for (int q = 0; q <= ntau; q++) {
        G.get_mat(q, M);
        for (int a = 0; a < s1; a++)
            for (int b = 0; b < s2; b++)
                write_row(fp, "mat", q, -1, a, b, M(a, b).real(), M(a, b).imag());
    }
    for (int i = 0; i <= nt; i++)
        for (int j = 0; j <= i; j++) {
            G.get_ret(i, j, M);
            for (int a = 0; a < s1; a++)
                for (int b = 0; b < s2; b++)
                    write_row(fp, "ret", i, j, a, b, M(a, b).real(), M(a, b).imag());
        }
    for (int i = 0; i <= nt; i++)
        for (int q = 0; q <= ntau; q++) {
            G.get_tv(i, q, M);
            for (int a = 0; a < s1; a++)
                for (int b = 0; b < s2; b++)
                    write_row(fp, "tv", i, q, a, b, M(a, b).real(), M(a, b).imag());
        }
    for (int i = 0; i <= nt; i++)
        for (int j = i; j <= nt; j++) {
            G.get_les(i, j, M);
            for (int a = 0; a < s1; a++)
                for (int b = 0; b < s2; b++)
                    write_row(fp, "les", i, j, a, b, M(a, b).real(), M(a, b).imag());
        }
}

} // namespace kbref

#endif // KBREF_REFERENCE_IO_HPP
