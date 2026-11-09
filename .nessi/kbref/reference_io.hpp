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

// Known data kinds. P1: weights, herm_matrix. P2: scalar, function,
// element, labels (see .omo/plans/02-phase2-utilities.md, section 3.5).
// P4: iteration (see .omo/plans/04-phase4-vie2-bubble.md, section 6.2).
// P6: pseudo (see .omo/plans/06-phase6-pseudo.md, P6-W5-T9).
inline bool known_data_kind(const std::string &k) {
    return k == "weights" || k == "herm_matrix" || k == "scalar" ||
           k == "function" || k == "element" || k == "labels" ||
           k == "iteration" || k == "pseudo";
}

inline bool has_meta(const Metadata &meta, const std::string &key) {
    for (size_t n = 0; n < meta.size(); n++)
        if (meta[n].first == key) return true;
    return false;
}

inline std::string get_meta(const Metadata &meta, const std::string &key) {
    for (size_t n = 0; n < meta.size(); n++)
        if (meta[n].first == key) return meta[n].second;
    return std::string();
}

// Write the metadata block. The P2 data kinds require the metadata
// `generator` and `case_set`; `function` and `element` require `nt`,
// `size1`, and `size2`.
inline void write_metadata(FILE *fp, const std::string &data_kind,
                           const Metadata &meta) {
    if (!known_data_kind(data_kind)) {
        throw std::runtime_error("kbref: unknown data_kind " + data_kind);
    }
    if (data_kind != "weights" && data_kind != "herm_matrix") {
        if (!has_meta(meta, "generator") || !has_meta(meta, "case_set"))
            throw std::runtime_error("kbref: missing generator/case_set");
    }
    if (data_kind == "function" || data_kind == "element") {
        if (!has_meta(meta, "nt") || !has_meta(meta, "size1") ||
            !has_meta(meta, "size2"))
            throw std::runtime_error("kbref: missing nt/size1/size2");
    }
    if (data_kind == "iteration") {
        const char *keys[] = {"algorithm", "niter", "ntau", "size1", "size2"};
        for (const char *key : keys)
            if (!has_meta(meta, key))
                throw std::runtime_error(std::string("kbref: missing ") + key);
        std::string alg = get_meta(meta, "algorithm");
        if (alg != "fixpoint" && alg != "cg")
            throw std::runtime_error("kbref: bad algorithm " + alg);
    }
    // P6 files: one case per file.
    if (data_kind == "pseudo") {
        const char *keys[] = {"generator", "case", "sig", "nt", "ntau",
                              "size1", "size2", "k"};
        for (const char *key : keys)
            if (!has_meta(meta, key))
                throw std::runtime_error(std::string("kbref: missing ") + key);
    }
    // P4 files: one case per file, the case and the statistics are
    // metadata only.
    if (get_meta(meta, "generator").compare(0, 7, "phase4_") == 0) {
        if (!has_meta(meta, "case") || !has_meta(meta, "sig"))
            throw std::runtime_error("kbref: missing case/sig");
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

inline void check_case_id(int case_id) {
    if (case_id < 0) throw std::runtime_error("kbref: negative case_id");
}

// scalar: component case_id -1 -1 -1 re im. Distances must be real.
inline void write_scalar_row(FILE *fp, const std::string &comp, int case_id,
                             std::complex<double> v) {
    bool dist = comp == "distance_ret" || comp == "distance_tv" ||
                comp == "distance_les" || comp == "distance_total";
    if (!dist && comp != "response")
        throw std::runtime_error("kbref: unknown scalar component " + comp);
    check_case_id(case_id);
    if (dist && v.imag() != 0.0)
        throw std::runtime_error("kbref: distance must be real");
    write_row(fp, comp.c_str(), case_id, -1, -1, -1, v.real(), v.imag());
}

// function: component case_id j a b re im. Here j is the NESSi time index
// -1..nt of the function, or -1 for an interpolated value.
inline void write_function_row(FILE *fp, const std::string &comp, int case_id,
                               int j, int a, int b, std::complex<double> v) {
    if (comp != "extrapolate" && comp != "interpolate")
        throw std::runtime_error("kbref: unknown function component " + comp);
    check_case_id(case_id);
    if (j < -1 || (comp == "interpolate" && j != -1) || a < 0 || b < 0)
        throw std::runtime_error("kbref: function index out of range");
    write_row(fp, comp.c_str(), case_id, j, a, b, v.real(), v.imag());
}

// element: component case_id j a b re im. For les_timediag and
// density_matrix, j is the NESSi contour step -1..nt. For the start-up
// source blocks startup_less_adv and startup_lmix_rmix, j is the NESSi
// block index 0..max(n,k) of C^<(j,n).
inline void write_element_row(FILE *fp, const std::string &comp, int case_id,
                              int tstp, int a, int b, std::complex<double> v) {
    bool startup = comp == "startup_less_adv" || comp == "startup_lmix_rmix";
    if (comp != "les_timediag" && comp != "density_matrix" && !startup)
        throw std::runtime_error("kbref: unknown element component " + comp);
    check_case_id(case_id);
    if (tstp < (startup ? 0 : -1) || a < 0 || b < 0)
        throw std::runtime_error("kbref: element index out of range");
    write_row(fp, comp.c_str(), case_id, tstp, a, b, v.real(), v.imag());
}

// labels: order (integer 0..5 in re) or alpha (complex).
inline void write_label_row(FILE *fp, const std::string &comp, int case_id,
                            std::complex<double> v) {
    check_case_id(case_id);
    if (comp == "order") {
        double r = v.real();
        if (v.imag() != 0.0 || r != std::floor(r) || r < 0.0 || r > 5.0)
            throw std::runtime_error("kbref: bad order label");
    } else if (comp != "alpha") {
        throw std::runtime_error("kbref: unknown label component " + comp);
    }
    write_row(fp, comp.c_str(), case_id, -1, -1, -1, v.real(), v.imag());
}

// iteration: component iter j a b re im. The component `g` is the
// Matsubara value G[j](a,b) after the iteration `iter` (iter = 0 is the
// initial guess); the real scalars rsold, c2, alpha, rsnew, and beta_cg
// (CG only) use j = a = b = -1 and im = 0.
inline void write_iteration_row(FILE *fp, const std::string &comp, int iter,
                                int j, int a, int b, std::complex<double> v) {
    if (comp == "g") {
        if (iter < 0 || j < 0 || a < 0 || b < 0)
            throw std::runtime_error("kbref: iteration index out of range");
    } else if (comp == "rsold" || comp == "c2" || comp == "alpha" ||
               comp == "rsnew" || comp == "beta_cg") {
        if (iter < 1 || j != -1 || a != -1 || b != -1)
            throw std::runtime_error("kbref: iteration index out of range");
        if (v.imag() != 0.0)
            throw std::runtime_error("kbref: iteration scalar must be real");
    } else {
        throw std::runtime_error("kbref: unknown iteration component " + comp);
    }
    write_row(fp, comp.c_str(), iter, j, a, b, v.real(), v.imag());
}

// pseudo: the independent storage of a herm_pseudo object (mat, gtr, tv,
// les; gtr is the stored ret_ triangle), the density matrix
// density(i,-1) (i = -1 for the Matsubara axis), and the start-up source
// blocks les_start_source(j,n), 0 <= n < j <= k, of the lesser Dyson/VIE2
// start-up procedure.
inline void write_pseudo_row(FILE *fp, const std::string &comp, int i, int j,
                             int a, int b, int k, std::complex<double> v) {
    bool ok;
    if (comp == "density")
        ok = i >= -1 && j == -1;
    else if (comp == "les_start_source")
        ok = 0 <= j && j < i && i <= k;
    else
        throw std::runtime_error("kbref: unknown pseudo component " + comp);
    if (!ok || a < 0 || b < 0)
        throw std::runtime_error("kbref: pseudo index out of range");
    write_row(fp, comp.c_str(), i, j, a, b, v.real(), v.imag());
}

// Write the independent storage domain of a herm_pseudo object in the
// fixed order mat, gtr, tv, les. For herm_pseudo, get_ret returns the
// stored greater component.
template <class GF, class Matrix>
void write_herm_pseudo(FILE *fp, GF &G, Matrix &M) {
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
            G.get_gtr(i, j, M);
            for (int a = 0; a < s1; a++)
                for (int b = 0; b < s2; b++)
                    write_row(fp, "gtr", i, j, a, b, M(a, b).real(), M(a, b).imag());
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
