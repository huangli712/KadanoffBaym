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
// P7: phase7_fourier, phase7_ness, phase7_example (see
// .omo/plans/07-phase7-fourier-ness.md, P7-W10-T8..T10).
// P8: moving, series (see .omo/plans/08-phase8-moving-parallel.md,
// P8-W0-T1).
// P5: observable, selected_green (see .omo/plans/05-phase5-examples.md,
// P5-W1-T9).
inline bool known_data_kind(const std::string &k) {
    return k == "weights" || k == "herm_matrix" || k == "scalar" ||
           k == "function" || k == "element" || k == "labels" ||
           k == "iteration" || k == "pseudo" || k == "phase7_fourier" ||
           k == "phase7_ness" || k == "phase7_example" || k == "moving" ||
           k == "series" || k == "observable" || k == "selected_green";
}

// P5 metadata of the data kinds observable and selected_green
static const char *const P5_META[] = {
    "generator", "case_set", "model", "approximation", "profile", "nt", "ntau",
    "h", "beta", "solve_order", "mats_max_iter", "bootstrap_max_iter",
    "corrector_steps", "size", "params"};
inline bool p5_model_ok(const std::string &m, const std::string &a);

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
    // P7 files
    if (data_kind == "phase7_fourier" || data_kind == "phase7_ness") {
        if (!has_meta(meta, "size_max"))
            throw std::runtime_error("kbref: missing size_max");
        if (data_kind == "phase7_ness" && !has_meta(meta, "nft_max"))
            throw std::runtime_error("kbref: missing nft_max");
    }
    if (data_kind == "phase7_example") {
        if (!has_meta(meta, "generator") || !has_meta(meta, "case"))
            throw std::runtime_error("kbref: missing generator/case");
        std::string c = get_meta(meta, "case");
        if (c != "siam" && c != "bethe" && c != "bethe_prop")
            throw std::runtime_error("kbref: bad case " + c);
    }
    // P8 files
    if (data_kind == "moving") {
        const char *keys[] = {"generator", "case_set", "case_id", "source",
                              "tc", "size", "sign", "tstp"};
        for (const char *key : keys)
            if (!has_meta(meta, key))
                throw std::runtime_error(std::string("kbref: missing ") + key);
    }
    if (data_kind == "series") {
        const char *keys[] = {"generator", "case_set", "components", "nsteps",
                              "nk", "size", "source"};
        for (const char *key : keys)
            if (!has_meta(meta, key))
                throw std::runtime_error(std::string("kbref: missing ") + key);
    }
    // P5 files
    if (data_kind == "observable" || data_kind == "selected_green") {
        for (const char *key : P5_META)
            if (!has_meta(meta, key))
                throw std::runtime_error(std::string("kbref: missing ") + key);
        if (!p5_model_ok(get_meta(meta, "model"), get_meta(meta, "approximation")))
            throw std::runtime_error("kbref: bad model/approximation");
        if (data_kind == "selected_green" && get_meta(meta, "model") != "hubbard")
            throw std::runtime_error("kbref: selected_green is only defined for hubbard");
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

// ---------------------------------------------------------------------
// P7 schema. All unused index fields must be 0.
//
// phase7_fourier (metadata size_max):
//   fft_input, fft_forward, fft_backward : i = N >= 1, 0 <= j < N.
//   dftcorr : i = theta index >= 0, j = order (1 linear, 3 cubic),
//             a = 0 theta, 1 corfac (both real), 2.. alpha_{a-2}
//             (a <= 3 for linear, a <= 5 for cubic).
//   adaptive : i = function id 0..2, j = omega index >= 0, a = 0 omega
//              (real), 1 dft_cplx, 2 adft_func.
//   matsubara_dft_result : i = case >= 0, j = m >= 0, a, b orbital.
//   matsubara_ft_result : i = 10 case + order (order 0, 1, 3), j = m
//                         (any integer), a, b orbital.
//   vie2_pcf20, vie2_pcf10 : i = case >= 0, j = tau index >= 0, a, b
//                            orbital.
// phase7_ness (metadata nft_max, size_max):
//   ret_time, les_time, ret_freq, les_freq : i = case >= 0,
//             0 <= j < nft_max, a, b orbital.
//   bath_ret, bath_les : i = bath type 0..3 (NONE, CONST, GAUSS, OHMIC),
//             j = omega index >= 0.
//   density, convolution_density : i = case >= 0, j = 0, a, b orbital.
//   bubble1, bubble2, resample, force_equilibrium, cntr_conversion :
//             i = case >= 0, time domain data, j = n (0 <= n < nft_max)
//             for the retarded and j = nft_max + n for the lesser
//             component, a, b orbital.
// phase7_example (metadata case = siam, bethe, bethe_prop, one case per
// file, i = case index 0, 1, 2):
//   gf_ret_time, gf_les_time, gf_ret_freq, gf_les_freq : j = grid index
//             >= 0 (bethe_prop: time only, j = t of G^R(t_nt, t_nt - t)
//             and G^<(t_nt - t, t_nt)).
//   density : j = 0 (bethe_prop: j = KB time step >= 0, 0 = Matsubara).
//   current : siam only, j = 0 ([SL*G]) or 1 ([G*SL]).
//   eint    : siam and bethe, j = 0.
//   iteration_error : bethe (a = 0, j = iteration >= 1) and bethe_prop
//             (a = stage 0..2, b = KB time step >= 0, j = iteration
//             >= 0); real value.
// ---------------------------------------------------------------------

struct Phase7Schema {
    std::string kind;
    int size_max;
    int nft_max;
    int case_id;
};

inline Phase7Schema phase7_schema(const std::string &kind, const Metadata &meta) {
    Phase7Schema s;
    s.kind = kind;
    s.size_max = has_meta(meta, "size_max") ? std::stoi(get_meta(meta, "size_max")) : 1;
    s.nft_max = has_meta(meta, "nft_max") ? std::stoi(get_meta(meta, "nft_max")) : 0;
    s.case_id = -1;
    if (kind == "phase7_example") {
        std::string c = get_meta(meta, "case");
        s.case_id = c == "siam" ? 0 : (c == "bethe" ? 1 : 2);
    }
    return s;
}

inline bool phase7_row_ok(const Phase7Schema &s, const std::string &c, int i,
                          int j, int a, int b, std::complex<double> v) {
    bool orb = 0 <= a && a < s.size_max && 0 <= b && b < s.size_max;
    if (s.kind == "phase7_fourier") {
        if (c == "fft_input" || c == "fft_forward" || c == "fft_backward")
            return i >= 1 && 0 <= j && j < i && a == 0 && b == 0;
        if (c == "dftcorr") {
            if (i < 0 || (j != 1 && j != 3) || b != 0) return false;
            if (a < 0 || a > (j == 1 ? 3 : 5)) return false;
            return a >= 2 || v.imag() == 0.0;
        }
        if (c == "adaptive")
            return 0 <= i && i <= 2 && j >= 0 && 0 <= a && a <= 2 && b == 0 &&
                   (a != 0 || v.imag() == 0.0);
        if (c == "matsubara_dft_result") return i >= 0 && j >= 0 && orb;
        if (c == "matsubara_ft_result") {
            int o = i % 10;
            return i >= 0 && (o == 0 || o == 1 || o == 3) && orb;
        }
        if (c == "vie2_pcf20" || c == "vie2_pcf10") return i >= 0 && j >= 0 && orb;
        throw std::runtime_error("kbref: unknown phase7_fourier component " + c);
    }
    if (s.kind == "phase7_ness") {
        if (c == "ret_time" || c == "les_time" || c == "ret_freq" || c == "les_freq")
            return i >= 0 && 0 <= j && j < s.nft_max && orb;
        if (c == "bath_ret" || c == "bath_les")
            return 0 <= i && i <= 3 && j >= 0 && a == 0 && b == 0;
        if (c == "density" || c == "convolution_density")
            return i >= 0 && j == 0 && orb;
        if (c == "bubble1" || c == "bubble2" || c == "resample" ||
            c == "force_equilibrium" || c == "cntr_conversion")
            return i >= 0 && 0 <= j && j < 2 * s.nft_max && orb;
        throw std::runtime_error("kbref: unknown phase7_ness component " + c);
    }
    if (s.kind == "phase7_example") {
        if (i != s.case_id) return false;
        bool prop = s.case_id == 2;
        if (c == "gf_ret_time" || c == "gf_les_time")
            return j >= 0 && a == 0 && b == 0;
        if (c == "gf_ret_freq" || c == "gf_les_freq")
            return !prop && j >= 0 && a == 0 && b == 0;
        if (c == "density")
            return (prop ? j >= 0 : j == 0) && a == 0 && b == 0 && v.imag() == 0.0;
        if (c == "current")
            return s.case_id == 0 && (j == 0 || j == 1) && a == 0 && b == 0 &&
                   v.imag() == 0.0;
        if (c == "eint")
            return s.case_id <= 1 && j == 0 && a == 0 && b == 0 && v.imag() == 0.0;
        if (c == "iteration_error") {
            if (v.imag() != 0.0) return false;
            if (s.case_id == 1) return j >= 1 && a == 0 && b == 0;
            if (s.case_id == 2) return j >= 0 && 0 <= a && a <= 2 && b >= 0;
            return false;
        }
        throw std::runtime_error("kbref: unknown phase7_example component " + c);
    }
    throw std::runtime_error("kbref: not a phase7 data kind " + s.kind);
}

inline void write_phase7_row(FILE *fp, const Phase7Schema &s, const std::string &c,
                             int i, int j, int a, int b, std::complex<double> v) {
    if (!phase7_row_ok(s, c, i, j, a, b, v))
        throw std::runtime_error("kbref: phase7 index out of range for " + c);
    write_row(fp, c.c_str(), i, j, a, b, v.real(), v.imag());
}

// ---------------------------------------------------------------------
// P8 schema.
//
// moving (one case per file; metadata generator, case_set, case_id,
// source, tc, size, sign, tstp): rows `component delti deltj a b re im`
// with component in (ret, less), 0 <= delti, deltj <= tc, 0 <= a, b <
// size. The whole square window must be present.
//
// series (metadata generator, case_set, components, nsteps, nk, size,
// source): rows `component step k a b re im`, 0 <= step < nsteps. A
// global scalar uses k = a = b = -1, a k-resolved scalar a = b = -1 and
// 0 <= k < nk, a matrix block 0 <= k < nk and 0 <= a, b < size. For the
// two-time quantities of the moving solvers and the truncated examples k
// is the relative time deltj (nk = tc + 1) of the leading row at the
// NESSi time step `step`; for GW k is the k point and the two-time blocks
// are G^R(t_step, t_0) and G^<(t_0, t_step) (the first column of the
// stored triangles). `step` is always the NESSi (0-based) time step, the
// Julia reader adds 1. The
// components of every case_set are fixed below; the metadata
// `components` lists the components in the file (comma separated).
// ---------------------------------------------------------------------

// Index class of a series component: 0 global scalar, 1 k-resolved
// scalar, 2 matrix block. Returns -1 for an unknown component.
inline int series_class(const std::string &case_set, const std::string &c) {
    if (case_set == "moving_solver") {
        if (c == "kinetic") return 0;
        if (c == "dyson_ret" || c == "dyson_less" || c == "dyson0_ret" ||
            c == "dyson0_less" || c == "bubble1_ret" || c == "bubble1_less" ||
            c == "bubble2_ret" || c == "bubble2_less" || c == "vie2_ret" ||
            c == "vie2_less")
            return 2;
        return -1;
    }
    if (case_set == "trunc_bethe") {
        if (c == "density" || c == "error" || c == "ekin") return 0;
        if (c == "g_ret" || c == "g_less" || c == "sigma_ret" || c == "sigma_less")
            return 2;
        return -1;
    }
    if (case_set == "ness2_bethe_trunc") {
        if (c == "density") return 0;
        if (c == "g_ret" || c == "g_less") return 2;
        return -1;
    }
    if (case_set == "gw") {
        if (c == "afield" || c == "density" || c == "current_nessi" ||
            c == "ekin" || c == "eint")
            return 0;
        if (c == "rho" || c == "sigma_hf" || c == "g_ret" || c == "g_less" ||
            c == "w_ret" || c == "w_less" || c == "sigma_gw_ret" ||
            c == "sigma_gw_less" || c == "p_ret" || c == "p_less")
            return 2;
        return -1;
    }
    throw std::runtime_error("kbref: unknown series case_set " + case_set);
}

struct SeriesSchema {
    std::string case_set;
    int nsteps;
    int nk;
    int size;
};

inline SeriesSchema series_schema(const Metadata &meta) {
    SeriesSchema s;
    s.case_set = get_meta(meta, "case_set");
    s.nsteps = std::stoi(get_meta(meta, "nsteps"));
    s.nk = std::stoi(get_meta(meta, "nk"));
    s.size = std::stoi(get_meta(meta, "size"));
    // Every listed component must be known
    std::string list = get_meta(meta, "components") + ",";
    size_t p = 0, q;
    while ((q = list.find(',', p)) != std::string::npos) {
        if (series_class(s.case_set, list.substr(p, q - p)) < 0)
            throw std::runtime_error("kbref: unknown series component " +
                                     list.substr(p, q - p));
        p = q + 1;
    }
    return s;
}

inline void write_series_row(FILE *fp, const SeriesSchema &s, const std::string &c,
                             int step, int k, int a, int b, std::complex<double> v) {
    int cls = series_class(s.case_set, c);
    bool ok = 0 <= step && step < s.nsteps;
    if (cls == 0) ok = ok && k == -1 && a == -1 && b == -1;
    else if (cls == 1) ok = ok && 0 <= k && k < s.nk && a == -1 && b == -1;
    else if (cls == 2)
        ok = ok && 0 <= k && k < s.nk && 0 <= a && a < s.size && 0 <= b && b < s.size;
    else ok = false;
    if (!ok) throw std::runtime_error("kbref: series index out of range for " + c);
    write_row(fp, c.c_str(), step, k, a, b, v.real(), v.imag());
}

// ---------------------------------------------------------------------
// P5 schema.
//
// observable: rows `component tstp j a b re im`, tstp = NESSi time step
// -1..nt. A scalar observable uses j = a = b = -1 and im = 0; the density
// matrix `rho` uses j = -1 and 0 <= a, b < size (complex). The allowed
// components of every model:
//   hubbard            n1 Ekin Epot Etot
//   fkm                den0 den1 total
//   holstein_bethe     rho Xph Pph XX Ekin Enx_MF Enx_corr Eph_cl Eph_corr Etot
//   holstein_impurity  rho (a = b = site) and the same scalars
//   holstein_nambu     rho XX Ekin Enx_MF Enx_corr Eph Etot
// selected_green (hubbard only): the four keys mat(0,-1,0,0),
// ret(3,1,0,0), les(1,3,0,0), tv(3,2,0,0) (NESSi indices).
// ---------------------------------------------------------------------

inline bool p5_model_ok(const std::string &m, const std::string &a) {
    if (m == "hubbard") return a == "2B" || a == "GW" || a == "TPP";
    if (m == "fkm") return a == "none";
    if (m == "holstein_bethe" || m == "holstein_nambu" || m == "holstein_impurity")
        return a == "Migdal" || a == "uMig";
    return false;
}

// 0: scalar, 2: density matrix, -1: not allowed
inline int observable_class(const std::string &m, const std::string &c) {
    if (m == "hubbard")
        return (c == "n1" || c == "Ekin" || c == "Epot" || c == "Etot") ? 0 : -1;
    if (m == "fkm") return (c == "den0" || c == "den1" || c == "total") ? 0 : -1;
    if (m == "holstein_bethe" || m == "holstein_impurity") {
        if (c == "rho") return 2;
        return (c == "Xph" || c == "Pph" || c == "XX" || c == "Ekin" || c == "Enx_MF" ||
                c == "Enx_corr" || c == "Eph_cl" || c == "Eph_corr" || c == "Etot")
                   ? 0
                   : -1;
    }
    if (m == "holstein_nambu") {
        if (c == "rho") return 2;
        return (c == "XX" || c == "Ekin" || c == "Enx_MF" || c == "Enx_corr" ||
                c == "Eph" || c == "Etot")
                   ? 0
                   : -1;
    }
    return -1;
}

inline void write_observable_row(FILE *fp, const Metadata &meta, const std::string &c,
                                 int tstp, int a, int b, std::complex<double> v) {
    int cls = observable_class(get_meta(meta, "model"), c);
    int nt = std::stoi(get_meta(meta, "nt"));
    int size = std::stoi(get_meta(meta, "size"));
    bool ok = -1 <= tstp && tstp <= nt;
    if (cls == 0) ok = ok && a == -1 && b == -1;
    else if (cls == 2) ok = ok && 0 <= a && a < size && 0 <= b && b < size;
    else ok = false;
    if (!ok) throw std::runtime_error("kbref: bad observable " + c);
    // scalar observables are real
    write_row(fp, c.c_str(), tstp, -1, a, b, v.real(), cls == 0 ? 0.0 : v.imag());
}

inline void write_selected_green_row(FILE *fp, const std::string &c, int i, int j, int a,
                                     int b, std::complex<double> v) {
    bool ok = a == 0 && b == 0 &&
              ((c == "mat" && i == 0 && j == -1) || (c == "ret" && i == 3 && j == 1) ||
               (c == "les" && i == 1 && j == 3) || (c == "tv" && i == 3 && j == 2));
    if (!ok) throw std::runtime_error("kbref: bad selected_green key " + c);
    write_row(fp, c.c_str(), i, j, a, b, v.real(), v.imag());
}

inline void write_moving_row(FILE *fp, const Metadata &meta, const std::string &c,
                             int delti, int deltj, int a, int b,
                             std::complex<double> v) {
    int tc = std::stoi(get_meta(meta, "tc"));
    int size = std::stoi(get_meta(meta, "size"));
    if ((c != "ret" && c != "less") || delti < 0 || delti > tc || deltj < 0 ||
        deltj > tc || a < 0 || a >= size || b < 0 || b >= size)
        throw std::runtime_error("kbref: moving index out of range for " + c);
    write_row(fp, c.c_str(), delti, deltj, a, b, v.real(), v.imag());
}

} // namespace kbref

#endif // KBREF_REFERENCE_IO_HPP
