// kbref generator: P2 incremental, partial lesser, equal-time, and density
// matrix convolutions.
//
// Usage: p2_convolution_utilities <output-dir>
//
// Variants: sig = fermi/bose, k = 1/5, with/without the contour function
// f. For variant v = (s * 2 + ki) * 2 + fm, base = 100 * v. Output (kbref-v1),
// all files start with "p2_convolution_utilities_<sig>_k<k>_<f|nof>_":
//
//   incr.dat           herm_matrix  C0 + alpha (A*f*B) via incr_convolution,
//                                   tstp = -1..NT, C0 = A          (base + 0)
//   incr_ret_less.dat  herm_matrix  alpha * convolution_timestep_les_retles,
//                                   only les(j,n), j <= n          (base + 1)
//   incr_less_adv.dat  herm_matrix  alpha * convolution_timestep_les_lesadv (base + 2)
//   incr_lmix_rmix.dat herm_matrix  alpha * convolution_timestep_les_tvvt   (base + 3)
//   incr_les.dat       herm_matrix  incr_convolution_les, C0 = 0          (base + 4)
//   startup.dat        element      start-up source blocks for n < k:
//                                   startup_less_adv  case base + 10 + n,
//                                   startup_lmix_rmix case base + 30 + n,
//                                   j = 0..max(n,k)
// and, for the variants without f only, "..._<sig>_k<k>_eq.dat":
//   eq.dat             element      les_timediag (base + 60),
//                                   density_matrix without f (base + 61),
//                                   density_matrix with f (base + 62),
//                                   j = tstp = -1..NT
// The labels file "p2_convolution_utilities_labels.dat" stores the order
// of all cases and alpha of the incremental cases.
//
// The hermitian NESSi convention Acc = A, Bcc = B is used. Inputs: see
// p2_common.hpp.
#include <cstdio>
#include <string>
#include <vector>

// CNTR_NO_EXTERN_TEMPLATES is defined by the CMake target.
#include "p2_common.hpp"

using namespace cntr;
using namespace p2;

typedef herm_matrix<double> GF;

static const std::string GEN = "p2_convolution_utilities";
static const cplx ALPHA(0.3, -0.7);

static std::string fname(const std::string &s) { return GEN + "_" + s; }

static kbref::Metadata element_meta(const std::string &name, int nt) {
    kbref::Metadata meta = base_meta(GEN, name);
    meta.push_back(std::make_pair("nt", kbref::fmt_int(nt)));
    meta.push_back(std::make_pair("size1", "2"));
    meta.push_back(std::make_pair("size2", "2"));
    meta.push_back(std::make_pair("beta", kbref::fmt_double(BETA)));
    meta.push_back(std::make_pair("h", kbref::fmt_double(H)));
    return meta;
}

static void write_block(FILE *fp, const char *comp, int case_id, int j, const cplx *x) {
    for (int a = 0; a < 2; a++)
        for (int b = 0; b < 2; b++)
            kbref::write_element_row(fp, comp, case_id, j, a, b, x[a * 2 + b]);
}

// One partial lesser convolution at time step n, accumulated into a zeroed
// buffer of max(n,k) + 1 blocks and multiplied by ALPHA.
static std::vector<cplx> partial(int op, int n, GF &C, GF &A, function<double> *f,
                                 GF &B, integration::Integrator<double> &I) {
    int k = I.get_k(), n1 = (n < k ? k : n);
    std::vector<cplx> buf((n1 + 1) * 4, cplx(0.0, 0.0));
    cplx *p = buf.data();
    if (f == nullptr) {
        if (op == 1)
            convolution_timestep_les_retles<double, GF, LARGESIZE>(n, p, C, A, A, B, B, I, BETA, H);
        else if (op == 2)
            convolution_timestep_les_lesadv<double, GF, LARGESIZE>(n, p, C, A, A, B, B, I, BETA, H);
        else
            convolution_timestep_les_tvvt<double, GF, LARGESIZE>(n, p, C, A, A, B, B, I, BETA, H);
    } else {
        if (op == 1)
            convolution_timestep_les_retles<double, GF, LARGESIZE>(n, p, C, A, A, f->ptr(0),
                                                                   B, B, I, BETA, H);
        else if (op == 2)
            convolution_timestep_les_lesadv<double, GF, LARGESIZE>(n, p, C, A, A, f->ptr(0),
                                                                   B, B, I, BETA, H);
        else
            convolution_timestep_les_tvvt<double, GF, LARGESIZE>(n, p, C, A, A, f->ptr(-1),
                                                                 B, B, I, BETA, H);
    }
    for (size_t l = 0; l < buf.size(); l++) buf[l] *= ALPHA;
    return buf;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <output-dir>\n", argv[0]);
        return 1;
    }
    const std::string outdir(argv[1]);
    Labels labels;
    const int korders[2] = {1, 5};

    for (int s = 0; s < 2; s++) {
        int sig = (s == 0) ? -1 : 1;
        std::string stag = (s == 0) ? "fermi" : "bose";
        GF A(NT, NTAU, 2, sig), B(NT, NTAU, 2, sig);
        build_green(A, 0, 2, sig, BETA, H);
        build_green(B, 1, 2, sig, BETA, H);
        function<double> f(NT, 2);
        build_function(f, 2, H);

        for (int ki = 0; ki < 2; ki++) {
            int k = korders[ki];
            integration::Integrator<double> &I = integration::I<double>(k);
            for (int fm = 0; fm < 2; fm++) {
                int base = 100 * ((s * 2 + ki) * 2 + fm);
                function<double> *fp = (fm == 1) ? &f : nullptr;
                std::string tag = stag + "_k" + std::to_string(k) + (fm ? "_f" : "_nof");
                cplx *f0 = fm ? f.ptr(-1) : nullptr;
                cplx *ft = fm ? f.ptr(0) : nullptr;

                // Full incremental convolution
                {
                    GF C(A);
                    for (int tstp = -1; tstp <= NT; tstp++)
                        incr_convolution<double, GF, LARGESIZE>(tstp, ALPHA, C, A, A, f0, ft,
                                                                B, B, I, BETA, H);
                    write_gf(outdir, GEN, fname(tag + "_incr"), base, C, BETA, H);
                    labels.add(base, k, ALPHA);
                }

                // Partial lesser convolutions, j <= n only
                const char *pname[4] = {"", "incr_ret_less", "incr_less_adv", "incr_lmix_rmix"};
                for (int op = 1; op <= 3; op++) {
                    GF C(NT, NTAU, 2, sig);
                    for (int n = 0; n <= NT; n++) {
                        std::vector<cplx> buf = partial(op, n, C, A, fp, B, I);
                        for (int j = 0; j <= n; j++)
                            element_set<double, LARGESIZE>(2, C.lesptr(j, n), buf.data() + j * 4);
                    }
                    write_gf(outdir, GEN, fname(tag + "_" + pname[op]), base + op, C, BETA, H);
                    labels.add(base + op, k, ALPHA);
                }

                // Lesser incremental convolution
                {
                    GF C(NT, NTAU, 2, sig);
                    for (int tstp = 0; tstp <= NT; tstp++) {
                        std::vector<bool> mask(tstp + 1, true);
                        incr_convolution_les<double, GF, LARGESIZE>(tstp, mask, ALPHA, C, A, A,
                                                                    f0, ft, B, B, I, BETA, H);
                    }
                    write_gf(outdir, GEN, fname(tag + "_incr_les"), base + 4, C, BETA, H);
                    labels.add(base + 4, k, ALPHA);
                }

                // Start-up source blocks for n < k
                {
                    std::string name = fname(tag + "_startup");
                    FILE *out = kbref::open_output(outdir, name + ".dat");
                    kbref::Metadata meta = element_meta(name, NT);
                    meta.push_back(std::make_pair("less_adv_case_id_offset",
                                                  kbref::fmt_int(base + 10)));
                    meta.push_back(std::make_pair("lmix_rmix_case_id_offset",
                                                  kbref::fmt_int(base + 30)));
                    kbref::write_metadata(out, "element", meta);
                    GF C(NT, NTAU, 2, sig);
                    for (int n = 0; n < k; n++) {
                        std::vector<cplx> b2 = partial(2, n, C, A, fp, B, I);
                        for (int j = 0; j <= k; j++)
                            write_block(out, "startup_less_adv", base + 10 + n, j, b2.data() + j * 4);
                        labels.add(base + 10 + n, k, ALPHA);
                    }
                    for (int n = 0; n < k; n++) {
                        std::vector<cplx> b3 = partial(3, n, C, A, fp, B, I);
                        for (int j = 0; j <= k; j++)
                            write_block(out, "startup_lmix_rmix", base + 30 + n, j, b3.data() + j * 4);
                        labels.add(base + 30 + n, k, ALPHA);
                    }
                    std::fclose(out);
                }

                // Equal-time convolutions (once per sig and k)
                if (fm == 0) {
                    std::string name = fname(stag + "_k" + std::to_string(k) + "_eq");
                    FILE *out = kbref::open_output(outdir, name + ".dat");
                    kbref::write_metadata(out, "element", element_meta(name, NT));
                    cdmatrix M(2, 2);
                    for (int tstp = -1; tstp <= NT; tstp++) {
                        convolution_les_timediag<double, GF>(tstp, M, A, B, I, BETA, H);
                        cplx x[4] = {M(0, 0), M(0, 1), M(1, 0), M(1, 1)};
                        write_block(out, "les_timediag", base + 60, tstp, x);
                    }
                    for (int tstp = -1; tstp <= NT; tstp++) {
                        convolution_density_matrix<double, GF>(tstp, M, A, A, B, B, BETA, H, k);
                        cplx x[4] = {M(0, 0), M(0, 1), M(1, 0), M(1, 1)};
                        write_block(out, "density_matrix", base + 61, tstp, x);
                    }
                    for (int tstp = -1; tstp <= NT; tstp++) {
                        convolution_density_matrix<double, GF>(tstp, M, A, A, f, B, B, BETA, H, k);
                        cplx x[4] = {M(0, 0), M(0, 1), M(1, 0), M(1, 1)};
                        write_block(out, "density_matrix", base + 62, tstp, x);
                    }
                    std::fclose(out);
                    labels.add(base + 60, k);
                    labels.add(base + 61, k);
                    labels.add(base + 62, k);
                }
            }
        }
    }
    labels.write(outdir, GEN, fname("labels"));
    return 0;
}
