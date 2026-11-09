// kbref generator: P2 utilities and GF algebra.
//
// Usage: p2_utilities <output-dir>
// Output (kbref-v1), all files start with "p2_utilities_":
//   extrap_gf_k{1..5}.dat   herm_matrix  extrapolate_timestep(NT-1, G, k)
//   extrapolate.dat         function     extrapolate_timestep(NT-1, f, k), k = 0..5
//   interpolate.dat         function     interpolation(NT, NT-0.37, f, k), k = 1..5
//   distance.dat            scalar       distance_norm2{_ret,_tv,_les,}(tstp, A, B)
//   t0_{fermi,bose}.dat     herm_matrix  set_t0_from_mat
//   tk_{fermi,bose}.dat     herm_matrix  set_tk_from_mat(G, KT)
//   hermitian.dat           herm_matrix  force_matsubara_hermitian
//   submatrix.dat           herm_matrix  set_submatrix
//   lmul_hermconj.dat       herm_matrix  left_multiply_hermconj(tstp, f, W)
//   rmul_hermconj.dat       herm_matrix  right_multiply_hermconj(tstp, f, W)
//   labels.dat              labels       order of the Integrator based cases
//
// Inputs: see p2_common.hpp.
#include <cstdio>
#include <string>
#include <vector>

// CNTR_NO_EXTERN_TEMPLATES is defined by the CMake target.
#include "p2_common.hpp"

using namespace cntr;
using namespace p2;

static const std::string GEN = "p2_utilities";
static const int KT = 3;
static const double TINTER = NT - 0.37;
static const double WEIGHT = 0.7;

static std::string fname(const std::string &s) { return GEN + "_" + s; }

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <output-dir>\n", argv[0]);
        return 1;
    }
    const std::string outdir(argv[1]);
    Labels labels;

    herm_matrix<double> A(NT, NTAU, 2, -1), B(NT, NTAU, 2, -1);
    build_green(A, 0, 2, -1, BETA, H);
    build_green(B, 1, 2, -1, BETA, H);
    function<double> f(NT, 2);
    build_function(f, 2, H);

    // Extrapolation of a herm_matrix, case_id = k - 1
    for (int k = 1; k <= 5; k++) {
        herm_matrix<double> G(A);
        extrapolate_timestep(NT - 1, G, k);
        write_gf(outdir, GEN, fname("extrap_gf_k" + std::to_string(k)), k - 1,
                 G, BETA, H);
        labels.add(k - 1, k);
    }

    // Extrapolation of a function, case_id = 10 + k
    {
        FILE *fp = kbref::open_output(outdir, fname("extrapolate") + ".dat");
        kbref::Metadata meta = base_meta(GEN, fname("extrapolate"));
        meta.push_back(std::make_pair("nt", kbref::fmt_int(NT)));
        meta.push_back(std::make_pair("size1", "2"));
        meta.push_back(std::make_pair("size2", "2"));
        meta.push_back(std::make_pair("n", kbref::fmt_int(NT - 1)));
        kbref::write_metadata(fp, "function", meta);
        for (int k = 0; k <= 5; k++) {
            function<double> g(f);
            extrapolate_timestep(NT - 1, g, k);
            cdmatrix M;
            for (int j = -1; j <= NT; j++) {
                g.get_value(j, M);
                for (int a = 0; a < 2; a++)
                    for (int b = 0; b < 2; b++)
                        kbref::write_function_row(fp, "extrapolate", 10 + k, j, a, b, M(a, b));
            }
            labels.add(10 + k, k);
        }
        std::fclose(fp);
    }

    // Interpolation of a function, case_id = 20 + k
    {
        FILE *fp = kbref::open_output(outdir, fname("interpolate") + ".dat");
        kbref::Metadata meta = base_meta(GEN, fname("interpolate"));
        meta.push_back(std::make_pair("nt", kbref::fmt_int(NT)));
        meta.push_back(std::make_pair("size1", "2"));
        meta.push_back(std::make_pair("size2", "2"));
        meta.push_back(std::make_pair("tstp", kbref::fmt_int(NT)));
        meta.push_back(std::make_pair("tinter", kbref::fmt_double(TINTER)));
        kbref::write_metadata(fp, "function", meta);
        for (int k = 1; k <= 5; k++) {
            cdmatrix M = interpolation(NT, TINTER, f, k);
            for (int a = 0; a < 2; a++)
                for (int b = 0; b < 2; b++)
                    kbref::write_function_row(fp, "interpolate", 20 + k, -1, a, b, M(a, b));
            labels.add(20 + k, k);
        }
        std::fclose(fp);
    }

    // Distances, case_id = 30 + tstp + 1 (no Integrator, no labels)
    {
        FILE *fp = kbref::open_output(outdir, fname("distance") + ".dat");
        kbref::Metadata meta = base_meta(GEN, fname("distance"));
        meta.push_back(std::make_pair("nt", kbref::fmt_int(NT)));
        meta.push_back(std::make_pair("case_id_offset", "31"));
        kbref::write_metadata(fp, "scalar", meta);
        for (int tstp = -1; tstp <= NT; tstp++) {
            int id = 31 + tstp;
            kbref::write_scalar_row(fp, "distance_ret", id, distance_norm2_ret(tstp, A, B));
            kbref::write_scalar_row(fp, "distance_tv", id, distance_norm2_tv(tstp, A, B));
            kbref::write_scalar_row(fp, "distance_les", id, distance_norm2_les(tstp, A, B));
            kbref::write_scalar_row(fp, "distance_total", id, distance_norm2(tstp, A, B));
        }
        std::fclose(fp);
    }

    // Boundary values from the Matsubara component, case_id 60..63
    for (int s = 0; s < 2; s++) {
        int sig = (s == 0) ? -1 : 1;
        std::string tag = (s == 0) ? "fermi" : "bose";
        herm_matrix<double> G(NT, NTAU, 2, sig);
        build_green(G, 0, 2, sig, BETA, H);
        herm_matrix<double> G0(NT, NTAU, 2, sig), Gk(NT, NTAU, 2, sig);
        G0.set_timestep(-1, G);
        Gk.set_timestep(-1, G);
        set_t0_from_mat(G0);
        set_tk_from_mat(Gk, KT);
        write_gf(outdir, GEN, fname("t0_" + tag), 60 + s, G0, BETA, H);
        kbref::Metadata extra;
        extra.push_back(std::make_pair("kt", kbref::fmt_int(KT)));
        write_gf(outdir, GEN, fname("tk_" + tag), 62 + s, Gk, BETA, H, extra);
    }

    // force_matsubara_hermitian, case_id 64: mat = A^M + i B^M
    {
        herm_matrix<double> G(A);
        cdmatrix Ma, Mb;
        for (int q = 0; q <= NTAU; q++) {
            A.get_mat(q, Ma);
            B.get_mat(q, Mb);
            cdmatrix M = Ma + cplx(0.0, 1.0) * Mb;
            G.set_mat(q, M);
        }
        force_matsubara_hermitian(G);
        write_gf(outdir, GEN, fname("hermitian"), 64, G, BETA, H);
    }

    // set_submatrix, case_id 65: G(i1,i2) <- A(j1,j2)
    {
        herm_matrix<double> G(NT, NTAU, 2, -1);
        std::vector<int> i1 = {0, 0, 1, 1}, i2 = {0, 1, 0, 1};
        std::vector<int> j1 = {1, 1, 0, 0}, j2 = {1, 0, 1, 0};
        G.set_submatrix(i1, i2, A, j1, j2);
        write_gf(outdir, GEN, fname("submatrix"), 65, G, BETA, H);
    }

    // Hermitian conjugate multiplication, case_id 66, 67
    {
        herm_matrix<double> L(A), R(A);
        for (int tstp = -1; tstp <= NT; tstp++) {
            L.left_multiply_hermconj(tstp, f, WEIGHT);
            R.right_multiply_hermconj(tstp, f, WEIGHT);
        }
        kbref::Metadata extra;
        extra.push_back(std::make_pair("weight", kbref::fmt_double(WEIGHT)));
        write_gf(outdir, GEN, fname("lmul_hermconj"), 66, L, BETA, H, extra);
        write_gf(outdir, GEN, fname("rmul_hermconj"), 67, R, BETA, H, extra);
    }

    labels.write(outdir, GEN, fname("labels"));
    return 0;
}
