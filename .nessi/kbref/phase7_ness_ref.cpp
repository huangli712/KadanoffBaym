// kbref generator: NESSi steady-state (ness2) Green's functions.
//
// Usage: phase7_ness_ref <output-dir>
// Output: <output-dir>/phase7_ness.dat (kbref-v1, data_kind phase7_ness,
// see reference_io.hpp for the index usage). nft_max = 512, size_max = 2.
//
// Inputs (the same definitions are used in the Julia test
// test/t_phase7_reference.jl; nothing is read back). Unless stated
// otherwise Nft = 256, h = 0.05, beta = 5, and the sign is FERMION.
//
// ret_time, les_time, ret_freq, les_freq (case id):
//   0  green_equilibrium_ness, gauss(0, 1), mu = 0.3, FFT_TRAPEZ;
//   1  green_equilibrium_ness, bethedos(-2, 2), mu = 0.1, FFT_TRAPEZ;
//   2  green_equilibrium_ness, gauss(0, 1), mu = 0.3, FFT_ADAPTIVE
//      (limit 100, nn 20), time domain only;
//   3  green_equilibrium_ness (BOSON), bethedos(1, 5), mu = 0, FFT_TRAPEZ;
//   4  dyson, H = 0.2, Sigma = 0.25 G1, BATH_GAUSS (eta 0.1, beta 5, mu 0);
//   5  dyson, H = [0.3, 0.2i; -0.2i, -0.4], Sigma = [0.3 G0, 0.1 G0;
//      0.1 G0, 0.2 G1], BATH_CONST (eta 0.05, beta 5, mu 0);
//   6  dyson, H = 0.4, Sigma = 0, BATH_OHMIC (eta 0.1, beta 5, mu 0);
//   7  dyson, H = 0, Sigma = 0.25 G1, BATH_NONE.
//   The Dyson equations use mu = 0 and FFT_TRAPEZ.
// bath_ret, bath_les: sigma_bath for the types 0..3 at
//   omega_j = -3 + 0.25 j (j = 0..24), h = 0.05, eta = 0.2, beta = 5,
//   mu = 0.1.
// density (case id): density_matrix of the cases 0, 1, 4, 5.
// convolution_density: 0 = [G0 * G1], 1 = [G5 * G5].
// bubble1, bubble2: 0 = Bubble(C, G0, G1) (scalar version);
//   1 = C(a,b) = Bubble(G5(a,b), G0(0,0)) for all a, b.
// resample: 0 = downsample(G0, 4) (Nft 64), 1 = upsample(h, G1, 2)
//   (Nft 512).
// force_equilibrium: 0 = G0 with les = 0, force_equilibrium(FERMION,
//   beta, 0.3, h); 1 = BOSON, gauss(3, 0.5), mu = -1, the same procedure.
// cntr_conversion: 0 = cntr2ness of cntr::green_equilibrium (gauss(0, 1),
//   mu = 0.3, nt = 128, ntau = 50, limit 100, nn 20) at tstp = nt;
//   1 = cntr2ness(ness2cntr(G0)) with nt = 137 at tstp = nt.
// The time data of the last five components is written for
// 0 <= n < Nft of the result, the lesser one with the offset nft_max.
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>

// CNTR_NO_EXTERN_TEMPLATES and CNTR_USE_NESS are defined by the target.
#include "cntr/cntr.hpp"
#include "cntr/ness2.hpp"
#include "reference_io.hpp"

// The header ness2_dyson_decl.hpp declares sigma_bath with double&
// arguments, the library defines (and the Dyson equation uses) the version
// with complex arguments, ness2_dyson_impl.hpp:25.
namespace ness2 {
void sigma_bath(cdouble &ret, cdouble &les, double omega, double h,
                dyson_bath_type bath_type, double bath_eta, double bath_beta,
                double bath_mu);
}

using namespace ness2;
typedef std::complex<double> cplx;

static const int NFT_MAX = 512;
static const cplx CI(0.0, 1.0);

static void write_gf(FILE *fp, const kbref::Phase7Schema &s, int id, herm_matrix_ness &G,
                     bool freq) {
    cdmatrix M;
    int Nft = G.Nft_, d = G.size1_;
    const char *names[4] = {"ret_time", "les_time", "ret_freq", "les_freq"};
    for (int part = 0; part < (freq ? 4 : 2); part++) {
        fft_domain dom = part < 2 ? fft_domain::time : fft_domain::freq;
        for (int n = 0; n < Nft; n++) {
            if (part % 2 == 0)
                G.get_ret(n, M, dom);
            else
                G.get_les(n, M, dom);
            for (int a = 0; a < d; a++)
                for (int b = 0; b < d; b++)
                    kbref::write_phase7_row(fp, s, names[part], id, n, a, b, M(a, b));
        }
    }
}

// Time domain data of an operation result: ret at j = n, les at
// j = NFT_MAX + n.
static void write_time(FILE *fp, const kbref::Phase7Schema &s, const char *comp, int id,
                       herm_matrix_ness &G) {
    cdmatrix M;
    int Nft = G.Nft_, d = G.size1_;
    for (int n = 0; n < Nft; n++) {
        G.get_ret(n, M, fft_domain::time);
        for (int a = 0; a < d; a++)
            for (int b = 0; b < d; b++)
                kbref::write_phase7_row(fp, s, comp, id, n, a, b, M(a, b));
    }
    for (int n = 0; n < Nft; n++) {
        G.get_les(n, M, fft_domain::time);
        for (int a = 0; a < d; a++)
            for (int b = 0; b < d; b++)
                kbref::write_phase7_row(fp, s, comp, id, NFT_MAX + n, a, b, M(a, b));
    }
}

static void write_matrix(FILE *fp, const kbref::Phase7Schema &s, const char *comp, int id,
                         cdmatrix &M) {
    for (int a = 0; a < M.rows(); a++)
        for (int b = 0; b < M.cols(); b++)
            kbref::write_phase7_row(fp, s, comp, id, 0, a, b, M(a, b));
}

// Sigma = [c00 A, c01 A; c01 A, c11 B] in the time domain
static void compose2(herm_matrix_ness &S, herm_matrix_ness &A, herm_matrix_ness &B,
                     double c00, double c01, double c11) {
    cdmatrix MA, MB, M(2, 2);
    for (int n = 0; n < S.Nft_; n++) {
        A.get_ret(n, MA, fft_domain::time);
        B.get_ret(n, MB, fft_domain::time);
        M(0, 0) = c00 * MA(0, 0);
        M(0, 1) = c01 * MA(0, 0);
        M(1, 0) = c01 * MA(0, 0);
        M(1, 1) = c11 * MB(0, 0);
        S.set_ret(n, M, fft_domain::time);
        A.get_les(n, MA, fft_domain::time);
        B.get_les(n, MB, fft_domain::time);
        M(0, 0) = c00 * MA(0, 0);
        M(0, 1) = c01 * MA(0, 0);
        M(1, 0) = c01 * MA(0, 0);
        M(1, 1) = c11 * MB(0, 0);
        S.set_les(n, M, fft_domain::time);
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: phase7_ness_ref <output-dir>\n");
        return 1;
    }
    const std::string outdir = argv[1];
    FILE *fp = kbref::open_output(outdir, "phase7_ness.dat");
    kbref::Metadata meta;
    meta.push_back(std::make_pair("generator", std::string("phase7_ness_ref")));
    meta.push_back(std::make_pair("case_set", std::string("equilibrium_dyson_bath_utils")));
    meta.push_back(std::make_pair("nft_max", kbref::fmt_int(NFT_MAX)));
    meta.push_back(std::make_pair("size_max", std::string("2")));
    kbref::write_metadata(fp, "phase7_ness", meta);
    kbref::Phase7Schema s = kbref::phase7_schema("phase7_ness", meta);

    const int Nft = 256;
    const double h = 0.05, beta = 5.0;

    // Equilibrium Green's functions
    gauss g01(0.0, 1.0);
    bethedos b22(-2.0, 2.0), b15(1.0, 5.0);
    herm_matrix_ness G0(Nft, 1), G1(Nft, 1), G2(Nft, 1), G3(Nft, 1);
    green_equilibrium_ness(FERMION, G0, g01, beta, 0.3, h, FFT_TRAPEZ);
    green_equilibrium_ness(FERMION, G1, b22, beta, 0.1, h, FFT_TRAPEZ);
    green_equilibrium_ness(FERMION, G2, g01, beta, 0.3, h, FFT_ADAPTIVE, 100, 20);
    green_equilibrium_ness(BOSON, G3, b15, beta, 0.0, h, FFT_TRAPEZ);
    write_gf(fp, s, 0, G0, true);
    write_gf(fp, s, 1, G1, true);
    write_gf(fp, s, 2, G2, false);
    write_gf(fp, s, 3, G3, true);

    // Dyson equations
    herm_matrix_ness G4(Nft, 1), G5(Nft, 2), G6(Nft, 1), G7(Nft, 1);
    {
        herm_matrix_ness S(Nft, 1);
        S = G1;
        S.smul(0.25, fft_domain::time);
        cdmatrix H(1, 1);
        H(0, 0) = 0.2;
        dyson(G4, 0.0, H, S, h, FFT_TRAPEZ, BATH_GAUSS, 0.1, beta, 0.0);
    }
    {
        herm_matrix_ness S(Nft, 2);
        compose2(S, G0, G1, 0.3, 0.1, 0.2);
        cdmatrix H(2, 2);
        H(0, 0) = 0.3;
        H(0, 1) = 0.2 * CI;
        H(1, 0) = -0.2 * CI;
        H(1, 1) = -0.4;
        dyson(G5, 0.0, H, S, h, FFT_TRAPEZ, BATH_CONST, 0.05, beta, 0.0);
    }
    {
        herm_matrix_ness S(Nft, 1);
        cdmatrix H(1, 1);
        H(0, 0) = 0.4;
        dyson(G6, 0.0, H, S, h, FFT_TRAPEZ, BATH_OHMIC, 0.1, beta, 0.0);
    }
    {
        herm_matrix_ness S(Nft, 1);
        S = G1;
        S.smul(0.25, fft_domain::time);
        cdmatrix H(1, 1);
        H(0, 0) = 0.0;
        dyson(G7, 0.0, H, S, h, FFT_TRAPEZ, BATH_NONE, 0.0, 0.0, 0.0);
    }
    write_gf(fp, s, 4, G4, true);
    write_gf(fp, s, 5, G5, true);
    write_gf(fp, s, 6, G6, true);
    write_gf(fp, s, 7, G7, true);

    // Bath self-energies
    for (int type = 0; type < 4; type++)
        for (int j = 0; j <= 24; j++) {
            cplx ret, les;
            sigma_bath(ret, les, -3.0 + 0.25 * j, h, static_cast<dyson_bath_type>(type), 0.2,
                       beta, 0.1);
            kbref::write_phase7_row(fp, s, "bath_ret", type, j, 0, 0, ret);
            kbref::write_phase7_row(fp, s, "bath_les", type, j, 0, 0, les);
        }

    // Density matrices
    {
        cdmatrix M;
        density_matrix(M, FERMION, G0);
        write_matrix(fp, s, "density", 0, M);
        density_matrix(M, FERMION, G1);
        write_matrix(fp, s, "density", 1, M);
        density_matrix(M, FERMION, G4);
        write_matrix(fp, s, "density", 4, M);
        density_matrix(M, FERMION, G5);
        write_matrix(fp, s, "density", 5, M);
        // NESSi transforms the inputs to the frequency domain, use copies
        herm_matrix_ness A(Nft, 1), B(Nft, 1), C(Nft, 2), D(Nft, 2);
        A = G0;
        B = G1;
        convolution_density_matrix(M, FERMION, A, B, h);
        write_matrix(fp, s, "convolution_density", 0, M);
        C = G5;
        D = G5;
        convolution_density_matrix(M, FERMION, C, D, h);
        write_matrix(fp, s, "convolution_density", 1, M);
    }

    // Bubbles
    {
        herm_matrix_ness C1(Nft, 1), C2(Nft, 1);
        Bubble1_ness(C1, G0, G1);
        Bubble2_ness(C2, G0, G1);
        write_time(fp, s, "bubble1", 0, C1);
        write_time(fp, s, "bubble2", 0, C2);
        herm_matrix_ness D1(Nft, 2), D2(Nft, 2);
        for (int a = 0; a < 2; a++)
            for (int b = 0; b < 2; b++) {
                Bubble1_ness(D1, a, b, G5, a, b, G0, 0, 0);
                Bubble2_ness(D2, a, b, G5, a, b, G0, 0, 0);
            }
        write_time(fp, s, "bubble1", 1, D1);
        write_time(fp, s, "bubble2", 1, D2);
    }

    // Resampling
    {
        herm_matrix_ness A(Nft, 1);
        A = G0;
        herm_matrix_ness R0 = downsample(A, 4);
        write_time(fp, s, "resample", 0, R0);
        herm_matrix_ness B(Nft, 1);
        B = G1;
        herm_matrix_ness R1 = upsample(h, B, 2);
        write_time(fp, s, "resample", 1, R1);
    }

    // force_equilibrium
    {
        herm_matrix_ness F0(Nft, 1);
        F0 = G0;
        F0.les_.set_zero(fft_domain::time);
        F0.les_.set_zero(fft_domain::freq);
        F0.force_equilibrium(FERMION, beta, 0.3, h);
        write_time(fp, s, "force_equilibrium", 0, F0);
        gauss g3(3.0, 0.5);
        herm_matrix_ness F1(Nft, 1);
        green_equilibrium_ness(BOSON, F1, g3, beta, -1.0, h, FFT_TRAPEZ);
        F1.les_.set_zero(fft_domain::time);
        F1.les_.set_zero(fft_domain::freq);
        F1.force_equilibrium(BOSON, beta, -1.0, h);
        write_time(fp, s, "force_equilibrium", 1, F1);
    }

    // Conversion between NESS and contour Green's functions
    {
        int nt = Nft / 2, ntau = 50;
        cntr::herm_matrix<double> Gc(nt, ntau, 1, FERMION);
        cntr::green_equilibrium(Gc, g01, beta, h, 0.3, 100, 20);
        herm_matrix_ness N0(Nft, 1);
        cntr2ness(N0, Gc, nt);
        write_time(fp, s, "cntr_conversion", 0, N0);
        int nt2 = Nft / 2 + 9;
        cntr::herm_matrix<double> Gc2(nt2, ntau, 1, FERMION);
        ness2cntr(Gc2, G0);
        herm_matrix_ness N1(Nft, 1);
        cntr2ness(N1, Gc2, nt2);
        write_time(fp, s, "cntr_conversion", 1, N1);
    }

    std::fclose(fp);
    return 0;
}
