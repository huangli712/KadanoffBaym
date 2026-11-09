// kbref generator: NESSi equilibrium Green's functions.
//
// Usage: p3_equilibrium_ref <output-dir>
// Output (kbref-v1, data_kind herm_matrix):
//   green_const_{fermion,boson}_{1x1,2x2}.dat   constant Hamiltonian
//   green_time_cf2_fermion.dat, green_time_cf4_{fermion,boson}.dat
//                                               time-dependent Hamiltonian
//   green_bethe_mat_fermion.dat, green_bethe_full_fermion.dat
//   green_custom_dos_fermion.dat                smooth DOS with finite support
//   green_single_pole_XX.dat
//
// Inputs (the same closed-form formulas are used in the Julia test
// test/nessi/t_phase3_equilibrium.jl; nothing is read back from the output):
//   * constant: H = 0.4 (1x1), H = [-0.3, 0.4 - 0.2i; 0.4 + 0.2i, 0.6] (2x2),
//     mu = 0.1 (fermions) or -1.0 (bosons), beta = 3, h = 0.05;
//   * time-dependent: H(t) = [1, 0.5 e^{-1.4 i t}; 0.5 e^{1.4 i t}, -1],
//     H(-1) = [1, 0.5; 0.5, -1], mu = 0.3 (fermions) or -1.5 (bosons),
//     beta = 2, h = 0.05, SolveOrder = 5, fixHam = true;
//   * Bethe DOS (V = 1), size 2, mu = 0.2, beta = 3, h = 0.05;
//   * custom DOS rho(e) = (15/16) (1 - e^2)^2 on [-1, 1], mu = 0.1,
//     beta = 3, h = 0.05, limit = 60, nn = 24;
//   * single pole XX: w = 1.4, beta = 20, h = 0.01, nt = ntau = 100.
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>

// CNTR_NO_EXTERN_TEMPLATES is defined by the CMake target.
#include "cntr/cntr.hpp"
#include "reference_io.hpp"

using namespace cntr;
typedef std::complex<double> cplx;

static const cplx CI(0.0, 1.0);

class custom_dos {
  public:
    double lo_;
    double hi_;
    custom_dos() {
        lo_ = -1.0;
        hi_ = 1.0;
    }
    double operator()(double x) {
        double y = 1.0 - x * x;
        return (y < 0.0) ? 0.0 : 15.0 / 16.0 * y * y;
    }
};

static void write_case(const std::string &outdir, const std::string &name,
                       herm_matrix<double> &G, double beta, double h,
                       double mu, const std::string &method) {
    FILE *fp = kbref::open_output(outdir, name + ".dat");
    kbref::Metadata meta;
    meta.push_back(std::make_pair("case", name));
    meta.push_back(std::make_pair("generator", std::string("p3_equilibrium_ref")));
    meta.push_back(std::make_pair("nt", kbref::fmt_int(G.nt())));
    meta.push_back(std::make_pair("ntau", kbref::fmt_int(G.ntau())));
    meta.push_back(std::make_pair("size1", kbref::fmt_int(G.size1())));
    meta.push_back(std::make_pair("size2", kbref::fmt_int(G.size2())));
    meta.push_back(std::make_pair("beta", kbref::fmt_double(beta)));
    meta.push_back(std::make_pair("h", kbref::fmt_double(h)));
    meta.push_back(std::make_pair("mu", kbref::fmt_double(mu)));
    meta.push_back(std::make_pair("sig", kbref::fmt_int(G.sig())));
    meta.push_back(std::make_pair("method", method));
    kbref::write_metadata(fp, "herm_matrix", meta);
    cdmatrix M;
    kbref::write_herm_matrix(fp, G, M);
    std::fclose(fp);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <output-dir>\n", argv[0]);
        return 1;
    }
    const std::string outdir(argv[1]);

    //
    // Constant Hamiltonian
    //
    {
        const int nt = 20, ntau = 40;
        const double beta = 3.0, h = 0.05;
        cdmatrix h1(1, 1), h2(2, 2);
        h1(0, 0) = 0.4;
        h2(0, 0) = -0.3;
        h2(0, 1) = cplx(0.4, -0.2);
        h2(1, 0) = cplx(0.4, 0.2);
        h2(1, 1) = 0.6;
        for (int sig = -1; sig <= 1; sig += 2) {
            const double mu = (sig == -1) ? 0.1 : -1.0;
            const std::string s = (sig == -1) ? "fermion" : "boson";
            herm_matrix<double> G1(nt, ntau, 1, sig), G2(nt, ntau, 2, sig);
            green_from_H(G1, mu, h1, beta, h);
            green_from_H(G2, mu, h2, beta, h);
            write_case(outdir, "green_const_" + s + "_1x1", G1, beta, h, mu, "const");
            write_case(outdir, "green_const_" + s + "_2x2", G2, beta, h, mu, "const");
        }
    }

    //
    // Time-dependent Hamiltonian
    //
    {
        const int nt = 40, ntau = 40, kt = 5;
        const double beta = 2.0, h = 0.05;
        function<double> eps(nt, 2);
        cdmatrix M(2, 2);
        M(0, 0) = 1.0;
        M(1, 1) = -1.0;
        M(0, 1) = 0.5;
        M(1, 0) = 0.5;
        eps.set_value(-1, M);
        for (int n = 0; n <= nt; n++) {
            double t = n * h;
            M(0, 1) = 0.5 * std::exp(-1.4 * CI * t);
            M(1, 0) = 0.5 * std::exp(1.4 * CI * t);
            eps.set_value(n, M);
        }
        herm_matrix<double> G2(nt, ntau, 2, -1), G4(nt, ntau, 2, -1), B4(nt, ntau, 2, 1);
        green_from_H(G2, 0.3, eps, beta, h, true, kt, 2);
        green_from_H(G4, 0.3, eps, beta, h, true, kt, 4);
        green_from_H(B4, -1.5, eps, beta, h, true, kt, 4);
        write_case(outdir, "green_time_cf2_fermion", G2, beta, h, 0.3, "cf2");
        write_case(outdir, "green_time_cf4_fermion", G4, beta, h, 0.3, "cf4");
        write_case(outdir, "green_time_cf4_boson", B4, beta, h, -1.5, "cf4");
    }

    //
    // DOS
    //
    {
        const int nt = 20, ntau = 40;
        const double beta = 3.0, h = 0.05;
        herm_matrix<double> Gm(nt, ntau, 2, -1), Gf(nt, ntau, 2, -1);
        green_equilibrium_mat_bethe(Gm, beta, 100, 20, 0.2);
        green_equilibrium_bethe(Gf, beta, h, 100, 20, 0.2);
        write_case(outdir, "green_bethe_mat_fermion", Gm, beta, h, 0.2, "bethe");
        write_case(outdir, "green_bethe_full_fermion", Gf, beta, h, 0.2, "bethe");
        //
        custom_dos dos;
        herm_matrix<double> Gc(nt, ntau, 1, -1);
        int limit = 60, nn = 24;
        double mu = 0.1;
        green_equilibrium(Gc, dos, beta, h, limit, nn, mu);
        write_case(outdir, "green_custom_dos_fermion", Gc, beta, h, mu, "custom_dos");
    }

    //
    // Bosonic single pole
    //
    {
        const int nt = 100, ntau = 100;
        const double w = 1.4, beta = 20.0, h = 0.01;
        herm_matrix<double> D(nt, ntau, 1, 1);
        green_single_pole_XX(D, w, beta, h);
        write_case(outdir, "green_single_pole_XX", D, beta, h, 0.0, "single_pole_XX");
    }
    return 0;
}
