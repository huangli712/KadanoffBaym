// P6 kbref generator: pseudo-particle Green's functions (herm_pseudo).
//
// Usage: pseudo_reference <output-dir>
//
// All inputs are closed-form and rebuilt in test/t_pseudo.jl (nothing
// is read back from the generated files). An input object with id w and
// scale s has the stored blocks (t = i h, tau = q beta / ntau)
//
//   mat(q)    = s val(w, 0, tau, 0)
//   gtr(i,j)  = s val(w, 1, t_i, t_j)     (j <= i, stored ret_ triangle)
//   tv(i,q)   = s val(w, 2, t_i, tau_q)
//   les(i,j)  = s val(w, 3, t_i, t_j)     (i <= j)
//
// with val(w,c,x,y)_ab = amp exp(-0.2x - 0.15y)
//   [cos(0.7x + 0.3y + 0.2(a+b) + w) + i sin(0.5x - 0.4y + 0.1ab + c)],
// amp = 0.3 + 0.1w + 0.05c + 0.1a - 0.07b. These objects are not physical
// Green's functions; they only fix the input data of the algorithms.
//
// The Hamiltonian of the Dyson cases is H(t) = H0 + 0.3 sin(2t), the
// contour function of the convolution cases is f(t) = F(t) (t = i h) and
// f(-1) = F(-0.5), F(t) = [1 + 0.2t, 0.1 + 0.3it; -0.2 + 0.05i, 0.8]. For
// size 1 the top-left corners are used.
//
// Every file contains one case (data_kind pseudo):
//   pseudo_storage.dat          storage of one input object
//   pseudo_convolution_*.dat    C = A * f * B (pseudo_convolution) and
//                               the density matrix of A * f * B
//   pseudodyson_*.dat           pseudodyson (matrix/function/scalar H),
//                               its density matrix (herm_pseudo::
//                               density_matrix), and the lesser
//                               start-up source blocks
//   pseudo_vie2_*.dat           pseudo_vie2 and the start-up source blocks
#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "cntr/cntr.hpp"
#include "reference_io.hpp"

using namespace cntr;

typedef std::complex<double> cplx;
typedef herm_pseudo<double> GP;

static const int NT = 10;
static const int NTAU = 30;
static const double BETA = 2.0;
static const double H = 0.02;
static const double LAM0 = 0.5;
static const std::string GEN = "pseudo_reference";

static cplx val(int w, int c, double x, double y, int a, int b) {
    double amp = 0.3 + 0.1 * w + 0.05 * c + 0.1 * a - 0.07 * b;
    double env = std::exp(-0.2 * x - 0.15 * y);
    return amp * env * cplx(std::cos(0.7 * x + 0.3 * y + 0.2 * (a + b) + w),
                            std::sin(0.5 * x - 0.4 * y + 0.1 * a * b + c));
}

static void build(GP &G, int w, double s) {
    const int nt = G.nt(), ntau = G.ntau(), d = G.size1();
    cdmatrix M(d, d);
    for (int q = 0; q <= ntau; q++) {
        double tau = q * BETA / ntau;
        for (int a = 0; a < d; a++)
            for (int b = 0; b < d; b++) M(a, b) = s * val(w, 0, tau, 0.0, a, b);
        G.set_mat(q, M);
    }
    for (int i = 0; i <= nt; i++) {
        for (int j = 0; j <= i; j++) {
            for (int a = 0; a < d; a++)
                for (int b = 0; b < d; b++) M(a, b) = s * val(w, 1, i * H, j * H, a, b);
            G.set_ret(i, j, M);
        }
        for (int q = 0; q <= ntau; q++) {
            double tau = q * BETA / ntau;
            for (int a = 0; a < d; a++)
                for (int b = 0; b < d; b++) M(a, b) = s * val(w, 2, i * H, tau, a, b);
            G.set_tv(i, q, M);
        }
        for (int j = i; j <= nt; j++) {
            for (int a = 0; a < d; a++)
                for (int b = 0; b < d; b++) M(a, b) = s * val(w, 3, i * H, j * H, a, b);
            G.set_les(i, j, M);
        }
    }
}

static cdmatrix hamiltonian(int d, double t) {
    cdmatrix M(2, 2);
    M(0, 0) = 0.3;
    M(0, 1) = cplx(0.2, -0.1);
    M(1, 0) = cplx(0.2, 0.1);
    M(1, 1) = -0.4;
    M += 0.3 * std::sin(2.0 * t) * cdmatrix::Identity(2, 2);
    return M.topLeftCorner(d, d);
}

static cdmatrix middle(int d, double t) {
    cdmatrix F(2, 2);
    F(0, 0) = 1.0 + 0.2 * t;
    F(0, 1) = cplx(0.1, 0.3 * t);
    F(1, 0) = cplx(-0.2, 0.05);
    F(1, 1) = 0.8;
    return F.topLeftCorner(d, d);
}

static FILE *open_case(const std::string &outdir, const std::string &name, int sig,
                       int d, int k, const kbref::Metadata &extra) {
    FILE *fp = kbref::open_output(outdir, name + ".dat");
    kbref::Metadata meta;
    meta.push_back(std::make_pair("generator", GEN));
    meta.push_back(std::make_pair("case_set", name));
    meta.push_back(std::make_pair("case", name));
    meta.push_back(std::make_pair("sig", kbref::fmt_int(sig)));
    meta.push_back(std::make_pair("nt", kbref::fmt_int(NT)));
    meta.push_back(std::make_pair("ntau", kbref::fmt_int(NTAU)));
    meta.push_back(std::make_pair("size1", kbref::fmt_int(d)));
    meta.push_back(std::make_pair("size2", kbref::fmt_int(d)));
    meta.push_back(std::make_pair("k", kbref::fmt_int(k)));
    meta.push_back(std::make_pair("beta", kbref::fmt_double(BETA)));
    meta.push_back(std::make_pair("h", kbref::fmt_double(H)));
    for (size_t n = 0; n < extra.size(); n++) meta.push_back(extra[n]);
    kbref::write_metadata(fp, "pseudo", meta);
    return fp;
}

static void write_block(FILE *fp, const char *comp, int i, int j, int d, int k,
                        const cplx *x) {
    for (int a = 0; a < d; a++)
        for (int b = 0; b < d; b++)
            kbref::write_pseudo_row(fp, comp, i, j, a, b, k, x[a * d + b]);
}

// Lesser start-up source blocks j = n+1, ..., k (n < k) of the Dyson
// (sign = +1, [Sigma * G]) or VIE2 (sign = -1, -[F * G]) start-up
// procedure: tv*vt + les*adv, see cntr_dyson_impl.hpp:921-927 and
// cntr_vie2_impl.hpp:874-881.
static void write_start_source(FILE *fp, GP &G, GP &A, GP &Acc, int k, double sign) {
    const int d = G.size1(), sg = d * d;
    integration::Integrator<double> &I = integration::I<double>(k);
    for (int n = 0; n < k; n++) {
        std::vector<cplx> buf((k + 1) * sg, cplx(0.0, 0.0));
        convolution_timestep_les_tvvt<double, GP, LARGESIZE>(n, buf.data(), G, A, Acc, G, G,
                                                              I, BETA, H);
        convolution_timestep_les_lesadv<double, GP, LARGESIZE>(n, buf.data(), G, A, Acc, G,
                                                                G, I, BETA, H);
        for (size_t l = 0; l < buf.size(); l++) buf[l] *= sign;
        for (int j = n + 1; j <= k; j++)
            write_block(fp, "les_start_source", j, n, d, k, buf.data() + j * sg);
    }
}

static void write_storage(const std::string &outdir) {
    GP A(NT, NTAU, 2, -1);
    build(A, 0, 1.0);
    kbref::Metadata extra;
    extra.push_back(std::make_pair("object", "0"));
    FILE *fp = open_case(outdir, "pseudo_storage", -1, 2, 1, extra);
    cdmatrix M;
    kbref::write_herm_pseudo(fp, A, M);
    std::fclose(fp);
}

static void write_convolution(const std::string &outdir, int d, int k, int sig, bool usef) {
    GP A(NT, NTAU, d, sig), Acc(NT, NTAU, d, sig), B(NT, NTAU, d, sig),
        Bcc(NT, NTAU, d, sig), C(NT, NTAU, d, sig);
    build(A, 1, 1.0);
    build(Acc, 2, 1.0);
    build(B, 3, 1.0);
    build(Bcc, 4, 1.0);
    function<double> f(NT, d);
    for (int i = -1; i <= NT; i++) {
        cdmatrix F = middle(d, i < 0 ? -0.5 : i * H);
        f.set_value(i, F);
    }
    integration::Integrator<double> &I = integration::I<double>(k);
    if (usef)
        pseudo_convolution<double>(C, A, Acc, f, B, Bcc, I, BETA, H);
    else
        pseudo_convolution<double>(C, A, Acc, B, Bcc, I, BETA, H);

    std::string name = "pseudo_convolution_size" + std::to_string(d) + "_k" +
                       std::to_string(k) + (sig == -1 ? "_fermi" : "_bose") +
                       (usef ? "_f" : "_nof");
    kbref::Metadata extra;
    extra.push_back(std::make_pair("f", usef ? "yes" : "no"));
    FILE *fp = open_case(outdir, name, sig, d, k, extra);
    cdmatrix M;
    kbref::write_herm_pseudo(fp, C, M);
    std::vector<cplx> rho(d * d);
    for (int tstp = -1; tstp <= NT; tstp++) {
        if (usef)
            convolution_density_matrix<double, GP>(tstp, rho.data(), A, Acc, f, B, Bcc, I,
                                                   BETA, H);
        else
            convolution_density_matrix<double, GP>(tstp, rho.data(), A, Acc, B, Bcc, I,
                                                   BETA, H);
        write_block(fp, "density", tstp, -1, d, k, rho.data());
    }
    std::fclose(fp);
}

// hpath: 0 = matrix, 1 = function, 2 = scalar (size 1 only)
static void write_dyson(const std::string &outdir, int d, int k, int sig, int hpath,
                        bool zero_sigma) {
    GP G(NT, NTAU, d, sig), S(NT, NTAU, d, sig);
    if (!zero_sigma) build(S, 5, 0.2);
    integration::Integrator<double> &I = integration::I<double>(k);
    const char *pname[3] = {"matrix", "function", "scalar"};
    if (hpath == 0) {
        cdmatrix H0 = hamiltonian(d, 0.0);
        std::vector<cdmatrix> Hv(NT + 1);
        for (int n = 0; n <= NT; n++) Hv[n] = hamiltonian(d, n * H);
        pseudodyson_mat(G, LAM0, H0, S, I, BETA);
        pseudodyson_start(G, LAM0, H0, Hv, S, I, BETA, H);
        for (int tstp = k + 1; tstp <= NT; tstp++)
            pseudodyson_timestep(tstp, G, LAM0, H0, Hv, S, I, BETA, H);
    } else if (hpath == 1) {
        function<double> Hf(NT, d);
        cdmatrix H0 = hamiltonian(d, 0.0);
        Hf.set_value(-1, H0);
        for (int n = 0; n <= NT; n++) {
            cdmatrix Hn = hamiltonian(d, n * H);
            Hf.set_value(n, Hn);
        }
        pseudodyson_mat(G, LAM0, Hf, S, I, BETA);
        pseudodyson_start(G, LAM0, Hf, S, I, BETA, H);
        for (int tstp = k + 1; tstp <= NT; tstp++)
            pseudodyson_timestep(tstp, G, LAM0, Hf, S, I, BETA, H);
    } else {
        cplx H0 = hamiltonian(1, 0.0)(0, 0);
        std::vector<cplx> Hv(NT + 1);
        for (int n = 0; n <= NT; n++) Hv[n] = hamiltonian(1, n * H)(0, 0);
        pseudodyson_mat(G, LAM0, H0, S, I, BETA);
        pseudodyson_start(G, LAM0, H0, Hv, S, I, BETA, H);
        for (int tstp = k + 1; tstp <= NT; tstp++)
            pseudodyson_timestep(tstp, G, LAM0, H0, Hv, S, I, BETA, H);
    }

    std::string name = "pseudodyson_size" + std::to_string(d) + "_k" + std::to_string(k) +
                       (sig == -1 ? "_fermi_" : "_bose_") + pname[hpath] +
                       (zero_sigma ? "_atomic" : "_sigma");
    kbref::Metadata extra;
    extra.push_back(std::make_pair("lambda0", kbref::fmt_double(LAM0)));
    extra.push_back(std::make_pair("hpath", pname[hpath]));
    extra.push_back(std::make_pair("sigma", zero_sigma ? "zero" : "formula"));
    FILE *fp = open_case(outdir, name, sig, d, k, extra);
    cdmatrix M;
    kbref::write_herm_pseudo(fp, G, M);
    // herm_pseudo::density_matrix of the solution
    for (int tstp = -1; tstp <= NT; tstp++) {
        cdmatrix R(d, d);
        G.density_matrix(tstp, R);
        std::vector<cplx> x(d * d);
        for (int a = 0; a < d; a++)
            for (int b = 0; b < d; b++) x[a * d + b] = R(a, b);
        write_block(fp, "density", tstp, -1, d, k, x.data());
    }
    write_start_source(fp, G, S, S, k, 1.0);
    std::fclose(fp);
}

static void write_vie2(const std::string &outdir, int d, int k, int sig, bool zero_f) {
    GP G(NT, NTAU, d, sig), F(NT, NTAU, d, sig), Fcc(NT, NTAU, d, sig), Q(NT, NTAU, d, sig);
    if (!zero_f) {
        build(F, 6, 0.2);
        build(Fcc, 7, 0.2);
    }
    build(Q, 8, 1.0);
    integration::Integrator<double> &I = integration::I<double>(k);
    pseudo_vie2<double>(G, F, Fcc, Q, I, BETA, H);

    std::string name = "pseudo_vie2_size" + std::to_string(d) + "_k" + std::to_string(k) +
                       (sig == -1 ? "_fermi" : "_bose") + (zero_f ? "_zero" : "_formula");
    kbref::Metadata extra;
    extra.push_back(std::make_pair("kernel", zero_f ? "zero" : "formula"));
    FILE *fp = open_case(outdir, name, sig, d, k, extra);
    cdmatrix M;
    kbref::write_herm_pseudo(fp, G, M);
    write_start_source(fp, G, F, Fcc, k, -1.0);
    std::fclose(fp);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <output-dir>\n", argv[0]);
        return 1;
    }
    const std::string outdir(argv[1]);
    // The Matrix version of pseudodyson_timestep prints a debug line to
    // std::cout (cntr_pseudodyson_impl.hpp:403); it is silenced here.
    std::cout.setstate(std::ios_base::failbit);

    write_storage(outdir);

    write_convolution(outdir, 1, 4, -1, false);
    write_convolution(outdir, 1, 4, -1, true);
    write_convolution(outdir, 2, 4, -1, false);
    write_convolution(outdir, 2, 4, 1, true);
    write_convolution(outdir, 2, 1, -1, true);

    write_dyson(outdir, 2, 3, -1, 0, true);
    write_dyson(outdir, 2, 4, -1, 0, false);
    write_dyson(outdir, 2, 4, 1, 1, false);
    write_dyson(outdir, 2, 1, -1, 1, false);
    write_dyson(outdir, 1, 3, -1, 2, false);
    write_dyson(outdir, 1, 3, -1, 0, false);
    write_dyson(outdir, 1, 3, -1, 1, false);

    write_vie2(outdir, 2, 3, -1, true);
    write_vie2(outdir, 2, 4, -1, false);
    write_vie2(outdir, 1, 2, 1, false);
    write_vie2(outdir, 2, 1, -1, false);
    write_vie2(outdir, 2, 5, 1, false);

    return 0;
}
