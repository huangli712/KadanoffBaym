// Phase 5 reference driver: Hubbard chain in the 2B, GW, and TPP
// approximations (see .omo/plans/05-phase5-examples.md, P5-W2).
//
// Usage: hubbard_chain_text <2B|GW|TPP> <quick|release> <output-dir>
//
// Single process transcription of NESSi examples/programs/
// hubbard_chain_{2b,gw,tpp}.cpp (no input file, no HDF5). The self-energy
// functions are taken unchanged from hubbard_chain_selfen_impl.cpp. The
// only deviation: the Hartree term uses Re(rho_ii) (the diagonal of the
// hermitian part of the density matrix), as in the KadanoffBaym example
// tutor/hubbard_chain (during the bootstrapping G^<(t,t) is anti-hermitian
// only up to the discretization error).
//
// Output: hubbard_<approx>_<profile>.dat (observable: n1, Ekin, Epot,
// Etot at the NESSi time steps 0..nt) and hubbard_<approx>_<profile>_green.dat
// (selected_green: mat(0,-1), ret(3,1), les(1,3), tv(3,2) of G_11).

#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

#include "cntr/cntr.hpp"
#include "hubbard_chain_selfen_decl.hpp"
#include "reference_io.hpp"

typedef std::complex<double> cplx;

struct Params {
    int Nt, Ntau, SolveOrder, MatsMaxIter, BootstrapMaxIter, CorrectorSteps;
    double h, beta, MatsMaxErr, BootstrapMaxErr;
    int Nsites, RampSite;
    double HoppingT, HubbardU, MuChem, RampW0;
};

// The profiles of tutor/hubbard_chain/hubbard_chain.toml
static Params profile_params(const std::string &profile) {
    Params p;
    p.Nsites = 2;
    p.HoppingT = 1.0;
    p.HubbardU = 1.0;
    p.MuChem = 0.0;
    p.RampSite = 1;
    p.RampW0 = 5.0;
    p.h = 0.05;
    p.beta = 20.0;
    p.MatsMaxIter = 100;
    p.MatsMaxErr = 1e-8;
    p.BootstrapMaxIter = 50;
    p.BootstrapMaxErr = 1e-10;
    p.CorrectorSteps = 3;
    if (profile == "quick") {
        p.Nt = 10;
        p.Ntau = 40;
        p.SolveOrder = 3;
    } else if (profile == "release") {
        p.Nt = 399;
        p.Ntau = 400;
        p.SolveOrder = 5;
    } else {
        throw std::runtime_error("unknown profile " + profile);
    }
    return p;
}

// hubb::Ham_MF with Re(rho_ii)
static void ham_mf_herm(int tstp, GREEN &G, CFUNC &U, cdmatrix &h0, CFUNC &hmf) {
    int n = G.size1();
    cdmatrix rho(n, n), Ut(n, n), hv(n, n);
    G.density_matrix(tstp, rho);
    U.get_value(tstp, Ut);
    hv = h0;
    for (int i = 0; i < n; i++) hv(i, i) += Ut(i, i) * rho(i, i).real();
    hmf.set_value(tstp, hv);
}

// hubbard_chain_2b.cpp:20-41
static double kinetic_energy(int tstp, cdmatrix &eps0, GREEN &G) {
    int n = G.size1();
    cdmatrix rho(n, n);
    G.density_matrix(tstp, rho);
    return ((eps0 * rho).trace()).real();
}

static double interaction_energy(int tstp, cdmatrix &eps0, CFUNC &eps_mf, GREEN &G,
                                 GREEN &Sigma, double beta, double h, int k) {
    int n = G.size1();
    cdmatrix rho(n, n), vmf(n, n);
    eps_mf.get_value(tstp, vmf);
    vmf = vmf - eps0;
    G.density_matrix(tstp, rho);
    double emf = 0.5 * ((vmf * rho).trace()).real();
    double ecorr = cntr::correlation_energy(tstp, G, Sigma, beta, h, k);
    return emf + ecorr;
}

struct State {
    GREEN G, Sigma, gtemp, Pol, PxU, UxP, Chi, Phi, UxPhi, PhixU, TPP;
};

// Self-energy at the NESSi time step tstp (-1 or > k)
static void update_sigma(const std::string &a, int tstp, State &s, CFUNC &Ut, const Params &p) {
    if (a == "2B") {
        hubb::Sigma_2B(tstp, s.G, Ut, s.Sigma);
    } else if (a == "GW") {
        hubb::Polarization(tstp, s.G, s.Pol);
        hubb::GenChi(tstp, p.h, p.beta, s.Pol, Ut, s.PxU, s.UxP, s.Chi, p.SolveOrder);
        hubb::Sigma_GW(tstp, s.G, Ut, s.Chi, s.Sigma);
    } else {
        hubb::GenTPP(tstp, p.h, p.beta, s.G, s.Phi, Ut, s.UxPhi, s.PhixU, s.TPP, p.SolveOrder);
        hubb::Sigma_TPP(tstp, s.G, Ut, s.TPP, s.Sigma);
    }
}

// Self-energy of the start-up time steps 0..k
static void update_sigma_start(const std::string &a, State &s, CFUNC &Ut, const Params &p) {
    int k = p.SolveOrder;
    if (a == "2B") {
        for (int t = 0; t <= k; t++) hubb::Sigma_2B(t, s.G, Ut, s.Sigma);
    } else if (a == "GW") {
        for (int t = 0; t <= k; t++) hubb::Polarization(t, s.G, s.Pol);
        hubb::GenChi(p.h, p.beta, s.Pol, Ut, s.PxU, s.UxP, s.Chi, k);
        for (int t = 0; t <= k; t++) hubb::Sigma_GW(t, s.G, Ut, s.Chi, s.Sigma);
    } else {
        hubb::GenTPP(p.h, p.beta, s.G, s.Phi, Ut, s.UxPhi, s.PhixU, s.TPP, k);
        for (int t = 0; t <= k; t++) hubb::Sigma_TPP(t, s.G, Ut, s.TPP, s.Sigma);
    }
}

int main(int argc, char *argv[]) {
    try {
        if (argc != 4)
            throw std::runtime_error("usage: hubbard_chain_text <2B|GW|TPP> <quick|release> <dir>");
        std::string approx = argv[1], profile = argv[2], dir = argv[3];
        if (approx != "2B" && approx != "GW" && approx != "TPP")
            throw std::runtime_error("unknown approximation " + approx);
        Params p = profile_params(profile);
        int N = p.Nsites, k = p.SolveOrder, Nt = p.Nt, Ntau = p.Ntau;
        double h = p.h, beta = p.beta;

        State s;
        s.G = GREEN(Nt, Ntau, N, FERMION);
        s.Sigma = GREEN(Nt, Ntau, N, FERMION);
        s.Pol = GREEN(Nt, Ntau, N, BOSON);
        s.PxU = GREEN(Nt, Ntau, N, BOSON);
        s.UxP = GREEN(Nt, Ntau, N, BOSON);
        s.Chi = GREEN(Nt, Ntau, N, BOSON);
        s.Phi = GREEN(Nt, Ntau, N, BOSON);
        s.UxPhi = GREEN(Nt, Ntau, N, BOSON);
        s.PhixU = GREEN(Nt, Ntau, N, BOSON);
        s.TPP = GREEN(Nt, Ntau, N, BOSON);

        cdmatrix eps0 = cdmatrix::Zero(N, N);
        for (int i = 0; i < N - 1; i++) {
            eps0(i, i + 1) = -p.HoppingT;
            eps0(i + 1, i) = -p.HoppingT;
        }
        CFUNC eps_mf(Nt, N), Ut(Nt, N);
        Ut.set_constant(p.HubbardU * cdmatrix::Identity(N, N));
        double mu = p.MuChem;
        cntr::green_from_H(s.G, mu, eps0, beta, h);
        cdmatrix DensM(N, N);
        s.G.density_matrix(-1, DensM);
        mu = mu + p.HubbardU * DensM.trace().real() / N;

        // Matsubara
        s.gtemp = GREEN(k, Ntau, N, FERMION);
        s.gtemp.set_timestep(-1, s.G);
        bool conv = false;
        for (int iter = 0; iter <= p.MatsMaxIter; iter++) {
            ham_mf_herm(-1, s.G, Ut, eps0, eps_mf);
            update_sigma(approx, -1, s, Ut, p);
            cntr::dyson_mat(s.G, mu, eps_mf, s.Sigma, beta, k);
            double err = cntr::distance_norm2(-1, s.G, s.gtemp);
            if (err < p.MatsMaxErr) {
                conv = true;
                break;
            }
            s.gtemp.set_timestep(-1, s.G);
        }
        if (!conv) throw std::runtime_error("Matsubara iteration not converged");

        // Quench and bootstrapping
        eps0(p.RampSite - 1, p.RampSite - 1) = p.RampW0;
        for (int t = 0; t <= k; t++) s.gtemp.set_timestep(t, s.G);
        conv = false;
        for (int iter = 0; iter <= p.BootstrapMaxIter; iter++) {
            for (int t = 0; t <= k; t++) ham_mf_herm(t, s.G, Ut, eps0, eps_mf);
            update_sigma_start(approx, s, Ut, p);
            cntr::dyson_start(s.G, mu, eps_mf, s.Sigma, beta, h, k);
            double err = 0.0;
            for (int t = 0; t <= k; t++) err += cntr::distance_norm2(t, s.G, s.gtemp);
            if (err < p.BootstrapMaxErr && iter > 2) {
                conv = true;
                break;
            }
            for (int t = 0; t <= k; t++) s.gtemp.set_timestep(t, s.G);
        }
        if (!conv) throw std::runtime_error("bootstrap iteration not converged");

        // Propagation
        for (int tstp = k + 1; tstp <= Nt; tstp++) {
            cntr::extrapolate_timestep(tstp - 1, s.G, k);
            for (int iter = 0; iter < p.CorrectorSteps; iter++) {
                ham_mf_herm(tstp, s.G, Ut, eps0, eps_mf);
                update_sigma(approx, tstp, s, Ut, p);
                cntr::dyson_timestep(tstp, s.G, mu, eps_mf, s.Sigma, beta, h, k);
            }
        }

        // Output
        std::string lc = approx == "2B" ? "2b" : (approx == "GW" ? "gw" : "tpp");
        std::string base = "hubbard_" + lc + "_" + profile;
        kbref::Metadata meta;
        meta.push_back(std::make_pair("generator", std::string("hubbard_chain_text")));
        meta.push_back(std::make_pair("case_set", base));
        meta.push_back(std::make_pair("model", std::string("hubbard")));
        meta.push_back(std::make_pair("approximation", approx));
        meta.push_back(std::make_pair("profile", profile));
        meta.push_back(std::make_pair("nt", std::to_string(Nt)));
        meta.push_back(std::make_pair("ntau", std::to_string(Ntau)));
        meta.push_back(std::make_pair("h", kbref::fmt_double(h)));
        meta.push_back(std::make_pair("beta", kbref::fmt_double(beta)));
        meta.push_back(std::make_pair("solve_order", std::to_string(k)));
        meta.push_back(std::make_pair("mats_max_iter", std::to_string(p.MatsMaxIter)));
        meta.push_back(std::make_pair("bootstrap_max_iter", std::to_string(p.BootstrapMaxIter)));
        meta.push_back(std::make_pair("corrector_steps", std::to_string(p.CorrectorSteps)));
        meta.push_back(std::make_pair("size", std::to_string(N)));
        meta.push_back(std::make_pair(
            "params", std::string("Nsites=2,HoppingT=1,HubbardU=1,MuChem=0,RampSite=1,RampW0=5")));

        FILE *fp = kbref::open_output(dir, base + ".dat");
        kbref::write_metadata(fp, "observable", meta);
        cdmatrix rho(N, N);
        for (int tstp = 0; tstp <= Nt; tstp++) {
            s.G.density_matrix(tstp, rho);
            double ekin = kinetic_energy(tstp, eps0, s.G);
            double epot = interaction_energy(tstp, eps0, eps_mf, s.G, s.Sigma, beta, h, k);
            kbref::write_observable_row(fp, meta, "n1", tstp, -1, -1, cplx(rho(0, 0).real(), 0.0));
            kbref::write_observable_row(fp, meta, "Ekin", tstp, -1, -1, cplx(ekin, 0.0));
            kbref::write_observable_row(fp, meta, "Epot", tstp, -1, -1, cplx(epot, 0.0));
            kbref::write_observable_row(fp, meta, "Etot", tstp, -1, -1, cplx(ekin + epot, 0.0));
        }
        std::fclose(fp);

        FILE *fg = kbref::open_output(dir, base + "_green.dat");
        kbref::write_metadata(fg, "selected_green", meta);
        cdmatrix x(N, N);
        s.G.get_mat(0, x);
        kbref::write_selected_green_row(fg, "mat", 0, -1, 0, 0, x(0, 0));
        s.G.get_ret(3, 1, x);
        kbref::write_selected_green_row(fg, "ret", 3, 1, 0, 0, x(0, 0));
        s.G.get_les(1, 3, x);
        kbref::write_selected_green_row(fg, "les", 1, 3, 0, 0, x(0, 0));
        s.G.get_tv(3, 2, x);
        kbref::write_selected_green_row(fg, "tv", 3, 2, 0, 0, x(0, 0));
        std::fclose(fg);
    } catch (std::exception &e) {
        std::cerr << "hubbard_chain_text: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
