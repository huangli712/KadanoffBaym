// Phase 5 reference driver: interaction quench of the Falicov-Kimball
// model on the Bethe lattice (see .omo/plans/05-phase5-examples.md,
// P5-W3-T4).
//
// Usage: fkm_bethe_quench_text none <quick|release> <output-dir>
//
// Transcription of NESSi libcntr/test/fkm_bethe_quench.cpp:34-127 (no
// HDF5). The release profile holds the Catch parameters, the quick
// profile the parameters of tutor/fkm_bethe_quench/fkm_bethe_quench.toml.
//
// Output: fkm_bethe_quench_<profile>.dat (observable: den0, den1, total at the NESSi
// time steps -1..nt).

#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

#include "cntr/cntr.hpp"
#include "reference_io.hpp"

typedef std::complex<double> cplx;

int main(int argc, char *argv[]) {
    try {
        if (argc != 4 || std::string(argv[1]) != "none")
            throw std::runtime_error("usage: fkm_bethe_quench_text none <quick|release> <dir>");
        std::string profile = argv[2], dir = argv[3];
        int nt, ntau, kt;
        if (profile == "quick") {
            nt = 10;
            ntau = 50;
            kt = 3;
        } else if (profile == "release") {
            nt = 50;
            ntau = 200;
            kt = 5;
        } else {
            throw std::runtime_error("unknown profile " + profile);
        }
        int tstp, itermax = 100, iter_rtime = 3, size = 1, sig = -1;
        double U0 = 1.0, U1 = 2.0, beta = 5.0, h = 0.01, mu, err, errmax = 1e-7;
        cntr::herm_matrix<double> G0, G1, G, Delta;
        cntr::function<double> f1, f0;

        G0 = cntr::herm_matrix<double>(nt, ntau, size, sig);
        G1 = cntr::herm_matrix<double>(nt, ntau, size, sig);
        Delta = cntr::herm_matrix<double>(nt, ntau, size, sig);
        G = cntr::herm_matrix<double>(nt, ntau, size, sig);
        cntr::green_equilibrium_mat_bethe(Delta, beta);
        f0 = cntr::function<double>(nt, size);
        f1 = cntr::function<double>(nt, size);
        mu = 0.0;
        f0[-1] = -U0 * 0.5;
        for (tstp = 0; tstp <= nt; tstp++) f0[tstp] = -U1 * 0.5;
        f1[-1] = U0 * 0.5;
        for (tstp = 0; tstp <= nt; tstp++) f1[tstp] = U1 * 0.5;

        bool matsubara_converged = false;
        tstp = -1;
        cntr::herm_matrix<double> gtemp(kt, ntau, size, -1);
        for (int iter = 0; iter <= itermax; iter++) {
            cntr::dyson_mat(G0, Delta, mu, f0, integration::I<double>(kt), beta, CNTR_MAT_FOURIER);
            cntr::dyson_mat(G1, Delta, mu, f1, integration::I<double>(kt), beta, CNTR_MAT_FOURIER);
            G.set_timestep(tstp, G0);
            G.smul(tstp, 0.5);
            G.incr_timestep(tstp, G1, 0.5);
            err = cntr::distance_norm2(tstp, G, gtemp);
            if (err < errmax && iter > 2) {
                matsubara_converged = true;
                break;
            }
            gtemp.set_timestep(tstp, G);
            Delta.set_timestep(tstp, G);
        }
        if (!matsubara_converged) throw std::runtime_error("Matsubara iteration not converged");
        bool start_converged = false;
        for (int iter = 0; iter <= itermax; iter++) {
            cntr::dyson_start(G0, mu, f0, Delta, integration::I<double>(kt), beta, h);
            cntr::dyson_start(G1, mu, f1, Delta, integration::I<double>(kt), beta, h);
            for (tstp = 0; tstp <= kt; tstp++) {
                G.set_timestep(tstp, G0);
                G.smul(tstp, 0.5);
                G.incr_timestep(tstp, G1, 0.5);
            }
            err = 0.0;
            for (tstp = 0; tstp <= kt; tstp++) err += cntr::distance_norm2(tstp, G, gtemp);
            if (err < errmax) {
                start_converged = true;
                break;
            }
            for (tstp = 0; tstp <= kt; tstp++) gtemp.set_timestep(tstp, G);
            for (tstp = 0; tstp <= kt; tstp++) Delta.set_timestep(tstp, G);
        }
        if (!start_converged) throw std::runtime_error("bootstrap iteration not converged");
        for (tstp = kt + 1; tstp <= nt; tstp++) {
            cntr::extrapolate_timestep(tstp - 1, Delta, integration::I<double>(kt));
            for (int iter = 1; iter <= iter_rtime; iter++) {
                cntr::dyson_timestep(tstp, G0, mu, f0, Delta, integration::I<double>(kt), beta, h);
                cntr::dyson_timestep(tstp, G1, mu, f1, Delta, integration::I<double>(kt), beta, h);
                G.set_timestep(tstp, G0);
                G.smul(tstp, 0.5);
                G.incr_timestep(tstp, G1, 0.5);
                Delta.set_timestep(tstp, G);
            }
        }

        kbref::Metadata meta;
        meta.push_back(std::make_pair("generator", std::string("fkm_bethe_quench_text")));
        meta.push_back(std::make_pair("case_set", "fkm_bethe_quench_" + profile));
        meta.push_back(std::make_pair("model", std::string("fkm")));
        meta.push_back(std::make_pair("approximation", std::string("none")));
        meta.push_back(std::make_pair("profile", profile));
        meta.push_back(std::make_pair("nt", std::to_string(nt)));
        meta.push_back(std::make_pair("ntau", std::to_string(ntau)));
        meta.push_back(std::make_pair("h", kbref::fmt_double(h)));
        meta.push_back(std::make_pair("beta", kbref::fmt_double(beta)));
        meta.push_back(std::make_pair("solve_order", std::to_string(kt)));
        meta.push_back(std::make_pair("mats_max_iter", std::to_string(itermax)));
        meta.push_back(std::make_pair("bootstrap_max_iter", std::to_string(itermax)));
        meta.push_back(std::make_pair("corrector_steps", std::to_string(iter_rtime)));
        meta.push_back(std::make_pair("size", std::string("1")));
        meta.push_back(std::make_pair("params", std::string("U0=1,U1=2,mu=0")));
        FILE *fp = kbref::open_output(dir, "fkm_bethe_quench_" + profile + ".dat");
        kbref::write_metadata(fp, "observable", meta);
        double maxdev = 0.0;
        for (tstp = -1; tstp <= nt; tstp++) {
            double d0 = G0.density_matrix(tstp).real();
            double d1 = G1.density_matrix(tstp).real();
            maxdev = std::max(maxdev, std::abs(d0 + d1 - 1.0));
            kbref::write_observable_row(fp, meta, "den0", tstp, -1, -1, cplx(d0, 0.0));
            kbref::write_observable_row(fp, meta, "den1", tstp, -1, -1, cplx(d1, 0.0));
            kbref::write_observable_row(fp, meta, "total", tstp, -1, -1, cplx(d0 + d1, 0.0));
        }
        std::fclose(fp);
        // Particle-hole symmetry: den0 + den1 = 1 holds only up to the
        // self-consistency threshold errmax = 1e-7 (NESSi gives 7.7e-8 in
        // the quick profile), the check uses 1e-6
        if (maxdev > 1e-6) throw std::runtime_error("den0 + den1 deviates from 1");
    } catch (std::exception &e) {
        std::cerr << "fkm_bethe_quench_text: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
