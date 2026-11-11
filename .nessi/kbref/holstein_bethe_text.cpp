// Phase 5 reference driver: Holstein model on the Bethe lattice, normal
// state, self-consistent (Migdal) and unrenormalized (uMig) Migdal
// approximations (see .omo/plans/05-phase5-examples.md, P5-W6-T3).
//
// Usage: holstein_bethe_text <Migdal|uMig> <quick|release> <output-dir>
//
// Transcription of NESSi examples/programs/Holstein_bethe_Migdal.cpp and
// Holstein_bethe_uMig.cpp (no input file, no HDF5); the self-energies
// and phonon routines are taken unchanged from Holstein_impurity_impl.cpp
// and Holstein_utils_impl.cpp. The only deviation: the density n_tot of
// the mean-field terms is 2 Re(rho) (as in tutor/holstein_bethe). The
// drives are read from tutor/holstein_bethe/dHopping_bethe_<profile>.dat
// (dEl_Ph_g = 0).
//
// Output: holstein_bethe_<approx>_<profile>.dat (observable at the NESSi
// time steps 0..nt).

#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cntr/cntr.hpp"
#include "Holstein_impurity_decl.hpp"
#include "Holstein_utils_decl.hpp"
#include "holstein_common.hpp"

typedef std::complex<double> cplx;

int main(int argc, char *argv[]) {
    try {
        if (argc != 4) throw std::runtime_error("usage: holstein_bethe_text <Migdal|uMig> <quick|release> <dir>");
        std::string approx = argv[1], profile = argv[2], dir = argv[3];
        if (approx != "Migdal" && approx != "uMig") throw std::runtime_error("unknown approximation " + approx);
        bool mig = approx == "Migdal";
        int Nt, Ntau, SolverOrder, MatsMaxIter, BootstrapMaxIter, CorrectorSteps;
        if (profile == "quick") {
            Nt = 8; Ntau = 40; SolverOrder = 3; MatsMaxIter = 100; BootstrapMaxIter = 50; CorrectorSteps = 3;
        } else if (profile == "release") {
            Nt = 400; Ntau = 500; SolverOrder = 5; MatsMaxIter = 800; BootstrapMaxIter = 20; CorrectorSteps = 5;
        } else {
            throw std::runtime_error("unknown profile " + profile);
        }
        double Hopping = 1.0, El_Ph_g = 0.5, Phfreq_w0 = 1.0, MuChem_MF = 0.5, beta = 10.0, h = 0.03;
        double MatsMaxErr = 1e-7, BootstrapMaxErr = 1e-7, MuChem, err;
        std::vector<double> dHopping = kbref::make_drive(1.0, 0.6, h, Nt);
        std::vector<double> dEl_Ph_g(Nt + 2, 0.0);
        int tstp, Norb = 1;

        GREEN G, Sigma, Hyb, Hyb_Sig, D, Pi, D0, D0_Pi, Pi_D0, gtemp;
        CFUNC g_elph_t, J_hop_t, h0_imp_t, h0_imp_MF_t, Xph_t, n_tot_t, Pph_t;
        CFUNC Ekin_t, Enx_MF_t, Enx_corr_t, Eph_qu_t, Eph_cl_t;
        cdmatrix h0_imp = cdmatrix::Zero(Norb, Norb), h0_imp_MF = cdmatrix::Zero(Norb, Norb);
        g_elph_t = CFUNC(Nt, Norb);
        J_hop_t = CFUNC(Nt, Norb);
        h0_imp_t = CFUNC(Nt, Norb);
        h0_imp_MF_t = CFUNC(Nt, Norb);
        G = GREEN(Nt, Ntau, Norb, FERMION);
        Hyb = GREEN(Nt, Ntau, Norb, FERMION);
        Sigma = GREEN(Nt, Ntau, Norb, FERMION);
        Hyb_Sig = GREEN(Nt, Ntau, Norb, FERMION);
        D = GREEN(Nt, Ntau, 1, BOSON);
        Pi = GREEN(Nt, Ntau, 1, BOSON);
        D0 = GREEN(Nt, Ntau, 1, BOSON);
        D0_Pi = GREEN(Nt, Ntau, 1, BOSON);
        Pi_D0 = GREEN(Nt, Ntau, 1, BOSON);
        Xph_t = CFUNC(Nt, 1);
        Pph_t = CFUNC(Nt, 1);
        n_tot_t = CFUNC(Nt, 1);
        Ekin_t = CFUNC(Nt, 1);
        Enx_MF_t = CFUNC(Nt, 1);
        Enx_corr_t = CFUNC(Nt, 1);
        Eph_qu_t = CFUNC(Nt, 1);
        Eph_cl_t = CFUNC(Nt, 1);

        h0_imp_MF(0, 0) = -MuChem_MF;
        h0_imp_MF_t.set_constant(h0_imp_MF);
        cdmatrix Iu = cdmatrix::Identity(1, 1);
        for (int it = -1; it <= Nt; it++) J_hop_t.set_value(it, (Hopping + dHopping[it + 1]) * Iu);
        for (int it = -1; it <= Nt; it++) g_elph_t.set_value(it, (El_Ph_g + dEl_Ph_g[it + 1]) * Iu);
        cntr::green_equilibrium_mat_bethe(G, beta);
        tstp = -1;
        Hyb.set_timestep(tstp, G);
        Hyb.right_multiply(tstp, J_hop_t);
        Hyb.left_multiply(tstp, J_hop_t);
        cntr::green_single_pole_XX(D0, Phfreq_w0, beta, h);
        for (int it = -1; it <= Nt; it++) D0.smul(it, 2.0);

        // Matsubara
        bool conv = false;
        gtemp = GREEN(SolverOrder, Ntau, Norb, FERMION);
        gtemp.set_timestep(tstp, G);
        bool FIXPOINT = false;
        for (int iter = 0; iter <= MatsMaxIter; iter++) {
            int meth = FIXPOINT ? CNTR_MAT_FIXPOINT : CNTR_MAT_FOURIER;
            if (mig) Hols::Sigma_Mig(tstp, G, Sigma, D0, D, Pi, D0_Pi, Pi_D0, g_elph_t, beta, h, SolverOrder, meth);
            else Hols::Sigma_uMig(tstp, G, D0, g_elph_t, Sigma);
            Hyb_Sig.set_timestep(tstp, Hyb);
            Hyb_Sig.incr_timestep(tstp, Sigma, 1.0);
            cntr::dyson_mat(G, 0.0, h0_imp_MF_t, Hyb_Sig, beta, SolverOrder, meth);
            G.smul(tstp, 0.5);
            G.incr_timestep(tstp, gtemp, 0.5);
            Hyb.set_timestep(tstp, G);
            Hyb.right_multiply(tstp, J_hop_t);
            Hyb.left_multiply(tstp, J_hop_t);
            err = cntr::distance_norm2(tstp, G, gtemp);
            if (err < MatsMaxErr && FIXPOINT) {
                conv = true;
                break;
            } else if (err < MatsMaxErr && !FIXPOINT) {
                FIXPOINT = true;
            }
            gtemp.set_timestep(tstp, G);
        }
        if (!conv) throw std::runtime_error("Matsubara iteration not converged");
        {
            cdmatrix rho_M = cdmatrix::Zero(1, 1);
            G.density_matrix(tstp, rho_M);
            rho_M(0, 0) = 2.0 * rho_M(0, 0).real();
            n_tot_t.set_value(tstp, rho_M);
            Hols::get_phonon_displace(tstp, Xph_t, n_tot_t, g_elph_t, D0, Phfreq_w0, SolverOrder, h);
            cdmatrix Xph_0(1, 1), g_elph_0(1, 1);
            Xph_t.get_value(tstp, Xph_0);
            g_elph_t.get_value(tstp, g_elph_0);
            MuChem = MuChem_MF + std::real(Xph_0(0, 0) * g_elph_0(0, 0));
            h0_imp(0, 0) = -MuChem;
            h0_imp_t.set_constant(h0_imp);
        }
        cntr::set_tk_from_mat(G, SolverOrder);
        cntr::set_tk_from_mat(Hyb, SolverOrder);

        // Bootstrapping
        conv = false;
        for (tstp = 0; tstp <= SolverOrder; tstp++) gtemp.set_timestep(tstp, G);
        for (int iter = 0; iter <= BootstrapMaxIter; iter++) {
            if (mig) Hols::Sigma_Mig(G, Sigma, D0, D, Pi, D0_Pi, Pi_D0, g_elph_t, beta, h, SolverOrder);
            else Hols::Sigma_uMig(G, D0, g_elph_t, Sigma, SolverOrder);
            for (tstp = 0; tstp <= SolverOrder; tstp++) {
                cdmatrix rho_M = cdmatrix::Zero(1, 1);
                G.density_matrix(tstp, rho_M);
                rho_M(0, 0) = 2.0 * rho_M(0, 0).real();
                n_tot_t.set_value(tstp, rho_M);
            }
            Hols::get_phonon_displace(Xph_t, n_tot_t, g_elph_t, D0, Phfreq_w0, SolverOrder, h);
            for (tstp = 0; tstp <= SolverOrder; tstp++) {
                cdmatrix Xph_tmp(1, 1), g_elph_tmp(1, 1), h0_tmp(1, 1);
                Xph_t.get_value(tstp, Xph_tmp);
                g_elph_t.get_value(tstp, g_elph_tmp);
                h0_tmp = h0_imp + Xph_tmp * g_elph_tmp;
                h0_imp_MF_t.set_value(tstp, h0_tmp);
                Hyb_Sig.set_timestep(tstp, Hyb);
                Hyb_Sig.incr_timestep(tstp, Sigma, 1.0);
            }
            cntr::dyson_start(G, 0.0, h0_imp_MF_t, Hyb_Sig, beta, h, SolverOrder);
            for (tstp = 0; tstp <= SolverOrder; tstp++) {
                Hyb.set_timestep(tstp, G);
                Hyb.right_multiply(tstp, J_hop_t);
                Hyb.left_multiply(tstp, J_hop_t);
            }
            err = 0.0;
            for (tstp = 0; tstp <= SolverOrder; tstp++) err += cntr::distance_norm2(tstp, G, gtemp);
            if (err < BootstrapMaxErr && iter > 2) {
                conv = true;
                break;
            }
            for (tstp = 0; tstp <= SolverOrder; tstp++) gtemp.set_timestep(tstp, G);
        }
        if (!conv) throw std::runtime_error("bootstrap iteration not converged");

        // Propagation
        for (tstp = SolverOrder + 1; tstp <= Nt; tstp++) {
            cntr::extrapolate_timestep(tstp - 1, G, SolverOrder);
            cntr::extrapolate_timestep(tstp - 1, Hyb, SolverOrder);
            for (int iter = 0; iter < CorrectorSteps; iter++) {
                cdmatrix rho_M = cdmatrix::Zero(1, 1);
                cdmatrix Xph_tmp(1, 1), g_elph_tmp(1, 1), h0_tmp(1, 1);
                if (mig) Hols::Sigma_Mig(tstp, G, Sigma, D0, D, Pi, D0_Pi, Pi_D0, g_elph_t, beta, h, SolverOrder);
                else Hols::Sigma_uMig(tstp, G, D0, g_elph_t, Sigma);
                G.density_matrix(tstp, rho_M);
                rho_M(0, 0) = 2.0 * rho_M(0, 0).real();
                n_tot_t.set_value(tstp, rho_M);
                Hols::get_phonon_displace(tstp, Xph_t, n_tot_t, g_elph_t, D0, Phfreq_w0, SolverOrder, h);
                Xph_t.get_value(tstp, Xph_tmp);
                g_elph_t.get_value(tstp, g_elph_tmp);
                h0_tmp = h0_imp + Xph_tmp * g_elph_tmp;
                h0_imp_MF_t.set_value(tstp, h0_tmp);
                Hyb_Sig.set_timestep(tstp, Hyb);
                Hyb_Sig.incr_timestep(tstp, Sigma, 1.0);
                cntr::dyson_timestep(tstp, G, 0.0, h0_imp_MF_t, Hyb_Sig, beta, h, SolverOrder);
                Hyb.set_timestep(tstp, G);
                Hyb.right_multiply(tstp, J_hop_t);
                Hyb.left_multiply(tstp, J_hop_t);
            }
        }

        // Energies and momentum
        cdmatrix Mat_tmp(1, 1);
        for (tstp = -1; tstp <= Nt; tstp++) {
            Mat_tmp(0, 0) = 4.0 * cntr::correlation_energy(tstp, G, Hyb, beta, h, SolverOrder);
            Ekin_t.set_value(tstp, Mat_tmp);
            cdmatrix Xph_tmp(1, 1), g_elph_tmp(1, 1), n_tmp(1, 1);
            Xph_t.get_value(tstp, Xph_tmp);
            g_elph_t.get_value(tstp, g_elph_tmp);
            n_tot_t.get_value(tstp, n_tmp);
            Mat_tmp(0, 0) = g_elph_tmp(0, 0) * Xph_tmp(0, 0) * n_tmp(0, 0);
            Enx_MF_t.set_value(tstp, Mat_tmp);
            Mat_tmp(0, 0) = 4.0 * cntr::correlation_energy(tstp, G, Sigma, beta, h, SolverOrder);
            Enx_corr_t.set_value(tstp, Mat_tmp);
        }
        Hols::get_phonon_momentum(Pph_t, n_tot_t, g_elph_t, D0, Phfreq_w0, SolverOrder, h);
        Hols::evaluate_phonon_energy_cl(Eph_cl_t, Xph_t, Pph_t, Phfreq_w0);
        if (mig) {
            Hols::evaluate_phonon_energy_qu(Eph_qu_t, D, Pi, SolverOrder, beta, h, Phfreq_w0);
        } else {
            for (tstp = -1; tstp <= Nt; tstp++) Pi.set_timestep_zero(tstp);
            Hols::evaluate_phonon_energy_qu(Eph_qu_t, D0, Pi, SolverOrder, beta, h, Phfreq_w0);
        }

        // Output
        std::string base = "holstein_bethe_" + std::string(mig ? "migdal" : "umig") + "_" + profile;
        kbref::Metadata meta = kbref::p5_metadata(
            "holstein_bethe_text", base, "holstein_bethe", approx, profile, Nt, Ntau, h, beta,
            SolverOrder, MatsMaxIter, BootstrapMaxIter, CorrectorSteps, 1,
            "Hopping=1,El_Ph_g=0.5,Phfreq_w0=1,MuChem_MF=0.5,dHopping=sin2(1,0.6)");
        FILE *fp = kbref::open_output(dir, base + ".dat");
        kbref::write_metadata(fp, "observable", meta);
        for (tstp = 0; tstp <= Nt; tstp++) {
            cdmatrix rho(1, 1), XX(1, 1), x(1, 1);
            G.density_matrix(tstp, rho);
            if (mig) D.get_les(tstp, tstp, XX);
            else D0.get_les(tstp, tstp, XX);
            XX(0, 0) *= cplx(0.0, 1.0);
            kbref::write_observable_row(fp, meta, "rho", tstp, 0, 0, rho(0, 0));
            double etot = 0.0;
            struct { const char *name; CFUNC *f; bool tot; } cols[] = {
                {"Xph", &Xph_t, false}, {"Pph", &Pph_t, false}, {"Ekin", &Ekin_t, true},
                {"Enx_MF", &Enx_MF_t, true}, {"Enx_corr", &Enx_corr_t, true},
                {"Eph_cl", &Eph_cl_t, true}, {"Eph_corr", &Eph_qu_t, true}};
            kbref::write_observable_row(fp, meta, "XX", tstp, -1, -1, cplx(XX(0, 0).real(), 0.0));
            for (auto &c : cols) {
                c.f->get_value(tstp, x);
                if (c.tot) etot += x(0, 0).real();
                kbref::write_observable_row(fp, meta, c.name, tstp, -1, -1, cplx(x(0, 0).real(), 0.0));
            }
            kbref::write_observable_row(fp, meta, "Etot", tstp, -1, -1, cplx(etot, 0.0));
        }
        std::fclose(fp);
    } catch (std::exception &e) {
        std::cerr << "holstein_bethe_text: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
