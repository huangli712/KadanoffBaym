// Phase 5 reference driver: Holstein model on the Bethe lattice in the
// Nambu (superconducting) form, Migdal and uMig (see
// .omo/plans/05-phase5-examples.md, P5-W6-T4).
//
// Usage: holstein_nambu_text <Migdal|uMig> <quick|release> <output-dir>
//
// Transcription of NESSi examples/programs/Holstein_bethe_Nambu_Migdal.cpp
// and Holstein_bethe_Nambu_uMig.cpp (no input file, no HDF5); the
// self-energies and phonon routines are taken unchanged from
// Holstein_impurity_impl.cpp and Holstein_utils_impl.cpp. The quick
// profile uses El_Ph_g = 1.0 (superconducting on the quick grid), the
// release profile the demo value 0.5. The drives are read from
// tutor/holstein_bethe/dHopping_nambu_<profile>.dat (sc_field = 0).
//
// Output: holstein_nambu_<approx>_<profile>.dat (observable at the NESSi
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
        if (argc != 4) throw std::runtime_error("usage: holstein_nambu_text <Migdal|uMig> <quick|release> <dir>");
        std::string approx = argv[1], profile = argv[2], dir = argv[3];
        if (approx != "Migdal" && approx != "uMig") throw std::runtime_error("unknown approximation " + approx);
        bool mig = approx == "Migdal";
        int Nt, Ntau, SolverOrder, MatsMaxIter, BootstrapMaxIter, CorrectorSteps;
        double El_Ph_g;
        if (profile == "quick") {
            Nt = 6; Ntau = 32; SolverOrder = 3; MatsMaxIter = 300; BootstrapMaxIter = 60; CorrectorSteps = 3;
            El_Ph_g = 1.0;
        } else if (profile == "release") {
            Nt = 400; Ntau = 500; SolverOrder = 5; MatsMaxIter = 800; BootstrapMaxIter = 20; CorrectorSteps = 5;
            El_Ph_g = 0.5;
        } else {
            throw std::runtime_error("unknown profile " + profile);
        }
        double Hopping = 1.0, Phfreq_w0 = 1.0, MuChem_MF = 0.5, beta = 10.0, h = 0.03;
        double MatsMaxErr = 1e-7, BootstrapMaxErr = 1e-7, err;
        std::vector<double> dHopping = kbref::make_drive(1.0, 0.6, h, Nt);
        std::vector<double> sc_field(Nt + 2, 0.0);
        int tstp, Norb = 2;
        cdmatrix sig3 = cdmatrix::Zero(2, 2);
        sig3(0, 0) = 1.0;
        sig3(1, 1) = -1.0;
        cdmatrix sig1 = cdmatrix::Zero(2, 2);
        sig1(0, 1) = 1.0;
        sig1(1, 0) = 1.0;

        GREEN G, Sigma, Hyb, gtemp, Hyb_Sig, D, Pi, D0, D0_Pi, Pi_D0;
        CFUNC g_elph_t, J_hop_t, h0_imp_t, Ekin_t, Enx_MF_t, Enx_corr_t, Eph_t;
        cdmatrix h0_imp;
        g_elph_t = CFUNC(Nt, Norb);
        J_hop_t = CFUNC(Nt, Norb);
        h0_imp = cdmatrix::Zero(Norb, Norb);
        h0_imp_t = CFUNC(Nt, Norb);
        G = GREEN(Nt, Ntau, Norb, FERMION);
        Hyb = GREEN(Nt, Ntau, Norb, FERMION);
        Sigma = GREEN(Nt, Ntau, Norb, FERMION);
        Hyb_Sig = GREEN(Nt, Ntau, Norb, FERMION);
        D = GREEN(Nt, Ntau, 1, BOSON);
        Pi = GREEN(Nt, Ntau, 1, BOSON);
        D0 = GREEN(Nt, Ntau, 1, BOSON);
        D0_Pi = GREEN(Nt, Ntau, 1, BOSON);
        Pi_D0 = GREEN(Nt, Ntau, 1, BOSON);
        Ekin_t = CFUNC(Nt, 1);
        Enx_MF_t = CFUNC(Nt, 1);
        Enx_corr_t = CFUNC(Nt, 1);
        Eph_t = CFUNC(Nt, 1);
        cdmatrix zero1 = cdmatrix::Zero(1, 1);
        Enx_MF_t.set_constant(zero1);

        g_elph_t.set_constant(El_Ph_g * sig3);
        J_hop_t.set_constant(Hopping * sig3);
        h0_imp = -MuChem_MF * sig3;
        h0_imp_t.set_constant(h0_imp);
        for (int it = -1; it <= Nt; it++) J_hop_t.set_value(it, (Hopping + dHopping[it + 1]) * sig3);
        for (int it = -1; it <= Nt; it++) h0_imp_t.set_value(it, h0_imp + sc_field[it + 1] * sig1);
        cntr::green_equilibrium_mat_bethe(G, beta);
        tstp = -1;
        Hyb.set_timestep(tstp, G);
        Hyb.right_multiply(tstp, J_hop_t);
        Hyb.left_multiply(tstp, J_hop_t);
        cntr::green_single_pole_XX(D0, Phfreq_w0, beta, h);
        for (int it = -1; it <= Nt; it++) D0.smul(it, 2.0);

        // Matsubara
        bool conv = false;
        gtemp = GREEN(SolverOrder, Ntau, Norb, -1);
        gtemp.set_timestep(tstp, G);
        bool FIXPOINT = false;
        for (int iter = 0; iter <= MatsMaxIter; iter++) {
            int meth = FIXPOINT ? CNTR_MAT_FIXPOINT : CNTR_MAT_FOURIER;
            if (mig) Hols::Sigma_Mig_sc(tstp, G, Sigma, D0, D, Pi, D0_Pi, Pi_D0, g_elph_t, beta, h, SolverOrder, meth);
            else Hols::Sigma_uMig_sc(tstp, G, D0, g_elph_t, Sigma);
            Hyb_Sig.set_timestep(tstp, Hyb);
            Hyb_Sig.incr_timestep(tstp, Sigma, 1.0);
            double SC_seed = (iter < 5 ? 0.1 : 0.0);
            cdmatrix h0_imp_sc = h0_imp + SC_seed * sig1;
            h0_imp_t.set_value(tstp, h0_imp_sc);
            cntr::dyson_mat(G, 0.0, h0_imp_t, Hyb_Sig, beta, SolverOrder, meth);
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
        cntr::set_tk_from_mat(G, SolverOrder);
        cntr::set_tk_from_mat(Hyb, SolverOrder);

        // Bootstrapping
        conv = false;
        for (tstp = 0; tstp <= SolverOrder; tstp++) gtemp.set_timestep(tstp, G);
        for (int iter = 0; iter <= BootstrapMaxIter; iter++) {
            if (mig) Hols::Sigma_Mig_sc(G, Sigma, D0, D, Pi, D0_Pi, Pi_D0, g_elph_t, beta, h, SolverOrder);
            else Hols::Sigma_uMig_sc(G, D0, g_elph_t, Sigma, SolverOrder);
            for (tstp = 0; tstp <= SolverOrder; tstp++) {
                Hyb_Sig.set_timestep(tstp, Hyb);
                Hyb_Sig.incr_timestep(tstp, Sigma, 1.0);
            }
            cntr::dyson_start(G, 0.0, h0_imp_t, Hyb_Sig, beta, h, SolverOrder);
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
                if (mig) Hols::Sigma_Mig_sc(tstp, G, Sigma, D0, D, Pi, D0_Pi, Pi_D0, g_elph_t, beta, h, SolverOrder);
                else Hols::Sigma_uMig_sc(tstp, G, D0, g_elph_t, Sigma);
                Hyb_Sig.set_timestep(tstp, Hyb);
                Hyb_Sig.incr_timestep(tstp, Sigma, 1.0);
                cntr::dyson_timestep(tstp, G, 0.0, h0_imp_t, Hyb_Sig, beta, h, SolverOrder);
                Hyb.set_timestep(tstp, G);
                Hyb.right_multiply(tstp, J_hop_t);
                Hyb.left_multiply(tstp, J_hop_t);
            }
        }
        if (!mig) {
            for (tstp = -1; tstp <= Nt; tstp++) D.set_timestep(tstp, D0);
        }

        // Energies
        cdmatrix Mat_tmp(1, 1);
        for (tstp = -1; tstp <= Nt; tstp++) {
            Mat_tmp(0, 0) = 2.0 * cntr::correlation_energy(tstp, G, Hyb, beta, h, SolverOrder);
            Ekin_t.set_value(tstp, Mat_tmp);
            Mat_tmp(0, 0) = 2.0 * cntr::correlation_energy(tstp, G, Sigma, beta, h, SolverOrder);
            Enx_corr_t.set_value(tstp, Mat_tmp);
        }
        if (!mig) {
            for (tstp = -1; tstp <= Nt; tstp++) Pi.set_timestep_zero(tstp);
        }
        Hols::evaluate_phonon_energy_qu(Eph_t, D, Pi, SolverOrder, beta, h, Phfreq_w0);

        // Output
        std::string base = "holstein_nambu_" + std::string(mig ? "migdal" : "umig") + "_" + profile;
        kbref::Metadata meta = kbref::p5_metadata(
            "holstein_nambu_text", base, "holstein_nambu", approx, profile, Nt, Ntau, h, beta,
            SolverOrder, MatsMaxIter, BootstrapMaxIter, CorrectorSteps, 2,
            "Hopping=1,El_Ph_g=" + kbref::fmt_double(El_Ph_g) +
                ",Phfreq_w0=1,MuChem_MF=0.5,sc_seed=0.1x5,dHopping=sin2(1,0.6)");
        FILE *fp = kbref::open_output(dir, base + ".dat");
        kbref::write_metadata(fp, "observable", meta);
        for (tstp = 0; tstp <= Nt; tstp++) {
            cdmatrix rho(Norb, Norb), XX(1, 1), ekin(1, 1), emf(1, 1), ecorr(1, 1), eph(1, 1);
            G.density_matrix(tstp, rho);
            D.get_les(tstp, tstp, XX);
            XX(0, 0) *= cplx(0.0, 1.0);
            Ekin_t.get_value(tstp, ekin);
            Enx_MF_t.get_value(tstp, emf);
            Enx_corr_t.get_value(tstp, ecorr);
            Eph_t.get_value(tstp, eph);
            for (int a = 0; a < Norb; a++)
                for (int b = 0; b < Norb; b++) kbref::write_observable_row(fp, meta, "rho", tstp, a, b, rho(a, b));
            double etot = (ekin(0, 0) + emf(0, 0) + ecorr(0, 0) + eph(0, 0)).real();
            kbref::write_observable_row(fp, meta, "XX", tstp, -1, -1, cplx(XX(0, 0).real(), 0.0));
            kbref::write_observable_row(fp, meta, "Ekin", tstp, -1, -1, cplx(ekin(0, 0).real(), 0.0));
            kbref::write_observable_row(fp, meta, "Enx_MF", tstp, -1, -1, cplx(emf(0, 0).real(), 0.0));
            kbref::write_observable_row(fp, meta, "Enx_corr", tstp, -1, -1, cplx(ecorr(0, 0).real(), 0.0));
            kbref::write_observable_row(fp, meta, "Eph", tstp, -1, -1, cplx(eph(0, 0).real(), 0.0));
            kbref::write_observable_row(fp, meta, "Etot", tstp, -1, -1, cplx(etot, 0.0));
        }
        std::fclose(fp);
    } catch (std::exception &e) {
        std::cerr << "holstein_nambu_text: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
