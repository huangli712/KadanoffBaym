// kbref generator: NESSi examples/programs/gw.cpp (with gw_latt_impl.cpp,
// gw_kpoints_impl.cpp, gw_selfene_impl.cpp) in one process, without MPI
// and HDF5.
//
// Usage: p8_gw_ref <output-dir>
// Output: <output-dir>/p8_gw.dat (kbref-v1, data_kind series, case_set gw).
//
// The MPI snapshot (distributed_timestep_array, gathered at the start of
// every iteration) is reproduced by copies of the time step of all k
// points. The only change to the physics of gw.cpp (user decision,
// 2026-11-10): the k-point density matrices used for the Hartree/Fock
// terms, the kinetic energy, and the current are replaced by their
// hermitian part (rho + rho^dagger)/2, as in tutor/gw/gw.jl (during the
// bootstrapping G^<(t,t) is anti-hermitian only up to the discretization
// error). rho_ of the k points (interaction energy) is not changed.
//
// Parameters (the same as gw_toml() in tutor/gw/validate.jl, NESSi
// conventions): Nk = 4, HoppingT = 1, HubbardU = 1, V = 0.25, MuChem = 0,
// beta = 5, Nt = 7, Ntau = 30, h = 0.02, SolverOrder = 5, MatsMaxIter =
// 60, MatsMaxErr = 1e-9, BootstrapMaxIter = 40, BootstrapMaxErr = 1e-10,
// TimeMaxErr = 1e-10, CorrectorSteps = 6, E(t) = E0 exp(-((t-t0)/w)^2)
// sin(omega (t-t0)) with E0 = 1, omega = 2, t0 = 0.05, w = 0.1.
//
// Records (step = NESSi time step 0..Nt, k = k point):
//   afield, density (local, one spin), current_nessi, ekin, eint
//   rho, sigma_hf (Hartree + Fock)                          per k
//   g_ret, g_less, w_ret, w_less, sigma_gw_ret, sigma_gw_less,
//   p_ret, p_less: X^R(t_step, 0) and X^<(0, t_step)       per k
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include <vector>

#include "cntr/cntr.hpp"
#include "reference_io.hpp"

typedef std::complex<double> cplx;
typedef cntr::herm_matrix<double> KGreen;
typedef cntr::herm_matrix_timestep<double> KGreenTstp;
typedef cntr::herm_matrix_timestep_view<double> KGreenView;
typedef cntr::function<double> KFunc;

static const int Nk = 4, Nt = 7, Ntau = 30, SolverOrder = 5, Norb = 1;
static const int MatsMaxIter = 60, BootstrapMaxIter = 40, CorrectorSteps = 6;
static const double HoppingT = 1.0, HubbardU = 1.0, Vnn = 0.25, MuChem = 0.0, beta = 5.0;
static const double h = 0.02, MatsMaxErr = 1e-9, BootstrapMaxErr = 1e-10, TimeMaxErr = 1e-10;
static const double E0 = 1.0, omega = 2.0, t0 = 0.05, width = 0.1;

// Lattice (gw_latt_impl.cpp)
struct Lattice {
    int nk, G;
    std::vector<double> kpoints, kweight;
    KFunc A;
    Lattice(std::vector<double> &efield) : A(Nt, 1) {
        nk = Nk;
        G = nk / 2;
        kpoints.resize(nk);
        kweight.resize(nk);
        for (int i = 0; i < nk; i++) {
            kpoints[i] = (i - G) * 2.0 * M_PI / nk;
            kweight[i] = 1.0 / nk;
        }
        int kt1 = (Nt >= SolverOrder ? SolverOrder : Nt);
        cdmatrix At(1, 1);
        At(0, 0) = 0.0;
        A.set_value(-1, At);
        for (int tstp = 0; tstp <= Nt; tstp++) {
            At(0, 0) = 0.0;
            int n1 = (tstp < kt1 ? kt1 : tstp);
            for (int n = 0; n <= n1; n++)
                At(0, 0) += integration::I<double>(kt1).gregory_weights(tstp, n) * efield[n + 1];
            At(0, 0) *= (-h);
            A.set_value(tstp, At);
        }
    }
    int add_kpoints(int k1, int s1, int k2, int s2) {
        int k12 = s1 * (k1 - G) + s2 * (k2 - G) + G;
        while (k12 < 0) k12 += nk;
        while (k12 >= nk) k12 -= nk;
        return k12;
    }
    double afield(int tstp) {
        cdmatrix a;
        A.get_value(tstp, a);
        return std::real(a(0, 0));
    }
    cdmatrix hk(int tstp, double kk) {
        cdmatrix m(1, 1);
        m(0, 0) = -2.0 * HoppingT * std::cos(kk - afield(tstp));
        return m;
    }
    cdmatrix vk(double kk) {
        cdmatrix m(1, 1);
        m(0, 0) = 2.0 * HoppingT * std::sin(kk);
        return m;
    }
    cdmatrix V(double q) {
        cdmatrix m(1, 1);
        m(0, 0) = HubbardU + 2.0 * Vnn * std::cos(q);
        return m;
    }
};

// k point (gw_kpoints_impl.cpp)
struct KPoint {
    double kk;
    KFunc rho, SH, SF, hk, hkeff, vertex;
    KGreen G, Sigma, P, PV, VP, W;
    KPoint(double k, Lattice &lat)
        : kk(k), rho(Nt, Norb), SH(Nt, Norb), SF(Nt, Norb), hk(Nt, Norb), hkeff(Nt, Norb),
          vertex(Nt, Norb), G(Nt, Ntau, Norb, FERMION), Sigma(Nt, Ntau, Norb, FERMION),
          P(Nt, Ntau, Norb, BOSON), PV(Nt, Ntau, Norb, BOSON), VP(Nt, Ntau, Norb, BOSON),
          W(Nt, Ntau, Norb, BOSON) {
        for (int t = -1; t <= Nt; t++) {
            cdmatrix m = lat.hk(t, kk);
            hk.set_value(t, m);
            m = lat.V(kk);
            vertex.set_value(t, m);
        }
    }
    void step_W(int tstp) {
        int n1 = (tstp == -1 || tstp > SolverOrder ? tstp : 0);
        int n2 = (tstp == -1 || tstp > SolverOrder ? tstp : SolverOrder);
        for (int n = n1; n <= n2; n++) {
            PV.set_timestep(n, P);
            PV.right_multiply(n, vertex);
            PV.smul(n, -1.0);
            VP.set_timestep(n, P);
            VP.left_multiply(n, vertex);
            VP.smul(n, -1.0);
        }
        KFunc Fsin(Nt, Norb);
        Fsin.set_zero();
        KGreen Q(Nt, Ntau, Norb, BOSON);
        cntr::vie2_timestep_sin(tstp, W, vertex, VP, PV, Fsin, Q, vertex, beta, h, SolverOrder);
    }
    void step_dyson(int tstp, int iter, Lattice &lat) {
        int n1 = (tstp == -1 || tstp > SolverOrder ? tstp : 0);
        int n2 = (tstp == -1 || tstp > SolverOrder ? tstp : SolverOrder);
        for (int n = n1; n <= n2; n++) {
            cdmatrix a, b, c;
            cdmatrix m = lat.hk(n, kk);
            hk.set_value(n, m);
            SH.get_value(n, a);
            SF.get_value(n, b);
            hk.get_value(n, c);
            cdmatrix s = a + b + c;
            hkeff.set_value(n, s);
        }
        if (tstp == -1 && iter == 1) {
            KGreenTstp tmp(-1, Ntau, Norb, G.sig());
            cdmatrix hk0;
            hkeff.get_value(-1, hk0);
            cntr::green_from_H(-1, tmp, MuChem, hk0, beta, h);
            G.set_timestep(-1, tmp);
        } else if (tstp == -1) {
            cntr::dyson_mat(G, MuChem, hkeff, Sigma, beta, SolverOrder);
        } else if (tstp <= SolverOrder) {
            cntr::dyson_start(G, MuChem, hkeff, Sigma, beta, h, SolverOrder);
        } else {
            cntr::dyson_timestep(tstp, G, MuChem, hkeff, Sigma, beta, h, SolverOrder);
        }
        for (int n = n1; n <= n2; n++) {
            cdmatrix m(Norb, Norb);
            G.density_matrix(n, m);
            rho.set_value(n, m);
        }
    }
    double step_dyson_with_error(int tstp, int iter, Lattice &lat) {
        KGreenTstp g;
        G.get_timestep(tstp, g);
        step_dyson(tstp, iter, lat);
        return cntr::distance_norm2(tstp, g, G);
    }
    double step_W_with_error(int tstp) {
        KGreenTstp w(tstp, Ntau, Norb, Norb, BOSON);
        W.get_timestep(tstp, w);
        step_W(tstp);
        return cntr::distance_norm2(tstp, w, W);
    }
};

// Snapshot of the time step of all k points (the MPI gather of gw.cpp)
struct Snapshot {
    std::vector<KGreenTstp> g, w;
    Snapshot(int tstp, std::vector<KPoint> &kps) {
        g.resize(Nk);
        w.resize(Nk);
        for (int k = 0; k < Nk; k++) {
            g[k] = KGreenTstp(tstp, Ntau, Norb, FERMION);
            w[k] = KGreenTstp(tstp, Ntau, Norb, BOSON);
            kps[k].G.get_timestep(tstp, g[k]);
            kps[k].W.get_timestep(tstp, w[k]);
        }
    }
};

// Hermitian part of the density matrices of the time step (set_density_k
// of gw_selfene_impl.cpp, see the header)
static void set_density_k(int tstp, std::vector<KPoint> &kps, std::vector<KFunc> &dens) {
    for (int k = 0; k < Nk; k++) {
        cdmatrix m(Norb, Norb);
        kps[k].G.density_matrix(tstp, m);
        cdmatrix s = 0.5 * (m + m.adjoint());
        dens[k].set_value(tstp, s);
    }
}

// One iteration at the time step tstp (the body of the loops of gw.cpp)
static void gw_iteration(int tstp, int iter, Lattice &lat, std::vector<KPoint> &kps,
                         std::vector<KFunc> &dens, double &err_ele, double &err_bos) {
    Snapshot snap(tstp, kps);
    std::vector<KGreenView> gv, wv;
    for (int k = 0; k < Nk; k++) {
        gv.push_back(KGreenView(tstp, snap.g[k]));
        wv.push_back(KGreenView(tstp, snap.w[k]));
    }
    set_density_k(tstp, kps, dens);
    // Hartree, Fock, GW
    cdmatrix vnl = lat.V(lat.kpoints[lat.G]);
    cdmatrix vloc = cdmatrix::Constant(1, 1, HubbardU);
    vnl -= vloc;
    for (int kk = 0; kk < Nk; kk++) {
        cdmatrix sh = cdmatrix::Zero(1, 1), sf = cdmatrix::Zero(1, 1), r;
        for (int k = 0; k < Nk; k++) {
            dens[k].get_value(tstp, r);
            sh(0, 0) += vloc(0, 0) * r(0, 0) * lat.kweight[k];
            sh(0, 0) += vnl(0, 0) * r(0, 0) * lat.kweight[k] * 2.0;
        }
        kps[kk].SH.set_value(tstp, sh);
        for (int q = 0; q < Nk; q++) {
            int kq = lat.add_kpoints(kk, 1, q, -1);
            dens[kq].get_value(tstp, r);
            cdmatrix Vq = lat.V(lat.kpoints[q]) - vloc;
            sf(0, 0) -= Vq(0, 0) * r(0, 0) * lat.kweight[q];
        }
        kps[kk].SF.set_value(tstp, sf);
        KGreenTstp stmp(tstp, Ntau, Norb, FERMION);
        kps[kk].Sigma.set_timestep_zero(tstp);
        for (int q = 0; q < Nk; q++) {
            int kq = lat.add_kpoints(kk, 1, q, -1);
            stmp.clear();
            cntr::Bubble2(tstp, stmp, 0, 0, gv[kq], gv[kq], 0, 0, wv[q], wv[q], 0, 0);
            kps[kk].Sigma.incr_timestep(tstp, stmp, lat.kweight[q]);
        }
    }
    // Dyson, polarization, W
    err_ele = 0.0;
    err_bos = 0.0;
    for (int qq = 0; qq < Nk; qq++) {
        err_ele += kps[qq].step_dyson_with_error(tstp, iter, lat);
        KGreenTstp ptmp(tstp, Ntau, Norb, BOSON);
        kps[qq].P.set_timestep_zero(tstp);
        for (int kk = 0; kk < Nk; kk++) {
            int kq = lat.add_kpoints(kk, 1, qq, -1);
            ptmp.clear();
            cntr::Bubble1(tstp, ptmp, 0, 0, gv[kq], gv[kq], 0, 0, gv[kk], gv[kk], 0, 0);
            kps[qq].P.incr_timestep(tstp, ptmp, -lat.kweight[kk]);
        }
        err_bos += kps[qq].step_W_with_error(tstp);
    }
    // gw.cpp calls set_density_k again on the gathered (pre-Dyson) time
    // step, which leaves density_k unchanged
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: p8_gw_ref <output-dir>\n");
        return 1;
    }
    std::vector<double> efield(Nt + 2, 0.0);
    for (int n = 0; n <= Nt; n++) {
        double t = n * h;
        efield[n + 1] = E0 * std::exp(-std::pow((t - t0) / width, 2)) * std::sin(omega * (t - t0));
    }
    Lattice lat(efield);
    std::vector<KPoint> kps;
    std::vector<KFunc> dens;
    for (int k = 0; k < Nk; k++) {
        kps.push_back(KPoint(lat.kpoints[k], lat));
        dens.push_back(KFunc(Nt, Norb));
    }

    // Noninteracting start
    for (int k = 0; k < Nk; k++) {
        cdmatrix hk0;
        kps[k].hkeff = kps[k].hk;
        kps[k].hkeff.get_value(-1, hk0);
        cntr::green_from_H(kps[k].G, MuChem, hk0, beta, h);
    }

    // Matsubara iterations
    bool converged = false;
    for (int iter = 0; iter <= MatsMaxIter; iter++) {
        double e1, e2;
        gw_iteration(-1, iter, lat, kps, dens, e1, e2);
        if (e1 < MatsMaxErr && e2 < MatsMaxErr) {
            converged = true;
            break;
        }
    }
    if (!converged) {
        std::fprintf(stderr, "Matsubara iteration not converged\n");
        return 1;
    }

    // Bootstrapping
    for (int tstp = 0; tstp <= SolverOrder; tstp++) {
        int ord = (SolverOrder < tstp - 1 ? SolverOrder : tstp - 1);
        for (int k = 0; k < Nk; k++) {
            if (tstp - 1 == -1) {
                cntr::set_t0_from_mat(kps[k].G);
                cntr::set_t0_from_mat(kps[k].W);
            } else {
                cntr::extrapolate_timestep(tstp - 1, kps[k].G, ord);
                cntr::extrapolate_timestep(tstp - 1, kps[k].W, ord);
            }
        }
        converged = false;
        for (int iter = 0; iter <= BootstrapMaxIter; iter++) {
            double e1 = 0.0, e2 = 0.0;
            for (int n = 0; n <= tstp; n++) gw_iteration(n, iter, lat, kps, dens, e1, e2);
            if (e1 < BootstrapMaxErr && e2 < BootstrapMaxErr && iter > 2) {
                converged = true;
                break;
            }
        }
        if (!converged) {
            std::fprintf(stderr, "bootstrapping not converged at %d\n", tstp);
            return 1;
        }
    }

    // Time stepping
    for (int tstp = SolverOrder + 1; tstp <= Nt; tstp++) {
        for (int k = 0; k < Nk; k++) {
            cntr::extrapolate_timestep(tstp - 1, kps[k].G, SolverOrder);
            cntr::extrapolate_timestep(tstp - 1, kps[k].W, SolverOrder);
        }
        for (int iter = 0; iter < CorrectorSteps; iter++) {
            double e1, e2;
            gw_iteration(tstp, iter, lat, kps, dens, e1, e2);
            if (e1 < TimeMaxErr && e2 < TimeMaxErr && iter > 1) break;
        }
    }

    // Output
    FILE *fp = kbref::open_output(argv[1], "p8_gw.dat");
    kbref::Metadata meta;
    meta.push_back(std::make_pair("generator", std::string("p8_gw_ref")));
    meta.push_back(std::make_pair("case_set", std::string("gw")));
    meta.push_back(std::make_pair(
        "components",
        std::string("afield,density,current_nessi,ekin,eint,rho,sigma_hf,g_ret,g_less,"
                    "w_ret,w_less,sigma_gw_ret,sigma_gw_less,p_ret,p_less")));
    meta.push_back(std::make_pair("nsteps", std::to_string(Nt + 1)));
    meta.push_back(std::make_pair("nk", std::to_string(Nk)));
    meta.push_back(std::make_pair("size", std::string("1")));
    meta.push_back(std::make_pair("source", std::string("gw.cpp (single process, hermitian rho)")));
    kbref::write_metadata(fp, "series", meta);
    kbref::SeriesSchema s = kbref::series_schema(meta);
    for (int tstp = 0; tstp <= Nt; tstp++) {
        double ekin = 0.0, curr = 0.0, eint = 0.0, nloc = 0.0;
        for (int k = 0; k < Nk; k++) {
            cdmatrix r, a, b, rr;
            dens[k].get_value(tstp, r);
            double w = lat.kweight[k];
            ekin += std::real((lat.hk(tstp, lat.kpoints[k]) * r).trace()) * w;
            curr += std::real((lat.vk(lat.kpoints[k]) * r).trace()) * w;
            nloc += std::real(r.trace()) * w;
            kps[k].rho.get_value(tstp, rr);
            kps[k].SH.get_value(tstp, a);
            kps[k].SF.get_value(tstp, b);
            eint += std::real(((a + b) * rr).trace()) * w;
            eint += cntr::correlation_energy(tstp, kps[k].G, kps[k].Sigma, beta, h, SolverOrder) * w;
            kbref::write_series_row(fp, s, "rho", tstp, k, 0, 0, rr(0, 0));
            kbref::write_series_row(fp, s, "sigma_hf", tstp, k, 0, 0, a(0, 0) + b(0, 0));
            cplx x;
            kps[k].G.get_ret(tstp, 0, x);
            kbref::write_series_row(fp, s, "g_ret", tstp, k, 0, 0, x);
            kps[k].G.get_les(0, tstp, x);
            kbref::write_series_row(fp, s, "g_less", tstp, k, 0, 0, x);
            kps[k].W.get_ret(tstp, 0, x);
            kbref::write_series_row(fp, s, "w_ret", tstp, k, 0, 0, x);
            kps[k].W.get_les(0, tstp, x);
            kbref::write_series_row(fp, s, "w_less", tstp, k, 0, 0, x);
            kps[k].Sigma.get_ret(tstp, 0, x);
            kbref::write_series_row(fp, s, "sigma_gw_ret", tstp, k, 0, 0, x);
            kps[k].Sigma.get_les(0, tstp, x);
            kbref::write_series_row(fp, s, "sigma_gw_less", tstp, k, 0, 0, x);
            kps[k].P.get_ret(tstp, 0, x);
            kbref::write_series_row(fp, s, "p_ret", tstp, k, 0, 0, x);
            kps[k].P.get_les(0, tstp, x);
            kbref::write_series_row(fp, s, "p_less", tstp, k, 0, 0, x);
        }
        kbref::write_series_row(fp, s, "afield", tstp, -1, -1, -1, cplx(lat.afield(tstp), 0.0));
        kbref::write_series_row(fp, s, "density", tstp, -1, -1, -1, cplx(nloc, 0.0));
        kbref::write_series_row(fp, s, "current_nessi", tstp, -1, -1, -1, cplx(curr, 0.0));
        kbref::write_series_row(fp, s, "ekin", tstp, -1, -1, -1, cplx(ekin, 0.0));
        kbref::write_series_row(fp, s, "eint", tstp, -1, -1, -1, cplx(eint, 0.0));
    }
    std::fclose(fp);
    return 0;
}
