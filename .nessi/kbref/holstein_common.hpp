// Shared helpers of the Phase 5 Holstein reference drivers
// (holstein_bethe_text.cpp, holstein_nambu_text.cpp,
// holstein_impurity_text.cpp).

#ifndef KBREF_HOLSTEIN_COMMON_HPP
#define KBREF_HOLSTEIN_COMMON_HPP

#include <cstdio>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "reference_io.hpp"

namespace kbref {

// Sin^2 drive pulse A sin^2(pi t / tend) of the NESSi Holstein demos
// (demo_Holstein*.py), evaluated at t = k h for k = -1..nt (NESSi tstp).
// It replaces the previously distributed data files tutor/*/d*.dat, in
// lockstep with the sin2 drive mode of tutor/common/example_config.jl.
// The multiplication order follows the Python reference.
inline std::vector<double> make_drive(double amplitude, double tend, double h, int nt) {
    std::vector<double> v(nt + 2, 0.0);
    for (int k = 1; k <= nt; ++k) {
        double t = k * h;
        if (t > 0.0 && t < tend)
            v[k + 1] = amplitude * std::sin(t * M_PI / tend) * std::sin(t * M_PI / tend);
    }
    return v;
}

// P5 metadata
inline Metadata p5_metadata(const std::string &generator, const std::string &case_set,
                            const std::string &model, const std::string &approx,
                            const std::string &profile, int nt, int ntau, double h, double beta,
                            int k, int mats, int boot, int corr, int size,
                            const std::string &params) {
    Metadata meta;
    meta.push_back(std::make_pair("generator", generator));
    meta.push_back(std::make_pair("case_set", case_set));
    meta.push_back(std::make_pair("model", model));
    meta.push_back(std::make_pair("approximation", approx));
    meta.push_back(std::make_pair("profile", profile));
    meta.push_back(std::make_pair("nt", std::to_string(nt)));
    meta.push_back(std::make_pair("ntau", std::to_string(ntau)));
    meta.push_back(std::make_pair("h", fmt_double(h)));
    meta.push_back(std::make_pair("beta", fmt_double(beta)));
    meta.push_back(std::make_pair("solve_order", std::to_string(k)));
    meta.push_back(std::make_pair("mats_max_iter", std::to_string(mats)));
    meta.push_back(std::make_pair("bootstrap_max_iter", std::to_string(boot)));
    meta.push_back(std::make_pair("corrector_steps", std::to_string(corr)));
    meta.push_back(std::make_pair("size", std::to_string(size)));
    meta.push_back(std::make_pair("params", params));
    return meta;
}

} // namespace kbref

#endif
