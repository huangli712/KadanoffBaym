// kbref generator: NESSi integration weights for k = 1..5.
//
// Usage: weights <output-dir>
// Output: <output-dir>/weights_k1_k5.dat (kbref-v1, data_kind weights)
//
// Component encoding (columns i j a b, unused columns are -1):
//   greg  k n j -1   gregory_weights(n, j),     n = 0..2k+3, j = 0..max(n,k)
//   bd    k l -1 -1  bd_weights(l),             l = 0..k+1
//   pint  k i j l    poly_integration(i, j, l), i, j, l = 0..k
//   pdiff k i l -1   poly_differentiation(i, l), i, l = 0..k
//   rcorr k m j l    rcorr(m, j, l),            k = 2..5, m = 1..k-1
// NESSi does not allocate rcorr for k = 1 and has no valid m = 0 entry.
#include <cstdio>
#include <string>

#include "cntr/integration.hpp"
#include "reference_io.hpp"

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <output-dir>\n", argv[0]);
        return 1;
    }
    const std::string outdir(argv[1]);
    FILE *fp = kbref::open_output(outdir, "weights_k1_k5.dat");

    kbref::Metadata meta;
    meta.push_back(std::make_pair("case", "weights_k1_k5"));
    meta.push_back(std::make_pair("kmin", "1"));
    meta.push_back(std::make_pair("kmax", "5"));
    kbref::write_metadata(fp, "weights", meta);

    // Rows are grouped by component, then by k.
    for (int k = 1; k <= 5; k++) {
        integration::Integrator<double> I(k);
        for (int n = 0; n <= 2 * k + 3; n++) {
            int jmax = (n > k) ? n : k;
            for (int j = 0; j <= jmax; j++)
                kbref::write_weight_row(fp, "greg", k, n, j, -1,
                                        I.gregory_weights(n, j));
        }
    }
    for (int k = 1; k <= 5; k++) {
        integration::Integrator<double> I(k);
        for (int l = 0; l <= k + 1; l++)
            kbref::write_weight_row(fp, "bd", k, l, -1, -1, I.bd_weights(l));
    }
    for (int k = 1; k <= 5; k++) {
        integration::Integrator<double> I(k);
        for (int i = 0; i <= k; i++)
            for (int j = 0; j <= k; j++)
                for (int l = 0; l <= k; l++)
                    kbref::write_weight_row(fp, "pint", k, i, j, l,
                                            I.poly_integration(i, j, l));
    }
    for (int k = 1; k <= 5; k++) {
        integration::Integrator<double> I(k);
        for (int i = 0; i <= k; i++)
            for (int l = 0; l <= k; l++)
                kbref::write_weight_row(fp, "pdiff", k, i, l, -1,
                                        I.poly_differentiation(i, l));
    }
    for (int k = 2; k <= 5; k++) {
        integration::Integrator<double> I(k);
        for (int m = 1; m <= k - 1; m++)
            for (int j = 0; j <= k; j++)
                for (int l = 0; l <= k; l++)
                    kbref::write_weight_row(fp, "rcorr", k, m, j, l,
                                            I.rcorr(m, j, l));
    }

    std::fclose(fp);
    return 0;
}
