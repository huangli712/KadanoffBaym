#!/usr/bin/env bash
#
# Build the NESSi side of the KB-vs-NESSi benchmarks (see README.md).
#
# Environment overrides:
#   NESSI_SRC   NESSi 2.0.0 tree        (default: ../nessi-2.0.0)
#   PREFIX      install prefix of the   (default: ../ppsc-port/nessi-install)
#               prebuilt libcntr
#   EIGEN_INC   Eigen include root     (default: ../eigen3-install/include)
#               (must resolve both eigen3/Eigen/Dense and Eigen/Core)
#   HDF5_INC    hdf5 include dir       (default: /opt/anaconda3/include)
#   HDF5_LIB    hdf5 lib dir           (default: /opt/anaconda3/lib)
#
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NESSI_SRC="${NESSI_SRC:-$HERE/../nessi-2.0.0}"
PREFIX="${PREFIX:-$HERE/../ppsc-port/nessi-install}"
EIGEN_INC="${EIGEN_INC:-$HERE/../eigen3-install/include}"
HDF5_INC="${HDF5_INC:-/opt/anaconda3/include}"
HDF5_LIB="${HDF5_LIB:-/opt/anaconda3/lib}"

CXXFLAGS="-O3 -std=c++17 -DCNTR_USE_HDF5 \
 -I$NESSI_SRC/libcntr -I$PREFIX/include \
 -I$EIGEN_INC -I$EIGEN_INC/eigen3 -I$HDF5_INC"
LDFLAGS="-L$PREFIX/lib -lcntr -lcntr_hdf5 -L$HDF5_LIB -lhdf5 \
 -Wl,-rpath,$PREFIX/lib -Wl,-rpath,$HDF5_LIB"
PROG="$NESSI_SRC/examples/programs"

g++ $CXXFLAGS "$HERE/nessi_dyson_bench.cpp" -o "$HERE/nessi_dyson_bench" $LDFLAGS
g++ $CXXFLAGS "$PROG/fkm_bethe_quench.cpp" -o "$HERE/fkm_bethe" $LDFLAGS
g++ $CXXFLAGS "$PROG/hubbard_chain_2b.cpp" "$PROG/hubbard_chain_selfen_impl.cpp" \
    -o "$HERE/hubbard_2b" $LDFLAGS
g++ $CXXFLAGS "$PROG/Holstein_bethe_Migdal.cpp" "$PROG/Holstein_utils_impl.cpp" \
    "$PROG/Holstein_impurity_impl.cpp" -o "$HERE/holstein_migdal" $LDFLAGS

echo "built: nessi_dyson_bench fkm_bethe hubbard_2b holstein_migdal"
