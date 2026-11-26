#!/usr/bin/env bash
#
# Regenerate all NESSi reference data used by the KadanoffBaym test suite.
#
# Usage: bash .nessi/kbref/generate.sh
#        bash .nessi/kbref/generate.sh --list
#        bash .nessi/kbref/generate.sh <model> <profile> --output-dir <dir>
#
# Without arguments all function-level generators are built in
# .nessi/kbref/build and write ref 1 plain text files into data/test
# (function-level reference data) and data/tutor (the siam, bethe, gw,
# and truncated Bethe example data). Running this script twice must
# produce byte-identical files.
#
# The example drivers: <model> is one of
# hubbard, fkm, holstein_bethe, holstein_nambu, holstein_impurity, all;
# <profile> is quick or release. The output directory must exist and be
# writable; it is passed unchanged to the executables.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

BUILD=".nessi/kbref/build"
TEST_OUT="data/test"
TUTOR_OUT="data/tutor"
EXAMPLE_MODELS="hubbard fkm holstein_bethe holstein_nambu holstein_impurity"

# The example drivers: target and approximations of a model
example_target() {
    case "$1" in
        hubbard) echo "hubbard_chain_text" ;;
        fkm) echo "fkm_bethe_quench_text" ;;
        holstein_bethe) echo "holstein_bethe_text" ;;
        holstein_nambu) echo "holstein_nambu_text" ;;
        holstein_impurity) echo "holstein_impurity_text" ;;
        *) return 1 ;;
    esac
}
example_approx() {
    case "$1" in
        hubbard) echo "2B GW TPP" ;;
        fkm) echo "none" ;;
        *) echo "Migdal uMig" ;;
    esac
}

if [ "$#" -gt 0 ]; then
    if [ "$1" = "--list" ]; then
        [ "$#" -eq 1 ] || { echo "generate.sh: --list takes no arguments" >&2; exit 2; }
        for m in ${EXAMPLE_MODELS}; do echo "${m}"; done
        exit 0
    fi
    [ "$#" -eq 4 ] || { echo "usage: generate.sh <model> <profile> --output-dir <dir>" >&2; exit 2; }
    MODEL="$1"; PROFILE="$2"
    [ "$3" = "--output-dir" ] || { echo "generate.sh: missing --output-dir" >&2; exit 2; }
    EXAMPLE_OUT="$4"
    case "${PROFILE}" in quick|release) ;; *) echo "generate.sh: unknown profile ${PROFILE}" >&2; exit 2 ;; esac
    if [ "${MODEL}" = "all" ]; then
        MODELS="${EXAMPLE_MODELS}"
    else
        example_target "${MODEL}" > /dev/null || { echo "generate.sh: unknown model ${MODEL}" >&2; exit 2; }
        MODELS="${MODEL}"
    fi
    [ -d "${EXAMPLE_OUT}" ] && [ -w "${EXAMPLE_OUT}" ] ||
        { echo "generate.sh: output directory ${EXAMPLE_OUT} missing or not writable" >&2; exit 2; }
    cmake -S .nessi/kbref -B "${BUILD}" > /dev/null
    for m in ${MODELS}; do
        t="$(example_target "${m}")"
        [ -f ".nessi/kbref/${t}.cpp" ] ||
            { echo "generate.sh: target ${t} is not registered (pending)" >&2; exit 3; }
        cmake --build "${BUILD}" --target "${t}" > /dev/null
    done
    for m in ${MODELS}; do
        t="$(example_target "${m}")"
        for a in $(example_approx "${m}"); do
            "${BUILD}/${t}" "${a}" "${PROFILE}" "${EXAMPLE_OUT}"
        done
    done
    exit 0
fi

# libcntr: Release build with ness=ON (required by the steady-state
# generators) and FFTW.
cmake -S .nessi/nessi-2.0.0/libcntr \
      -B .nessi/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="${ROOT}/.nessi/install" \
      -DCMAKE_INCLUDE_PATH="${ROOT}/.nessi/eigen3-install/include" \
      -DEigen3_DIR="${ROOT}/.nessi/eigen3-install/share/eigen3/cmake" \
      -Domp=OFF -Dmpi=OFF -Dhdf5=OFF -Dness=ON \
      -DBUILD_TEST=OFF -DBUILD_DOC=OFF > /dev/null
cmake --build .nessi/build --parallel 2 > /dev/null
cmake --install .nessi/build > /dev/null

cmake -S .nessi/kbref -B "${BUILD}" > /dev/null
cmake --build "${BUILD}" --target weights > /dev/null
cmake --build "${BUILD}" --target convolution > /dev/null
TOOL_TARGETS="utilities differentiation convolution_utilities response"
for t in ${TOOL_TARGETS}; do
    cmake --build "${BUILD}" --target "${t}" > /dev/null
done
DYSON_TARGETS="dyson_ref equilibrium_ref"
for t in ${DYSON_TARGETS}; do
    cmake --build "${BUILD}" --target "${t}" > /dev/null
done
REALTIME_TARGETS="vie2 bubble observables"
for t in ${REALTIME_TARGETS}; do
    cmake --build "${BUILD}" --target "${t}" > /dev/null
done
cmake --build "${BUILD}" --target pseudo_reference > /dev/null
STEADY_FOURIER_TARGETS="fourier_ref steady_ref"
EXAMPLE_REF_TARGETS="examples_siam_ref examples_bethe_ref examples_bethe_prop_ref"
MOVING_TARGETS="moving_ref moving_solver_ref"
EXAMPLE_SERIES_TARGETS="trunc_bethe_ref steady_bethe_trunc_ref gw_ref"
for t in ${STEADY_FOURIER_TARGETS} ${EXAMPLE_REF_TARGETS} ${MOVING_TARGETS} \
         ${EXAMPLE_SERIES_TARGETS}; do
    cmake --build "${BUILD}" --target "${t}" > /dev/null
done

mkdir -p "${TEST_OUT}" "${TUTOR_OUT}"
rm -f "${TEST_OUT}"/weights_k1_k5.dat "${TEST_OUT}"/convolution_type*_size*_k*.dat

"${BUILD}/weights" "${TEST_OUT}"
"${BUILD}/convolution" "${TEST_OUT}"

# Tools: each generator owns the files ${TEST_OUT}/<target>_*.dat
for t in ${TOOL_TARGETS}; do
    rm -f "${TEST_OUT}/${t}"_*.dat
    "${BUILD}/${t}" "${TEST_OUT}"
done

# Dyson: all files are in ${TEST_OUT}/dyson
rm -rf "${TEST_OUT}/dyson"
mkdir -p "${TEST_OUT}/dyson"
for t in ${DYSON_TARGETS}; do
    "${BUILD}/${t}" "${TEST_OUT}/dyson"
done

# Real time: all files are in ${TEST_OUT}/realtime
rm -rf "${TEST_OUT}/realtime"
mkdir -p "${TEST_OUT}/realtime"
for t in ${REALTIME_TARGETS}; do
    "${BUILD}/${t}" "${TEST_OUT}/realtime"
done

# Pseudo particles: all files are in ${TEST_OUT}/pseudo
rm -rf "${TEST_OUT}/pseudo"
mkdir -p "${TEST_OUT}/pseudo"
"${BUILD}/pseudo_reference" "${TEST_OUT}/pseudo"

# Steady state and Fourier: into data/test
rm -f "${TEST_OUT}"/fourier.dat "${TEST_OUT}"/steady.dat
for t in ${STEADY_FOURIER_TARGETS}; do
    "${BUILD}/${t}" "${TEST_OUT}"
done

# Whole model runs: into data/tutor
rm -f "${TUTOR_OUT}"/examples_*.dat
for t in ${EXAMPLE_REF_TARGETS}; do
    "${BUILD}/${t}" "${TUTOR_OUT}"
done

# Moving-window solvers: into data/test
rm -f "${TEST_OUT}"/moving_case*.dat "${TEST_OUT}"/moving_solver.dat
for t in ${MOVING_TARGETS}; do
    "${BUILD}/${t}" "${TEST_OUT}"
done

# Truncated Bethe lattices and GW: into data/tutor
rm -f "${TUTOR_OUT}"/trunc_bethe.dat "${TUTOR_OUT}"/steady_bethe_trunc.dat "${TUTOR_OUT}"/gw.dat
for t in ${EXAMPLE_SERIES_TARGETS}; do
    "${BUILD}/${t}" "${TUTOR_OUT}"
done

cd "${TEST_OUT}"
ls -1 *.dat dyson/*.dat realtime/*.dat pseudo/*.dat | LC_ALL=C sort | xargs sha256sum
cd "${ROOT}/${TUTOR_OUT}"
ls -1 *.dat | LC_ALL=C sort | xargs sha256sum
