#!/usr/bin/env bash
#
# Regenerate all NESSi reference data used by the KadanoffBaym test suite.
#
# Usage: bash .nessi/kbref/generate.sh
#        bash .nessi/kbref/generate.sh --list
#        bash .nessi/kbref/generate.sh <model> <profile> --output-dir <dir>
#
# Without arguments the generators of the phases 1-4 and 6-8 are built in
# .nessi/kbref/build and write kbref-v1 plain text files into
# test/data. Running this script twice must produce byte-identical
# files.
#
# Phase 5 (.omo/plans/05-phase5-examples.md): <model> is one of
# hubbard, fkm, holstein_bethe, holstein_nambu, holstein_impurity, all;
# <profile> is quick or release. The output directory must exist and be
# writable; it is passed unchanged to the executables.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

BUILD=".nessi/kbref/build"
OUTDIR="test/data"
P5_MODELS="hubbard fkm holstein_bethe holstein_nambu holstein_impurity"

# Phase 5: target and approximations of a model
p5_target() {
    case "$1" in
        hubbard) echo "hubbard_chain_text" ;;
        fkm) echo "fkm_bethe_quench_text" ;;
        holstein_bethe) echo "holstein_bethe_text" ;;
        holstein_nambu) echo "holstein_nambu_text" ;;
        holstein_impurity) echo "holstein_impurity_text" ;;
        *) return 1 ;;
    esac
}
p5_approx() {
    case "$1" in
        hubbard) echo "2B GW TPP" ;;
        fkm) echo "none" ;;
        *) echo "Migdal uMig" ;;
    esac
}

if [ "$#" -gt 0 ]; then
    if [ "$1" = "--list" ]; then
        [ "$#" -eq 1 ] || { echo "generate.sh: --list takes no arguments" >&2; exit 2; }
        for m in ${P5_MODELS}; do echo "${m}"; done
        exit 0
    fi
    [ "$#" -eq 4 ] || { echo "usage: generate.sh <model> <profile> --output-dir <dir>" >&2; exit 2; }
    MODEL="$1"; PROFILE="$2"
    [ "$3" = "--output-dir" ] || { echo "generate.sh: missing --output-dir" >&2; exit 2; }
    P5_OUT="$4"
    case "${PROFILE}" in quick|release) ;; *) echo "generate.sh: unknown profile ${PROFILE}" >&2; exit 2 ;; esac
    if [ "${MODEL}" = "all" ]; then
        MODELS="${P5_MODELS}"
    else
        p5_target "${MODEL}" > /dev/null || { echo "generate.sh: unknown model ${MODEL}" >&2; exit 2; }
        MODELS="${MODEL}"
    fi
    [ -d "${P5_OUT}" ] && [ -w "${P5_OUT}" ] ||
        { echo "generate.sh: output directory ${P5_OUT} missing or not writable" >&2; exit 2; }
    cmake -S .nessi/kbref -B "${BUILD}" > /dev/null
    for m in ${MODELS}; do
        t="$(p5_target "${m}")"
        [ -f ".nessi/kbref/${t}.cpp" ] ||
            { echo "generate.sh: target ${t} is not registered (pending)" >&2; exit 3; }
        cmake --build "${BUILD}" --target "${t}" > /dev/null
    done
    for m in ${MODELS}; do
        t="$(p5_target "${m}")"
        for a in $(p5_approx "${m}"); do
            "${BUILD}/${t}" "${a}" "${PROFILE}" "${P5_OUT}"
        done
    done
    exit 0
fi

# libcntr: the build of .omo/plans/01-phase1-foundation.md, section 3.2,
# with ness=ON (required by the P7 generators, P7-W10-T3). It does not
# change the P1..P6 reference data.
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
P2_TARGETS="p2_utilities p2_differentiation p2_convolution_utilities p2_response"
for t in ${P2_TARGETS}; do
    cmake --build "${BUILD}" --target "${t}" > /dev/null
done
P3_TARGETS="p3_dyson_ref p3_equilibrium_ref"
for t in ${P3_TARGETS}; do
    cmake --build "${BUILD}" --target "${t}" > /dev/null
done
P4_TARGETS="phase4_vie2 phase4_bubble phase4_observables"
for t in ${P4_TARGETS}; do
    cmake --build "${BUILD}" --target "${t}" > /dev/null
done
cmake --build "${BUILD}" --target pseudo_reference > /dev/null
P7_TARGETS="phase7_fourier_ref phase7_ness_ref phase7_examples_siam_ref phase7_examples_bethe_ref phase7_examples_bethe_prop_ref"
for t in ${P7_TARGETS}; do
    cmake --build "${BUILD}" --target "${t}" > /dev/null
done
P8_TARGETS="p8_trunc_bethe_ref p8_moving_solver_ref p8_ness2_bethe_trunc_ref p8_moving_ref p8_gw_ref"
for t in ${P8_TARGETS}; do
    cmake --build "${BUILD}" --target "${t}" > /dev/null
done

mkdir -p "${OUTDIR}"
rm -f "${OUTDIR}"/weights_k1_k5.dat "${OUTDIR}"/convolution_type*_size*_k*.dat

"${BUILD}/weights" "${OUTDIR}"
"${BUILD}/convolution" "${OUTDIR}"

# P2: each generator owns the files ${OUTDIR}/<target>_*.dat
for t in ${P2_TARGETS}; do
    rm -f "${OUTDIR}/${t}"_*.dat
    "${BUILD}/${t}" "${OUTDIR}"
done

# P3: all files are in ${OUTDIR}/phase3
rm -rf "${OUTDIR}/phase3"
mkdir -p "${OUTDIR}/phase3"
for t in ${P3_TARGETS}; do
    "${BUILD}/${t}" "${OUTDIR}/phase3"
done

# P4: all files are in ${OUTDIR}/phase4
rm -rf "${OUTDIR}/phase4"
mkdir -p "${OUTDIR}/phase4"
for t in ${P4_TARGETS}; do
    "${BUILD}/${t}" "${OUTDIR}/phase4"
done

# P6: all files are in ${OUTDIR}/phase6
rm -rf "${OUTDIR}/phase6"
mkdir -p "${OUTDIR}/phase6"
"${BUILD}/pseudo_reference" "${OUTDIR}/phase6"

# P7: each generator writes one file ${OUTDIR}/phase7_*.dat
rm -f "${OUTDIR}"/phase7_*.dat
for t in ${P7_TARGETS}; do
    "${BUILD}/${t}" "${OUTDIR}"
done

# P8: each generator writes one file ${OUTDIR}/p8_*.dat
rm -f "${OUTDIR}"/p8_*.dat
for t in ${P8_TARGETS}; do
    "${BUILD}/${t}" "${OUTDIR}"
done

cd "${OUTDIR}"
ls -1 *.dat phase3/*.dat phase4/*.dat phase6/*.dat | LC_ALL=C sort | xargs sha256sum
