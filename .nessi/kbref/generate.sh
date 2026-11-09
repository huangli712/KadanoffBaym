#!/usr/bin/env bash
#
# Regenerate all NESSi reference data used by the KadanoffBaym test suite.
#
# Usage: bash .nessi/kbref/generate.sh
#
# The generators are built in .nessi/kbref/build and write kbref-v1 plain
# text files into test/nessi/data. Running this script twice must produce
# byte-identical files.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

BUILD=".nessi/kbref/build"
OUTDIR="test/nessi/data"

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

cd "${OUTDIR}"
ls -1 *.dat phase3/*.dat phase4/*.dat | LC_ALL=C sort | xargs sha256sum
