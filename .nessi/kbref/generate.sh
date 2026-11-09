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

mkdir -p "${OUTDIR}"
rm -f "${OUTDIR}"/weights_k1_k5.dat "${OUTDIR}"/convolution_type*_size*_k*.dat

"${BUILD}/weights" "${OUTDIR}"
"${BUILD}/convolution" "${OUTDIR}"

cd "${OUTDIR}"
ls -1 *.dat | LC_ALL=C sort | xargs sha256sum
