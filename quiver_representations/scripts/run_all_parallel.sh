#!/bin/bash
# Parallel execution of RAD (radical) computations using GNU parallel.
#
# Environment variables:
#   NUM_WORKERS - number of parallel jobs (default: 64)
#   GC_INITIAL_HEAP_SIZE - Macaulay2 GC heap size (default: not set)
#
# This script expects to be run from a directory containing a 'jobs' subdirectory
# with rad.m2 files.

set -euo pipefail

NUM_WORKERS=${NUM_WORKERS:-64}

# Optional: halt all jobs if one fails (set HALT_ON_FAIL=1 to enable)
HALT_OPT=""
if [ "${HALT_ON_FAIL:-}" = "1" ]; then
    HALT_OPT="--halt now,fail=1"
fi

# Build a stable list of jobs
find jobs -type f -name 'rad.m2' -printf '%h\n' | sort -V > joblist.txt

if [ ! -s joblist.txt ]; then
    echo "No rad.m2 jobs found in jobs/ directory"
    exit 1
fi

echo "Found $(wc -l < joblist.txt) jobs, running with $NUM_WORKERS workers"

# Run jobs in parallel
parallel -j "$NUM_WORKERS" --joblog run.log --eta $HALT_OPT '
  dir={};
  echo "START dir=$dir pid=$$ t=$(date -Is)" | tee "$dir/parallel.meta"

  bash -lc "cd \"$dir\"; M2 --script rad.m2" \
    >"$dir/rad_out.txt" 2>"$dir/rad_err.txt"
  rc=$?

  echo "END   dir=$dir rc=$rc t=$(date -Is)" | tee -a "$dir/parallel.meta"
  exit $rc
' :::: joblist.txt
