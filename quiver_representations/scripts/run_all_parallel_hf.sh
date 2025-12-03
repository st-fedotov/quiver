#!/bin/bash
# Parallel execution of Hilbert function computations using GNU parallel.
#
# Environment variables:
#   NUM_WORKERS - number of parallel jobs (default: 64)
#   GC_INITIAL_HEAP_SIZE - Macaulay2 GC heap size (default: not set)
#
# This script expects to be run from a directory containing a 'jobs' subdirectory
# with hilbert.m2 files.

set -euo pipefail

NUM_WORKERS=${NUM_WORKERS:-64}

# Build a stable list of jobs
find jobs -type f -name 'hilbert.m2' -printf '%h\n' | sort -V > hilbert_joblist.txt

if [ ! -s hilbert_joblist.txt ]; then
    echo "No hilbert.m2 jobs found in jobs/ directory"
    exit 1
fi

echo "Found $(wc -l < hilbert_joblist.txt) jobs, running with $NUM_WORKERS workers"

# Run jobs in parallel
parallel -j "$NUM_WORKERS" --joblog hilbert_run.log --eta '
  dir={};
  echo "START dir=$dir pid=$$ t=$(date -Is)" | tee "$dir/hilbert_parallel.meta"

  bash -lc "cd \"$dir\"; M2 --script hilbert.m2" \
    >"$dir/hilbert_out.txt" 2>"$dir/hilbert_err.txt"
  rc=$?

  echo "END   dir=$dir rc=$rc t=$(date -Is)" | tee -a "$dir/hilbert_parallel.meta"
  exit $rc
' :::: hilbert_joblist.txt
