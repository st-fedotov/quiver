#!/bin/bash
# Parallel execution of RAD (radical) computations using GNU parallel.
#
# Environment variables:
#   NUM_WORKERS - number of parallel jobs (default: 64)
#   GC_INITIAL_HEAP_SIZE - Macaulay2 GC heap size (default: not set)
#   MEM_MAX - optional per-job memory cap, e.g. "50G". When set, each M2 job runs inside a
#             transient systemd user scope with MemoryMax=$MEM_MAX. Empty/unset = no cap
#             (default; behaviour unchanged). Needs a usable systemd user manager:
#             XDG_RUNTIME_DIR must point at the user runtime dir (enable linger for
#             headless/detached use).
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

# Optional per-job memory cap (set MEM_MAX, e.g. MEM_MAX=50G). Exported so the per-job
# shells parallel spawns inherit it. Empty = no wrapper, i.e. run M2 directly as before.
CAP=""
if [ -n "${MEM_MAX:-}" ]; then
    CAP="systemd-run --scope --user -p MemoryMax=$MEM_MAX"
fi
export CAP

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

  $CAP bash -lc "cd \"$dir\"; M2 --script rad.m2" \
    >"$dir/rad_out.txt" 2>"$dir/rad_err.txt"
  rc=$?

  echo "END   dir=$dir rc=$rc t=$(date -Is)" | tee -a "$dir/parallel.meta"
  exit $rc
' :::: joblist.txt
