# Use environment variables with defaults
NUM_WORKERS=${NUM_WORKERS:-64}

# 0) Build a stable list once
find jobs -type f -name 'hilbert.m2' -printf '%h\n' | sort -V > hilbert_joblist.txt

# 1) Run exactly those jobs (no in-run retries)
parallel -j $NUM_WORKERS --joblog hilbert_run.log --eta '
  dir={};
  echo "START dir=$dir pid=$$ t=$(date -Is)" | tee "$dir/hilbert_parallel.meta"

  bash -lc "cd \"$dir\"; M2 --script hilbert.m2" \
    >"$dir/hilbert_out.txt" 2>"$dir/hilbert_err.txt"
  rc=$?

  echo "END   dir=$dir rc=$rc t=$(date -Is)" | tee -a "$dir/hilbert_parallel.meta"
  exit $rc
' :::: hilbert_joblist.txt
