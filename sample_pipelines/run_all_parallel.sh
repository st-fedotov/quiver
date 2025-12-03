# Use environment variables with defaults
NUM_WORKERS=${NUM_WORKERS:-64}

# 0) Build a stable list once
find jobs -type f -name 'rad.m2' -printf '%h\n' | sort -V > joblist.txt

# 1) Run exactly those jobs (no in-run retries)
parallel -j $NUM_WORKERS --joblog run.log --eta '
  dir={};
  echo "START dir=$dir pid=$$ t=$(date -Is)" | tee "$dir/parallel.meta"

  bash -lc "cd \"$dir\"; M2 --script rad.m2" \
    >"$dir/rad_out.txt" 2>"$dir/rad_err.txt"
  rc=$?

  echo "END   dir=$dir rc=$rc t=$(date -Is)" | tee -a "$dir/parallel.meta"
  exit $rc
' :::: joblist.txt
