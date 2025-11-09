"""
Hilbert function and multi-degree space dimension computations.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

def load_manifest(run_root: Path) -> Dict[str, Dict]:
    doc = yaml.safe_load((run_root / "manifest.yaml").read_text(encoding="utf-8", errors="ignore")) or {}

    def _norm_vec(obj):
        if obj is None: return None
        if isinstance(obj, dict):
            pairs = [(int(k), int(v)) for k, v in obj.items()]
            pairs.sort(key=lambda t: t[0])
            return [v for _, v in pairs]
        if isinstance(obj, (list, tuple)):
            return [int(x) for x in obj]
        return None

    out = {}
    for j in (doc.get("jobs") or []):
        jid = str(j.get("id", ""))
        out[jid] = {
            "id": jid,
            "quiver": j.get("quiver"),
            "module": j.get("module"),
            "ambient_dim": _norm_vec(j.get("ambient_dim")),
            "target_dim":  _norm_vec(j.get("target_dim")),
        }
    return out

def relpath(from_root: Path, p: Path) -> str:
    return os.path.relpath(p, start=from_root)


# ----------------------------- B) results parsing -----------------------------
# Ring comes from the original script (rad.m2), GB(Ifullsat) generators from rad_out.txt.

_RX_RING_BLOCK = re.compile(r'^\s*(R\s*=\s*QQ\[[\s\S]*?\];)\s*$', re.M)

def parse_ring_block_from_script(m2_path: Path) -> str:
    """
    Extract the exact 'R = QQ[ ... ];' block from rad.m2 by bracket matching.
    Preserves every character verbatim; raises if brackets don’t balance or ';' is missing.
    """
    txt = m2_path.read_text(encoding="utf-8", errors="strict")

    start = txt.find("R = QQ[")
    if start < 0:
        raise ValueError(f"'R = QQ[' not found in {m2_path}")

    i = start + len("R = QQ[")
    depth = 1  # we are inside the initial '['
    n = len(txt)

    # scan until the matching closing ']' for the initial '['
    while i < n and depth > 0:
        ch = txt[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        i += 1

    if depth != 0:
        raise ValueError(f"Unbalanced brackets in ring declaration in {m2_path}")

    # expect a trailing semicolon after the closing ']'
    j = i
    while j < n and txt[j].isspace():
        j += 1
    if j >= n or txt[j] != ";":
        raise ValueError(f"Missing ';' after ring 'R = QQ[...]' in {m2_path}")

    ring_block = txt[start:j+1]  # include the trailing ';'
    return ring_block

_RX_GB_HDR      = re.compile(r'^\s*===== GROEBNER BASIS \(Ifullsat\) =====\s*$')
_RX_SECTION_HDR = re.compile(r'^\s*===== ')
_RX_PROMPT_LINE = re.compile(r'^\s*i\d+\s*:\s')

# keep existing imports/regexes except replace parse_ifullsat_gb with the version below

# Replace your parse_ifullsat_gb with this more permissive version

# ----------------------------- B) results parsing -----------------------------
# Ring comes from the original script (rad.m2), GB(Ifullsat) generators from rad_out.txt.

_HDR_IFULLSAT = "===== GROEBNER BASIS (Ifullsat) ====="
_RX_PROMPT_LINE = re.compile(r'^\s*i\d+\s*:\s')

def parse_ifullsat_gb(rad_out: Path) -> list[str]:
    """
    Capture generator lines under the Ifullsat header.
    Robust to the header appearing:
      - as a standalone line, or
      - embedded in a prompt/stdio line just before the standalone header.
    """
    lines = rad_out.read_text(encoding="utf-8", errors="ignore").splitlines()
    n = len(lines)
    i = 0

    # find the first line that contains the header substring
    while i < n and _HDR_IFULLSAT not in lines[i]:
        i += 1
    if i == n:
        raise ValueError(f"'{_HDR_IFULLSAT}' not found in {rad_out}")

    # if the very next line is the standalone header, skip it too
    j = i + 1
    if j < n and lines[j].strip() == _HDR_IFULLSAT:
        i = j

    # begin reading AFTER the header line we settled on
    i += 1

    # skip immediate prompts/empties
    while i < n and (not lines[i].strip() or _RX_PROMPT_LINE.match(lines[i])):
        i += 1

    out: list[str] = []
    while i < n:
        raw = lines[i]
        line = raw.strip()

        # stop conditions
        if not line:
            break
        if "=====" in line:           # next section header
            break
        if _RX_PROMPT_LINE.match(raw):
            break

        # filter obvious noise
        if "<<" in line or line.startswith("--"):
            i += 1
            continue

        out.append(line)
        i += 1

    return out


# ----------------------------- C) multidegrees -----------------------------

def enumerate_multidegrees(n_vertices: int, r_max: int) -> Iterable[Tuple[int, ...]]:
    for r in itertools.product(range(r_max + 1), repeat=n_vertices):
        if any(r):
            yield r

import os, shutil, datetime
from pathlib import Path
from textwrap import dedent
from itertools import combinations, product
from typing import Dict, List, Tuple
import yaml  # requires PyYAML

# You still provide:
#   parse_ifullsat_gb(rad_out_path: Path) -> List[str]


def _read_manifest_exact(source_dir: Path) -> Dict[str, Dict]:
    """
    Read {source}/manifest.yaml with EXACT required shape:
      batch: {...}
      jobs: [ { id: "...", ambient_dim: {0:..}, target_dim: {0:..}, ... }, ... ]
    Returns: dict id -> job-meta (each a dict).
    """
    man_path = source_dir / "manifest.yaml"
    if not man_path.exists():
        raise FileNotFoundError(f"Missing manifest file: {man_path}")

    with man_path.open("r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)

    if not isinstance(manifest, dict):
        raise ValueError(f"{man_path}: top-level must be a mapping")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError(f"{man_path}: must contain non-empty 'jobs' list")

    meta_by_id: Dict[str, Dict] = {}
    for rec in jobs:
        if not isinstance(rec, dict):
            raise ValueError(f"{man_path}: each jobs[] entry must be a mapping")
        jid = rec.get("id")
        if not isinstance(jid, str) or not jid:
            raise ValueError(f"{man_path}: each job must have string 'id'")
        meta_by_id[jid] = rec
    return meta_by_id


def _emit_ring_vertex_order(
    ambient: Dict[int, int],
    target: Dict[int, int],
    *,
    prefix: str = "p_",
) -> Tuple[str, List[int]]:
    """
    Single-line Macaulay2 ring in strict vertex order v=0..n-1.
    Variables: p_(v,i1,...,ik). Multigrading is one-hot per vertex via Degrees.
    """
    n = len(ambient)
    if n != len(target):
        raise ValueError(f"ambient_dim and target_dim length mismatch ({n} vs {len(target)})")

    # Validate dense keys 0..n-1 exactly
    if set(ambient.keys()) != set(range(n)) or set(target.keys()) != set(range(n)):
        raise ValueError(f"keys must be exactly 0..{n-1} for ambient_dim and target_dim")

    # Per-vertex blocks (vertex order)
    blocks: List[List[str]] = []
    for v in range(n):
        N = int(ambient[v]); k = int(target[v])
        if N <= 0 or k <= 0:
            blocks.append([])
            continue
        vars_v = [f"{prefix}({v}," + ",".join(map(str, I)) + ")" for I in combinations(range(N), k)]
        blocks.append(vars_v)

    ordered_vars = [nm for block in blocks for nm in block]
    if not ordered_vars:
        raise ValueError("All vertices have k==0 or N==0: no Plücker variables for the ring.")

    block_sizes = [len(block) for block in blocks]
    num_blocks  = len(blocks)

    # Degrees: one-hot per block
    deg_rows = []
    for j, block in enumerate(blocks):
        for _ in block:
            row = ["0"] * num_blocks
            row[j] = "1"
            deg_rows.append("{ " + ", ".join(row) + " }")

    vars_m2 = ", ".join(ordered_vars)
    mo_m2   = "{ " + ", ".join(str(b) for b in block_sizes) + " }"
    degs_m2 = "{ " + ", ".join(deg_rows) + " }"

    ring_block = f"R = QQ[{vars_m2}, MonomialOrder => {mo_m2}, Degrees => {degs_m2}];\nuse R;"
    return ring_block, block_sizes


def _emit_hilbert_script(
    *,
    ring_block: str,
    gb_lines: List[str],
    degrees_vertex: List[Tuple[int, ...]],
    csv_path: str,
    header_cols: List[str],
) -> str:
    """
    Emit the Macaulay2 script that reuses Ifullsat via forceGB and writes HF CSV.
    Multidegrees are in vertex order and CSV header is r0,r1,...,hf.
    """
    if not gb_lines:
        return 'error "parse_ifullsat_gb returned no generators for Ifullsat; cannot compute HF.";\n'

    gb_row   = ", ".join(gb_lines)
    degs_txt = ",\n".join("  { " + ", ".join(map(str, d)) + " }" for d in degrees_vertex)
    header   = ",".join(header_cols + ["hf"])

    return f'''-- === Auto Hilbert (blocks in vertex order: v=0,1,2,...) ===
{ring_block}

-- Reuse Ifullsat Groebner basis from RAD (declare; do not recompute)
Glist = matrix {{ {{ {gb_row} }} }};
if numRows Glist == 0 or numColumns Glist == 0 then (
  Igb = ideal 0_R;
) else (
  G   = forceGB Glist;
  Igb = ideal flatten entries gens G;
);
Q = R / Igb;

Degs = {{
{degs_txt}
}};

OUT = openOut "{csv_path}";
OUT << "{header}" << endl;

for t from 0 to #Degs-1 do (
  k = Degs#t;
  d = hilbertFunction(k, Q);
  for i from 0 to #k-1 do (
    if i>0 then OUT << ",";
    OUT << k#i
  );
  OUT << "," << d << endl;
);
close OUT;

<< "Wrote " << #Degs << " rows to {csv_path}" << endl;
exit 0;
'''


def write_hilbert_batch(
    source: str | Path,
    dest: str | Path | None = None,
    r_max: int = 2,
    *,
    script_name: str = "hilbert.m2",
    out_stdout: str = "hilbert_out.txt",
    out_stderr: str = "hilbert_err.txt",
    docker_image: str = "m2-ppa",
    msys_no_pathconv: bool = True,
) -> Path:
    """
    Build a Hilbert-only run from an existing RAD run (EXACT manifest format):
      • reads {source}/manifest.yaml with 'batch' + 'jobs: [ ... ]'
      • reads jobs/*/rad_out.txt for Ifullsat GB strings (via parse_ifullsat_gb)
      • rebuilds the ring in vertex order (0..n-1) from dict dim-vectors
      • enumerates multidegrees (0..r_max)^n in vertex order; writes hf.csv per job
      • emits run_all.sh (docker) and zips dest
    Errors are explicit; nothing is silently skipped.
    """
    source = Path(source).resolve()
    if dest is None:
        dest = Path(f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    dest = Path(dest).resolve()
    jobs_dir = dest / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    meta_by_id = _read_manifest_exact(source)

    job_dirs = sorted([p for p in (source / "jobs").iterdir() if p.is_dir()])
    if not job_dirs:
        raise ValueError(f"No job folders found under {source}/jobs")

    new_jobs_manifest: List[Dict] = []

    for parent in job_dirs:
        parent_id = parent.name.split("_", 1)[0]
        meta = meta_by_id.get(parent_id)
        if meta is None:
            raise ValueError(f"Manifest has no entry for job '{parent_id}'")

        ambient = meta.get("ambient_dim")
        target  = meta.get("target_dim")
        if not isinstance(ambient, dict) or not isinstance(target, dict):
            raise TypeError(f"{parent_id}: ambient_dim and target_dim must be dict[int,int]")

        n = len(ambient)
        if n == 0 or n != len(target):
            raise ValueError(f"{parent_id}: ambient_dim and target_dim must have same positive length")
        if set(ambient.keys()) != set(range(n)) or set(target.keys()) != set(range(n)):
            raise ValueError(f"{parent_id}: keys must be exactly 0..{n-1} for ambient_dim and target_dim")

        rad_out = parent / "rad_out.txt"
        if not rad_out.exists():
            raise FileNotFoundError(f"{parent_id}: missing {rad_out} (needed to reuse Ifullsat GB)")

        gb_lines = parse_ifullsat_gb(rad_out)  # list[str], must be non-empty

        if not gb_lines:
            print(f"WARNING! {parent_id}: parsed zero Ifullsat generators from {rad_out}")

        ring_block, block_sizes = _emit_ring_vertex_order(ambient, target, prefix="p_")
        degs_vertex = [tuple(t) for t in product(range(r_max + 1), repeat=n)]
        header_cols = [f"r{i}" for i in range(n)]

        job_id  = f"{parent_id}__hf"
        job_dir = jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        m2_text = _emit_hilbert_script(
            ring_block=ring_block,
            gb_lines=gb_lines,
            degrees_vertex=degs_vertex,
            csv_path="hf.csv",
            header_cols=header_cols,
        )
        (job_dir / script_name).write_text(m2_text, encoding="utf-8")

        # meta.yaml
        parent_run_rel = os.path.relpath(source, start=dest)
        parent_job_rel = os.path.relpath(parent, start=dest)
        (job_dir / "meta.yaml").write_text(dedent(f'''\
            parent_run_dir: "{parent_run_rel}"
            parent_job_dir: "{parent_job_rel}"
            parent_job_id: "{parent_id}"
            r_max: {r_max}
            num_degrees: {len(degs_vertex)}
            quiver: "{meta.get('quiver')}"
            module: "{meta.get('module')}"
            ambient_dim: {ambient}
            target_dim: {target}
            block_sizes_vertex_order: {block_sizes}
        '''), encoding="utf-8")

        new_jobs_manifest.append({
            "id": job_id,
            "parent_job_id": parent_id,
            "parent_job_dir": parent_job_rel,
            "r_max": r_max,
            "num_degrees": len(degs_vertex),
            "quiver": meta.get("quiver"),
            "module": meta.get("module"),
            "ambient_dim": ambient,
            "target_dim": target,
            "block_sizes_vertex_order": block_sizes,
        })

    # batch manifest for the Hilbert run
    (dest / "manifest.yaml").write_text(
        dedent(f'''\
            source:
              parent_run_dir: "{os.path.relpath(source, start=dest)}"
              r_max: {r_max}
            jobs:
        ''') + "".join(
            dedent(f'''\
                - id: "{j['id']}"
                  parent_job_id: "{j['parent_job_id']}"
                  parent_job_dir: "{j['parent_job_dir']}"
                  r_max: {j['r_max']}
                  num_degrees: {j['num_degrees']}
                  quiver: "{j.get('quiver')}"
                  module: "{j.get('module')}"
                  target_dim: {j["target_dim"]}
                  ambient_dim: {j["ambient_dim"]}
                  block_sizes_vertex_order: {j["block_sizes_vertex_order"]}
            ''') for j in new_jobs_manifest
        ),
        encoding="utf-8"
    )

    # launcher
    env_prefix = "MSYS_NO_PATHCONV=1 " if msys_no_pathconv else ""
    launcher_sh = f'''#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"

for d in "$ROOT/jobs"/*/; do
  echo "==> Running $(basename "$d") started"
  {env_prefix}docker run -i --rm \\
    -v "$ROOT":/home/m2/work \\
    -w "/home/m2/work/jobs/$(basename "$d")" \\
    {docker_image} \\
    < "$d/{script_name}" > "$d/{out_stdout}" 2> "$d/{out_stderr}" || echo "    FAILED"
done

echo "All Hilbert jobs attempted."
'''
    (dest / "run_all.sh").write_text(launcher_sh, encoding="utf-8")
    os.chmod(dest / "run_all.sh", 0o755)

    shutil.make_archive(str(dest), "zip", root_dir=dest.parent, base_dir=dest.name)
    print(f"Created archive: {dest.with_suffix('.zip')}")
    return dest
