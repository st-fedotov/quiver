"""
Parsers for Macaulay2 output files.
"""

import re
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


import re, json, csv
from pathlib import Path

# ---------------------------- regexes for booleans (old + new sections) ----------------------------

_BOOL = {"true": True, "false": False}

# ----- booleans -----
RX_I_EQ         = re.compile(r"^\s*I1 == Ifull \?\s*(true|false)\s*$", re.I)
RX_I1_RAD       = re.compile(r"^\s*I1 == radical\(I1\) \?\s*(true|false)\s*$", re.I)
RX_IFULL_RAD    = re.compile(r"^\s*Ifull == radical\(Ifull\) \?\s*(true|false)\s*$", re.I)
RX_RADS_EQ      = re.compile(r"^\s*rad\(I1\) == rad\(Ifull\) \?\s*(true|false)\s*$", re.I)

# saturated + ordinary-vs-saturated
RX_I1_SAT_EQ    = re.compile(r"^\s*I1 == I1sat \?\s*(true|false)\s*$", re.I)
RX_IFULL_SAT_EQ = re.compile(r"^\s*Ifull == Ifullsat \?\s*(true|false)\s*$", re.I)
RX_SAT_EQ       = re.compile(r"^\s*I1sat == Ifullsat \?\s*(true|false)\s*$", re.I)
RX_I1SAT_RAD    = re.compile(r"^\s*I1sat == radical\(I1sat\) \?\s*(true|false)\s*$", re.I)

# original generator counts (if you print them)
RX_ORIG_I1      = re.compile(r"^\s*orig gens I1\s*=\s*(\d+)\s*$", re.I)
RX_ORIG_IFULL   = re.compile(r"^\s*orig gens Ifull\s*=\s*(\d+)\s*$", re.I)

# ----- GB column counts (PARSE THESE LINES DIRECTLY) -----
RX_GB_I1        = re.compile(r"^\s*#cols\s+gens\s+gb\(I1\)\s*=\s*(\d+)\s*$", re.I)
RX_GB_IFULL     = re.compile(r"^\s*#cols\s+gens\s+gb\(Ifull\)\s*=\s*(\d+)\s*$", re.I)
RX_GB_I1SAT     = re.compile(r"^\s*#cols\s+gens\s+gb\(I1sat\)\s*=\s*(\d+)\s*$", re.I)
RX_GB_IFULLSAT  = re.compile(r"^\s*#cols\s+gens\s+gb\(Ifullsat\)\s*=\s*(\d+)\s*$", re.I)

# GB headers to count entries below them
RX_GB_HEADER       = re.compile(r"^\s*===== GROEBNER BASIS \((I1|Ifull|I1sat|Ifullsat)\) =====\s*$")

# ----- GB degree/provenance metrics for Ifullsat -----
RX_IFULLSAT_METRICS = re.compile(
    r"^\s*IFULLSAT_METRICS\s+deg1=(\d+)\s+deg2=(\d+)\s+deggt2=(\d+)\s+from_orig=(\d+)\s*$", re.I
)

# ----- fast diagnostics for X = V(Ifullsat) -----
RX_X_EMPTY   = re.compile(r"^\s*X_EMPTY\s+isEmpty=(0|1)\s*$", re.I)
RX_X_IRR     = re.compile(r"^\s*X_IRREDUCIBLE\s+irreducible=(0|1)\s*$", re.I)
RX_X_EQDIM   = re.compile(r"^\s*X_EQDIM\s+equidimensional=(0|1)\s*$", re.I)
RX_X_COMPS   = re.compile(
    r"^\s*X_COMPONENTS\s+num=(\d+)\s+compProjDims=\{([^}]*)\}\s+compAffDims=\{([^}]*)\}\s*$", re.I
)
RX_X_CONN    = re.compile(r"^\s*X_CONNECTED\s+connected=(0|1)\s*$", re.I)

def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")

def _parse_rad_out(txt: str) -> dict:
    res = {
        # comparisons (unsaturated)
        "I1_eq_Ifull": None,
        "I1_eq_radI1": None,
        "Ifull_eq_radIfull": None,
        "radI1_eq_radIfull": None,
        # comparisons (saturated + ordinary-vs-saturated)
        "I1_eq_I1sat": None,
        "Ifull_eq_Ifullsat": None,
        "I1sat_eq_Ifullsat": None,
        "I1sat_eq_radI1sat": None,
        # original gens (if printed)
        "orig_gens_I1": None,
        "orig_gens_Ifull": None,
        # GB sizes (explicit)
        "GB_I1": None, "GB_Ifull": None, "GB_I1sat": None, "GB_Ifullsat": None,
        # GB degree/provenance metrics for Ifullsat
        "n_deg1": None, "n_deg2": None, "n_deggt2": None, "n_same": None,
        # geometric diagnostics for X = V(Ifullsat)
        "is_empty": None,
        "is_irreducible": None,
        "is_equidimensional": None,
        "irred_proj_dims": None,   # list[int]
        "irred_aff_dims": None,    # list[int]
        "is_connected": None,
    }


    for line in txt.splitlines():
        m = RX_I_EQ.search(line)
        if m: res["I1_eq_Ifull"] = _BOOL[m.group(1).lower()]; continue

        m = RX_I1_RAD.search(line)
        if m: res["I1_eq_radI1"] = _BOOL[m.group(1).lower()]; continue
        m = RX_IFULL_RAD.search(line)
        if m: res["Ifull_eq_radIfull"] = _BOOL[m.group(1).lower()]; continue
        m = RX_RADS_EQ.search(line)
        if m: res["radI1_eq_radIfull"] = _BOOL[m.group(1).lower()]; continue

        m = RX_I1_SAT_EQ.search(line)
        if m: res["I1_eq_I1sat"] = _BOOL[m.group(1).lower()]; continue
        m = RX_IFULL_SAT_EQ.search(line)
        if m: res["Ifull_eq_Ifullsat"] = _BOOL[m.group(1).lower()]; continue
        m = RX_SAT_EQ.search(line)
        if m: res["I1sat_eq_Ifullsat"] = _BOOL[m.group(1).lower()]; continue
        m = RX_I1SAT_RAD.search(line)
        if m: res["I1sat_eq_radI1sat"] = _BOOL[m.group(1).lower()]; continue

        m = RX_ORIG_I1.search(line)
        if m: res["orig_gens_I1"] = int(m.group(1)); continue
        m = RX_ORIG_IFULL.search(line)
        if m: res["orig_gens_Ifull"] = int(m.group(1)); continue

        m = RX_GB_I1.search(line)
        if m: res["GB_I1"] = int(m.group(1)); continue
        m = RX_GB_IFULL.search(line)
        if m: res["GB_Ifull"] = int(m.group(1)); continue
        m = RX_GB_I1SAT.search(line)
        if m: res["GB_I1sat"] = int(m.group(1)); continue
        m = RX_GB_IFULLSAT.search(line)
        if m: res["GB_Ifullsat"] = int(m.group(1)); continue

        # --- Ifullsat GB degree/provenance metrics ---
        m = RX_IFULLSAT_METRICS.search(line)
        if m:
            res["n_deg1"]   = int(m.group(1))
            res["n_deg2"]   = int(m.group(2))
            res["n_deggt2"] = int(m.group(3))
            res["n_same"]   = int(m.group(4))
            continue

        # --- fast diagnostics for X = V(Ifullsat) ---
        m = RX_X_EMPTY.search(line)
        if m:
            res["is_empty"] = (m.group(1) == "1")
            continue
        m = RX_X_IRR.search(line)
        if m:
            res["is_irreducible"] = (m.group(1) == "1")
            continue
        m = RX_X_EQDIM.search(line)
        if m:
            res["is_equidimensional"] = (m.group(1) == "1")
            continue
        m = RX_X_COMPS.search(line)
        if m:
            # num = int(m.group(1))  # not strictly needed; we trust the lists
            def _nums(s):
                s = s.strip()
                if not s:
                    return []
                return [int(t) for t in re.findall(r"-?\d+", s)]
            res["irred_proj_dims"] = _nums(m.group(2))
            res["irred_aff_dims"]  = _nums(m.group(3))
            continue
        m = RX_X_CONN.search(line)
        if m:
            res["is_connected"] = (m.group(1) == "1")
            continue

    return res

# ---------------------------- manifest helpers ----------------------------

def _normalize_dim_vector(obj) -> list | None:
    """
    Accepts dict[int->int], dict[str->int], or list/tuple of ints; returns list in key order.
    Returns None if obj is missing/unusable.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        try:
            keys = list(obj.keys())
            # convert numeric strings to ints
            def key_to_int(k):
                try:
                    return int(k)
                except Exception:
                    return k
            pairs = [(key_to_int(k), v) for k, v in obj.items()]
            # keep only int keys and sort
            pairs = [(int(k), int(v)) for k, v in pairs if isinstance(k, int)]
            if not pairs:
                return None
            pairs.sort(key=lambda t: t[0])
            return [v for _, v in pairs]
        except Exception:
            return None
    if isinstance(obj, (list, tuple)):
        try:
            return [int(x) for x in obj]
        except Exception:
            return None
    return None


def _load_manifest(run_root: Path) -> dict:
    """
    Returns mapping:
      { "000": {"quiver": "...", "module": "...",
                "ambient_dim": [...], "target_dim": [...] }, ... }
    Accepts several key spellings for dims (robust to emitter variants).
    """
    manifest_path = run_root / "manifest.yaml"
    try:
        import yaml
    except Exception:
        raise RuntimeError("PyYAML is required to read manifest.yaml")

    doc = yaml.safe_load(manifest_path.read_text(encoding="utf-8", errors="ignore")) or {}
    out = {}
    for j in (doc.get("jobs") or []):
        jid = str(j.get("id", ""))
        name_q = j.get("quiver")
        name_m = j.get("module")

        ambient = (
            j.get("ambient_dim")
        )
        target  = (
            j.get("target_dim")
        )

        ambient_vec = _normalize_dim_vector(ambient)
        target_vec  = _normalize_dim_vector(target)

        out[jid] = {
            "quiver": name_q,
            "module": name_m,
            "ambient_dim": ambient_vec,
            "target_dim": target_vec,
        }
        # allow lookups without leading zeros too
        out.setdefault(jid.lstrip("0") or "0", out[jid])
    return out


# ---------------------------- main entry + table ----------------------------

def parse_quiver_jobs_rad(path: str = ".", write_json_path: str | None = None,
                          write_csv_path: str | None = "parsed.csv") -> dict:
    """
    STRICT MODE: `path` is the run root containing:
      - manifest.yaml
      - jobs/<NNN_*>/rad_out.txt (and optionally rad_err.txt)
    """
    run_root = Path(path).resolve()
    names = _load_manifest(run_root)
    jobs_dir = run_root / "jobs"

    rows = []
    for job_path in sorted(p for p in jobs_dir.iterdir() if p.is_dir()):
        label = job_path.name                      # e.g., '007_xxx'
        job_num = label.split("_", 1)[0]          # '007'
        # display JUST THE NUMBER, without leading zeros if you prefer:
        task_number = int(job_num) if job_num.isdigit() else job_num

        meta = names.get(job_num) or names.get(job_num.lstrip("0") or "0") or {}
        quiver = meta.get("quiver")
        module = meta.get("module")
        ambient_dim = meta.get("ambient_dim")
        target_dim  = meta.get("target_dim")

        out_file = job_path / "rad_out.txt"
        err_file = job_path / "rad_err.txt"

        if not out_file.exists():
            rows.append({
                "task": task_number,
                "quiver": quiver, "module": module,
                "ambient_dim": ambient_dim, "target_dim": target_dim,
                "status": "missing_rad_out",
                # comparisons:
                "I1_eq_Ifull": None, "I1_eq_radI1": None,
                "I1_eq_I1sat": None, "Ifull_eq_Ifullsat": None,
                "I1sat_eq_Ifullsat": None, "I1sat_eq_radI1sat": None,
                # gens + GB sizes:
                "orig_gens_I1": None, "orig_gens_Ifull": None,
                "GB_I1": None, "GB_Ifull": None, "GB_I1sat": None, "GB_Ifullsat": None,
                # GB metrics for Ifullsat
                "n_deg1": None, "n_deg2": None, "n_deggt2": None, "n_same": None,
                # geometric diagnostics for X
                "is_empty": None, "is_irreducible": None, "is_equidimensional": None,
                "irred_proj_dims": None, "irred_aff_dims": None, "is_connected": None,
                "stderr_nonempty": (err_file.exists() and err_file.stat().st_size > 0),
            })

            continue

        txt = _read_text(out_file)
        res = _parse_rad_out(txt)

        rows.append({
            "task": task_number,
            "quiver": quiver, "module": module,
            "ambient_dim": ambient_dim, "target_dim": target_dim,
            "status": "ok",
            # comparisons:
            "I1_eq_Ifull": res["I1_eq_Ifull"],
            "I1_eq_radI1": res["I1_eq_radI1"],
            "I1_eq_I1sat": res["I1_eq_I1sat"],
            "Ifull_eq_Ifullsat": res["Ifull_eq_Ifullsat"],
            "I1sat_eq_Ifullsat": res["I1sat_eq_Ifullsat"],
            "I1sat_eq_radI1sat": res["I1sat_eq_radI1sat"],
            # gens + GB sizes:
            "orig_gens_I1": res["orig_gens_I1"],
            "orig_gens_Ifull": res["orig_gens_Ifull"],
            "GB_I1": res.get("GB_I1"),
            "GB_Ifull": res.get("GB_Ifull"),
            "GB_I1sat": res.get("GB_I1sat"),
            "GB_Ifullsat": res.get("GB_Ifullsat"),
            # GB metrics for Ifullsat
            "n_deg1": res.get("n_deg1"),
            "n_deg2": res.get("n_deg2"),
            "n_deggt2": res.get("n_deggt2"),
            "n_same": res.get("n_same"),
            # geometric diagnostics for X
            "is_empty": res.get("is_empty"),
            "is_irreducible": res.get("is_irreducible"),
            "is_equidimensional": res.get("is_equidimensional"),
            "irred_proj_dims": res.get("irred_proj_dims"),
            "irred_aff_dims": res.get("irred_aff_dims"),
            "is_connected": res.get("is_connected"),
            "stderr_nonempty": (err_file.exists() and err_file.stat().st_size > 0),
        })

    result = {
        "run_root": str(run_root),
        "jobs_root": str(jobs_dir),
        "results": rows,
    }
    if write_json_path:
        Path(write_json_path).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if write_csv_path:
        csv_path = Path(write_csv_path)
        # columns requested by you
        fieldnames = [
            "task", "quiver", "module", "ambient_dim", "target_dim",
            "I1_eq_Ifull", "I1_eq_radI1", "I1_eq_I1sat", "Ifull_eq_Ifullsat",
            "I1sat_eq_Ifullsat", "I1sat_eq_radI1sat",
            "orig_gens_I1", "orig_gens_Ifull",
            "GB_I1", "GB_Ifull", "GB_I1sat", "GB_Ifullsat",
            "n_deg1", "n_deg2", "n_deggt2", "n_same",
            "is_empty", "is_irreducible", "is_equidimensional",
            "irred_dims", "is_connected",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                # stringify complex fields
                irred_dims = " ".join(str(x) for x in (r.get("irred_proj_dims") or []))
                row = dict(r)  # shallow copy
                row["ambient_dim"]   = " ".join(str(x) for x in (r.get("ambient_dim") or []))
                row["target_dim"]    = " ".join(str(x) for x in (r.get("target_dim")  or []))
                row["irred_dims"]    = irred_dims
                # booleans as TRUE/FALSE in CSV
                for k in ["is_empty","is_irreducible","is_equidimensional","is_connected",
                          "I1_eq_Ifull","I1_eq_radI1","I1_eq_I1sat","Ifull_eq_Ifullsat",
                          "I1sat_eq_Ifullsat","I1sat_eq_radI1sat","stderr_nonempty"]:
                    v = row.get(k)
                    if v is True:  row[k] = "TRUE"
                    elif v is False: row[k] = "FALSE"
                    elif v is None: row[k] = ""
                w.writerow(row)

    return result


def print_results_table_rad(parsed: dict) -> None:
    def fmt_bool(v):
        if v is True:  return "✔"
        if v is False: return "✘"
        return "—"

    def fmt_dim(vec):
        return "—" if not vec else "[" + ", ".join(str(x) for x in vec) + "]"

    headers = [
        "Task", "Quiver", "Module", "dim(M)", "e (target)",
        "I1=Ifull", "I1=rad(I1)", "I1=I1sat", "Ifull=Ifullsat",
        "I1sat=Ifullsat", "rad(I1sat)=I1sat",
        "#G(I1)", "#G(Ifull)", "GB(I1)", "GB(Ifull)", "GB(I1sat)", "GB(Ifullsat)",
        # quick GB metrics (Ifullsat)
        "deg1", "deg2", "deg>2", "from_orig",
        # geometric diagnostics for X
        "X empty", "X irreducible", "X equidim", "X comp proj dims", "X connected",
    ]

    rows = parsed.get("results", [])
    table = []
    for r in rows:
        irred_dims = r.get("irred_proj_dims")
        irred_dims_str = "—" if not irred_dims else "[" + " ".join(str(x) for x in irred_dims) + "]"
        table.append([
            r.get("task", ""),
            r.get("quiver") or "",
            r.get("module") or "",
            fmt_dim(r.get("ambient_dim")),
            fmt_dim(r.get("target_dim")),
            fmt_bool(r.get("I1_eq_Ifull")),
            fmt_bool(r.get("I1_eq_radI1")),
            fmt_bool(r.get("I1_eq_I1sat")),
            fmt_bool(r.get("Ifull_eq_Ifullsat")),
            fmt_bool(r.get("I1sat_eq_Ifullsat")),
            fmt_bool(r.get("I1sat_eq_radI1sat")),
            (r.get("orig_gens_I1") if r.get("orig_gens_I1") is not None else "—"),
            (r.get("orig_gens_Ifull") if r.get("orig_gens_Ifull") is not None else "—"),
            (r.get("GB_I1") if r.get("GB_I1") is not None else "—"),
            (r.get("GB_Ifull") if r.get("GB_Ifull") is not None else "—"),
            (r.get("GB_I1sat") if r.get("GB_I1sat") is not None else "—"),
            (r.get("GB_Ifullsat") if r.get("GB_Ifullsat") is not None else "—"),
            (r.get("n_deg1") if r.get("n_deg1") is not None else "—"),
            (r.get("n_deg2") if r.get("n_deg2") is not None else "—"),
            (r.get("n_deggt2") if r.get("n_deggt2") is not None else "—"),
            (r.get("n_same") if r.get("n_same") is not None else "—"),
            fmt_bool(r.get("is_empty")),
            fmt_bool(r.get("is_irreducible")),
            fmt_bool(r.get("is_equidimensional")),
            irred_dims_str,
            fmt_bool(r.get("is_connected")),
        ])

    widths = [max(len(str(x)) for x in col) for col in zip(headers, *table)] if table else [len(h) for h in headers]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in table:
        print("  ".join(str(cell).ljust(w) for cell, w in zip(row, widths)))


import re
from pathlib import Path

def collect_minimal_generators_markdown(path: str = "run_results",
                                        write_path: str = "ifullsat_minimal_gens.md") -> str:
    """
    No changes to M2 logs required.
    Extracts the block printed after:
        '===== MINIMAL GENERATORS (Ifullsat) ====='
    Skips 'iNN :' / 'oNN =' lines, stops at headers/prompts/tags/comments or lines containing
    '<<' or '=' (which indicate M2 code, not polynomial output). Outputs a Markdown file.
    """
    run_root = Path(path).resolve()
    jobs_dir = run_root / "jobs"
    manifest = _load_manifest(run_root)

    RX_HEAD   = re.compile(r"^\s*===== MINIMAL GENERATORS \(Ifullsat\) =====\s*$")
    RX_NEXTH  = re.compile(r"^\s*=====")                 # next header
    RX_PROMPT = re.compile(r"^\s*i\s*\d+\s*:\s")         # 'i46 : ...'
    RX_ECHO   = re.compile(r"^\s*o\s*\d+\s*=")           # 'o46 = stdio'
    RX_TAG    = re.compile(r"^\s*(IFULLSAT_|X_)")        # your tagged lines
    RX_COMMENT= re.compile(r"^\s*--")                    # M2 comment
    RX_CODE   = re.compile(r"<<|=")                     # strong code cue; safe for polys

    parts: list[str] = []
    parts.append("# Minimal generators of Ifullsat\n")

    for job_path in sorted(p for p in jobs_dir.iterdir() if p.is_dir()):
        label   = job_path.name
        job_num = label.split("_", 1)[0]
        meta    = manifest.get(job_num) or manifest.get(job_num.lstrip("0") or "0") or {}

        quiver  = meta.get("quiver") or ""
        module  = meta.get("module") or ""
        tgt     = meta.get("target_dim") or []

        parts.append(f"## {job_num} — {quiver or '(no quiver)'}\n")
        parts.append(f"representation: {module or '(none)'}\n")
        parts.append(f"target_dim: [{' '.join(str(x) for x in (tgt or []))}]\n")

        out_file = job_path / "rad_out.txt"
        if not out_file.exists():
            parts.append("_rad_out.txt missing_\n")
            continue

        txt = _read_text(out_file)
        lines = txt.splitlines()

        gens: list[str] = []
        i = 0

        # 1) Find printed header
        while i < len(lines) and not RX_HEAD.match(lines[i]):
            i += 1
        if i == len(lines):
            parts.append("_no 'MINIMAL GENERATORS (Ifullsat)' block found_\n")
            continue

        i += 1
        # 2) Skip blanks + immediate prompts/echoes
        while i < len(lines) and (lines[i].strip() == "" or RX_PROMPT.match(lines[i]) or RX_ECHO.match(lines[i])):
            i += 1

        # 3) Collect polynomial lines until a boundary
        while i < len(lines):
            s = lines[i]
            if (RX_NEXTH.match(s) or RX_PROMPT.match(s) or RX_ECHO.match(s) or
                RX_TAG.match(s) or RX_COMMENT.match(s) or RX_CODE.search(s)):
                break
            st = s.strip()
            if st:
                gens.append(st)
            i += 1

        if gens:
            parts.append("```m2")
            parts.extend(gens)
            parts.append("```\n")
        else:
            parts.append("_no minimal generators captured (Ifullsat)_\n")

    md = "\n".join(parts).rstrip() + "\n"
    if write_path:
        Path(write_path).write_text(md, encoding="utf-8")
    return md


def parse_ring_block(rad_out: Path) -> str:
    txt = rad_out.read_text(encoding="utf-8", errors="ignore")
    m = _RX_RING_BLOCK.search(txt)
    if not m:
        raise ValueError(f"No multi-line 'R = QQ[...] ;' ring block found in {rad_out}")
    ring_block = m.group(1).strip()
    if not ring_block.endswith(";"):
        ring_block += ";"
    return ring_block


