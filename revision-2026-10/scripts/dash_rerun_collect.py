"""DASH re-run collector (revision v4 follow-up).

Read-only on the research folder. Writes everything to --out and finally one
long-format CSV:  DASH_rerun_results.csv
columns: section,item,key,value,status,source,note

Sub-commands
  provenance : environment, inventory+hashes, artefact check, original/November
               input identification (+ staging copies for latency runs), Study I
               candidate files, training-log hits, training-split/scaler
               reconstruction, threshold re-check, test-split overlap and
               test_metrics replication.
  merge      : summarise latency run directories (raw parquet) and write the
               final CSV (provenance rows + latency rows).

Nothing in the research folder is modified. Model artefacts are never re-saved.
"""
import argparse, ast, csv, glob, hashlib, json, os, platform, re, shutil, subprocess, sys, time, traceback, warnings
from datetime import datetime, timezone
from pathlib import Path

ROWS = []
FEATURE_COLS = ["r_t", "sigma_hat", "OFI_t", "Imbalance_t", "VolSpike_t", "msg_count"]
H, ALPHA = 30, 2.0
TAG = "H30_a2_L94"
PINNED = "53353760db37921a8da5e231169d9dcec3027cff"
ART_FILES = ["s1_xgb_H30_a2_L94.json", "s2_xgb_H30_a2_L94.json", "s1_rf_H30_a2_L94.pkl", "s2_rf_H30_a2_L94.pkl",
             "artifacts/s1_lr_H30_a2_L94.pkl", "s2_lr_H30_a2_L94.pkl", "scaler_H30_a2_L94.pkl", "thresholds_H30_a2_L94.json"]
SKIP_DIRS = {".git", "venv", ".venv", "env", "__pycache__", "node_modules", "site-packages", ".ipynb_checkpoints", ".idea", ".vscode"}
HASH_EXT = {".parquet", ".csv", ".json", ".pkl", ".pth", ".py", ".ipynb", ".feather", ".txt", ".log", ".md", ".zip", ".gz", ".tar"}
TEXT_EXT = {".txt", ".log", ".out", ".md", ".ipynb", ".csv"}
# v3 manuscript "Test Set" column (tab:warn), used only to check whether it can be reproduced
V3_TEST = {"XGBoost": (0.1729, 0.5571, 0.2639, 0.4981, 0.9499, 0.6535),
           "RandomForest": (0.1654, 0.7305, 0.2697, 0.4996, 0.9187, 0.6473),
           "Logistic": (0.1844, 0.5768, 0.2795, 0.5056, 0.8916, 0.6453)}
MODEL_KEY = {"XGBoost": "xgb", "RandomForest": "rf", "Logistic": "lr"}


def row(section, item, key, value, status="", source="", note=""):
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float):
        value = repr(value)
    ROWS.append(dict(section=section, item=str(item), key=str(key), value="" if value is None else str(value),
                     status=status, source=str(source), note=note))


def err(section, item, e):
    row("ERROR", f"{section}:{item}", type(e).__name__, str(e)[:500], "ERROR", "", traceback.format_exc(limit=3)[-800:])


def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def utc(ts=None):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat() if ts else datetime.now(timezone.utc).isoformat()


def write_rows(path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["section", "item", "key", "value", "status", "source", "note"])
        w.writeheader()
        w.writerows(ROWS)


def read_rows(path):
    with open(path, encoding="utf-8-sig") as f:
        ROWS.extend(csv.DictReader(f))


def flush_idx(p):
    m = re.search(r"flush_(\d+)\.parquet$", str(p))
    return int(m.group(1)) if m else None


# ----------------------------------------------------------------------------- env
def step_env(root, pinned):
    S = "ENV"
    row(S, "run", "time_utc", utc())
    row(S, "run", "collector_sha256", sha256(__file__))
    row(S, "host", "hostname", platform.node())
    row(S, "host", "platform", platform.platform())
    row(S, "host", "python", platform.python_version())
    row(S, "host", "cpu_count", os.cpu_count())
    try:
        if Path("/proc/cpuinfo").exists():
            m = re.search(r"model name\s*:\s*(.+)", Path("/proc/cpuinfo").read_text())
            row(S, "host", "cpu_model", m.group(1).strip() if m else "")
        else:
            row(S, "host", "cpu_model", platform.processor())
        if Path("/proc/meminfo").exists():
            m = re.search(r"MemTotal:\s*(\d+)", Path("/proc/meminfo").read_text())
            row(S, "host", "mem_total_bytes", int(m.group(1)) * 1024 if m else "")
    except Exception as e:
        err(S, "cpu", e)
    import importlib.metadata as md
    for p in ["numpy", "pandas", "pyarrow", "scikit-learn", "xgboost", "joblib", "redis", "psutil", "torch", "scipy"]:
        try:
            row(S, "package", p, md.version(p))
        except Exception:
            row(S, "package", p, "", "MISSING")
    for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        row(S, "threads", v, os.environ.get(v, ""))
    row(S, "paths", "research_root", root)
    row(S, "paths", "pinned_repo", pinned)
    for label, repo in [("research_root", root), ("pinned_repo", pinned)]:
        if (Path(repo) / ".git").exists() and shutil.which("git"):
            try:
                head = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True, timeout=60).strip()
                dirty = subprocess.check_output(["git", "-C", repo, "status", "--porcelain"], text=True, timeout=120)
                row("GIT", label, "HEAD", head, "MATCH_PINNED" if head == PINNED else "DIFFERENT_FROM_PINNED")
                row("GIT", label, "uncommitted_entries", len([l for l in dirty.splitlines() if l.strip()]))
            except Exception as e:
                err("GIT", label, e)
        else:
            row("GIT", label, "HEAD", "", "NOT_A_GIT_REPO_OR_NO_GIT")


def step_git_history(root):
    if not ((Path(root) / ".git").exists() and shutil.which("git")):
        return
    targets = ART_FILES + ["train_2.py", "test_metrics.py", "test_metrics2.py", "labeling.py", "data/labeled_train.parquet", "config.py"]
    for t in targets:
        try:
            out = subprocess.check_output(["git", "-C", root, "log", "--follow", "--format=%H|%aI|%s", "--", t],
                                          text=True, timeout=120).strip().splitlines()
            row("GIT_HISTORY", t, "n_commits", len(out))
            for i, line in enumerate(out[:8]):
                row("GIT_HISTORY", t, f"commit_{i}", line)
        except Exception as e:
            err("GIT_HISTORY", t, e)


# ----------------------------------------------------------------------------- inventory
def walk(root, out):
    out = Path(out).resolve()
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.lower().startswith(("dash_rerun", "_rerun", ".venv", "venv"))
                  and not (Path(dp, d) / "pyvenv.cfg").exists() and Path(dp, d).resolve() != out]
        for fn in fns:
            yield Path(dp) / fn


def step_inventory(root, out, max_hash_mb):
    S = "INVENTORY"
    files = []
    for p in walk(root, out):
        try:
            st = p.stat()
        except Exception:
            continue
        ext = p.suffix.lower()
        rel = os.path.relpath(p, root)
        h = ""
        if ext in HASH_EXT and st.st_size <= max_hash_mb * 1024 * 1024:
            try:
                h = sha256(p)
            except Exception as e:
                h = f"ERR:{type(e).__name__}"
        files.append(dict(path=p, rel=rel, size=st.st_size, mtime=st.st_mtime, sha=h, ext=ext))
        row(S, rel, "file", st.st_size, "", h, f"mtime_utc={utc(st.st_mtime)}")
    row("SUMMARY", "inventory", "files_listed", len(files))
    return files


def step_artefacts(root, pinned, ref):
    exp = json.loads((Path(ref) / "original_expected_sha256.json").read_text())
    for label, repo in [("research_root", root), ("pinned_repo", pinned)]:
        for n, h in exp["artifacts"].items():
            p = Path(repo) / n
            if not p.exists():
                row("ARTEFACT", f"{label}:{n}", "sha256", "", "MISSING")
                continue
            a = sha256(p)
            row("ARTEFACT", f"{label}:{n}", "sha256", a, "MATCH" if a == h else "DIFFERENT", str(p))
        for n, h in exp["repo_py"].items():
            p = Path(repo) / n
            if p.exists():
                a = sha256(p)
                row("REPO_PY", f"{label}:{n}", "sha256", a, "MATCH" if a == h else "DIFFERENT")
            else:
                row("REPO_PY", f"{label}:{n}", "sha256", "", "MISSING")


# ----------------------------------------------------------------------------- inputs
def identify_sets(files, ref, out, stage):
    import pandas as pd, numpy as np
    exp = json.loads((Path(ref) / "original_expected_sha256.json").read_text())
    orig = {v: k for k, v in exp["data_files"].items()}
    nov_map = json.loads((Path(ref) / "november_file_hashes.json").read_text())
    nov = {v: k for k, v in nov_map.items()}
    status = {}
    for label, want, total in [("original75", orig, len(exp["data_files"])), ("november2025", nov, len(nov_map))]:
        found = {}
        for f in files:
            if f["ext"] == ".parquet" and f["sha"] in want and want[f["sha"]] not in found:
                found[want[f["sha"]]] = f["path"]
        S = "INPUT_" + label.upper()
        row(S, label, "files_found", len(found), "COMPLETE" if len(found) == total else ("PARTIAL" if found else "NOT_FOUND"),
            "", f"expected {total} files by SHA-256")
        if found:
            dirs = sorted({str(Path(p).parent) for p in found.values()})
            for d in dirs[:10]:
                row(S, label, "found_in_dir", d)
        status[label] = dict(found=len(found), total=total, dir="")
        if len(found) == total:
            dst = Path(out) / "stage" / label
            dst.mkdir(parents=True, exist_ok=True)
            if stage:
                for name, p in found.items():
                    t = dst / name
                    if not t.exists():
                        shutil.copy2(p, t)
            status[label]["dir"] = str(dst)
            try:
                fs = sorted(found.items(), key=lambda kv: flush_idx(kv[0]))
                df = pd.concat([pd.read_parquet(p) for _, p in fs], ignore_index=True)
                row(S, label, "rows", len(df))
                row(S, label, "columns", "|".join(map(str, df.columns)))
                X = np.ascontiguousarray(df[FEATURE_COLS].to_numpy(np.float32))
                ih = hashlib.sha256(X.tobytes()).hexdigest()
                mh = hashlib.sha256("".join(f"{n}:{sha256(p)}\n" for n, p in fs).encode()).hexdigest()
                if label == "original75":
                    row(S, label, "input_array_sha256", ih, "MATCH" if ih == exp["input_array_sha256"] else "DIFFERENT")
                    row(S, label, "manifest_sha256", mh, "MATCH" if mh == exp["data_file_manifest_sha256"] else "DIFFERENT")
                else:
                    row(S, label, "input_array_sha256", ih, "MATCH" if ih.startswith("e1cd7d2a19eacd3b") else "DIFFERENT")
                    row(S, label, "manifest_sha256", mh, "MATCH" if mh.startswith("202f9ad9f7af9705") else "DIFFERENT")
                for c in ["sec", "ts", "timestamp", "time", "t"]:
                    if c in df.columns:
                        s = df[c]
                        row(S, label, f"{c}_min", s.min())
                        row(S, label, f"{c}_max", s.max())
                        try:
                            unit = "s" if float(s.max()) < 1e11 else ("ms" if float(s.max()) < 1e14 else "us")
                            row(S, label, f"{c}_min_utc", pd.to_datetime(s.min(), unit=unit, utc=True), "", "", f"assumed unit={unit}")
                            row(S, label, f"{c}_max_utc", pd.to_datetime(s.max(), unit=unit, utc=True), "", "", f"assumed unit={unit}")
                        except Exception:
                            pass
                        row(S, label, f"{c}_monotonic", bool(s.is_monotonic_increasing))
                        row(S, label, f"{c}_duplicates", int(s.duplicated().sum()))
                        break
            except Exception as e:
                err(S, label, e)
    (Path(out) / "stage_status.json").write_text(json.dumps(status, indent=2))
    return status


# ----------------------------------------------------------------------------- Study I candidates
def step_study1(files, root):
    import pandas as pd
    S = "STUDY1_CANDIDATE"
    stored = [0.513659, 0.503215, 0.500254]
    n = 0
    for f in files:
        if f["ext"] not in {".csv", ".parquet"} or f["size"] > 1_500_000_000:
            continue
        try:
            if f["ext"] == ".csv":
                with open(f["path"], encoding="utf-8-sig", errors="ignore") as fh:
                    cols = [c.strip() for c in fh.readline().split(",")]
            else:
                import pyarrow.parquet as pq
                cols = pq.read_schema(f["path"]).names
            if not ("engine" in cols and any(c in cols for c in ["total_ms", "latency_ms", "dash_p99", "p99"])):
                if not ({"dash_p99", "redis_p99"} <= set(cols)):
                    continue
            d = pd.read_csv(f["path"], low_memory=False) if f["ext"] == ".csv" else pd.read_parquet(f["path"])
            n += 1
            it = f["rel"]
            row(S, it, "rows", len(d), "", f["sha"], f"mtime_utc={utc(f['mtime'])}")
            row(S, it, "columns", "|".join(map(str, d.columns)))
            taus = set()
            for c in ["tau", "tau_s1", "tau_conf", "thr", "threshold"]:
                if c in d.columns:
                    vals = sorted(pd.to_numeric(d[c], errors="coerce").dropna().unique().tolist())
                    taus |= set(vals)
                    row(S, it, f"unique_{c}", "|".join(f"{v:.6f}" for v in vals[:20]))
            for c in ["model", "k", "K", "trial", "repeat", "run_id", "engine", "symbol", "regime", "N"]:
                if c in d.columns:
                    u = d[c].dropna().unique()
                    row(S, it, f"n_unique_{c}", len(u), "", "", "|".join(map(str, sorted(map(str, u))[:15])))
            match = any(abs(t - s) < 5e-6 for t in taus for s in stored)
            outer = None
            if "trial" in d.columns:
                outer = d["trial"].nunique() <= 10
            row(S, it, "verdict", "",
                "CANDIDATE_MATCH" if (match and outer) else ("THRESHOLD_MATCH_ONLY" if match else ("NO_THRESHOLD_COLUMN" if not taus else "THRESHOLD_MISMATCH")),
                "", "match needs stored model-specific thresholds AND <=10 outer trials")
        except Exception as e:
            err(S, f["rel"], e)
    row("SUMMARY", "study1", "candidate_files_examined", n)


# ----------------------------------------------------------------------------- logs
PATTERNS = [r"Split: train=\d+", r"Loaded data: .*rows=\d+", r"test rows: [\d,]+", r"chosen_thr=", r"Stage-1: Precision=",
            r"Stage-2: Precision=", r"Loaded tuned thresholds", r"loaded [\d,]+ rows", r"Labeling complete", r"\[OK\] saved",
            r"Fetching historical data from S3"]


def step_logs(files):
    S = "TRAINING_LOG_HIT"
    rx = re.compile("|".join(PATTERNS))
    hits = 0
    for f in files:
        if f["ext"] not in TEXT_EXT or f["size"] > 50_000_000 or f["ext"] == ".csv":
            continue
        try:
            with open(f["path"], encoding="utf-8", errors="ignore") as fh:
                for i, line in enumerate(fh, 1):
                    if rx.search(line):
                        row(S, f["rel"], f"line_{i}", line.strip()[:400], "", f["sha"], f"mtime_utc={utc(f['mtime'])}")
                        hits += 1
                        if hits > 400:
                            return
        except Exception as e:
            err(S, f["rel"], e)
    row("SUMMARY", "logs", "hits", hits)


# ----------------------------------------------------------------------------- training reconstruction
def load_helpers(pinned):
    import numpy as np, pandas as pd
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from typing import Optional, Dict, Any, List, Tuple
    src = (Path(pinned) / "train_2.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(src)
    names = {"infer_label_col", "sort_time_if_possible", "train_val_split_time_with_gap", "fit_scaler", "_safe_div", "_f_beta", "choose_threshold"}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = dict(np=np, pd=pd, precision_recall_curve=precision_recall_curve, average_precision_score=average_precision_score,
              Optional=Optional, Dict=Dict, Any=Any, List=List, Tuple=Tuple)
    exec(compile(ast.Module(body=body, type_ignores=[]), "train_2_helpers", "exec"), ns)
    fn = next(n for n in body if n.name == "choose_threshold")
    fsrc = ast.get_source_segment(src, fn)
    fixed = fsrc.replace("prec2 = prec[1:]", "prec2 = prec[:-1]").replace("rec2 = rec[1:]", "rec2 = rec[:-1]").replace("def choose_threshold(", "def choose_threshold_fixed(")
    assert fixed != fsrc, "indexing lines not found"
    exec(compile(fixed, "fixed", "exec"), ns)
    return ns


def forward_return(r, h):
    import numpy as np
    r = r.to_numpy()
    cs = np.concatenate([[0.0], np.cumsum(r)])
    out = np.full(len(r), np.nan)
    idx = np.arange(len(r) - h)
    out[idx] = cs[idx + h + 1] - cs[idx + 1]
    return out


def label(df):
    import numpy as np
    df = df.copy()
    df["future_r"] = forward_return(df["r_t"], H)
    thr = ALPHA * np.sqrt(H) * df["sigma_hat"]
    df["y"] = 0
    df.loc[df["future_r"] >= thr, "y"] = 2
    df.loc[df["future_r"] <= -thr, "y"] = 1
    return df[df["future_r"].notna()].reset_index(drop=True)


def candidate_datasets(files, root):
    """yield (name, dataframe, how) for files/dirs containing the six features."""
    import pandas as pd, pyarrow.parquet as pq
    flush_dirs = {}
    for f in files:
        if f["ext"] == ".parquet" and flush_idx(f["path"]) is not None:
            flush_dirs.setdefault(str(Path(f["path"]).parent), []).append(f["path"])
    for d, fs in sorted(flush_dirs.items()):
        try:
            if not all(c in pq.read_schema(fs[0]).names for c in FEATURE_COLS):
                continue
            fs = sorted(fs, key=flush_idx)
            df = pd.concat([pd.read_parquet(p) for p in fs], ignore_index=True)
            rel = os.path.relpath(d, root)
            row("TRAINING_SOURCE", rel, "flush_files", len(fs), "", "", f"rows={len(df)}")
            yield f"{rel}|last230000|flush_order", df.tail(230000).reset_index(drop=True), "labeling.py style (MinIO loader, no time sort)"
            if "sec" in df.columns:
                yield f"{rel}|last230000|sorted_sec", df.sort_values("sec", kind="mergesort").tail(230000).reset_index(drop=True), "test_metrics.py style (time-sorted)"
            yield f"{rel}|full", df, "whole directory"
        except Exception as e:
            err("TRAINING_SOURCE", d, e)
    for f in files:
        if f["ext"] not in {".parquet", ".csv", ".feather"} or flush_idx(f["path"]) is not None or f["size"] > 2_000_000_000:
            continue
        try:
            if f["ext"] == ".parquet":
                cols = pq.read_schema(f["path"]).names
            elif f["ext"] == ".csv":
                with open(f["path"], encoding="utf-8-sig", errors="ignore") as fh:
                    cols = [c.strip() for c in fh.readline().split(",")]
            else:
                cols = list(pd.read_feather(f["path"]).columns)
            if not all(c in cols for c in FEATURE_COLS):
                continue
            df = pd.read_parquet(f["path"]) if f["ext"] == ".parquet" else (pd.read_csv(f["path"]) if f["ext"] == ".csv" else pd.read_feather(f["path"]))
            row("TRAINING_SOURCE", f["rel"], "standalone_file", len(df), "", f["sha"], f"mtime_utc={utc(f['mtime'])}")
            yield f"{f['rel']}|file", df, "standalone file (train_2 --data)"
        except Exception as e:
            err("TRAINING_SOURCE", f["rel"], e)


def split_variants(n):
    out = []
    for vr in (0.2, 0.1, 0.15, 0.25, 0.3):
        for gap in (30, 0):
            split = max(1, min(int(n * (1 - vr)), n - 1))
            out.append((f"train_2 val_ratio={vr} gap={gap}", 0, max(1, split - gap), split, n))
    ntr, nva = int(n * 0.70), int(n * 0.15)
    out.append(("70/15/15 (test_metrics)", 0, ntr, ntr + H, ntr + H + nva))
    return out


def step_training(files, root, pinned, out):
    import numpy as np, pandas as pd, joblib
    ns = load_helpers(pinned)
    sc = joblib.load(Path(pinned) / f"scaler_{TAG}.pkl")
    mu0, sd0 = np.asarray(sc["mu"], np.float64), np.asarray(sc["sd"], np.float64)
    row("SCALER", "artefact", "feat_cols", "|".join(sc.get("feat_cols", [])))
    row("SCALER", "artefact", "mu", "|".join(f"{x:.9g}" for x in mu0))
    row("SCALER", "artefact", "sd", "|".join(f"{x:.9g}" for x in sd0))
    best_overall = None
    tested = 0
    for name, raw, how in candidate_datasets(files, root):
        try:
            lab_col = None
            try:
                lab_col = ns["infer_label_col"](raw)
            except Exception:
                pass
            if lab_col:
                df = ns["sort_time_if_possible"](raw).rename(columns={lab_col: "y"})
                lab_how = f"existing label column '{lab_col}'; train_2 time sort"
            else:
                df = label(raw)
                lab_how = "labels recomputed as labeling.py (H=30, alpha=2), NaN tail dropped"
            n = len(df)
            X = df[FEATURE_COLS].to_numpy(np.float32)
            best = None
            for vname, a, tr_end, va0, va1 in split_variants(n):
                mu = X[a:tr_end].mean(axis=0).astype(np.float64)
                sd = X[a:tr_end].std(axis=0).astype(np.float64) + 1e-8
                rmu = float(np.max(np.abs(mu - mu0) / sd0))
                rsd = float(np.max(np.abs(sd - sd0) / sd0))
                score = max(rmu, rsd)
                if best is None or score < best[0]:
                    best = (score, vname, tr_end, va0, va1, rmu, rsd)
            tested += 1
            score, vname, tr_end, va0, va1, rmu, rsd = best
            st = "SCALER_MATCH" if score < 1e-4 else ("NEAR" if score < 1e-2 else "NO_MATCH")
            row("TRAINING_CANDIDATE", name, "best_split_variant", vname, st, "", f"{how}; {lab_how}; n_labelled={n}; max_rel_mu={rmu:.3g}; max_rel_sd={rsd:.3g}")
            if best_overall is None or score < best_overall[0]:
                best_overall = (score, name, df, vname, tr_end, va0, va1, st, lab_how)
            if "sorted_sec" in name or name.endswith("|file"):
                test_metrics_replication(name, df, pinned, ns)
        except Exception as e:
            err("TRAINING_CANDIDATE", name, e)
    row("SUMMARY", "training", "candidates_tested", tested)
    if best_overall is None:
        row("TRAINING_SPLIT", "none", "status", "", "NO_CANDIDATE_DATA", "", "no file/dir with the six feature columns found")
        return
    score, name, df, vname, tr_end, va0, va1, st, lab_how = best_overall
    S = "TRAINING_SPLIT"
    row(S, name, "status", st, st, "", f"best scaler agreement over all candidates; max relative diff={score:.3g}")
    row(S, name, "variant", vname)
    row(S, name, "n_labelled_rows", len(df))
    row(S, name, "train_rows", f"0..{tr_end - 1}", "", "", f"count={tr_end}")
    row(S, name, "validation_rows", f"{va0}..{va1 - 1}", "", "", f"count={va1 - va0}")
    for c in ["sec", "ts", "timestamp", "time", "t"]:
        if c in df.columns:
            for lab_, i in [("train_first", 0), ("train_last", tr_end - 1), ("val_first", va0), ("val_last", va1 - 1)]:
                v = df[c].iloc[i]
                row(S, name, f"{c}_{lab_}", v)
            break
    else:
        row(S, name, "time_column", "", "ABSENT", "", "labelled data has no time column; dates cannot be derived from it")
    if st != "SCALER_MATCH":
        row(S, name, "threshold_recheck", "", "SKIPPED", "", "no dataset reproduces the scaler; validation set unidentified")
        return
    threshold_recheck(name, df, va0, va1, pinned, ns)
    # overlap of test_metrics 70/15/15 test interval with the identified validation interval
    n = len(df)
    t0 = int(n * 0.70) + H + int(n * 0.15) + H
    ov = max(0, min(n, va1) - max(t0, va0))
    row("TESTSPLIT_OVERLAP", name, "test_metrics_test_rows", f"{t0}..{n - 1}", "", "", f"count={n - t0}")
    row("TESTSPLIT_OVERLAP", name, "overlap_with_validation_rows", ov, "OVERLAP" if ov > 0 else "DISJOINT")


def load_models(pinned, model):
    import joblib
    P = Path(pinned)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        if model == "XGBoost":
            from xgboost import XGBClassifier
            s1, s2 = XGBClassifier(), XGBClassifier()
            s1.load_model(str(P / f"s1_xgb_{TAG}.json")); s2.load_model(str(P / f"s2_xgb_{TAG}.json"))
        elif model == "RandomForest":
            s1, s2 = joblib.load(P / f"s1_rf_{TAG}.pkl"), joblib.load(P / f"s2_rf_{TAG}.pkl")
        else:
            s1, s2 = joblib.load(P / "artifacts" / f"s1_lr_{TAG}.pkl"), joblib.load(P / f"s2_lr_{TAG}.pkl")
    return s1, s2, [str(w.message)[:200] for w in ws]


def prf(y, pred):
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return float(p), float(r), float(f)


def threshold_recheck(name, df, va0, va1, pinned, ns):
    import numpy as np, joblib
    S = "THRESHOLD_RECHECK"
    thr = json.loads((Path(pinned) / f"thresholds_{TAG}.json").read_text())
    sc = joblib.load(Path(pinned) / f"scaler_{TAG}.pkl")
    mu, sd = np.asarray(sc["mu"], np.float32), np.asarray(sc["sd"], np.float32)
    va = df.iloc[va0:va1]
    X = (va[FEATURE_COLS].to_numpy(np.float32) - mu) / sd
    y3 = va["y"].astype(int).to_numpy()
    y1 = (y3 != 0).astype(int)
    shock = y3 != 0
    y2 = (y3[shock] == 2).astype(int)
    row(S, name, "validation_rows", len(va), "", "", f"shock_rows={int(shock.sum())}")
    for model in ["Logistic", "RandomForest", "XGBoost"]:
        try:
            s1, s2, w = load_models(pinned, model)
            if w:
                row(S, model, "load_warnings", " | ".join(w), "WARN")
            for stage, yt, pp in [("s1", y1, s1.predict_proba(X)[:, 1]), ("s2", y2, s2.predict_proba(X[shock])[:, 1])]:
                st = thr["models"][MODEL_KEY[model]][stage]
                leg = ns["choose_threshold"](yt, pp, policy="f1")
                fix = ns["choose_threshold_fixed"](yt, pp, policy="f1")
                it = f"{model}:{stage}"
                row(S, it, "stored_thr", st["thr"])
                row(S, it, "legacy_routine_thr_on_identified_val", leg["thr"],
                    "REPRODUCES_STORED" if abs(leg["thr"] - st["thr"]) < 1e-9 else "DIFFERS_FROM_STORED")
                row(S, it, "legacy_routine_reported_PRF", f"{leg['precision']:.6f}|{leg['recall']:.6f}|{leg['f1']:.6f}", "", "", "what the defective routine writes")
                row(S, it, "stored_PRF_in_json", f"{st['precision']:.6f}|{st['recall']:.6f}|{st['f1']:.6f}")
                a = prf(yt, (pp >= st["thr"]).astype(int))
                row(S, it, "achieved_PRF_at_stored_thr", f"{a[0]:.6f}|{a[1]:.6f}|{a[2]:.6f}", "", "", "true metrics at the stored operating point")
                row(S, it, "corrected_routine_thr", fix["thr"], "", "", "F1-max with prec[:-1]/rec[:-1]")
                b = prf(yt, (pp >= fix["thr"]).astype(int))
                row(S, it, "achieved_PRF_at_corrected_thr", f"{b[0]:.6f}|{b[1]:.6f}|{b[2]:.6f}")
                row(S, it, "gate_rate_stored_vs_corrected", f"{float((pp >= st['thr']).mean()):.6f}|{float((pp >= fix['thr']).mean()):.6f}")
            if model == "XGBoost":
                try:
                    b = s2.get_booster()
                    row(S, "XGBoost:s2", "booster_attributes", json.dumps(b.attributes()))
                    row(S, "XGBoost:s2", "n_trees", b.num_boosted_rounds())
                    row(S, "XGBoost:s1", "n_trees", s1.get_booster().num_boosted_rounds())
                except Exception as e:
                    err(S, "xgb_trees", e)
        except Exception as e:
            err(S, model, e)


def test_metrics_replication(name, df, pinned, ns):
    """Replicates test_metrics.py logic (70/15/15) on a candidate and compares with v3 tab:warn test column."""
    import numpy as np, joblib
    S = "TEST_METRICS_REPLICATION"
    thr = json.loads((Path(pinned) / f"thresholds_{TAG}.json").read_text())
    sc = joblib.load(Path(pinned) / f"scaler_{TAG}.pkl")
    mu, sd = np.asarray(sc["mu"], np.float32), np.asarray(sc["sd"], np.float32)
    n = len(df)
    t0 = int(n * 0.70) + H + int(n * 0.15) + H
    te = df.iloc[t0:]
    X = (te[FEATURE_COLS].to_numpy(np.float32) - mu) / sd
    y = te["y"].astype(int).to_numpy()
    row(S, name, "test_rows", len(te))
    for model in ["XGBoost", "RandomForest", "Logistic"]:
        try:
            s1, s2, _ = load_models(pinned, model)
            k = MODEL_KEY[model]
            p1 = s1.predict_proba(X)[:, 1]
            pred1 = (p1 >= thr["models"][k]["s1"]["thr"]).astype(int)
            a = prf((y > 0).astype(int), pred1)
            m = pred1 == 1
            b = (0.0, 0.0, 0.0)
            if m.sum():
                b = prf((y[m] == 2).astype(int), (s2.predict_proba(X[m])[:, 1] >= thr["models"][k]["s2"]["thr"]).astype(int))
            got = tuple(round(v, 4) for v in a + b)
            exp = V3_TEST[model]
            same = all(abs(g - e) < 6e-5 for g, e in zip(got, exp))
            row(S, f"{name}:{model}", "S1_P|R|F1|S2_P|R|F1", "|".join(f"{v:.4f}" for v in got),
                "REPRODUCES_V3_TEST_COLUMN" if same else "DIFFERS_FROM_V3", "", "v3=" + "|".join(f"{v:.4f}" for v in exp))
        except Exception as e:
            err(S, f"{name}:{model}", e)


# ----------------------------------------------------------------------------- latency merge
REF_LAPTOP = {1: (0.888235, 2, 0.087366), 100: (1.434662, 4, 0.234205)}  # Table 7 laptop, XGBoost


def merge_latency(out):
    import numpy as np, pandas as pd
    for pf in sorted(glob.glob(str(Path(out) / "latency_*" / "preflight.json"))):
        try:
            j = json.loads(Path(pf).read_text())
            row("LATENCY_PREFLIGHT", Path(pf).parent.name, "status", j.get("status"), j.get("status"), "", "; ".join(j.get("issues", []))[:400])
        except Exception as e:
            err("LATENCY_PREFLIGHT", pf, e)
    runs = sorted(glob.glob(str(Path(out) / "latency_*" / "extension_*")))
    row("SUMMARY", "latency", "run_directories", len(runs))
    for rd in runs:
        rd = Path(rd)
        tag = f"{rd.parent.name}/{rd.name}"
        try:
            meta = json.loads((rd / "metadata.json").read_text())
            for k in ["run_status", "env_label", "dataset_status", "sha_check", "input_sha256", "manifest_sha256", "python", "platform",
                      "cpu_count", "repo_commit", "end_utc", "error"]:
                if k in meta:
                    row("LATENCY_META", tag, k, meta[k])
            rs = meta.get("redis_server", {}) or {}
            row("LATENCY_META", tag, "redis_version", rs.get("redis_version", ""))
            rp = meta.get("redis_persistence", {}) or {}
            row("LATENCY_META", tag, "aof_enabled", rp.get("aof_enabled", ""))
            for p, v in (meta.get("versions") or {}).items():
                row("LATENCY_META", tag, f"version_{p}", v)
            for m, info in (meta.get("model_load") or {}).items():
                row("LATENCY_META", tag, f"model_load_{m}", json.dumps(info)[:400])
            fs = sorted(rd.glob("lat__*.parquet"))
            if not fs:
                row("LATENCY_SUMMARY", tag, "status", "", "NO_RECORDS")
                continue
            d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
            row("LATENCY_SUMMARY", tag, "records", len(d), "", "", f"failures={int((~d.success).sum())}")
            mism = gate = 0
            for (m, k, r), g in d.groupby(["model", "K", "repeat"]):
                p = g.pivot(index="request_sequence", columns="engine", values="prediction")
                q = g.pivot(index="request_sequence", columns="engine", values="stage2_fired")
                mism += int(p[["B1", "B2"]].ne(p.A, axis=0).sum().sum())
                gate += int(q[["B1", "B2"]].ne(q.A, axis=0).sum().sum())
            row("LATENCY_SUMMARY", tag, "prediction_mismatches_vs_A", mism, "OK" if mism == 0 else "MISMATCH")
            row("LATENCY_SUMMARY", tag, "stage2_mismatches_vs_A", gate, "OK" if gate == 0 else "MISMATCH")
            gg = d[d.success].groupby(["model", "K", "repeat", "engine"])
            per = pd.DataFrame(dict(n=gg.size(), p50=gg.latency_ms.quantile(.5), p99=gg.latency_ms.quantile(.99),
                                    mean=gg.latency_ms.mean(), state_mean=gg.state_ms.mean(), stage2=gg.stage2_fired.mean())).reset_index()
            per.to_csv(rd / "per_repeat_recomputed.csv", index=False)
            for x in per.itertuples():
                row("LATENCY_PER_REPEAT", f"{tag}:{x.model}:K{x.K}:r{x.repeat}:{x.engine}", "n|p50|p99|mean|state_mean|stage2",
                    f"{x.n}|{x.p50:.6f}|{x.p99:.6f}|{x.mean:.6f}|{x.state_mean:.6f}|{x.stage2:.4f}")
            for (m, k), g in per.groupby(["model", "K"]):
                p99 = g.pivot(index="repeat", columns="engine", values="p99")
                p50 = g.pivot(index="repeat", columns="engine", values="p50")
                rr = p99.B1 / p99.A
                cv = float(p99.B2.std(ddof=1) / p99.B2.mean())
                med = float(rr.median())
                it = f"{tag}:{m}:K{k}"
                status = "NOISY_INCONCLUSIVE" if cv > 0.25 else ("SCOPE_LIMITED" if med <= 1.10 else "DESCRIPTIVE")
                row("LATENCY_SUMMARY", it, "median_ratio_p99_B1_A", round(med, 6), status, "", f"repeats>1={int((rr > 1).sum())}/{len(rr)}; ratios=" + "|".join(f"{v:.4f}" for v in rr))
                row("LATENCY_SUMMARY", it, "B2_cv", round(cv, 6), "NOISY" if cv > 0.25 else "OK")
                row("LATENCY_SUMMARY", it, "mean_p99_A|B1|B2", "|".join(f"{p99[e].mean():.6f}" for e in ["A", "B1", "B2"]), "", "", "SD=" + "|".join(f"{p99[e].std(ddof=1):.6f}" for e in ["A", "B1", "B2"]))
                row("LATENCY_SUMMARY", it, "mean_p50_A|B1|B2", "|".join(f"{p50[e].mean():.6f}" for e in ["A", "B1", "B2"]))
                pr = p50.B1 / p50.A
                row("LATENCY_SUMMARY", it, "p50_ratio_B1_A_min|max|n_gt1", f"{pr.min():.4f}|{pr.max():.4f}|{int((pr > 1).sum())}")
                row("LATENCY_SUMMARY", it, "stage2_rate", round(float(g.stage2.mean()), 4))
                if m == "XGBoost" and "original" in str(meta.get("dataset_status", "")) and int(k) in REF_LAPTOP:
                    ref = REF_LAPTOP[int(k)]
                    row("LATENCY_SUMMARY", it, "reference_2026_10_02_laptop_median|gt1|cv", f"{ref[0]}|{ref[1]}|{ref[2]}", "REFERENCE", "", "original laptop run (Table 7) for comparison only; not pooled")
        except Exception as e:
            err("LATENCY", tag, e)


def dataset_info(name, pattern, out):
    import pandas as pd, numpy as np
    dsf = Path(out) / "rows_datasets.csv"
    if dsf.exists():
        read_rows(dsf)
    S = "DATASET"
    try:
        fs = sorted(glob.glob(pattern), key=lambda f: flush_idx(f) if flush_idx(f) is not None else -1)
        row(S, name, "files", len(fs), "", pattern)
        if not fs:
            row(S, name, "status", "", "NOT_FOUND")
        else:
            df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
            row(S, name, "rows", len(df))
            X = np.ascontiguousarray(df[FEATURE_COLS].to_numpy(np.float32))
            row(S, name, "input_array_sha256", hashlib.sha256(X.tobytes()).hexdigest())
            if "sec" in df.columns:
                s = df["sec"]
                if pd.api.types.is_datetime64_any_dtype(s):
                    conv = lambda v: pd.Timestamp(v)
                else:
                    unit = "s" if float(s.max()) < 1e11 else ("ms" if float(s.max()) < 1e14 else "us")
                    conv = lambda v: pd.to_datetime(v, unit=unit, utc=True)
                for lab, part in [("full", s), ("first8200_used", s.iloc[:8200])]:
                    row(S, name, f"{lab}_utc_range", f"{conv(part.min())} .. {conv(part.max())}")
                    row(S, name, f"{lab}_duplicate_ts", int(part.duplicated().sum()))
                row(S, name, "monotonic", bool(s.is_monotonic_increasing))
            row(S, name, "nan_or_inf", int((~np.isfinite(X)).sum()))
    except Exception as e:
        err(S, name, e)
    write_rows(dsf)
    print(f"[collect] dataset-info {name}: done")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["provenance", "merge", "dataset-info"])
    ap.add_argument("--name")
    ap.add_argument("--glob")
    ap.add_argument("--root")
    ap.add_argument("--pinned")
    ap.add_argument("--ref", default=str(Path(__file__).resolve().parent / "ref"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-hash-mb", type=int, default=2048)
    ap.add_argument("--no-stage", action="store_true")
    ap.add_argument("--skip-training", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    prov = out / "rows_provenance.csv"
    if a.command == "provenance":
        root, pinned = str(Path(a.root).resolve()), str(Path(a.pinned).resolve())
        t = time.time()
        for name, fn in [("env", lambda: step_env(root, pinned)), ("git_history", lambda: step_git_history(root))]:
            try:
                fn()
            except Exception as e:
                err("STEP", name, e)
        files = []
        try:
            files = step_inventory(root, out, a.max_hash_mb)
        except Exception as e:
            err("STEP", "inventory", e)
        for name, fn in [("artefacts", lambda: step_artefacts(root, pinned, a.ref)),
                         ("inputs", lambda: identify_sets(files, a.ref, out, not a.no_stage)),
                         ("study1", lambda: step_study1(files, root)),
                         ("logs", lambda: step_logs(files))] + ([] if a.skip_training else [("training", lambda: step_training(files, root, pinned, out))]):
            print(f"[collect] {name} ...", flush=True)
            try:
                fn()
            except Exception as e:
                err("STEP", name, e)
            write_rows(prov)
        row("SUMMARY", "provenance", "seconds", round(time.time() - t, 1))
        write_rows(prov)
        print(f"[collect] provenance rows: {len(ROWS)} -> {prov}")
    elif a.command == "dataset-info":
        dataset_info(a.name, a.glob, out)
    else:
        if prov.exists():
            read_rows(prov)
        dsf = out / "rows_datasets.csv"
        if dsf.exists():
            read_rows(dsf)
        try:
            merge_latency(out)
        except Exception as e:
            err("STEP", "merge_latency", e)
        final = out / "DASH_rerun_results.csv"
        write_rows(final)
        n_err = sum(1 for r in ROWS if r["status"] == "ERROR")
        print(f"[collect] final CSV: {final}  rows={len(ROWS)}  errors={n_err}")


if __name__ == "__main__":
    main()
