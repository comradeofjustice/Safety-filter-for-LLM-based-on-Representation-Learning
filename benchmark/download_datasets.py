#!/usr/bin/env python3
"""
Download safety evaluation datasets from ModelScope, OpenI, or GitHub.
Saves each as data.parquet in benchmark/<dataset_name>/.
Logs to benchmark/download_log_02.log.
"""

import os
import sys
import time
import json
import logging
import subprocess
import pandas as pd

BENCHMARK_DIR = "/root/autodl-tmp/llm-safety-classifier/benchmark"
LOG_FILE = os.path.join(BENCHMARK_DIR, "download_log_02.log")
MODELSCOPE_CACHE = os.path.expanduser("~/.cache/modelscope/hub/datasets")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def save_and_log(df, output_dir, dataset_name, source, elapsed):
    """Save dataframe as parquet and log metadata."""
    parquet_path = os.path.join(output_dir, "data.parquet")
    df.to_parquet(parquet_path, index=False)
    file_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
    num_samples = len(df)
    columns = list(df.columns)

    logger.info(f"  Dataset: {dataset_name}  |  Source: {source}")
    logger.info(f"  Samples: {num_samples}")
    logger.info(f"  Columns: {len(columns)} - {columns}")
    logger.info(f"  File size: {file_size_mb:.2f} MB")
    logger.info(f"  Download time: {elapsed:.1f}s")
    logger.info(f"  Saved to: {parquet_path}")

    return {
        "dataset": dataset_name,
        "source": source,
        "samples": num_samples,
        "columns": columns,
        "file_size_mb": f"{file_size_mb:.2f}",
        "time_seconds": f"{elapsed:.1f}",
        "status": "success",
    }


def fail_result(dataset_name, error_msg):
    logger.error(f"  FAILED: {dataset_name} - {error_msg}")
    return {
        "dataset": dataset_name,
        "source": "N/A",
        "samples": 0,
        "columns": [],
        "file_size_mb": "0",
        "time_seconds": "0",
        "status": f"failed: {error_msg[:200]}",
    }


# ─── 1. XGuard-Train-Open-200K via ModelScope ──────────────────────────

def download_xguard():
    dataset_name = "Alibaba-AAIG/XGuard-Train-Open-200K"
    output_dir = os.path.join(BENCHMARK_DIR, "XGuard-Train-Open-200K")
    os.makedirs(output_dir, exist_ok=True)
    start = time.time()

    logger.info(f"\n{'─' * 50}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Output:  {output_dir}")

    try:
        from modelscope.msdatasets import MsDataset

        logger.info("Loading via MsDataset (subset_name='xguard_train_open_200k')...")
        ds = MsDataset.load(
            dataset_name,
            subset_name="xguard_train_open_200k",
        )

        if hasattr(ds, "to_pandas"):
            df = ds["train"].to_pandas() if "train" in ds else ds.to_pandas()
        else:
            rows = list(ds)
            df = pd.DataFrame(rows)

        return save_and_log(df, output_dir, dataset_name, "ModelScope", time.time() - start)

    except Exception as e:
        return _try_openi_xguard(dataset_name, output_dir, start)


def _try_openi_xguard(dataset_name, output_dir, start):
    """Fallback: try OpenI community."""
    logger.info("ModelScope failed, trying OpenI...")
    try:
        # OpenI / pcl.ac.cn mirror
        repo_url = "https://openi.pcl.ac.cn/kafm/XGuard-Train-Open-200K.git"
        clone_dir = os.path.join(output_dir, "_clone")

        if not os.path.exists(clone_dir):
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, clone_dir],
                check=True, capture_output=True, text=True, timeout=120,
            )
        else:
            subprocess.run(
                ["git", "-C", clone_dir, "pull"],
                check=True, capture_output=True, text=True, timeout=60,
            )

        # Find data files in clone
        jsonl_files = []
        for root, _, files in os.walk(clone_dir):
            for f in files:
                if f.endswith(".jsonl") or f.endswith(".json"):
                    jsonl_files.append(os.path.join(root, f))

        if not jsonl_files:
            return fail_result(dataset_name, "No JSONL/JSON files in OpenI clone")

        dfs = []
        for fp in jsonl_files:
            logger.info(f"  Reading: {fp}")
            rows = []
            with open(fp, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            if rows:
                dfs.append(pd.DataFrame(rows))

        if not dfs:
            return fail_result(dataset_name, "Empty data files from OpenI")

        df = pd.concat(dfs, ignore_index=True)
        # Clean up clone
        subprocess.run(["rm", "-rf", clone_dir])
        return save_and_log(df, output_dir, dataset_name, "OpenI", time.time() - start)

    except Exception as e2:
        # Also try direct JSONL download
        return _try_jsonl_download(
            dataset_name, output_dir, start,
            "https://openi.pcl.ac.cn/kafm/XGuard-Train-Open-200K/raw/branch/main/data/train.jsonl",
        )


def _try_jsonl_download(dataset_name, output_dir, start, *urls):
    """Try downloading raw JSONL from URLs."""
    import urllib.request

    for url in urls:
        try:
            logger.info(f"  Trying direct download: {url}")
            tmp_path = os.path.join(output_dir, "_temp.jsonl")
            urllib.request.urlretrieve(url, tmp_path)
            rows = []
            with open(tmp_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            df = pd.DataFrame(rows)
            os.remove(tmp_path)
            if len(df) > 0:
                return save_and_log(df, output_dir, dataset_name, "Direct URL", time.time() - start)
        except Exception:
            continue

    return fail_result(dataset_name, "All sources exhausted")


# ─── 2. CRiskEval via GitHub ───────────────────────────────────────

def download_criskeval():
    dataset_name = "CRiskEval"
    output_dir = os.path.join(BENCHMARK_DIR, "CRiskEval")
    os.makedirs(output_dir, exist_ok=True)
    start = time.time()

    logger.info(f"\n{'─' * 50}")
    logger.info(f"Dataset: {dataset_name} (GitHub: tjunlp-lab/CRiskEval)")
    logger.info(f"Output:  {output_dir}")

    try:
        repo_url = "https://github.com/tjunlp-lab/CRiskEval.git"
        clone_dir = os.path.join(output_dir, "_clone")

        if not os.path.exists(clone_dir):
            logger.info("Cloning from GitHub...")
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, clone_dir],
                check=True, capture_output=True, text=True, timeout=120,
            )

        # Look for data files
        data_files = []
        for root, _, files in os.walk(clone_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in (".json", ".jsonl", ".csv", ".tsv", ".parquet"):
                    data_files.append(os.path.join(root, f))

        if not data_files:
            return fail_result(dataset_name, "No data files found in repo")

        logger.info(f"  Found {len(data_files)} data files: {[os.path.basename(f) for f in data_files]}")

        # Try to assemble dataframe
        dfs = []
        for fp in data_files:
            if fp.endswith(".csv"):
                df_chunk = pd.read_csv(fp)
            elif fp.endswith(".tsv"):
                df_chunk = pd.read_csv(fp, sep="\t")
            elif fp.endswith(".parquet"):
                df_chunk = pd.read_parquet(fp)
            elif fp.endswith(".jsonl"):
                rows = []
                with open(fp, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))
                df_chunk = pd.DataFrame(rows)
            elif fp.endswith(".json"):
                with open(fp, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    df_chunk = pd.DataFrame(data)
                else:
                    df_chunk = pd.DataFrame([data])
            else:
                continue

            if len(df_chunk) > 0:
                logger.info(f"  Loaded {len(df_chunk)} rows from {os.path.basename(fp)}")
                dfs.append(df_chunk)

        if not dfs:
            return fail_result(dataset_name, "Could not parse any data files")

        df = pd.concat(dfs, ignore_index=True)
        subprocess.run(["rm", "-rf", clone_dir])
        return save_and_log(df, output_dir, dataset_name, "GitHub", time.time() - start)

    except Exception as e:
        return fail_result(dataset_name, f"{type(e).__name__}: {e}")


# ─── 3. MM-SafetyBench via ModelScope / GitHub ──────────────────────

def download_mmsafety():
    dataset_name = "MM-SafetyBench"
    output_dir = os.path.join(BENCHMARK_DIR, "MM-SafetyBench")
    os.makedirs(output_dir, exist_ok=True)
    start = time.time()

    logger.info(f"\n{'─' * 50}")
    logger.info(f"Dataset: {dataset_name} (PKU-Alignment)")
    logger.info(f"Output:  {output_dir}")

    # Try ModelScope first
    try:
        from modelscope.msdatasets import MsDataset
        logger.info("Trying ModelScope: PKU-Alignment/MM-SafetyBench...")
        ds = MsDataset.load("PKU-Alignment/MM-SafetyBench")
        if hasattr(ds, "to_pandas"):
            df = ds["train"].to_pandas() if "train" in ds else ds.to_pandas()
        else:
            df = pd.DataFrame(list(ds))
        if len(df) > 0:
            return save_and_log(df, output_dir, dataset_name, "ModelScope", time.time() - start)
    except Exception as e:
        logger.info(f"  ModelScope failed: {e}")

    # Try GitHub fallback
    try:
        logger.info("Trying GitHub: AI45Lab/MM-SafetyBench...")
        repo_url = "https://github.com/AI45Lab/MM-SafetyBench.git"
        clone_dir = os.path.join(output_dir, "_clone")

        if not os.path.exists(clone_dir):
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, clone_dir],
                check=True, capture_output=True, text=True, timeout=120,
            )

        # Collect data files
        data_files = []
        for root, _, files in os.walk(clone_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in (".json", ".jsonl", ".csv", ".tsv", ".parquet", ".txt"):
                    data_files.append(os.path.join(root, f))

        if not data_files:
            return fail_result(dataset_name, "No data files in GitHub repo")

        logger.info(f"  Found {len(data_files)} data/annotation files")

        # MM-SafetyBench has text prompts in .txt and annotations in .csv
        # Filter to relevant directories
        text_files = [f for f in data_files if "SD" in f or "TYPO" in f or "text" in f.lower()]
        csv_files = [f for f in data_files if f.endswith(".csv")]

        # Build from CSV annotations if available
        if csv_files:
            dfs = []
            for fp in csv_files:
                df_chunk = pd.read_csv(fp)
                if len(df_chunk) > 0:
                    logger.info(f"  Loaded {len(df_chunk)} rows from {os.path.basename(fp)}")
                    dfs.append(df_chunk)
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                subprocess.run(["rm", "-rf", clone_dir])
                return save_and_log(df, output_dir, dataset_name, "GitHub", time.time() - start)

        # Fallback: build DataFrame from text files
        if text_files:
            rows = []
            for fp in text_files:
                scenario = os.path.basename(os.path.dirname(fp))
                with open(fp, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            rows.append({"scenario": scenario, "prompt": line})
            if rows:
                df = pd.DataFrame(rows)
                subprocess.run(["rm", "-rf", clone_dir])
                return save_and_log(df, output_dir, dataset_name, "GitHub", time.time() - start)

        subprocess.run(["rm", "-rf", clone_dir])
        return fail_result(dataset_name, "Could not extract data from GitHub repo")

    except Exception as e:
        return fail_result(dataset_name, f"{type(e).__name__}: {e}")


# ─── Main ───────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("Safety Dataset Download — Started")
    logger.info(f"Output: {BENCHMARK_DIR}")
    logger.info("=" * 60)

    results = []

    # 1. XGuard
    results.append(download_xguard())

    # 2. CRiskEval
    results.append(download_criskeval())

    # 3. MM-SafetyBench
    results.append(download_mmsafety())

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("Download Summary")
    logger.info("=" * 60)
    for r in results:
        icon = "[OK]" if r["status"] == "success" else "[FAIL]"
        logger.info(
            f"  {icon} {r['dataset']} | {r['source']} | "
            f"{r['samples']} samples, {r['file_size_mb']} MB, {r['time_seconds']}s"
        )

    failures = [r for r in results if r["status"] != "success"]
    if failures:
        logger.warning(f"\n{failures} dataset(s) failed. Check the log for details.")
        sys.exit(1)
    else:
        logger.info("\nAll datasets downloaded successfully.")


if __name__ == "__main__":
    main()
