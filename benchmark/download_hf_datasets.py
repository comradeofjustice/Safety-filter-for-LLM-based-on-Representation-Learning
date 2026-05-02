#!/usr/bin/env python3
"""Download safety evaluation datasets from HuggingFace mirror and save as parquet.

Datasets:
  1. walledai/XSTest         — over-safety detection, 450 prompts (NAACL 2024)
  2. lmsys/toxic-chat        — real-user toxicity detection, 10K+ samples
  3. nvidia/Aegis-AI-Content-Safety-Dataset-2.0 — content safety, 33K (NAACL 2025)
  4. PKU-Alignment/BeaverTails — safety alignment, 330K+, 14 harm categories
  5. LibrAI/do-not-answer    — LLM refusal eval, 939 prompts, 61 harm types

Uses HF_ENDPOINT=https://hf-mirror.com.  Falls back to GitHub raw for gated datasets.
"""

import os
import sys
import time
import logging
import pandas as pd
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names

BENCHMARK_DIR = "/root/autodl-tmp/llm-safety-classifier/benchmark"
LOG_FILE = os.path.join(BENCHMARK_DIR, "download_log_03.log")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def save_parquet(df, output_dir, dataset_name, start_time, extra_info=None):
    """Save DataFrame as parquet and log result."""
    os.makedirs(output_dir, exist_ok=True)
    parquet_path = os.path.join(output_dir, "data.parquet")
    df.to_parquet(parquet_path, index=False)

    elapsed = time.time() - start_time
    size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
    samples = len(df)
    fields = list(df.columns)

    logger.info(
        f"DONE dataset={dataset_name} samples={samples} "
        f"fields={fields} size_mb={size_mb:.1f} time_s={elapsed:.0f} "
        f"({fmt_time(elapsed)})"
    )
    if extra_info:
        logger.info(f"  Extra: {extra_info}")

    return {
        "dataset": dataset_name,
        "samples": samples,
        "fields": fields,
        "size_mb": round(size_mb, 1),
        "time_s": round(elapsed, 1),
        "status": "success",
    }


# ---------------------------------------------------------------------------
# Per-dataset downloaders
# ---------------------------------------------------------------------------

def download_xstest():
    """walledai/XSTest — gated on HF; use GitHub CSV fallback."""
    start_time = time.time()
    logger.info("START dataset=XSTest source=github-fallback")

    url = ("https://raw.githubusercontent.com/paul-rottger/"
           "exaggerated-safety/main/xstest_v2_prompts.csv")

    try:
        df = pd.read_csv(url)
    except Exception as e:
        logger.warning(f"  GitHub fallback failed: {e}")
        elapsed = time.time() - start_time
        logger.info(f"DONE dataset=XSTest samples=0 fields=[] "
                    f"size_mb=0 time_s={elapsed:.0f} status=failed")
        return None

    output_dir = os.path.join(BENCHMARK_DIR, "XSTest")
    return save_parquet(df, output_dir, "XSTest", start_time,
                        extra_info="GitHub fallback (gated on HF)")


def download_toxic_chat():
    """lmsys/toxic-chat — has configs: toxicchat0124, toxicchat1123."""
    start_time = time.time()
    logger.info("START dataset=ToxicChat source=huggingface path=lmsys/toxic-chat")

    configs = ["toxicchat0124", "toxicchat1123"]
    all_dfs = []

    for config in configs:
        try:
            # Get splits for this config
            splits = get_dataset_split_names("lmsys/toxic-chat", config)
            logger.info(f"  Config={config} splits={splits}")
        except Exception as e:
            logger.warning(f"  Could not list splits for config={config}: {e}")
            splits = ["train"]

        for split in splits:
            try:
                ds = load_dataset("lmsys/toxic-chat", config, split=split)
                df = ds.to_pandas()
                df["_config"] = config
                df["_split"] = split
                all_dfs.append(df)
                logger.info(f"  Loaded config={config} split={split} -> {len(df)} rows")
            except Exception as e:
                logger.warning(f"  SKIP config={config} split={split}: {e}")

    if not all_dfs:
        elapsed = time.time() - start_time
        logger.info(f"DONE dataset=ToxicChat samples=0 fields=[] "
                    f"size_mb=0 time_s={elapsed:.0f} status=failed")
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    output_dir = os.path.join(BENCHMARK_DIR, "ToxicChat")
    return save_parquet(combined, output_dir, "ToxicChat", start_time)


def download_aegis():
    """nvidia/Aegis-AI-Content-Safety-Dataset-2.0 — 3 splits."""
    start_time = time.time()
    logger.info("START dataset=AegisAI-Content-Safety-2.0 source=huggingface "
                "path=nvidia/Aegis-AI-Content-Safety-Dataset-2.0")

    splits = get_dataset_split_names("nvidia/Aegis-AI-Content-Safety-Dataset-2.0")
    logger.info(f"  Splits: {splits}")
    all_dfs = []

    for split in splits:
        ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split=split)
        df = ds.to_pandas()
        df["_split"] = split
        all_dfs.append(df)
        logger.info(f"  Loaded split={split} -> {len(df)} rows")

    combined = pd.concat(all_dfs, ignore_index=True)
    output_dir = os.path.join(BENCHMARK_DIR, "AegisAI-Content-Safety-2.0")
    return save_parquet(combined, output_dir, "AegisAI-Content-Safety-2.0", start_time)


def download_beavertails():
    """PKU-Alignment/BeaverTails — 4 splits: 330k_train/test, 30k_train/test."""
    start_time = time.time()
    logger.info("START dataset=BeaverTails source=huggingface "
                "path=PKU-Alignment/BeaverTails")

    splits = get_dataset_split_names("PKU-Alignment/BeaverTails")
    logger.info(f"  Splits: {splits}")
    all_dfs = []

    for split in splits:
        ds = load_dataset("PKU-Alignment/BeaverTails", split=split)
        df = ds.to_pandas()
        df["_split"] = split
        all_dfs.append(df)
        logger.info(f"  Loaded split={split} -> {len(df)} rows")

    combined = pd.concat(all_dfs, ignore_index=True)
    output_dir = os.path.join(BENCHMARK_DIR, "BeaverTails")
    return save_parquet(combined, output_dir, "BeaverTails", start_time)


def download_do_not_answer():
    """LibrAI/do-not-answer — 1 split (train), 939 prompts."""
    start_time = time.time()
    logger.info("START dataset=DoNotAnswer source=huggingface "
                "path=LibrAI/do-not-answer")

    splits = get_dataset_split_names("LibrAI/do-not-answer")
    logger.info(f"  Splits: {splits}")
    all_dfs = []

    for split in splits:
        ds = load_dataset("LibrAI/do-not-answer", split=split)
        df = ds.to_pandas()
        df["_split"] = split
        all_dfs.append(df)
        logger.info(f"  Loaded split={split} -> {len(df)} rows")

    combined = pd.concat(all_dfs, ignore_index=True)
    output_dir = os.path.join(BENCHMARK_DIR, "DoNotAnswer")
    return save_parquet(combined, output_dir, "DoNotAnswer", start_time)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("HF Mirror Safety Dataset Download — download_log_03")
    logger.info(f"HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    logger.info("=" * 60)

    downloaders = [
        ("XSTest", download_xstest),
        ("ToxicChat", download_toxic_chat),
        ("AegisAI-Content-Safety-2.0", download_aegis),
        ("BeaverTails", download_beavertails),
        ("DoNotAnswer", download_do_not_answer),
    ]

    results = []
    overall_start = time.time()

    for name, download_fn in downloaders:
        logger.info(f"\n{'─' * 55}")
        result = download_fn()
        if result is None:
            results.append({
                "dataset": name, "samples": 0, "fields": [],
                "size_mb": 0, "time_s": 0, "status": "failed",
            })
        else:
            results.append(result)

    # Summary
    total_time = time.time() - overall_start
    total_samples = sum(r["samples"] for r in results)
    total_size = sum(r["size_mb"] for r in results)
    logger.info(f"\n{'=' * 60}")
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    for r in results:
        icon = "OK" if r["status"] == "success" else "FAIL"
        logger.info(
            f"  [{icon}] {r['dataset']}: {r['samples']} samples, "
            f"{r['size_mb']} MB, {r['time_s']}s"
        )
    logger.info(
        f"  TOTAL: {total_samples} samples, {total_size:.1f} MB, "
        f"{fmt_time(total_time)}"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
