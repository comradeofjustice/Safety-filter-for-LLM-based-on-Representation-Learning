#!/usr/bin/env python3
"""Download safety evaluation datasets from HF mirror / GitHub and save as parquet.

Datasets:
  1. walledai/XSTest         — over-safety detection, 450 prompts (NAACL 2024)
  2. lmsys/toxic-chat        — real-user toxicity detection, 10K+ samples
  3. nvidia/Aegis-AI-Content-Safety-Dataset-2.0 — content safety, 33K (NAACL 2025)
  4. PKU-Alignment/BeaverTails — safety alignment, 330K+, 14 harm categories
  5. LibrAI/do-not-answer    — LLM refusal eval, 939 prompts, 61 harm types

Sources: HF mirror (hf-mirror.com), GitHub raw (gated fallback), HF cache (offline).
"""

import os
import sys
import io
import time
import json
import logging
import glob
import urllib.request

import pandas as pd

# ----- config -----
BENCHMARK_DIR = "/root/autodl-tmp/llm-safety-classifier/benchmark"
LOG_FILE = os.path.join(BENCHMARK_DIR, "download_log_03.log")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def save_parquet(df, dataset_name, start_time, extra_info=None):
    """Save DataFrame as parquet and log result."""
    output_dir = os.path.join(BENCHMARK_DIR, dataset_name)
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
        logger.info(f"  Note: {extra_info}")

    return {
        "dataset": dataset_name,
        "samples": samples,
        "fields": fields,
        "size_mb": round(size_mb, 1),
        "time_s": round(elapsed, 1),
        "status": "success",
    }


def log_fail(dataset_name, start_time, reason):
    elapsed = time.time() - start_time
    logger.info(
        f"DONE dataset={dataset_name} samples=0 fields=[] "
        f"size_mb=0 time_s={elapsed:.0f} status=failed ({reason})"
    )
    return None


# ---------------------------------------------------------------------------
# 1. walledai/XSTest — gated on HF, use GitHub CSV fallback
# ---------------------------------------------------------------------------
def download_xstest():
    start_time = time.time()
    logger.info("START dataset=XSTest source=github-fallback "
                "repo=paul-rottger/exaggerated-safety")

    url = ("https://raw.githubusercontent.com/paul-rottger/"
           "exaggerated-safety/main/xstest_prompts.csv")
    try:
        with urllib.request.urlopen(url, timeout=60) as f:
            raw = f.read()
        df = pd.read_csv(io.BytesIO(raw))
        logger.info(f"  Downloaded CSV from GitHub: {len(df)} rows")
    except Exception as e:
        return log_fail("XSTest", start_time, f"github-csv:{e}")

    return save_parquet(df, "XSTest", start_time,
                        extra_info="GitHub fallback (HF gated)")


# ---------------------------------------------------------------------------
# 2. lmsys/toxic-chat — cached from previous run, use offline mode
# ---------------------------------------------------------------------------
def download_toxic_chat():
    start_time = time.time()
    logger.info("START dataset=ToxicChat source=huggingface-cache "
                "path=lmsys/toxic-chat")

    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from datasets import load_dataset, get_dataset_split_names

        configs = ["toxicchat0124", "toxicchat1123"]
        all_dfs = []

        for cfg in configs:
            try:
                splits = get_dataset_split_names("lmsys/toxic-chat", cfg)
            except Exception:
                splits = ["train", "test"]

            for split in splits:
                try:
                    ds = load_dataset("lmsys/toxic-chat", cfg, split=split)
                    df = ds.to_pandas()
                    df["_config"] = cfg
                    df["_split"] = split
                    all_dfs.append(df)
                    logger.info(f"  Loaded {cfg}/{split}: {len(df)} rows")
                except Exception as e:
                    logger.warning(f"  SKIP {cfg}/{split}: {e}")

        if not all_dfs:
            return log_fail("ToxicChat", start_time, "no-splits-loaded")

        combined = pd.concat(all_dfs, ignore_index=True)
        return save_parquet(combined, "ToxicChat", start_time,
                            extra_info=f"configs={configs}")

    except Exception as e:
        return log_fail("ToxicChat", start_time, str(e)[:200])
    finally:
        os.environ["HF_HUB_OFFLINE"] = "0"


# ---------------------------------------------------------------------------
# 3. nvidia/Aegis-AI-Content-Safety-Dataset-2.0 — download JSONs via mirror
# ---------------------------------------------------------------------------
def download_aegis():
    start_time = time.time()
    logger.info("START dataset=AegisAI-Content-Safety-2.0 source=huggingface-mirror "
                "path=nvidia/Aegis-AI-Content-Safety-Dataset-2.0")

    from huggingface_hub import snapshot_download

    try:
        snapshot_dir = snapshot_download(
            "nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
            repo_type="dataset",
            allow_patterns=["*.json"],
            max_workers=1,
        )
        logger.info(f"  Downloaded to: {snapshot_dir}")

        all_dfs = []
        json_files = sorted(glob.glob(f"{snapshot_dir}/*.json"))
        for fp in json_files:
            fn = os.path.basename(fp)
            with open(fp, "r") as f:
                data = json.load(f)
            df = pd.DataFrame(data if isinstance(data, list) else [data])
            df["_source_file"] = fn
            all_dfs.append(df)
            logger.info(f"  Loaded {fn}: {len(df)} rows")

        combined = pd.concat(all_dfs, ignore_index=True)
        return save_parquet(combined, "AegisAI-Content-Safety-2.0", start_time,
                            extra_info=f"{len(json_files)} JSON files via snapshot_download")

    except Exception as e:
        return log_fail("AegisAI-Content-Safety-2.0", start_time, str(e)[:200])


# ---------------------------------------------------------------------------
# 4. PKU-Alignment/BeaverTails — cached, use offline mode
# ---------------------------------------------------------------------------
def download_beavertails():
    start_time = time.time()
    logger.info("START dataset=BeaverTails source=huggingface-cache "
                "path=PKU-Alignment/BeaverTails")

    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from datasets import load_dataset

        splits = ["330k_train", "330k_test", "30k_train", "30k_test"]
        all_dfs = []

        for split in splits:
            try:
                ds = load_dataset("PKU-Alignment/BeaverTails", split=split)
                df = ds.to_pandas()
                df["_split"] = split
                all_dfs.append(df)
                logger.info(f"  Loaded {split}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"  SKIP {split}: {e}")

        if not all_dfs:
            return log_fail("BeaverTails", start_time, "no-splits-loaded")

        combined = pd.concat(all_dfs, ignore_index=True)
        return save_parquet(combined, "BeaverTails", start_time)

    except Exception as e:
        return log_fail("BeaverTails", start_time, str(e)[:200])
    finally:
        os.environ["HF_HUB_OFFLINE"] = "0"


# ---------------------------------------------------------------------------
# 5. LibrAI/do-not-answer — cached, use offline mode
# ---------------------------------------------------------------------------
def download_do_not_answer():
    start_time = time.time()
    logger.info("START dataset=DoNotAnswer source=huggingface-cache "
                "path=LibrAI/do-not-answer")

    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from datasets import load_dataset

        ds = load_dataset("LibrAI/do-not-answer", split="train")
        df = ds.to_pandas()
        df["_split"] = "train"
        logger.info(f"  Loaded train: {len(df)} rows")

        return save_parquet(df, "DoNotAnswer", start_time)

    except Exception as e:
        return log_fail("DoNotAnswer", start_time, str(e)[:200])
    finally:
        os.environ["HF_HUB_OFFLINE"] = "0"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("HF Mirror Safety Dataset Download — download_log_03")
    logger.info(f"HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    logger.info(f"Output dir: {BENCHMARK_DIR}")
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

    for name, fn in downloaders:
        logger.info(f"\n{'─' * 55}")
        result = fn()
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
