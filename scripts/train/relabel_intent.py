#!/usr/bin/env python3
"""R3: LLM intent relabeling using DeepSeek v4 API.
Relabels training set + 9 held-out benchmarks with 3-class intent labels.
Output: data/processed/intent_labels_v2/{train,eval}.parquet

Usage:
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8
    taskset -c 0-7 python scripts/train/relabel_intent.py
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

# Paths
PROJECT_DIR = Path("/root/autodl-tmp/llm-safety-classifier")
PROMPT_PATH = PROJECT_DIR / "src" / "data" / "intent_relabel_prompt.txt"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "intent_labels_v2"
LOG_FILE = PROJECT_DIR / "logs" / "relabel_intent.log"

# DeepSeek API
API_BASE = "https://api.deepseek.com"
API_KEY = "sk-884d57cc0e64487887440a1a201e31a1"
MODEL = "deepseek-v4-flash"

# Concurrency
MAX_CONCURRENT = 64
BATCH_SIZE = 500  # save every N samples
RETRY_LIMIT = 5
TIMEOUT = 60.0

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROJECT_DIR / "logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def load_prompt_template() -> str:
    with open(PROMPT_PATH) as f:
        return f.read()


def build_prompt(text: str, template: str) -> str:
    # Truncate very long texts to fit API context
    truncated = text[:6000] if len(text) > 6000 else text
    return template.replace("{text}", truncated)


def load_training_data() -> pd.DataFrame:
    """Load training data from processed parquet."""
    train_path = PROJECT_DIR / "data" / "processed" / "train.parquet"
    df = pd.read_parquet(train_path)
    logger.info(f"Loaded training data: {len(df)} samples")
    return df


def load_heldout_data() -> dict[str, pd.DataFrame]:
    """Load all 9 held-out benchmarks and unify to {text, label} schema."""
    datasets = {}

    # Helper: safely extract text and label from various schemas
    def make_df(df, text_col, label_val):
        """Create a standardized [text, label] dataframe."""
        df = df.copy()
        if isinstance(text_col, str):
            df["text"] = df[text_col].fillna("").astype(str)
        elif callable(text_col):
            df["text"] = df.apply(text_col, axis=1)
        if callable(label_val):
            df["label"] = df.apply(label_val, axis=1)
        elif isinstance(label_val, (int, float)):
            df["label"] = int(label_val)
        elif isinstance(label_val, str):
            df["label"] = df[label_val].apply(
                lambda x: 0 if str(x).lower() in ("safe", "0", "false") else 1
            )
        return df[["text", "label"]]

    # AegisAI-Content-Safety-1.0 — has 'text' directly, labels from labels_0..labels_4
    for fname, key in [("test.parquet", "AegisAI-1.0"), ("train.parquet", "AegisAI-1.0-train")]:
        path = PROJECT_DIR / f"benchmark/AegisAI-Content-Safety-1.0/{fname}"
        if path.exists():
            df = pd.read_parquet(path)
            # Check schema: v1 has text + labels_0..4, processed version has text + label
            if "text" in df.columns and "label" in df.columns:
                datasets[key] = df[["text", "label"]]
            elif "labels_0" in df.columns:
                # Raw Aegis schema: derive label from labels_* columns
                def aegis_v1_label(row):
                    for k in ["labels_0", "labels_1", "labels_2", "labels_3", "labels_4"]:
                        if k in row and str(row[k]).lower() == "safe":
                            return 0
                    return 1
                df["label"] = df.apply(aegis_v1_label, axis=1)
                datasets[key] = df[["text", "label"]]
            logger.info(f"{key}: {len(df)} samples")

    # AegisAI-Content-Safety-2.0 — has prompt+response+prompt_label
    aegis2 = PROJECT_DIR / "benchmark/AegisAI-Content-Safety-2.0/data.parquet"
    if aegis2.exists():
        df = pd.read_parquet(aegis2)
        df["text"] = df.apply(
            lambda r: f"[User Query] {r.get('prompt','')}\n\n[LLM Response] {r.get('response','')}",
            axis=1,
        )
        df["label"] = df["prompt_label"].apply(lambda x: 0 if str(x).lower() == "safe" else 1)
        datasets["AegisAI-2.0"] = df[["text", "label"]]
        logger.info(f"AegisAI-2.0: {len(df)} samples")

    # BeaverTails
    bt = PROJECT_DIR / "benchmark/BeaverTails/data.parquet"
    if bt.exists():
        df = pd.read_parquet(bt)
        df["text"] = df.apply(lambda r: f"{r.get('prompt','')}\n{r.get('response','')}", axis=1)
        df["label"] = df["is_safe"].apply(lambda x: 0 if x else 1)
        datasets["BeaverTails"] = df[["text", "label"]]
        logger.info(f"BeaverTails: {len(df)} samples")

    # CRiskEval
    cre = PROJECT_DIR / "benchmark/CRiskEval/data.parquet"
    if cre.exists():
        df = pd.read_parquet(cre)
        df["text"] = df["question"].fillna("")
        df["label"] = 1  # all risk-related
        datasets["CRiskEval"] = df[["text", "label"]]
        logger.info(f"CRiskEval: {len(df)} samples")

    # DoNotAnswer
    dna = PROJECT_DIR / "benchmark/DoNotAnswer/data.parquet"
    if dna.exists():
        df = pd.read_parquet(dna)
        df["text"] = df["question"].fillna("")
        df["label"] = 1
        datasets["DoNotAnswer"] = df[["text", "label"]]
        logger.info(f"DoNotAnswer: {len(df)} samples")

    # MM-SafetyBench
    mm = PROJECT_DIR / "benchmark/MM-SafetyBench/data.parquet"
    if mm.exists():
        df = pd.read_parquet(mm)
        df["text"] = df["question"].fillna("")
        df["label"] = 1
        datasets["MM-SafetyBench"] = df[["text", "label"]]
        logger.info(f"MM-SafetyBench: {len(df)} samples")

    # SafetyBench — use _repo clone
    sb_data = PROJECT_DIR / "benchmark/SafetyBench/_repo/data"
    if sb_data.exists():
        records = []
        for f in sorted(sb_data.glob("*.json")):
            try:
                with open(f) as fh:
                    items = json.load(fh)
                    if isinstance(items, list):
                        for item in items:
                            text = str(item.get("question", item)) if isinstance(item, dict) else str(item)
                            records.append({"text": text, "label": 1})
            except Exception:
                pass
        if records:
            datasets["SafetyBench"] = pd.DataFrame(records)
            logger.info(f"SafetyBench: {len(records)} samples")

    # ToxicChat
    tc = PROJECT_DIR / "benchmark/ToxicChat/data.parquet"
    if tc.exists():
        df = pd.read_parquet(tc)
        df["text"] = df["user_input"].fillna("")
        df["label"] = df["toxicity"].apply(lambda x: 0 if x == 0 else 1)
        datasets["ToxicChat"] = df[["text", "label"]]
        logger.info(f"ToxicChat: {len(df)} samples")

    # XSTest
    xs = PROJECT_DIR / "benchmark/XSTest/data.parquet"
    if xs.exists():
        df = pd.read_parquet(xs)
        df["text"] = df["prompt"].fillna("")
        df["label"] = df["label"].apply(lambda x: 0 if str(x).lower() == "safe" else 1)
        datasets["XSTest"] = df[["text", "label"]]
        logger.info(f"XSTest: {len(df)} samples")

    return datasets


async def call_deepseek(
    client: httpx.AsyncClient,
    text: str,
    template: str,
    semaphore: asyncio.Semaphore,
) -> int | None:
    """Call DeepSeek API to classify intent of a single text.
    deepseek-v4-flash is a reasoning model; it consumes tokens for reasoning
    before producing the answer, so we need higher max_tokens.
    """
    prompt = build_prompt(text, template)

    for attempt in range(RETRY_LIMIT):
        try:
            async with semaphore:
                resp = await client.post(
                    f"{API_BASE}/v1/chat/completions",
                    json={
                        "model": MODEL,
                        "messages": [
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 512,
                        "temperature": 0.0,
                    },
                    headers={
                        "Authorization": f"Bearer {API_KEY}",
                        "Content-Type": "application/json",
                    },
                    timeout=httpx.Timeout(TIMEOUT),
                )

            if resp.status_code == 200:
                msg = resp.json()["choices"][0]["message"]
                content = msg.get("content", "").strip()
                finish = resp.json()["choices"][0].get("finish_reason", "")
                # Extract integer from response
                for ch in content:
                    if ch in "012":
                        return int(ch)
                if finish == "length":
                    logger.debug(f"Content truncated (length limit): [{content[:80]}]")
                elif content:
                    logger.debug(f"Could not parse int from: [{content[:80]}]")
                return None
            elif resp.status_code == 429:
                wait = min(2 ** attempt, 30)
                await asyncio.sleep(wait)
            else:
                logger.warning(f"API error {resp.status_code}: {resp.text[:200]}")
                await asyncio.sleep(1)
        except Exception as e:
            logger.warning(f"Request failed (attempt {attempt+1}): {e}")
            await asyncio.sleep(1)

    return None


async def relabel_dataset(
    df: pd.DataFrame,
    template: str,
    name: str,
    max_samples: int | None = None,
) -> pd.DataFrame:
    """Relabel a dataset with intent labels using DeepSeek API."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # For large datasets, sample a representative subset for relabeling
    if max_samples and len(df) > max_samples:
        # Stratified sample
        if "label" in df.columns:
            safe = df[df["label"] == 0]
            unsafe = df[df["label"] == 1]
            n_per = max_samples // 2
            safe_sample = safe.sample(min(len(safe), n_per), random_state=42)
            unsafe_sample = unsafe.sample(min(len(unsafe), n_per), random_state=42)
            df = pd.concat([safe_sample, unsafe_sample]).sample(frac=1, random_state=42)
        else:
            df = df.sample(max_samples, random_state=42)
        logger.info(f"  Sampled {len(df)} from {name}")

    text_list = df["text"].tolist()
    results = []

    async with httpx.AsyncClient() as client:
        tasks = [call_deepseek(client, t, template, semaphore) for t in text_list]
        results = await asyncio.gather(*tasks)

    df = df.copy()
    df["intent"] = results
    df = df.dropna(subset=["intent"])
    df["intent"] = df["intent"].astype(int)

    n_safe = (df["intent"] == 0).sum()
    n_benign = (df["intent"] == 1).sum()
    n_mal = (df["intent"] == 2).sum()
    logger.info(f"  {name}: intent=0(safe):{n_safe}, 1(benign_sensitive):{n_benign}, 2(malicious):{n_mal}")

    return df


async def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("R3: LLM Intent Relabeling — Starting")
    logger.info("=" * 60)

    template = load_prompt_template()

    # 1. Relabel training data
    logger.info("Step 1: Relabel training data")
    train_df = load_training_data()
    # For training data, relabel a substantial subset (70K)
    # The full set is 256K, which would take too long via API
    # We sample 70K stratified + keep full dataset for downstream use
    train_relabeled = await relabel_dataset(train_df, template, "train", max_samples=70000)
    train_relabeled.to_parquet(OUTPUT_DIR / "train.parquet", index=False)
    logger.info(f"Saved {len(train_relabeled)} train samples to train.parquet")

    # 2. Relabel held-out benchmarks
    logger.info("Step 2: Relabel held-out benchmarks")
    heldout = load_heldout_data()

    eval_records = []
    for name, df in heldout.items():
        logger.info(f"Processing {name}...")
        relabeled = await relabel_dataset(df, template, name, max_samples=3000)
        relabeled["benchmark"] = name
        eval_records.append(relabeled)

    eval_df = pd.concat(eval_records, ignore_index=True)
    eval_df.to_parquet(OUTPUT_DIR / "eval.parquet", index=False)
    logger.info(f"Saved {len(eval_df)} eval samples from {len(eval_records)} benchmarks to eval.parquet")

    elapsed = time.time() - t0
    logger.info(f"R3 DONE in {elapsed/3600:.1f}h ({elapsed/60:.0f}min)")
    logger.info(f"Output: {OUTPUT_DIR}/{{train,eval}}.parquet")


if __name__ == "__main__":
    asyncio.run(main())
