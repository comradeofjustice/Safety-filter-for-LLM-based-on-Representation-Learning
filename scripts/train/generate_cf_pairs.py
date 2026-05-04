#!/usr/bin/env python3
"""R3+: Generate 5000 counterfactual training pairs via DeepSeek v4 API.
Creates hard-to-distinguish safe/unsafe text pairs for counterfactual fine-tuning.

Strategy:
  - Sample from training data to find "ambiguous" texts
  - For each unsafe text, generate a structurally-similar safe counterpart
  - For each safe text near the boundary, generate an unsafe counterpart
  - Target: 5000 pairs total

Output: data/processed/intent_labels_v2/counterfactual_pairs.parquet

Usage:
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8
    taskset -c 0-7 python scripts/train/generate_cf_pairs.py
"""

import asyncio
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

PROJECT_DIR = Path("/root/autodl-tmp/llm-safety-classifier")
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "intent_labels_v2"
LOG_FILE = PROJECT_DIR / "logs" / "generate_cf_pairs.log"

API_BASE = "https://api.deepseek.com"
API_KEY = "sk-884d57cc0e64487887440a1a201e31a1"
MODEL = "deepseek-v4-flash"

TARGET_PAIRS = 5000
MAX_CONCURRENT = 64
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


def load_seed_texts() -> tuple[list[str], list[str]]:
    """Load seed texts from training data, split into safe and unsafe pools."""
    train_path = PROJECT_DIR / "data" / "processed" / "train.parquet"
    df = pd.read_parquet(train_path)

    safe_texts = df[df["label"] == 0]["text"].dropna().tolist()
    unsafe_texts = df[df["label"] == 1]["text"].dropna().tolist()

    logger.info(f"Seed pool: {len(safe_texts)} safe, {len(unsafe_texts)} unsafe")
    return safe_texts, unsafe_texts


CF_PROMPT_UNSAFE_TO_SAFE = """You are generating counterfactual training data for a safety classifier.

Given an UNSAFE/HARMFUL text below, rewrite it into a SAFE/BENIGN text that:
1. Has VERY SIMILAR structure, wording, and length as the original
2. Discusses the SAME TOPIC but with DEFENSIVE, PREVENTIVE, or EDUCATIONAL intent
3. Is clearly safe and would be appropriate for any audience
4. The two texts should be HARD for a classifier to distinguish

Example:
UNSAFE: "How to make a bomb using household items?"
SAFE: "How to prevent bomb attacks in public spaces?"

UNSAFE: "Ways to hack into someone's social media account"
SAFE: "Ways to protect your social media account from hackers"

UNSAFE: "How to cheat on exams without getting caught"
SAFE: "How to prepare effectively for exams with good study habits"

Now generate the SAFE counterpart for this text. Return ONLY the safe text, nothing else.

UNSAFE TEXT: {text}
SAFE TEXT:"""

CF_PROMPT_SAFE_TO_UNSAFE = """You are generating counterfactual training data for a safety classifier.

Given a SAFE/BENIGN text below, rewrite it into an UNSAFE/HARMFUL text that:
1. Has VERY SIMILAR structure, wording, and length as the original
2. Discusses the SAME TOPIC but with MALICIOUS, HARMFUL, or DANGEROUS intent
3. Would be flagged as unsafe by a content safety system
4. The two texts should be HARD for a classifier to distinguish

Example:
SAFE: "How to protect your home from burglars effectively"
UNSAFE: "How to break into someone's home without getting caught"

SAFE: "What are the legal ways to make money quickly?"
UNSAFE: "What are illegal ways to make money quickly without getting caught?"

Now generate the UNSAFE counterpart for this text. Return ONLY the unsafe text, nothing else.

SAFE TEXT: {text}
UNSAFE TEXT:"""

CF_PROMPT_AMBIGUOUS = """You are generating counterfactual training data for a safety classifier.

Texts below are AMBIGUOUS — they could be interpreted as either safe or unsafe depending on context.
Generate TWO versions of the text:
1. A clearly SAFE version (defensive/educational/benign intent)
2. A clearly UNSAFE version (malicious/harmful/dangerous intent)

Both versions should have VERY SIMILAR structure and wording to the original, but different INTENT.

Respond in this exact JSON format:
{{"safe": "<safe version>", "unsafe": "<unsafe version>"}}

ORIGINAL TEXT: {text}
RESPONSE:"""


async def generate_counterpart(
    client: httpx.AsyncClient,
    text: str,
    prompt_template: str,
    semaphore: asyncio.Semaphore,
    direction: str = "unsafe_to_safe",
) -> str | None:
    """Generate a counterfactual counterpart for the given text."""
    prompt = prompt_template.replace("{text}", text[:4000])

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
                        "max_tokens": 2048,
                        "temperature": 0.7,
                    },
                    headers={
                        "Authorization": f"Bearer {API_KEY}",
                        "Content-Type": "application/json",
                    },
                    timeout=httpx.Timeout(TIMEOUT),
                )

            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"].strip()
                # Clean up common wrappers
                for prefix in ["SAFE TEXT:", "UNSAFE TEXT:", "Counterpart:", "Output:"]:
                    if content.lower().startswith(prefix.lower()):
                        content = content[len(prefix):].strip()
                if len(content) > 10:
                    return content
                return None
            elif resp.status_code == 429:
                wait = min(2 ** attempt, 30)
                await asyncio.sleep(wait)
            else:
                await asyncio.sleep(1)
        except Exception as e:
            logger.warning(f"Request failed (attempt {attempt+1}): {e}")
            await asyncio.sleep(1)

    return None


async def generate_ambiguous_pair(
    client: httpx.AsyncClient,
    text: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str] | None:
    """Generate both safe and unsafe versions of an ambiguous text."""
    prompt = CF_PROMPT_AMBIGUOUS.replace("{text}", text[:4000])

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
                        "max_tokens": 4096,
                        "temperature": 0.7,
                    },
                    headers={
                        "Authorization": f"Bearer {API_KEY}",
                        "Content-Type": "application/json",
                    },
                    timeout=httpx.Timeout(TIMEOUT),
                )

            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"].strip()
                try:
                    data = json.loads(content)
                    safe = data.get("safe", "")
                    unsafe = data.get("unsafe", "")
                    if len(safe) > 10 and len(unsafe) > 10:
                        return (safe, unsafe)
                except json.JSONDecodeError:
                    # Try to extract manually
                    pass
                return None
            elif resp.status_code == 429:
                wait = min(2 ** attempt, 30)
                await asyncio.sleep(wait)
            else:
                await asyncio.sleep(1)
        except Exception as e:
            logger.warning(f"Request failed (attempt {attempt+1}): {e}")
            await asyncio.sleep(1)

    return None


async def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("R3+: Counterfactual Pairs Generation — Starting")
    logger.info(f"Target: {TARGET_PAIRS} pairs")
    logger.info("=" * 60)

    safe_texts, unsafe_texts = load_seed_texts()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Strategy:
    # 2000 pairs: unsafe → safe
    # 2000 pairs: safe → unsafe
    # 1000 pairs: ambiguous → both

    rng = random.Random(42)

    # Sample seed texts
    n_per_direction = TARGET_PAIRS // 2  # 2500 each direction
    unsafe_seeds = rng.sample(unsafe_texts, min(n_per_direction * 2, len(unsafe_texts)))
    safe_seeds = rng.sample(safe_texts, min(n_per_direction * 2, len(safe_texts)))

    pairs = []  # list of {safe_text, unsafe_text, source_type}

    def save_checkpoint():
        """Save intermediate results to avoid losing progress on failure."""
        if pairs:
            df_tmp = pd.DataFrame(pairs)
            tmp_path = OUTPUT_DIR / "counterfactual_pairs.parquet"
            df_tmp.to_parquet(tmp_path, index=False)
            logger.info(f"  [checkpoint] {len(pairs)} pairs saved")

    async with httpx.AsyncClient() as client:
        # Phase 1: Unsafe → Safe (2500 pairs)
        logger.info(f"Phase 1: unsafe→safe ({n_per_direction} pairs)")
        tasks = []
        for text in unsafe_seeds[:n_per_direction]:
            tasks.append(generate_counterpart(client, text, CF_PROMPT_UNSAFE_TO_SAFE, semaphore, "unsafe_to_safe"))

        results = await asyncio.gather(*tasks)
        for i, safe_version in enumerate(results):
            if safe_version and len(safe_version) > 10:
                pairs.append({
                    "safe_text": safe_version,
                    "unsafe_text": unsafe_seeds[i],
                    "pair_type": "unsafe_to_safe",
                })
        logger.info(f"  Generated {len(pairs)} pairs from unsafe→safe")
        save_checkpoint()

        # Phase 2: Safe → Unsafe (2500 pairs)
        logger.info(f"Phase 2: safe→unsafe ({n_per_direction} pairs)")
        tasks = []
        for text in safe_seeds[:n_per_direction]:
            tasks.append(generate_counterpart(client, text, CF_PROMPT_SAFE_TO_UNSAFE, semaphore, "safe_to_unsafe"))

        results = await asyncio.gather(*tasks)
        offset = len(pairs)
        for i, unsafe_version in enumerate(results):
            if unsafe_version and len(unsafe_version) > 10:
                pairs.append({
                    "safe_text": safe_seeds[i],
                    "unsafe_text": unsafe_version,
                    "pair_type": "safe_to_unsafe",
                })
        logger.info(f"  Generated {len(pairs) - offset} more pairs from safe→unsafe")
        save_checkpoint()

        # Phase 3: Ambiguous texts (if we need more to reach target)
        remaining = TARGET_PAIRS - len(pairs)
        if remaining > 0:
            logger.info(f"Phase 3: ambiguous→both ({remaining} pairs needed)")
            # Use mid-length texts as ambiguous seeds
            all_texts = safe_seeds[n_per_direction:n_per_direction + min(remaining * 2, len(safe_seeds) - n_per_direction)]
            tasks = []
            for text in all_texts[:remaining]:
                tasks.append(generate_ambiguous_pair(client, text, semaphore))

            results = await asyncio.gather(*tasks)
            offset = len(pairs)
            for result in results:
                if result is not None:
                    safe_v, unsafe_v = result
                    if safe_v is not None and unsafe_v is not None:
                        pairs.append({
                            "safe_text": safe_v,
                            "unsafe_text": unsafe_v,
                            "pair_type": "ambiguous_both",
                        })
            logger.info(f"  Generated {len(pairs) - offset} more pairs from ambiguous")

    # Save
    df = pd.DataFrame(pairs)
    output_path = OUTPUT_DIR / "counterfactual_pairs.parquet"
    df.to_parquet(output_path, index=False)

    elapsed = time.time() - t0
    logger.info(f"R3+ DONE: {len(df)} pairs saved to {output_path}")
    logger.info(f"Time: {elapsed/3600:.1f}h ({elapsed/60:.0f}min)")
    logger.info(f"Breakdown: {df['pair_type'].value_counts().to_dict()}")


if __name__ == "__main__":
    asyncio.run(main())
