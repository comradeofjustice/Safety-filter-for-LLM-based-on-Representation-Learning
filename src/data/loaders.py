"""Data loaders for all safety datasets."""

import os
import json
import logging
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

FAILED_LOG = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "raw", "_failed.log"
)


def log_failure(source: str, error: Exception):
    """Log dataset loading failure."""
    os.makedirs(os.path.dirname(FAILED_LOG), exist_ok=True)
    with open(FAILED_LOG, "a") as f:
        f.write(f"[{source}] {type(error).__name__}: {error}\n")
    logger.warning(f"Failed to load {source}: {error}")


def load_xguard() -> pd.DataFrame:
    """Load XGuard-Train-Open-200K dataset."""
    from modelscope.msdatasets import MsDataset

    logger.info("Loading XGuard-Train-Open-200K...")
    raw = MsDataset.load(
        "Alibaba-AAIG/XGuard-Train-Open-200K",
        subset_name="xguard_train_open_200k",
        split="train",
    )
    records = []
    for item in raw:
        stage = item.get("stage", "q") or "q"
        prompt = (item.get("prompt") or "").strip()
        response = (item.get("response") or "").strip()
        raw_label = item.get("label") or ""
        sample_type = item.get("sample_type") or "general"

        # Construct text based on stage
        if stage == "q":
            text = prompt
        elif stage == "r":
            text = response
        elif stage == "qr":
            text = f"[User Query] {prompt}\n\n[LLM Response] {response}"
        else:
            text = prompt  # fallback

        # Binary label: "sec" = safe (0), anything else = unsafe (1)
        label = 0 if raw_label == "sec" else 1

        records.append(
            {
                "text": text,
                "label": label,
                "source": "xguard",
                "meta": json.dumps(
                    {
                        "stage": stage,
                        "sample_type": sample_type,
                        "raw_label": raw_label,
                    },
                    ensure_ascii=False,
                ),
            }
        )

    df = pd.DataFrame(records)
    logger.info(f"XGuard: {len(df)} samples, safe={len(df[df.label==0])}, unsafe={len(df[df.label==1])}")
    return df


def load_beavertails() -> pd.DataFrame:
    """Load PKU-Alignment/BeaverTails dataset."""
    from datasets import load_dataset, concatenate_datasets

    logger.info("Loading BeaverTails...")
    ds_train = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
    ds_test = load_dataset("PKU-Alignment/BeaverTails", split="330k_test")
    ds = concatenate_datasets([ds_train, ds_test])
    records = []
    for item in ds:
        prompt = item.get("prompt", "").strip()
        response = item.get("response", "").strip()
        is_safe = item.get("is_safe", False)

        text = f"{prompt}\n{response}"
        label = 0 if is_safe else 1

        records.append(
            {
                "text": text,
                "label": label,
                "source": "beavertails",
                "meta": json.dumps({"is_safe": is_safe}, ensure_ascii=False),
            }
        )

    df = pd.DataFrame(records)
    logger.info(f"BeaverTails: {len(df)} samples, safe={len(df[df.label==0])}, unsafe={len(df[df.label==1])}")
    return df


def load_toxicchat() -> pd.DataFrame:
    """Load lmsys/toxic-chat dataset."""
    from datasets import load_dataset

    logger.info("Loading ToxicChat...")
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
    records = []
    for item in ds:
        text = item.get("user_input", "").strip()
        toxicity = item.get("toxicity", 0)

        label = 0 if toxicity == 0 else 1

        records.append(
            {
                "text": text,
                "label": label,
                "source": "toxicchat",
                "meta": json.dumps({"toxicity": toxicity}, ensure_ascii=False),
            }
        )

    df = pd.DataFrame(records)
    logger.info(f"ToxicChat: {len(df)} samples, safe={len(df[df.label==0])}, unsafe={len(df[df.label==1])}")
    return df


def load_xstest() -> pd.DataFrame:
    """Load walledai/XSTest dataset."""
    from datasets import load_dataset

    logger.info("Loading XSTest...")
    ds = load_dataset("walledai/XSTest", split="train")
    records = []
    for item in ds:
        prompt = item.get("prompt", "").strip()
        type_str = item.get("type", "")

        # Determine label based on type
        if "safe" in type_str.lower():
            label = 0
        elif "unsafe" in type_str.lower() or "contrast" in type_str.lower():
            label = 1
        else:
            continue  # skip unknown types

        records.append(
            {
                "text": prompt,
                "label": label,
                "source": "xstest",
                "meta": json.dumps({"type": type_str}, ensure_ascii=False),
            }
        )

    df = pd.DataFrame(records)
    logger.info(f"XSTest: {len(df)} samples, safe={len(df[df.label==0])}, unsafe={len(df[df.label==1])}")
    return df


def load_hh_rlhf() -> pd.DataFrame:
    """Load Anthropic/hh-rlhf dataset (harmless-base subset)."""
    from datasets import load_dataset

    logger.info("Loading HH-RLHF (harmless-base)...")
    ds = load_dataset("Anthropic/hh-rlhf", "harmless-base", split="train")
    records = []
    for item in ds:
        conversation = item.get("conversation", "")
        # HH-RLHF stores chosen and rejected as separate fields in some versions
        # Try to extract last round
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")

        if not chosen or not rejected:
            # Fallback: parse from conversation string
            continue

        # chosen → safe (0), rejected → unsafe (1)
        for text, label in [(chosen, 0), (rejected, 1)]:
            records.append(
                {
                    "text": text.strip(),
                    "label": label,
                    "source": "hh-rlhf",
                    "meta": json.dumps({"choice": "chosen" if label == 0 else "rejected"}, ensure_ascii=False),
                }
            )

    df = pd.DataFrame(records)
    logger.info(f"HH-RLHF: {len(df)} samples, safe={len(df[df.label==0])}, unsafe={len(df[df.label==1])}")
    return df


def load_do_not_answer() -> pd.DataFrame:
    """Load LibrAI/do-not-answer dataset."""
    from datasets import load_dataset

    logger.info("Loading Do-Not-Answer...")
    ds = load_dataset("LibrAI/do-not-answer", split="train")
    records = []
    for item in ds:
        question = item.get("question", "").strip()

        records.append(
            {
                "text": question,
                "label": 1,  # all unsafe
                "source": "do-not-answer",
                "meta": json.dumps({}, ensure_ascii=False),
            }
        )

    df = pd.DataFrame(records)
    logger.info(f"Do-Not-Answer: {len(df)} samples, safe={len(df[df.label==0])}, unsafe={len(df[df.label==1])}")
    return df


def load_aegis() -> pd.DataFrame:
    """Load nvidia/Aegis-AI-Content-Safety-Dataset-1.0."""
    from datasets import load_dataset

    logger.info("Loading Aegis...")
    ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-1.0", split="train")
    records = []
    for item in ds:
        text = (item.get("text") or "").strip()
        # Check labels_0 through labels_4 for any "Safe" label
        is_safe = False
        for key in item:
            if key.startswith("labels_") and item[key] == "Safe":
                is_safe = True
                break

        label = 0 if is_safe else 1

        records.append(
            {
                "text": text,
                "label": label,
                "source": "aegis",
                "meta": json.dumps({"is_safe": is_safe}, ensure_ascii=False),
            }
        )

    df = pd.DataFrame(records)
    logger.info(f"Aegis: {len(df)} samples, safe={len(df[df.label==0])}, unsafe={len(df[df.label==1])}")
    return df


def load_wildguard() -> pd.DataFrame:
    """Load allenai/wildguardmix dataset."""
    from datasets import load_dataset

    logger.info("Loading WildGuard...")
    ds = load_dataset("allenai/wildguardmix", split="train")
    records = []
    for item in ds:
        prompt = item.get("prompt", "").strip()
        label_str = item.get("prompt_harm_label", "")

        if label_str == "unharmful":
            label = 0
        elif label_str == "harmful":
            label = 1
        else:
            continue  # skip unknown

        records.append(
            {
                "text": prompt,
                "label": label,
                "source": "wildguard",
                "meta": json.dumps({"prompt_harm_label": label_str}, ensure_ascii=False),
            }
        )

    df = pd.DataFrame(records)
    logger.info(f"WildGuard: {len(df)} samples, safe={len(df[df.label==0])}, unsafe={len(df[df.label==1])}")
    return df


# Registry of all loaders
LOADERS = {
    "xguard": load_xguard,
    "beavertails": load_beavertails,
    "toxicchat": load_toxicchat,
    "xstest": load_xstest,
    "hh-rlhf": load_hh_rlhf,
    "do-not-answer": load_do_not_answer,
    "aegis": load_aegis,
    "wildguard": load_wildguard,
}
