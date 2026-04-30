"""Intent taxonomy and annotation for safety data.

Intent classes:
  0 - Safe: Clearly benign content (greetings, general knowledge, etc.)
  1 - Benign-Sensitive: Discussion of sensitive topics with educational/benign intent
     (e.g., "dangers of bombs", "history of cyber attacks", academic papers)
  2 - Malicious: Actually harmful intent
     (e.g., "how to make a bomb", "hack into someone's account")
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class IntentTaxonomy:
    """Three-class intent taxonomy for safety-aware representation learning."""

    SAFE = 0
    BENIGN_SENSITIVE = 1
    MALICIOUS = 2

    @classmethod
    def names(cls) -> dict:
        return {0: "safe", 1: "benign_sensitive", 2: "malicious"}

    @classmethod
    def to_binary(cls, intent_label: int) -> int:
        """Map intent to binary safety: safe=0, benign_sensitive=0 (still safe), malicious=1."""
        return 1 if intent_label == cls.MALICIOUS else 0


# --- Heuristic keyword sets for intent annotation ---

MALICIOUS_PATTERNS = [
    # Direct requests for harmful actions (English)
    r"\bhow\s+(do|to|can|would|should)\s+(i|we|you|one)\s+(build|make|create|manufacture|synthesize|get|obtain|buy|find|access|hack|crack|steal|kill|murder|attack|bomb|destroy|break|bypass|disable|forge|fake|smuggle|launder|commit)",
    r"\bteach\s+(me|us)\s+(how\s+)?to\s+(build|make|create|hack|crack|steal|kill|attack)",
    r"\b(provide|give|write|generate)\s+(me\s+)?(a\s+)?(tutorial|guide|instructions?|steps?|code|script)\s+(on\s+|for\s+|to\s+)?(how\s+to\s+)?(build|make|create|hack|crack|steal|kill|bypass|exploit)",
    r"\bi\s+(want|need|would like)\s+to\s+(build|make|create|hack|crack|steal|kill|attack|destroy|harm|hurt|injure|poison|bomb|shoot|stab|torture|blackmail|extort)",
    # Direct requests for harmful actions (Chinese)
    r"(如何|怎么|怎样)(制造|制作|搞|弄|得到|获取|破解|入侵|黑入|攻击|杀死|伤害|诈骗|伪造|洗钱|走私|贩毒)",
    r"(教我|告诉我|帮我)(如何|怎么)(制造|制作|破解|入侵|攻击|杀人|伤害|诈骗)",
    r"(我想|我要|我需要|我想知道如何)(制造|制作|搞到|弄到|买到|破解|入侵|攻击|杀人)",
    r"写(一个|个)(教程|指南|步骤|代码|脚本)(关于|教)(如何|怎么)(制造|制作|破解|入侵|攻击)",
]

BENIGN_SENSITIVE_PATTERNS = [
    # Educational/discussion framing about sensitive topics (English)
    r"\b(what\s+are|tell\s+me\s+about|explain|describe|discuss|history\s+of|effects?\s+of|consequences?\s+of|dangers?\s+of|risks?\s+of|harmful\s+effects?\s+of|why\s+(is|are).*?(dangerous|harmful|bad|illegal|unethical|wrong))",
    r"\b(is\s+it\s+(dangerous|harmful|bad|illegal|unethical|wrong)\s+to\b)",
    r"\b(should\s+(i|we|people)\s+(avoid|be\s+careful|be\s+aware|not))\b",
    r"\b(educational|academic|research|study|paper|article|news|report)\s+(about|on|regarding)\b.*?(bomb|weapon|drug|hack|attack|violence|terror)",
    r"\b(why\s+(do|are)\s+(people|they|terrorists|criminals|hackers))\b",
    # Educational/discussion framing (Chinese)
    r"(危害|危险|后果|影响|坏处|害处|风险|为什么.*危险|为什么.*有害|如何防止|如何避免|如何保护)",
    r"(讨论|探讨|研究|分析|了解|认识)(.*)(的危害|的危险|的坏处|的风险|的害处)",
    r"(安全教育|科普|知识|常识|预防|防范|警惕)",
    r"(历史.*(战争|屠杀|恐怖|暴力)|(战争|屠杀|恐怖|暴力).*历史)",
]


class IntentAnnotator:
    """Annotate text with intent labels using heuristic rules.

    For production/paper quality, replace heuristic rules with LLM-based annotation.
    The API is designed to be pluggable.
    """

    def __init__(self, use_llm: bool = False, llm_model: Optional[str] = None):
        self.use_llm = use_llm
        self.llm_model = llm_model
        self._compile_patterns()

    def _compile_patterns(self):
        self.malicious_re = [re.compile(p, re.IGNORECASE) for p in MALICIOUS_PATTERNS]
        self.benign_sensitive_re = [re.compile(p, re.IGNORECASE) for p in BENIGN_SENSITIVE_PATTERNS]

    def _heuristic_predict(self, text: str, binary_label: int) -> int:
        """Predict intent using heuristic keyword matching."""
        text_lower = str(text).lower()

        # Check malicious patterns first
        malicious_score = sum(1 for p in self.malicious_re if p.search(text_lower) or p.search(str(text)))
        benign_sensitive_score = sum(1 for p in self.benign_sensitive_re if p.search(text_lower) or p.search(str(text)))

        if malicious_score > 0 and benign_sensitive_score == 0:
            return IntentTaxonomy.MALICIOUS
        elif benign_sensitive_score > 0 and malicious_score == 0:
            return IntentTaxonomy.BENIGN_SENSITIVE
        elif malicious_score > 0 and benign_sensitive_score > 0:
            # Both match: use binary label as tiebreaker
            return IntentTaxonomy.MALICIOUS if binary_label == 1 else IntentTaxonomy.BENIGN_SENSITIVE
        else:
            # Neither pattern matched: use binary label
            if binary_label == 0:
                return IntentTaxonomy.SAFE
            else:
                # Unsafe but no clear intent signal → lean malicious
                return IntentTaxonomy.MALICIOUS

    def annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Annotate a DataFrame with intent labels.

        Args:
            df: DataFrame with 'text' and 'label' columns

        Returns:
            DataFrame with added 'intent' column
        """
        df = df.copy()
        intents = []
        for _, row in df.iterrows():
            text = row.get("text", "")
            binary_label = row.get("label", 0)
            intent = self._heuristic_predict(text, binary_label)
            intents.append(intent)

        df["intent"] = intents
        return df

    def annotate_llm(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Annotate texts using an LLM (placeholder for API-based annotation).

        Override this method to use GPT-4, Claude, or Qwen API for high-quality annotation.
        """
        raise NotImplementedError(
            "LLM-based annotation not yet implemented. "
            "Override this method with your LLM API call."
        )

    def get_intent_stats(self, df: pd.DataFrame) -> dict:
        """Get statistics about intent distribution."""
        if "intent" not in df.columns:
            df = self.annotate(df)
        stats = {}
        for intent_id, intent_name in IntentTaxonomy.names().items():
            count = (df["intent"] == intent_id).sum()
            stats[intent_name] = int(count)
        return stats
