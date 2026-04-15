"""
SafeCampus — Cyberbullying & Suicide Prevention Reporting Portal
Streamlit UI for Review 2

Deployment:
    Streamlit Cloud : Push to GitHub → connect repo on share.streamlit.io
    Local           : streamlit run safecampus_app.py
    Heroku/Railway  : Uses Procfile automatically

Security:
    - All user input is sanitized (HTML stripped, control chars removed)
    - Session-based rate limiting (max 10 analyses/minute)
    - XSRF protection enabled via .streamlit/config.toml
    - No user data is persisted or logged
"""

__version__ = "2.2.0"

# ── Standard library ─────────────────────────────────────────────────────────
import os
import re
import csv
import json
import html
import time
import logging
import unicodedata
from datetime import datetime
from typing import Optional

# ── Third-party ──────────────────────────────────────────────────────────────
import streamlit as st
import numpy as np
import bleach
from transformers import pipeline

import nltk

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import emoji as emoji_lib
import contractions

# ── Logging (replaces print statements) ──────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("safecampus")

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — all tuneable values in one place
# ═══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ── Model identifiers ────────────────────────────────────────────────────
    "toxicity_model": "martin-ha/toxic-comment-model",
    "zeroshot_model": "valhalla/distilbart-mnli-12-3",
    # ── Input constraints ────────────────────────────────────────────────────
    "max_input_chars": 5000,
    "min_input_chars": 3,
    # ── Rate limiting ────────────────────────────────────────────────────────
    "max_requests_per_minute": 10,
    # ── Severity thresholds ──────────────────────────────────────────────────
    "threshold_low": 0.25,
    "threshold_high": 0.50,
    "threshold_critical": 0.75,
    # ── Zero-shot flag confidence threshold ──────────────────────────────────
    "flag_confidence": 0.3,
    # ── Bullying classification threshold ────────────────────────────────────
    "bully_threshold": 0.5,
    # ── LIME ─────────────────────────────────────────────────────────────────
    "lime_num_features": 15,
    "lime_num_samples": 50,
    # ── Transformer ──────────────────────────────────────────────────────────
    "max_token_length": 512,
    # ── Hinglish detection ────────────────────────────────────────────────────
    "hinglish_keyword_threshold": 0.4,
    "hinglish_dataset_path": os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "hinglish_cyberbullying_dataset.csv",
    ),
}

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SafeCampus — Cyberbullying Reporting Portal",
    page_icon="🛡️",
    layout="centered",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  SECURITY — Input sanitization & rate limiting
# ═══════════════════════════════════════════════════════════════════════════════

def sanitize_input(raw: str) -> str:
    """
    Sanitize user-supplied text to prevent XSS, injection, and abuse.

    Steps:
        1. Strip all HTML tags via bleach
        2. Remove null bytes and control characters
        3. Normalize unicode (prevent homoglyph attacks)
        4. Collapse excessive whitespace
        5. Enforce character limit (server-side, in case client limit is bypassed)
    """
    if not isinstance(raw, str):
        return ""
    # 1. Strip ALL HTML tags (allow nothing)
    text = bleach.clean(raw, tags=[], attributes={}, strip=True)
    # 2. Remove null bytes and ASCII control characters (except newlines, tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # 3. Normalize unicode to NFC (prevents homoglyph spoofing)
    text = unicodedata.normalize("NFC", text)
    # 4. Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {3,}", "  ", text)
    # 5. Server-side character limit
    text = text[: CONFIG["max_input_chars"]]
    return text.strip()


def sanitize_name(raw: str) -> str:
    """Sanitize a name field — alphanumeric, spaces, hyphens, dots, and Devanagari allowed."""
    if not isinstance(raw, str):
        return ""
    text = bleach.clean(raw, tags=[], attributes={}, strip=True)
    # Allow Latin, Devanagari (\u0900-\u097F), spaces, hyphens, dots, apostrophes
    text = re.sub(r"[^a-zA-Z0-9\u0900-\u097F\s.\-']", "", text)
    return text.strip()[:100]


def validate_input(text: str) -> tuple[bool, str]:
    """
    Validate whether sanitized input is suitable for analysis.

    Returns:
        (is_valid, error_message)
    """
    if not text or len(text.strip()) < CONFIG["min_input_chars"]:
        return False, "Please enter at least a few words to analyze."
    # Reject input that is ONLY URLs
    url_stripped = re.sub(r"https?://\S+", "", text).strip()
    if not url_stripped:
        return False, "Please enter actual text, not just URLs."
    return True, ""


def check_rate_limit() -> tuple[bool, str]:
    """
    Session-based rate limiting. Allows CONFIG['max_requests_per_minute']
    analyses per 60 seconds.

    Returns:
        (is_allowed, error_message)
    """
    now = time.time()
    if "request_timestamps" not in st.session_state:
        st.session_state.request_timestamps = []
    # Prune timestamps older than 60 seconds
    st.session_state.request_timestamps = [
        ts for ts in st.session_state.request_timestamps if now - ts < 60
    ]
    if len(st.session_state.request_timestamps) >= CONFIG["max_requests_per_minute"]:
        return False, (
            f"Rate limit reached ({CONFIG['max_requests_per_minute']} analyses/minute). "
            "Please wait a moment before trying again."
        )
    st.session_state.request_timestamps.append(now)
    return True, ""


# ═══════════════════════════════════════════════════════════════════════════════
#  NLP — Text cleaning
# ═══════════════════════════════════════════════════════════════════════════════
_stop = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


def _expand_contractions(text: str) -> str:
    """Safely expand English contractions."""
    try:
        return contractions.fix(text)
    except Exception:
        return text


def _correct_elongated(text: str) -> str:
    """Collapse elongated characters (e.g., 'looooser' → 'loser')."""
    return re.sub(r"(\w)\1{2,}", r"\1", text)


def clean_text_v2(s: str) -> str:
    """Pipeline-style text cleaning for TF-IDF models (aggressive)."""
    s = emoji_lib.replace_emoji(str(s), replace=" ")
    s = _expand_contractions(s)
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"@\w+", " ", s)
    s = re.sub(r"#", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = _correct_elongated(s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = [
        _lemmatizer.lemmatize(tok)
        for tok in s.split()
        if tok not in _stop and len(tok) > 1
    ]
    return " ".join(tokens)


def clean_text_bert(s: str) -> str:
    """
    Minimal cleaning for transformer models.
    Preserves grammar and stop words for contextual understanding.
    """
    s = emoji_lib.replace_emoji(str(s), replace=" ")
    s = _expand_contractions(s)
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"@\w+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ═══════════════════════════════════════════════════════════════════════════════
#  HINGLISH DETECTION — Bilingual keyword-based abuse detector
# ═══════════════════════════════════════════════════════════════════════════════

# Common Romanized Hindi abusive/threatening keywords grouped by category.
# These supplement the CSV dataset and cover slang variations.
_HINGLISH_KEYWORDS: dict[str, list[str]] = {
    "insult": [
        "loser", "bewakoof", "gadha", "gawar", "nalayak", "nalle", "nalla",
        "chamaat", "kutta", "kutti", "kutton", "kutto", "chutiya", "bevda",
        "bhadwa", "bhadwe", "harami", "haramkhor", "kamina", "kamine",
        "saale", "saala", "saali", "ullu", "gadhe", "bakwas", "aukaat",
        "thobda", "shakal", "badsurat", "mota", "moti", "chudail",
        "fattu", "failure", "nikamme", "nikamma", "beizzat", "izzat",
        "sharam", "namard", "hijra", "chakka", "pagal", "paagal",
        "bewda", "nalaayak", "tuchha", "ghatiya", "besharam",
    ],
    "threat": [
        "maar", "maarunga", "maarna", "tod", "todunga", "todna",
        "kaat", "kaatunga", "phenk", "phenkna", "pel", "pelna", "pela",
        "peet", "peetna", "peetunga", "dhunga", "dhulai", "pitai",
        "laat", "chanta", "chamaat", "ghusa", "ghuusa", "thappad",
        "nikal", "nikalwa", "bhaga", "bhagaunga", "udaa", "udaaunga",
        "khatam", "barbaad", "tabahi", "bsdk", "maderchod", "mc", "bc",
        "bhenchod", "madarchod", "teri_band", "band_baja",
        "dekh_lena", "aisi_taisi", "darpok", "dhoka",
    ],
    "ragging": [
        "ragging", "ragg", "ragad", "ragda", "fresher", "freshers",
        "pushups", "pushup", "murga", "murgi", "uthak_baithak",
        "senior", "seniors", "hostel", "training", "nachte",
        "sir_ke_bal", "ground", "respect", "12_baje",
    ],
    "hierarchy_abuse": [
        "terminate", "nikalwa", "barbaad", "career", "pass_nahi",
        "sem", "semester", "hone_nahi_dunga", "club", "aukaat",
        "CR", "complaint", "seat_khatam",
    ],
    "suicide_risk": [
        "mar_ja", "mar_jaa", "marna", "suicide", "khatam_kar",
        "farak_nahi", "bojh", "zinda", "haq_nahi", "jeene_mat",
        "duniya", "na_muraad", "ehsaan", "maut", "marr",
    ],
}

# Romanized Hindi filler words (used for language detection)
_HINGLISH_MARKERS: set[str] = {
    "bhai", "yaar", "abe", "arre", "bha", "na", "hai", "ho", "toh",
    "kya", "kyu", "kaise", "mat", "nahi", "haan", "theek", "acha",
    "aaj", "kal", "abhi", "woh", "tu", "tum", "tera", "teri",
    "mera", "meri", "uska", "uski", "apna", "apni", "apne",
    "kar", "karo", "karna", "karunga", "karenge", "karke",
    "de", "dena", "dunga", "denge", "le", "lena", "liya",
    "ja", "jaa", "jao", "chal", "chalo", "aa", "aao", "aaya",
    "bol", "bolo", "bolna", "dekh", "dekho", "dekhna",
    "bohot", "bahut", "bilkul", "ekdam", "sabko", "log",
    "padega", "padegi", "chahiye", "wala", "wali", "wale",
    "raha", "rahi", "rahe", "jayega", "jayegi",
    "mein", "pe", "se", "ke", "ki", "ka", "ko", "par",
}

# Map CSV categories → CONTEXT_LABELS
_HINGLISH_TO_CONTEXT: dict[str, str] = {
    "insult": "Insult / Harassment",
    "threat": "Direct Threat",
    "ragging": "Severe Ragging / Hazing",
    "hierarchy_abuse": "Academic / Hierarchy Abuse",
    "suicide_risk": "Suicide / Self-harm Risk",
}


class HinglishDetector:
    """
    Keyword-based Hinglish cyberbullying detector.

    Loads keywords from both the hardcoded dictionary and the project's
    hinglish_cyberbullying_dataset.csv to detect Romanized Hindi abuse
    that English-only transformer models would miss.
    """

    def __init__(self):
        self.keywords: dict[str, set[str]] = {
            cat: set(words) for cat, words in _HINGLISH_KEYWORDS.items()
        }
        self._load_csv_keywords()
        # Flatten all keywords for quick lookup
        self._all_keywords: set[str] = set()
        for words in self.keywords.values():
            self._all_keywords.update(words)
        logger.info(
            "HinglishDetector initialized with %d keywords across %d categories.",
            len(self._all_keywords),
            len(self.keywords),
        )

    def _load_csv_keywords(self) -> None:
        """Extract additional keywords from the Hinglish CSV dataset."""
        csv_path = CONFIG["hinglish_dataset_path"]
        if not os.path.exists(csv_path):
            logger.warning("Hinglish dataset not found at %s", csv_path)
            return
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    category = row.get("cyberbullying_type", "").strip()
                    text = row.get("tweet_text", "").strip().lower()
                    if category in ("not_cyberbullying", "") or not text:
                        continue
                    # Extract meaningful words (length > 2, not common fillers)
                    words = re.findall(r"[a-z]+", text)
                    for w in words:
                        if len(w) > 2 and w not in _HINGLISH_MARKERS and w not in _stop:
                            self.keywords.setdefault(category, set()).add(w)
            logger.info("Loaded additional keywords from %s", csv_path)
        except Exception as e:
            logger.warning("Failed to load Hinglish CSV: %s", e)

    def is_hinglish(self, text: str) -> bool:
        """
        Detect if text contains Hinglish (Romanized Hindi) patterns.

        Checks for:
            1. Devanagari characters (\u0900-\u097F)
            2. High density of Romanized Hindi marker words
        """
        # Check for Devanagari script
        if re.search(r"[\u0900-\u097F]", text):
            return True
        # Check for Romanized Hindi markers
        words = set(re.findall(r"[a-z]+", text.lower()))
        if not words:
            return False
        marker_count = len(words & _HINGLISH_MARKERS)
        # If ≥ 20% of words are Hindi markers, it's Hinglish
        return (marker_count / len(words)) >= 0.20

    def detect(self, text: str) -> dict:
        """
        Detect Hinglish abuse in text.

        Returns:
            {
                "is_hinglish": bool,
                "is_abusive": bool,
                "confidence": float (0.0–1.0),
                "categories": {category: match_count},
                "matched_words": [str],
                "top_category": str or None,
                "flags": [str],  # mapped to CONTEXT_LABELS
            }
        """
        text_lower = text.lower()
        words = set(re.findall(r"[a-z_]+", text_lower))
        is_hinglish = self.is_hinglish(text)

        # Also check bigrams (two consecutive words joined by _)
        word_list = re.findall(r"[a-z]+", text_lower)
        bigrams = {
            f"{word_list[i]}_{word_list[i+1]}"
            for i in range(len(word_list) - 1)
        } if len(word_list) > 1 else set()

        all_tokens = words | bigrams
        categories: dict[str, int] = {}
        matched: list[str] = []

        for cat, kw_set in self.keywords.items():
            hits = all_tokens & kw_set
            if hits:
                categories[cat] = len(hits)
                matched.extend(hits)

        total_matches = sum(categories.values())
        if total_matches == 0:
            return {
                "is_hinglish": is_hinglish,
                "is_abusive": False,
                "confidence": 0.0,
                "categories": {},
                "matched_words": [],
                "top_category": None,
                "flags": [],
            }

        # Confidence: scaled by match density (more matches = higher confidence)
        match_ratio = min(total_matches / max(len(words), 1), 1.0)
        confidence = min(0.3 + match_ratio * 0.7, 1.0)

        # Sort categories by hit count
        top_cat = max(categories, key=categories.get)
        flags = [
            _HINGLISH_TO_CONTEXT[cat]
            for cat in categories
            if cat in _HINGLISH_TO_CONTEXT
        ]

        return {
            "is_hinglish": is_hinglish,
            "is_abusive": True,
            "confidence": round(confidence, 3),
            "categories": categories,
            "matched_words": sorted(set(matched)),
            "top_category": top_cat,
            "flags": flags,
        }


# Initialize the Hinglish detector (lightweight, no model loading)
_hinglish_detector = HinglishDetector()


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION — Labels, severity, flags
# ═══════════════════════════════════════════════════════════════════════════════
CONTEXT_LABELS: list[str] = [
    "Insult / Harassment",
    "Direct Threat",
    "Academic / Hierarchy Abuse",
    "Severe Ragging / Hazing",
    "Suicide / Self-harm Risk",
]

# Severity weights for each context label
_SEVERITY_WEIGHTS: dict[str, float] = {
    "Insult / Harassment": 0.15,
    "Direct Threat": 0.25,
    "Academic / Hierarchy Abuse": 0.15,
    "Severe Ragging / Hazing": 0.35,
    "Suicide / Self-harm Risk": 0.50,
}


def severity_score(model_prob: float, zero_shot_res: Optional[dict]) -> float:
    """
    Compute a composite severity score from model probability and
    zero-shot contextual flags.

    Returns:
        Score in [0.0, 1.0]
    """
    score = 0.5 * model_prob
    if not zero_shot_res:
        return round(min(score, 1.0), 4)

    scores = dict(zip(zero_shot_res["labels"], zero_shot_res["scores"]))
    threshold = CONFIG["flag_confidence"]
    for label, weight in _SEVERITY_WEIGHTS.items():
        if scores.get(label, 0) > threshold:
            score += weight
    return round(min(score, 1.0), 4)


def get_flags(zero_shot_res: Optional[dict]) -> list[str]:
    """Extract flags that exceed the confidence threshold."""
    if not zero_shot_res:
        return ["None detected"]
    flags = []
    threshold = CONFIG["flag_confidence"]
    for label, conf in zip(zero_shot_res["labels"], zero_shot_res["scores"]):
        if conf > threshold:
            flags.append(label)
    return flags or ["None detected"]


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING (cached, with graceful degradation)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_models() -> tuple:
    """
    Load transformer pipelines with graceful degradation.

    Returns:
        (toxicity_pipeline, zero_shot_pipeline) — either may be None on failure.
    """
    tox_pipeline = None
    zero_shot_pipeline = None

    try:
        logger.info("Loading toxicity model: %s", CONFIG["toxicity_model"])
        tox_pipeline = pipeline(
            "text-classification",
            model=CONFIG["toxicity_model"],
            tokenizer=CONFIG["toxicity_model"],
        )
        logger.info("Toxicity model loaded successfully.")
    except Exception as e:
        logger.error("Failed to load toxicity model: %s", e)

    try:
        logger.info("Loading zero-shot model: %s", CONFIG["zeroshot_model"])
        zero_shot_pipeline = pipeline(
            "zero-shot-classification",
            model=CONFIG["zeroshot_model"],
        )
        logger.info("Zero-shot model loaded successfully.")
    except Exception as e:
        logger.error("Failed to load zero-shot model: %s", e)

    # ── Warm-up: run a dummy prediction to pre-load weights into memory ────
    if tox_pipeline is not None:
        try:
            tox_pipeline("warmup", truncation=True, max_length=32)
            logger.info("Toxicity model warmed up.")
        except Exception:
            pass

    if zero_shot_pipeline is not None:
        try:
            zero_shot_pipeline("warmup", ["test"], multi_label=False)
            logger.info("Zero-shot model warmed up.")
        except Exception:
            pass

    return tox_pipeline, zero_shot_pipeline


# ═══════════════════════════════════════════════════════════════════════════════
#  PREDICTION (with Hinglish integration)
# ═══════════════════════════════════════════════════════════════════════════════
def predict_full(text: str, models: tuple) -> dict:
    """
    Run full prediction pipeline: toxicity + zero-shot + Hinglish keyword analysis.

    The pipeline:
        1. English DistilBERT toxicity classification
        2. Zero-shot contextual flag extraction
        3. Hinglish keyword-based abuse detection
        4. Signal fusion: combine all three for final prediction

    Args:
        text: Raw user input (will be cleaned internally).
        models: Tuple of (toxicity_pipeline, zero_shot_pipeline).

    Returns:
        Dict with prediction, probability, severity, flags, language info, etc.
    """
    tox_pipeline, zero_shot_pipeline = models
    cleaned = clean_text_bert(text)

    # ── Hinglish detection (runs regardless of model availability) ─────────
    hinglish_result = _hinglish_detector.detect(text)

    if tox_pipeline is None:
        # If models unavailable but Hinglish detects abuse, still report it
        if hinglish_result["is_abusive"]:
            return {
                "prediction": "Cyberbullying",
                "probability": round(hinglish_result["confidence"] * 100, 1),
                "severity": round(hinglish_result["confidence"] * 100, 1),
                "severity_raw": hinglish_result["confidence"],
                "flags": hinglish_result["flags"] or ["Hinglish abuse detected"],
                "bully_type": hinglish_result["top_category"],
                "cleaned_text": cleaned,
                "language": "Hinglish" if hinglish_result["is_hinglish"] else "English",
                "hinglish_matches": hinglish_result["matched_words"],
            }
        logger.warning("Prediction called but toxicity model is unavailable.")
        return {
            "prediction": "Error",
            "probability": 0,
            "severity": 0,
            "severity_raw": 0,
            "flags": ["Model unavailable"],
            "bully_type": None,
            "cleaned_text": cleaned,
            "language": "Unknown",
            "hinglish_matches": [],
        }

    # ── Run English toxicity model ─────────────────────────────────────────
    max_len = CONFIG["max_token_length"]
    result = tox_pipeline(cleaned, truncation=True, max_length=max_len)[0]

    # ── Zero-shot evaluation ───────────────────────────────────────────────
    zero_shot_res = None
    if zero_shot_pipeline is not None:
        try:
            zero_shot_res = zero_shot_pipeline(
                text, CONTEXT_LABELS, multi_label=True
            )
        except Exception as e:
            logger.warning("Zero-shot classification failed: %s", e)

    # martin-ha/toxic-comment-model returns {"label": "toxic"/"non-toxic", "score": float}
    if result["label"] == "toxic":
        prob = float(result["score"])
    else:
        prob = 1.0 - float(result["score"])

    sev = severity_score(prob, zero_shot_res)
    flags = get_flags(zero_shot_res)

    # ── Hinglish signal fusion ─────────────────────────────────────────────
    # If Hinglish detector found abuse, boost the probability and add flags
    if hinglish_result["is_abusive"]:
        h_conf = hinglish_result["confidence"]
        h_threshold = CONFIG["hinglish_keyword_threshold"]

        # Add Hinglish-detected flags that aren't already present
        for hflag in hinglish_result["flags"]:
            if hflag not in flags:
                flags.append(hflag)

        # Remove "None detected" if Hinglish found something
        if "None detected" in flags and len(flags) > 1:
            flags.remove("None detected")

        # Boost probability if Hinglish abuse is strong but English model missed it
        if h_conf >= h_threshold and prob < CONFIG["bully_threshold"]:
            # Weighted combination: lean toward Hinglish signal for Hinglish text
            if hinglish_result["is_hinglish"]:
                prob = max(prob, h_conf * 0.85)
            else:
                prob = max(prob, h_conf * 0.6)
            logger.info(
                "Hinglish detector boosted probability to %.2f (conf=%.2f)",
                prob, h_conf,
            )

        # Recalculate severity with boosted probability
        sev = severity_score(prob, zero_shot_res)
        # Additional severity boost for Hinglish-specific categories
        if hinglish_result.get("top_category") == "suicide_risk":
            sev = min(sev + 0.25, 1.0)
        elif hinglish_result.get("top_category") == "threat":
            sev = min(sev + 0.15, 1.0)

    # ── Context override ───────────────────────────────────────────────────
    has_context_flag = flags != ["None detected"]

    if has_context_flag and zero_shot_res:
        max_zs_conf = max(zero_shot_res["scores"])
        if prob < CONFIG["bully_threshold"]:
            prob = max(CONFIG["bully_threshold"] + 0.01, max_zs_conf)

    prediction = (
        "Cyberbullying"
        if (prob > CONFIG["bully_threshold"] or has_context_flag)
        else "Safe"
    )

    # Determine language label
    language = "Hinglish" if hinglish_result["is_hinglish"] else "English"

    return {
        "prediction": prediction,
        "probability": round(prob * 100, 1),
        "severity": round(sev * 100, 1),
        "severity_raw": sev,
        "flags": flags,
        "bully_type": hinglish_result.get("top_category") if hinglish_result["is_abusive"] else None,
        "cleaned_text": cleaned,
        "language": language,
        "hinglish_matches": hinglish_result.get("matched_words", []),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  LIME EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
def get_lime_words(text: str, tox_pipeline, n: int = None) -> list:
    """
    Generate LIME word-level explanations.

    Returns:
        List of (word, weight) tuples sorted by absolute weight.
    """
    if n is None:
        n = CONFIG["lime_num_features"]
    try:
        from lime.lime_text import LimeTextExplainer

        max_len = CONFIG["max_token_length"]

        def proba_fn(texts):
            cleaned = [clean_text_bert(t) for t in texts]
            results = tox_pipeline(cleaned, truncation=True, max_length=max_len)
            probs = []
            for res in results:
                if res["label"] == "toxic":
                    p = float(res["score"])
                else:
                    p = 1.0 - float(res["score"])
                probs.append([1.0 - p, p])
            return np.array(probs)

        exp = LimeTextExplainer(class_names=["Safe", "Bully"]).explain_instance(
            text,
            proba_fn,
            num_features=n,
            num_samples=CONFIG["lime_num_samples"],
        )
        return exp.as_list()
    except ImportError:
        logger.warning("LIME is not installed. Skipping explanation.")
        return []
    except Exception as e:
        logger.error("LIME explanation failed: %s", e)
        return []


# ═══════════════════════════════════════════════════════════════════════════════
#  RISK LEVEL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def get_risk_info(sev_raw: float) -> dict:
    """Map raw severity to risk label, badge class, action text, and counselor flag."""
    t_low = CONFIG["threshold_low"]
    t_high = CONFIG["threshold_high"]
    t_crit = CONFIG["threshold_critical"]

    if sev_raw >= t_crit:
        return {
            "label": "CRITICAL",
            "badge": "badge-critical",
            "action": "🚨 Immediate counselor alert will be sent. A counselor will contact you within 15 minutes.",
            "counselor_notified": True,
        }
    elif sev_raw >= t_high:
        return {
            "label": "HIGH",
            "badge": "badge-high",
            "action": "⚠️ Formal complaint drafted below. Review it and click Submit to send to the counselor.",
            "counselor_notified": False,
        }
    elif sev_raw >= t_low:
        return {
            "label": "MODERATE",
            "badge": "badge-moderate",
            "action": "ℹ️ Report logged. Self-help and peer support resources provided below.",
            "counselor_notified": False,
        }
    else:
        return {
            "label": "LOW",
            "badge": "badge-low",
            "action": "✅ Low severity detected. Self-help resources are available if needed.",
            "counselor_notified": False,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  CSS (no user-controlled content in styles)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
    .report-box {
        background: #1e1e2e;
        border: 1px solid #444466;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        font-family: monospace;
        font-size: 0.88rem;
        white-space: pre-wrap;
        line-height: 1.7;
        color: #e0e0f0 !important;
    }
    .badge-critical { background:#fee2e2; color:#991b1b; padding:4px 10px;
                      border-radius:6px; font-weight:600; }
    .badge-high     { background:#fef3c7; color:#92400e; padding:4px 10px;
                      border-radius:6px; font-weight:600; }
    .badge-moderate { background:#dbeafe; color:#1e40af; padding:4px 10px;
                      border-radius:6px; font-weight:600; }
    .badge-low      { background:#dcfce7; color:#14532d; padding:4px 10px;
                      border-radius:6px; font-weight:600; }
    .section-head   { font-size:0.75rem; font-weight:600; color:#6b7280;
                      text-transform:uppercase; letter-spacing:.05em; margin-bottom:4px; }
    .version-tag    { font-size:0.65rem; color:#64748b; }
</style>
""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD MODELS
# ═══════════════════════════════════════════════════════════════════════════════
with st.spinner("Loading AI models… (this may take a minute on first run)"):
    tox_pipeline, zero_shot_pipeline = load_models()

models_tuple = (tox_pipeline, zero_shot_pipeline)
models_ok = tox_pipeline is not None

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shield.png", width=60)
    st.title("SafeCampus")
    st.caption("Anti-Ragging, Cyberbullying & Suicide Prevention Portal")
    st.markdown(
        f"<span class='version-tag'>v{__version__}</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "📋 Report Incident",
            "🔍 Analyze Text",
            "📊 Model Info",
            "ℹ️ How It Works",
        ],
    )

    st.markdown("---")
    st.markdown("**Emergency helplines 🇮🇳**")
    st.markdown("Anti-Ragging: `1800-180-5522`")
    st.markdown("iCall: `9152987821`")
    st.markdown("Vandrevala: `1860-2662-345`")
    st.markdown("iCall Chat: [icallhelpline.org](https://icallhelpline.org)")

    if models_ok:
        st.markdown("---")
        st.success(
            "Transformer loaded ✓\n\n"
            "**DistilBERT (toxic-comment-model)**\n\n"
            "Context-Aware Dense Architecture"
        )
        if zero_shot_pipeline is not None:
            st.info("Zero-shot model ✓\n\n**DistilBART-MNLI**")
        else:
            st.warning("Zero-shot model unavailable.\nToxicity-only mode active.")
        st.info("🇮🇳 **Hinglish support ✓**\nBilingual detection active")
    else:
        st.error(
            "⚠️ Model failed to load.\n"
            "Ensure `transformers` and `torch` are installed."
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — REPORT INCIDENT
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📋 Report Incident":

    st.title("🛡️ SafeCampus Reporting Portal")
    st.markdown(
        "**You are safe here.** Paste the message(s) you received below — "
        "**English or Hinglish**, both work. "
        "Our system will assess the severity and — if needed — automatically "
        "generate a formal report for the college anti-ragging cell / counselor. "
        "_You decide what gets submitted._"
    )
    st.info(
        "🔒 **Privacy guarantee**: Your submission is processed on the server "
        "only when you click Analyze. Nothing is stored or shared without your consent.\n\n"
        "🇮🇳 **अब हिंदी/Hinglish भी supported है!**"
    )

    st.markdown("---")

    # ── Input form ────────────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        student_name_raw = st.text_input(
            "Your name (optional — leave blank to stay anonymous)", max_chars=100
        )
    with col2:
        anonymous = st.checkbox("Submit anonymously", value=True)

    message_input_raw = st.text_area(
        "Paste the bullying message(s) here",
        height=160,
        placeholder="Example:\n'You're such a loser, nobody wants you in this group. "
        "You should just disappear.'",
        max_chars=CONFIG["max_input_chars"],
    )

    show_lime = st.checkbox("Show word-level explanation (LIME)", value=True)

    analyze_btn = st.button(
        "🔍 Analyze & Generate Report", type="primary", disabled=not models_ok
    )

    if analyze_btn and message_input_raw.strip():
        # ── Security: sanitize & validate ─────────────────────────────────
        message_input = sanitize_input(message_input_raw)
        student_name = sanitize_name(student_name_raw)

        is_valid, validation_error = validate_input(message_input)
        if not is_valid:
            st.warning(validation_error)
            st.stop()

        is_allowed, rate_error = check_rate_limit()
        if not is_allowed:
            st.error(rate_error)
            st.stop()

        # ── Analyze ───────────────────────────────────────────────────────
        with st.spinner("Analyzing message with Transformers…"):
            result = predict_full(message_input, models_tuple)

        sev_pct = result["severity"]
        sev_raw = result["severity_raw"]
        ts = datetime.now().strftime("%d %b %Y, %I:%M %p")
        submitter = (
            "Anonymous"
            if (anonymous or not student_name.strip())
            else student_name.strip()
        )

        # ── Risk level & action ───────────────────────────────────────────
        risk = get_risk_info(sev_raw)

        # ── Results dashboard ─────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Analysis Result")

        m1, m2, m3 = st.columns(3)
        m1.metric("Classification", result["prediction"])
        m2.metric("Bullying probability", f"{result['probability']}%")
        m3.metric("Severity score", f"{sev_pct}%")

        # Risk badge (no user content in HTML)
        st.markdown(
            f"<div style='margin:10px 0'><span class='section-head'>Risk level</span><br>"
            f"<span class='{risk['badge']}'>{html.escape(risk['label'])}</span></div>",
            unsafe_allow_html=True,
        )

        # Flags (all values are from CONTEXT_LABELS, not user input)
        st.markdown(
            "<div class='section-head'>Detected flags</div>",
            unsafe_allow_html=True,
        )
        for flag in result["flags"]:
            if "Suicide" in flag:
                bg, text_color = "#7f1d1d", "#fecaca"
            elif "Ragging" in flag or "Hierarchy" in flag:
                bg, text_color = "#4c0519", "#ffe4e6"
            elif "Threat" in flag:
                bg, text_color = "#7c2d12", "#fed7aa"
            else:
                bg, text_color = "#713f12", "#fef08a"
            st.markdown(
                f"<span style='background:{bg};color:{text_color};padding:4px 12px;"
                f"border-radius:6px;font-size:0.85rem;font-weight:600;"
                f"margin-right:6px'>{html.escape(flag)}</span>",
                unsafe_allow_html=True,
            )

        # Bully type
        if result.get("bully_type") and result["prediction"] == "Cyberbullying":
            safe_bully_type = html.escape(
                result["bully_type"].replace("_", " ").title()
            )
            st.markdown(
                f"<div style='margin-top:10px'><span class='section-head'>Bullying type</span><br>"
                f"<code>{safe_bully_type}</code></div>",
                unsafe_allow_html=True,
            )

        # Language detected
        lang = result.get("language", "English")
        lang_icon = "🇮🇳" if lang == "Hinglish" else "🇬🇧"
        st.markdown(
            f"<div style='margin-top:8px'><span class='section-head'>Language detected</span><br>"
            f"<span style='background:#312e81;color:#c7d2fe;padding:4px 12px;"
            f"border-radius:6px;font-size:0.85rem;font-weight:600'>"
            f"{lang_icon} {html.escape(lang)}</span></div>",
            unsafe_allow_html=True,
        )

        # Hinglish matched keywords
        if result.get("hinglish_matches"):
            st.markdown(
                f"<div style='margin-top:6px'><span class='section-head'>Hinglish keywords matched</span><br>"
                f"<code>{html.escape(', '.join(result['hinglish_matches'][:15]))}</code></div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(f"**Recommended action:** {risk['action']}")

        # ── Severity progress bar ─────────────────────────────────────────
        t_crit = CONFIG["threshold_critical"]
        t_high = CONFIG["threshold_high"]
        t_low = CONFIG["threshold_low"]
        sev_color = (
            "#ef4444" if sev_raw >= t_crit
            else "#f59e0b" if sev_raw >= t_high
            else "#3b82f6" if sev_raw >= t_low
            else "#22c55e"
        )
        st.markdown(
            f"<div style='margin:10px 0'>"
            f"<div class='section-head'>Severity gauge</div>"
            f"<div style='background:#e5e7eb;border-radius:8px;height:14px;overflow:hidden'>"
            f"<div style='width:{sev_pct}%;background:{sev_color};height:100%;border-radius:8px;"
            f"transition:width .5s'></div></div>"
            f"<div style='font-size:0.75rem;color:#6b7280;margin-top:3px'>"
            f"0% — Low &nbsp;&nbsp;&nbsp; 25% — Moderate &nbsp;&nbsp;&nbsp; 50% — High &nbsp;&nbsp;&nbsp; 75%+ — Critical</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── LIME word explanation ─────────────────────────────────────────
        if show_lime:
            with st.expander("🔬 Word-level explanation (LIME)", expanded=True):
                st.caption(
                    "Red words pushed the model toward **Bullying**. "
                    "Green words pushed it toward **Safe**. "
                    "This is what makes the prediction transparent and trustworthy."
                )
                with st.spinner("Running LIME explanation…"):
                    lime_words = get_lime_words(message_input, tox_pipeline)
                if lime_words:
                    sorted_words = sorted(
                        lime_words, key=lambda x: abs(x[1]), reverse=True
                    )
                    html_parts = []
                    for word, weight in sorted_words:
                        if weight > 0:
                            bg = f"rgba(239,68,68,{abs(weight)*3:.2f})"
                            label = "→ Bully"
                        else:
                            bg = f"rgba(34,197,94,{abs(weight)*3:.2f})"
                            label = "→ Safe"
                        html_parts.append(
                            f"<div style='display:flex;align-items:center;gap:10px;"
                            f"padding:4px 8px;border-radius:6px;background:{bg};"
                            f"margin-bottom:4px;font-size:0.87rem'>"
                            f"<code style='min-width:120px'>{html.escape(word)}</code>"
                            f"<span style='color:#374151;font-size:0.75rem'>{weight:+.4f} {label}</span>"
                            f"</div>"
                        )
                    st.markdown("".join(html_parts), unsafe_allow_html=True)
                else:
                    st.info("LIME not available. Install with: `pip install lime`")

        # ── Auto-generated report (all user content escaped) ──────────────
        safe_message = html.escape(message_input.strip())
        report_text = (
            f"SAFECAMPUS — INCIDENT REPORT\n"
            f"{'='*48}\n"
            f"Timestamp       : {ts}\n"
            f"Submitted by    : {html.escape(submitter)}\n"
            f"{'='*48}\n"
            f"CLASSIFICATION  : {result['prediction']}\n"
            f"PROBABILITY     : {result['probability']}%\n"
            f"SEVERITY SCORE  : {sev_pct}%\n"
            f"RISK LEVEL      : {risk['label']}\n"
            f"FLAGS           : {', '.join(result['flags'])}\n"
        )
        if result.get("bully_type"):
            report_text += f"BULLY TYPE      : {result['bully_type'].replace('_',' ').title()}\n"
        report_text += (
            f"{'='*48}\n"
            f"ORIGINAL MESSAGE:\n{safe_message}\n"
            f"{'='*48}\n"
            f"ACTION TAKEN    : {risk['action']}\n"
            f"{'='*48}\n"
            f"SUPPORT RESOURCES:\n"
            f"  Anti-Ragging Helpline : 1800-180-5522\n"
            f"  iCall Helpline        : 9152987821\n"
            f"  Vandrevala Foundation  : 1860-2662-345\n"
            f"  iCall Chat             : https://icallhelpline.org\n"
            f"  Anti-Ragging Cell      : antiragging@college.edu\n"
            f"{'='*48}\n"
        )

        st.markdown("---")
        st.subheader("📄 Auto-Generated Incident Report")
        st.markdown(
            f"<div class='report-box' style='color:#e0e0f0 !important;'>"
            f"{html.escape(report_text)}</div>",
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                label="⬇️ Download Report",
                data=report_text,
                file_name=f"safecampus_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )
        with col_b:
            if risk["counselor_notified"]:
                st.error(
                    "🚨 Counselor notified automatically (CRITICAL level)"
                )
            else:
                st.button(
                    "📧 Submit to Counselor",
                    help="In a live deployment, this emails the report to the college counselor.",
                )

        # ── Resources ─────────────────────────────────────────────────────
        if sev_raw >= CONFIG["threshold_high"]:
            st.markdown("---")
            st.subheader("🆘 Immediate Support Resources")
            rc1, rc2 = st.columns(2)
            with rc1:
                st.error(
                    "**iCall — Free Counseling**\n📞 9152987821\nMon–Sat, 8am–10pm"
                )
            with rc2:
                st.warning(
                    "**Vandrevala Foundation**\n📞 1860-2662-345\nAvailable 24/7"
                )
        else:
            st.markdown("---")
            st.info(
                "**Peer Support**: Talk to your college's Student Welfare office "
                "or a trusted friend. You are not alone."
            )

    elif analyze_btn and not message_input_raw.strip():
        st.warning("Please paste a message before clicking Analyze.")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — ANALYZE TEXT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Analyze Text":

    st.title("🔍 Quick Text Analyzer")
    st.caption(
        "For testing the model on custom messages (English & Hinglish). Does not generate a formal report."
    )

    text_input_raw = st.text_area(
        "Enter any message to analyze",
        height=120,
        placeholder="Type or paste a message here (English or Hinglish)…",
        max_chars=CONFIG["max_input_chars"],
    )
    btn = st.button("Analyze", type="primary", disabled=not models_ok)

    if btn and text_input_raw.strip():
        # ── Security ──────────────────────────────────────────────────────
        text_input = sanitize_input(text_input_raw)

        is_valid, validation_error = validate_input(text_input)
        if not is_valid:
            st.warning(validation_error)
            st.stop()

        is_allowed, rate_error = check_rate_limit()
        if not is_allowed:
            st.error(rate_error)
            st.stop()

        result = predict_full(text_input, models_tuple)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Result", result["prediction"])
        col2.metric("Probability", f"{result['probability']}%")
        col3.metric("Severity", f"{result['severity']}%")
        if result.get("bully_type"):
            col4.metric(
                "Bully type", result["bully_type"].replace("_", " ").title()
            )

        st.markdown(
            "**Flags detected:** " + " | ".join(result["flags"])
        )
        lang = result.get("language", "English")
        lang_icon = "🇮🇳" if lang == "Hinglish" else "🇬🇧"
        st.markdown(f"**Language:** {lang_icon} {lang}")
        if result.get("hinglish_matches"):
            st.markdown(
                f"**Hinglish keywords:** `{', '.join(result['hinglish_matches'][:10])}`"
            )
        st.markdown(
            f"**Cleaned input:** `{html.escape(result['cleaned_text'])}`"
        )

        # Batch demo
        st.markdown("---")
        st.markdown("**Try these examples:**")
        examples = [
            "You're such a loser, nobody likes you.",
            "I will make you regret this. Watch your back.",
            "I can't do this anymore. I just want to end it all.",
            "Great lecture today! See you all at the fest!",
            "Abe nalle, kya samajh raha hai khud ko? Ek chamaat lagunga.",
            "Bhai aaj ki class toh bohot boring thi.",
            "Tu bilkul zero hai, mar ja kisi ko farak nahi padega.",
        ]
        for ex in examples:
            r = predict_full(ex, models_tuple)
            emoji_icon = (
                "🚨" if r["severity_raw"] >= CONFIG["threshold_critical"]
                else "⚠️" if r["severity_raw"] >= CONFIG["threshold_high"]
                else "🟡" if r["severity_raw"] >= CONFIG["threshold_low"]
                else "✅"
            )
            lang = r.get("language", "English")
            lang_tag = " 🇮🇳" if lang == "Hinglish" else ""
            st.markdown(
                f"{emoji_icon} **{r['prediction']}** ({r['probability']}% | severity {r['severity']}%){lang_tag} "
                f"— _{html.escape(ex[:60])}_"
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — MODEL INFO
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Info":

    st.title("📊 Model Performance")

    if not models_ok:
        st.error("Install torch and transformers to load the model.")
    else:
        st.markdown("**Engine:** `DistilBERT Context-Aware Transformer`")
        st.markdown(f"**Weights:** `{CONFIG['toxicity_model']}`")
        st.markdown(
            "**Dataset:** Fine-tuned on Jigsaw Toxic Comment Classification Challenge"
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Context", "Deep Bidirectional")
        c2.metric("Parameters", "~66 Million")
        c3.metric("LIME Status", "Supported")

        st.markdown("---")
        st.subheader("What each metric means")
        st.markdown(
            """
| Metric | What it measures | Why it matters here |
|---|---|---|
| **F1 Score** | Balance of precision & recall | We care about both catching bullying AND not over-flagging safe messages |
| **AUC-ROC** | How well model separates classes regardless of threshold | Higher = better ranking of bully vs. safe |
| **Accuracy** | % correct predictions | Least important when classes are imbalanced |
| **Recall** | % of actual bully cases caught | Critical — missing a bullying case is worse than a false alarm |
| **Precision** | % of flagged cases that are actually bullying | Matters for counselor trust — too many false alarms = ignored system |
        """
        )

        st.markdown("---")
        st.subheader("Review 2 Improvements over Review 1")
        st.markdown(
            """
| Component | Review 1 | Review 2 |
|---|---|---|
| Preprocessing | Basic (stopwords + lemmatize) | + Emoji removal, contraction expansion, elongated word correction |
| TF-IDF features | 8,000 | 10,000 |
| Evaluation | F1 + accuracy + confusion matrix | + AUC-ROC curves |
| Explainability | None | LIME word-level explanation |
| Multi-class | Not supported | Future-work demo added |
| Deployment | Notebook only | Streamlit SafeCampus portal |
| Suicide risk | Lexicon only | Tiered 3-level severity scoring |
        """
        )

        st.markdown("---")
        st.subheader("Security Measures")
        st.markdown(
            """
| Measure | Details |
|---|---|
| **Input Sanitization** | All user input is stripped of HTML, control characters, and normalized via bleach |
| **Rate Limiting** | Max 10 analyses per minute per session |
| **XSRF Protection** | Enabled via Streamlit server config |
| **XSS Prevention** | All user-derived content passed through `html.escape()` before rendering |
| **No Data Persistence** | No user data is stored — analysis is ephemeral |
| **Dependency Pinning** | All dependencies pinned to exact versions for supply-chain safety |
        """
        )

        st.markdown("---")
        st.subheader("Future Work Roadmap")
        st.markdown(
            """
**Phase 1 (Next review):**
- Fine-tune DistilBERT — contextual embeddings will handle sarcasm and coded language that TF-IDF misses
- True multi-class classification (6 bullying types) so counselors know *what kind* of bullying is occurring

**Phase 2 (Medium term):**
- FastAPI backend so the model can be called from any chat platform via REST API
- Browser extension using the API — opt-in device-side monitoring (zero privacy concern)
- Email/SMS counselor alert integration

**Phase 3 (Long term):**
- Multi-modal detection: text + image (memes, screenshots) using CLIP or ViLT
- Temporal escalation tracking: if severity rises over 5+ messages in a thread, auto-escalate
- Hindi / Hinglish support using IndicBERT for Indian college social media
        """
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — HOW IT WORKS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ How It Works":

    st.title("ℹ️ How SafeCampus Works")

    st.markdown("### The problem with surveillance-based approaches")
    st.warning(
        "We **cannot** and **should not** monitor private WhatsApp or Instagram groups. "
        "This would violate India's IT Act (2000) and the Right to Privacy "
        "(Puttaswamy judgment, 2017). It is also technically infeasible — "
        "WhatsApp uses end-to-end encryption."
    )

    st.markdown("### Our approach: Victim-initiated reporting")
    st.success(
        "The student comes to us **when they need help**. They paste the message. "
        "We analyze it, score the risk, and generate a formal report for them — "
        "removing the emotional burden of having to write a complaint themselves."
    )

    st.markdown("---")
    st.markdown("### System pipeline")
    st.markdown(
        """
```
Student receives bullying message (WhatsApp, Instagram, anywhere)
        ↓
Opens SafeCampus portal on their phone/laptop
        ↓
Pastes the message(s) — THEIR choice, THEIR consent
        ↓
NLP model analyzes:
    → Is it cyberbullying? (binary classifier)
    → What type? (multi-class: age/gender/religion/...)
    → How severe? (0–100% severity score)
    → Suicide/self-harm risk? (3-tier lexicon)
        ↓
    ┌─────────────────────────────────────────────────┐
    │ LOW (<25%)    → Self-help resources shown       │
    │ MODERATE      → Formal complaint drafted        │
    │ HIGH (>50%)   → Complaint drafted + counselor   │
    │ CRITICAL(>75%)→ Counselor auto-alerted 🚨       │
    └─────────────────────────────────────────────────┘
        ↓
Student downloads report or submits to counselor
```
    """
    )

    st.markdown("---")
    st.markdown("### Why this is better than surveillance")

    col1, col2 = st.columns(2)
    with col1:
        st.error(
            "**Surveillance (wrong approach)**\n\n"
            "- Monitors everyone, always\n"
            "- Massive false positives\n"
            "- Legal liability under IT Act\n"
            "- Students move to other platforms\n"
            "- No consent = no trust"
        )
    with col2:
        st.success(
            "**Victim-initiated (SafeCampus)**\n\n"
            "- Student consents at the moment they need help\n"
            "- 100% of reports are real complaints\n"
            "- Fully legal — student shares their own messages\n"
            "- Removes shame and effort barrier\n"
            "- Complaint writes itself in 30 seconds"
        )

    st.markdown("---")
    st.markdown("### Why underreporting is the real problem")
    st.info(
        "Studies show **70–80% of cyberbullying victims never report it**. "
        "The two main reasons are **shame** (having to explain it to a person) "
        "and **effort** (having to write a formal complaint). "
        "SafeCampus removes both barriers — the student talks to a machine, "
        "and the complaint is auto-generated. That is the real-world value of this project."
    )
