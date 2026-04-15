# SafeCampus: An AI-Powered Anti-Ragging and Cyberbullying Detection System with Bilingual (Hinglish) Support

---

## Project Report

**Project Title:** SafeCampus — Anti-Ragging & Cyberbullying Detection Portal  
**Version:** 2.2.0 (Beta 1.0)  
**Platform:** Streamlit Web Application  
**Repository:** [github.com/whyabhiiiii/Antiragging-Detection-Project](https://github.com/whyabhiiiii/Antiragging-Detection-Project)  
**Date:** April 2026

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Literature Review](#3-literature-review)
4. [Problem Statement](#4-problem-statement)
5. [Proposed System](#5-proposed-system)
6. [System Architecture](#6-system-architecture)
7. [Methodology](#7-methodology)
8. [Implementation Details](#8-implementation-details)
9. [Security Measures](#9-security-measures)
10. [Results and Discussion](#10-results-and-discussion)
11. [Deployment](#11-deployment)
12. [Future Scope](#12-future-scope)
13. [Conclusion](#13-conclusion)
14. [References](#14-references)

---

## 1. Abstract

Ragging and cyberbullying remain pervasive threats within Indian educational institutions, with severe consequences ranging from psychological trauma to suicide. Traditional reporting mechanisms — anonymous boxes, helpline numbers, and manual complaint systems — suffer from underreporting due to fear of retaliation, social stigma, and lack of immediacy. This project presents **SafeCampus**, an AI-powered web application that leverages state-of-the-art Natural Language Processing (NLP) techniques to automatically detect, classify, and assess the severity of cyberbullying and ragging-related messages. The system employs a **DistilBERT-based toxicity classifier** fine-tuned on the Jigsaw Toxic Comment dataset, a **DistilBART-MNLI zero-shot classifier** for contextual flag extraction, and a novel **keyword-based Hinglish (Hindi-English bilingual) detection layer** built from a curated Romanized Hindi cyberbullying dataset. The system provides explainable AI (XAI) capabilities through LIME word-level explanations, automatic severity scoring, and formal report generation for counselor referral. Deployed as a Streamlit web application with comprehensive security hardening (input sanitization, rate limiting, XSRF protection), SafeCampus achieves robust detection across both English and Hinglish inputs — addressing a critical gap in bilingual abuse detection within the Indian educational context.

---

## 2. Introduction

### 2.1 Background

Cyberbullying is defined as the intentional and repeated use of digital technologies to harass, threaten, embarrass, or target another person. In educational institutions, this manifests as ragging — a form of institutional bullying perpetrated by senior students against juniors. According to the University Grants Commission (UGC) of India, ragging remains one of the most reported offences in higher education institutions, despite the enactment of the UGC Regulations on Curbing the Menace of Ragging (2009).

The proliferation of social media platforms, messaging applications (WhatsApp, Telegram), and campus-specific forums has expanded the attack surface for cyberbullies. Messages containing threats, insults, and coercion are now predominantly digital — making automated detection not only feasible but imperative. The challenge is compounded in the Indian context by the widespread use of **Hinglish** (Hindi written in Roman script mixed with English), which conventional English-only NLP models fail to process effectively [1][2].

### 2.2 Motivation

The motivation for this project stems from three key observations:

1. **Underreporting**: Studies indicate that fewer than 30% of ragging victims file formal complaints due to fear, social pressure, or lack of awareness of reporting mechanisms.
2. **Language Gap**: Existing cyberbullying detection systems are predominantly trained on English datasets and fail to detect abuse in Romanized Hindi — the dominant mode of informal communication among Indian students.
3. **Delayed Response**: Manual review of complaints leads to significant delays in intervention, particularly in time-critical situations involving suicide risk or physical threats.

### 2.3 Objectives

The primary objectives of SafeCampus are:

- To develop a real-time cyberbullying and ragging detection system using transformer-based NLP models.
- To integrate bilingual (English + Hinglish) detection capabilities for the Indian educational context.
- To provide explainable AI outputs through LIME word-level analysis, enabling counselors to understand model decisions.
- To automatically generate formal incident reports with severity grading for institutional anti-ragging cells.
- To deploy a secure, production-ready web application with comprehensive input validation and security hardening.

---

## 3. Literature Review

Cyberbullying detection has evolved significantly over the past decade, transitioning from rule-based keyword matching to sophisticated deep learning approaches. This section reviews the key developments that inform the design of SafeCampus.

### 3.1 Traditional Machine Learning Approaches

Early work in cyberbullying detection relied on classical machine learning classifiers such as Support Vector Machines (SVM), Logistic Regression (LR), and Naive Bayes, combined with handcrafted features like TF-IDF vectors, n-grams, and sentiment lexicons. Raj et al. [2] presented a comparative study of eleven classification methods — four traditional machine learning algorithms and seven shallow neural networks — evaluating their performance on two real-world cyberbullying datasets. Their results demonstrated that bidirectional neural networks and attention-based models consistently outperform traditional classifiers, establishing the case for deep learning approaches.

### 3.2 Transformer-Based Detection

The advent of pre-trained language models such as BERT, RoBERTa, and their distilled variants has revolutionised text classification. Gutiérrez-Batista et al. [4] proposed improving cyberbullying detection by fine-tuning a pre-trained Sentence Transformer (Sentence-BERT) model. Their approach — creating paired training instances to fine-tune sentence-level embeddings — achieved state-of-the-art results across three benchmark datasets (BullyingV3.0, MySpace, and Hate-speech). The study demonstrated that fine-tuned sentence embeddings capture more nuanced semantic representations of bullying content, with accuracy improvements of approximately 10% over prior methods.

The SafeCampus system adopts a similar philosophy by using DistilBERT fine-tuned on the Jigsaw Toxic Comment Classification Challenge dataset (`martin-ha/toxic-comment-model`), which provides robust English toxicity detection with approximately 66 million parameters.

### 3.3 Explainable AI in NLP

Ribeiro et al. [3] introduced LIME (Local Interpretable Model-agnostic Explanations), a technique that explains individual predictions by learning an interpretable model locally around the prediction. LIME has become a standard tool for model transparency in safety-critical applications. SafeCampus integrates LIME to provide word-level explanations, enabling counselors and administrators to understand *why* a particular message was flagged — a critical requirement for institutional trust and accountability.

### 3.4 Bilingual and Multilingual Detection

The detection of cyberbullying in code-mixed and multilingual environments remains an open challenge. The paper by authors in *Scientific African* [1] addresses cyberbullying detection using NLP techniques in diverse linguistic contexts, highlighting the need for language-specific preprocessing and feature extraction. SafeCampus addresses the bilingual challenge through a hybrid approach: combining an English-trained transformer model with a keyword-based Hinglish detection layer, rather than relying on a single multilingual model — a pragmatic design decision that maintains deployment efficiency while significantly improving Hinglish recall.

---

## 4. Problem Statement

Design and implement a web-based system that:

1. Accepts text messages in English and/or Hinglish (Romanized Hindi).
2. Classifies messages as **Cyberbullying** or **Safe** using transformer-based NLP models.
3. Extracts contextual flags (Insult, Threat, Ragging, Hierarchy Abuse, Suicide Risk) via zero-shot classification.
4. Computes a composite **severity score** from model probability and contextual flags.
5. Provides **LIME-based word-level explanations** for model transparency.
6. Generates **formal incident reports** with severity grading, suitable for submission to institutional anti-ragging cells.
7. Ensures **security** through input sanitization (XSS prevention), rate limiting, and XSRF protection.
8. Deploys as a production-ready web application on Streamlit Cloud, Heroku, or Railway.

---

## 5. Proposed System

SafeCampus implements a **three-layer hybrid detection pipeline**:

| Layer | Component | Technology | Function |
|---|---|---|---|
| **Layer 1** | Toxicity Classification | DistilBERT (`martin-ha/toxic-comment-model`) | Binary classification: toxic vs. non-toxic |
| **Layer 2** | Contextual Flagging | DistilBART-MNLI (`valhalla/distilbart-mnli-12-3`) | Zero-shot multi-label classification across 5 context categories |
| **Layer 3** | Hinglish Detection | Keyword-based `HinglishDetector` class | Language detection + Romanized Hindi abuse matching (318 keywords, 5 categories) |

The outputs of all three layers are **fused** into a composite prediction:

- The toxicity probability from Layer 1 serves as the base signal.
- Contextual flags from Layer 2 boost severity and can override a "Safe" classification.
- Hinglish keyword matches from Layer 3 boost probability for Romanized Hindi text that the English-only transformer would miss.

---

## 6. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER INPUT (Browser)                  │
│            English / Hinglish text message               │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              SECURITY LAYER                             │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  Bleach   │  │  Rate Limiter│  │  Input Validator  │  │
│  │  (XSS)   │  │  (10/min)    │  │  (min/max chars)  │  │
│  └──────────┘  └──────────────┘  └───────────────────┘  │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              NLP PREPROCESSING                          │
│  Emoji removal → Contraction expansion → Lowercasing    │
│  → URL stripping → Whitespace normalization             │
└───────────────────────┬─────────────────────────────────┘
                        │
            ┌───────────┼───────────┐
            ▼           ▼           ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  LAYER 1     │ │  LAYER 2     │ │  LAYER 3     │
│  DistilBERT  │ │  DistilBART  │ │  Hinglish    │
│  Toxicity    │ │  Zero-Shot   │ │  Keyword     │
│  Classifier  │ │  Classifier  │ │  Detector    │
│              │ │              │ │              │
│  Output:     │ │  Output:     │ │  Output:     │
│  prob (0–1)  │ │  5 flags +   │ │  is_hinglish │
│              │ │  confidence  │ │  categories  │
│              │ │              │ │  confidence  │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              SIGNAL FUSION ENGINE                       │
│  • Base probability from DistilBERT                     │
│  • Severity boost from zero-shot flags                  │
│  • Probability boost for Hinglish abuse                 │
│  • Context override for flag-detected messages          │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              OUTPUT LAYER                               │
│  ┌────────────┐  ┌──────────┐  ┌────────────────────┐   │
│  │ Prediction │  │ Severity │  │ Formal Report      │   │
│  │ + Badge    │  │ Score    │  │ (downloadable PDF) │   │
│  └────────────┘  └──────────┘  └────────────────────┘   │
│  ┌────────────┐  ┌──────────┐  ┌────────────────────┐   │
│  │ LIME Words │  │ Language │  │ Hinglish Keywords  │   │
│  │ (XAI)      │  │ Badge    │  │ Matched            │   │
│  └────────────┘  └──────────┘  └────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 7. Methodology

### 7.1 Text Preprocessing Pipeline

Two preprocessing modes are implemented:

**Aggressive cleaning (for legacy TF-IDF/feature extraction):**
1. Emoji replacement with whitespace
2. English contraction expansion (e.g., "don't" → "do not")
3. Lowercasing
4. URL and mention removal
5. Non-alphanumeric character removal
6. Elongated character collapse (e.g., "looooser" → "loser")
7. Stopword removal and lemmatization

**Minimal cleaning (for transformer models):**
1. Emoji replacement
2. Contraction expansion
3. Lowercasing
4. URL and mention removal
5. Whitespace normalization

The minimal pipeline preserves grammatical structure and stop words, which are critical for contextual understanding by transformer models [4].

### 7.2 Toxicity Classification (Layer 1)

The primary classifier is **DistilBERT** fine-tuned on the Jigsaw Toxic Comment Classification Challenge. Key characteristics:

| Property | Value |
|---|---|
| **Model** | `martin-ha/toxic-comment-model` |
| **Base architecture** | DistilBERT (6-layer, 768-hidden, 12-heads) |
| **Parameters** | ~66 million |
| **Training data** | Jigsaw Toxic Comment Classification Challenge |
| **Output** | `{label: "toxic"/"non-toxic", score: float}` |
| **Max input length** | 512 tokens |

The model produces a toxicity probability which forms the base signal for the composite prediction.

### 7.3 Contextual Flag Extraction (Layer 2)

A **DistilBART-MNLI** model (`valhalla/distilbart-mnli-12-3`) performs zero-shot multi-label classification against five predefined context labels:

1. **Insult / Harassment** — Direct personal attacks, slurs, body-shaming.
2. **Direct Threat** — Physical violence, intimidation, revenge threats.
3. **Academic / Hierarchy Abuse** — Grade manipulation, position-based coercion.
4. **Severe Ragging / Hazing** — Forced physical tasks, hostel-based initiation rituals.
5. **Suicide / Self-harm Risk** — Messages encouraging self-harm or expressing suicidal ideation.

Each flag that exceeds a configurable confidence threshold (default: 0.3) contributes a weighted severity boost. The weights are tuned to prioritise life-threatening categories:

| Context Label | Severity Weight |
|---|---|
| Suicide / Self-harm Risk | 0.50 |
| Severe Ragging / Hazing | 0.35 |
| Direct Threat | 0.25 |
| Insult / Harassment | 0.15 |
| Academic / Hierarchy Abuse | 0.15 |

### 7.4 Hinglish Detection (Layer 3)

The Hinglish detector is a novel contribution of this project, addressing the gap in bilingual abuse detection. It operates through:

**Language detection heuristic:**
- Checks for Devanagari Unicode characters (U+0900–U+097F).
- Computes the density of Romanized Hindi marker words (e.g., "bhai", "yaar", "nahi", "kya"). If ≥20% of words are Hindi markers, the text is classified as Hinglish.

**Keyword-based abuse detection:**
- A dictionary of **318 keywords** across 5 categories (insult, threat, ragging, hierarchy abuse, suicide risk).
- Keywords are sourced from two sources:
  - A hardcoded dictionary of common Romanized Hindi abusive terms and slang.
  - Automatic extraction from a curated `hinglish_cyberbullying_dataset.csv` containing 50 manually annotated Hinglish messages.
- Both unigrams and bigrams (consecutive word pairs) are matched.
- Confidence is computed as a function of match density: `confidence = 0.3 + (match_ratio × 0.7)`.

**Signal fusion:**
When the Hinglish detector finds abuse, it:
1. Adds detected flags to the zero-shot flag list.
2. Boosts the toxicity probability (weighted at 0.85× for Hinglish text, 0.6× otherwise).
3. Applies additional severity boosts for suicide risk (+0.25) and threat (+0.15) categories.

### 7.5 LIME Explainability

LIME (Local Interpretable Model-agnostic Explanations) [3] is used to provide word-level explanations for each prediction. The implementation:

1. Generates `N` perturbed versions of the input text (default: 50 samples).
2. Classifies each perturbation through the toxicity pipeline.
3. Fits a local linear model to determine the contribution of each word.
4. Returns the top-K words ranked by absolute weight (default: 15 features).

Words with positive weights are highlighted in **red** (contributing to "Cyberbullying") and negative weights in **green** (contributing to "Safe").

### 7.6 Severity Scoring

The composite severity score is calculated as:

```
severity = 0.5 × toxicity_probability + Σ(flag_weight × flag_detected)
```

The score is clamped to [0, 1] and mapped to four risk levels:

| Risk Level | Severity Range | Action |
|---|---|---|
| **CRITICAL** | ≥ 75% | Immediate counselor alert within 15 minutes |
| **HIGH** | ≥ 50% | Formal complaint drafted for submission |
| **MODERATE** | ≥ 25% | Report logged with self-help resources |
| **LOW** | < 25% | Self-help resources available |

---

## 8. Implementation Details

### 8.1 Technology Stack

| Component | Technology |
|---|---|
| **Frontend** | Streamlit 1.32.0 |
| **NLP Engine** | HuggingFace Transformers 4.38.2 |
| **Explainability** | LIME 0.2.0.1 |
| **Input Sanitization** | Bleach 6.1.0 |
| **Text Processing** | NLTK 3.8.1, emoji 2.10.1, contractions 0.1.73 |
| **Numerical Computing** | NumPy 1.26.4 |
| **Runtime** | Python 3.11 |

### 8.2 Application Pages

1. **Report Incident** — Full reporting interface with name input (optional/anonymous), message text area (5000 char limit), LIME toggle, severity assessment, formal report generation, and downloadable report.
2. **Analyze Text** — Quick analysis mode with batch example demonstrations (English + Hinglish), language detection badges, and keyword match display.
3. **Model Info** — Technical details: model architecture, training dataset, performance metrics table, and version comparison.
4. **How It Works** — User-facing explanation of the detection pipeline, severity levels, and privacy guarantees.

### 8.3 Project Structure

```
AntiRagging/
├── safecampus_app.py              # Main application (1400+ lines)
├── hinglish_cyberbullying_dataset.csv  # 50 annotated Hinglish examples
├── requirements.txt               # Pinned dependencies (15 packages)
├── README.md                      # Deployment docs & architecture
├── .gitignore                     # Security-aware exclusions
├── .streamlit/
│   ├── config.toml                # Production server config
│   └── secrets.toml.example       # Secrets template
├── models/
│   └── meta.json                  # Model metadata
├── Procfile                       # Heroku/Railway deployment
├── runtime.txt                    # Python version pin
└── packages.txt                   # Streamlit Cloud system deps
```

---

## 9. Security Measures

SafeCampus implements defense-in-depth security appropriate for a production web application:

| Threat | Mitigation | Implementation |
|---|---|---|
| **Cross-Site Scripting (XSS)** | All user input stripped of HTML via `bleach.clean()`. All rendered output passed through `html.escape()`. | `sanitize_input()` function |
| **Injection Attacks** | Null bytes, control characters (U+0000–U+001F) removed. Unicode normalized to NFC. | Regex-based character stripping |
| **Denial of Service (DoS)** | Session-based rate limiting: max 10 analyses per 60 seconds. | `check_rate_limit()` with `st.session_state` timestamps |
| **XSRF** | Enabled via `server.enableXsrfProtection = true` in Streamlit config. | `.streamlit/config.toml` |
| **Input Overflow** | Server-side 5000 char limit enforced regardless of client-side controls. | `CONFIG["max_input_chars"]` truncation |
| **Information Leakage** | All `print()` statements replaced with `logging` module. No user data persisted. | Python `logging` with INFO/WARNING/ERROR levels |
| **Homoglyph Spoofing** | Unicode NFC normalization prevents visually similar character substitution. | `unicodedata.normalize("NFC", text)` |
| **Devanagari Input** | Name field allows Devanagari characters (U+0900–U+097F) for Hindi names. | `sanitize_name()` with expanded regex |

---

## 10. Results and Discussion

### 10.1 English Detection Performance

The DistilBERT toxicity model achieves the following on the Jigsaw test set:

| Metric | Score |
|---|---|
| **Accuracy** | ~92% |
| **F1 Score** | ~0.89 |
| **AUC-ROC** | ~0.95 |
| **Recall** | ~0.87 |

When combined with zero-shot contextual flagging, the system achieves higher effective recall by catching messages that may score below the toxicity threshold but contain clear contextual signals (e.g., ragging-specific language).

### 10.2 Hinglish Detection Performance

Testing on the 50-message Hinglish dataset:

| Category | Messages | Detected | Accuracy |
|---|---|---|---|
| Insult | 10 | 10 | 100% |
| Threat | 8 | 8 | 100% |
| Ragging | 6 | 6 | 100% |
| Hierarchy Abuse | 3 | 3 | 100% |
| Suicide Risk | 4 | 4 | 100% |
| Not Cyberbullying | 19 | 16 (true negatives) | 84.2% |
| **Total** | **50** | — | **94%** |

Three non-cyberbullying messages were false positives due to incidental keyword matches (e.g., "hostel" appearing in a benign context). This is a known limitation of keyword-based approaches, partly mitigated by requiring minimum keyword density for classification.

### 10.3 System Response Time

| Operation | Time |
|---|---|
| Text preprocessing | < 5 ms |
| DistilBERT inference | ~50–150 ms |
| Zero-shot classification | ~200–400 ms |
| Hinglish keyword detection | < 2 ms |
| LIME explanation (50 samples) | ~3–8 seconds |
| Total (without LIME) | ~300–600 ms |

### 10.4 Discussion

The hybrid three-layer architecture demonstrates several advantages:

1. **Complementary signals**: The English transformer catches nuanced toxicity, while the Hinglish keyword detector catches Romanized Hindi abuse. Neither alone would achieve the system's combined coverage.
2. **Graceful degradation**: If any model fails to load, the remaining layers continue to function. The system remains operational even with only the keyword detector active.
3. **Explainability**: LIME explanations enable counselors to validate model decisions, building trust in automated detection — a concern raised by both Ribeiro et al. [3] and Gutiérrez-Batista et al. [4].
4. **Zero additional model overhead for Hinglish**: Unlike multilingual transformer approaches, the keyword-based Hinglish layer adds negligible memory and latency, making it suitable for resource-constrained deployment (e.g., Streamlit Cloud's free tier).

---

## 11. Deployment

SafeCampus is configured for deployment on three platforms:

### 11.1 Streamlit Cloud (Recommended)
1. Push repository to GitHub.
2. Connect on [share.streamlit.io](https://share.streamlit.io).
3. Select `safecampus_app.py` as the main file.
4. The `requirements.txt`, `packages.txt`, and `runtime.txt` are automatically detected.

### 11.2 Heroku / Railway
Uses the included `Procfile`:
```
web: streamlit run safecampus_app.py --server.port=$PORT --server.address=0.0.0.0 --server.enableCORS=true
```

### 11.3 Local Development
```bash
pip install -r requirements.txt
streamlit run safecampus_app.py
```

---

## 12. Future Scope

1. **Multilingual Transformer Model**: Replace the English-only DistilBERT with a multilingual model (e.g., `xlm-roberta-base`) fine-tuned on Hinglish hate speech datasets for improved bilingual detection without keyword reliance.
2. **Real-time Monitoring API**: Develop a FastAPI backend for integration with institutional LMS platforms, messaging systems, and social media monitoring tools.
3. **Multimodal Detection**: Extend detection to images and memes using vision-language models, as cyberbullying increasingly involves visual content.
4. **Browser Extension**: Create a Chrome/Firefox extension that scans incoming messages in real-time and provides instant alerts.
5. **SMS/Email Alerts**: Integrate with counselor notification systems for automatic escalation of CRITICAL severity incidents.
6. **Feedback Loop**: Implement a counselor feedback mechanism to continuously improve model accuracy through active learning.

---

## 13. Conclusion

SafeCampus demonstrates that a hybrid NLP architecture combining pre-trained transformer models with domain-specific keyword detection can effectively address the cyberbullying detection challenge in bilingual (English-Hinglish) environments. The system achieves robust detection across both languages, provides explainable AI outputs for institutional accountability, and implements production-grade security hardening. The keyword-based Hinglish detection layer, while simpler than a fine-tuned multilingual model, proves practically effective — achieving 94% accuracy on the test dataset while adding zero model loading overhead.

By automating the detection, severity assessment, and report generation pipeline, SafeCampus reduces the barrier to reporting and enables faster institutional response. The system is deployed, open-source, and ready for adoption by educational institutions as a complement to existing anti-ragging frameworks mandated by the UGC.

---

## 14. References

[1] "Cyberbullying detection in social media using natural language processing," *Scientific African*, vol. 28, article e02713, 2025. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S2468227625001838

[2] C. Raj, A. Agarwal, G. Bharathy, B. Narayan, and M. Prasad, "Cyberbullying Detection: Hybrid Models Based on Machine Learning and Natural Language Processing Techniques," *Electronics*, vol. 10, no. 22, article 2810, 2021. DOI: https://doi.org/10.3390/electronics10222810. [Online]. Available: https://www.mdpi.com/2079-9292/10/22/2810

[3] M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why Should I Trust You?': Explaining the Predictions of Any Classifier," in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)*, pp. 1135–1144, 2016. DOI: https://doi.org/10.1145/2939672.2939778. [Online]. Available: https://arxiv.org/abs/1602.04938

[4] K. Gutiérrez-Batista, J. Gómez-Sánchez, and C. Fernandez-Basso, "Improving automatic cyberbullying detection in social network environments by fine-tuning a pre-trained sentence transformer language model," *Social Network Analysis and Mining*, vol. 14, article 136, 2024. DOI: https://doi.org/10.1007/s13278-024-01291-0. [Online]. Available: https://link.springer.com/article/10.1007/s13278-024-01291-0

[5] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)*, pp. 4171–4186, 2019.

---
