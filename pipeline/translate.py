# pipeline/translate.py
"""
Step 4: Translate Kannada text to Hindi using Sarvam AI's Translate API.

Uses the Sarvam-Translate (Mayura v1) model via REST API for Kannada → Hindi.
Requires a Sarvam AI API key (set via SARVAM_API_KEY environment variable).

API docs: https://docs.sarvam.ai/api-reference-docs/translate
"""

import json
import os
import requests
from typing import Optional

from pipeline.utils import setup_logger

logger = setup_logger("translate")

# Sarvam AI translation API endpoint
SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"


def translate_text(
    text: str,
    output_dir: str,
    source_lang: str = "kn-IN",
    target_lang: str = "hi-IN",
    api_key: Optional[str] = None,
) -> dict:
    """
    Translate text from Kannada to Hindi using Sarvam AI Translate API.
    
    Args:
        text:        Source text in Kannada.
        output_dir:  Directory to save translation output.
        source_lang: BCP-47 source language code (default: 'kn-IN' for Kannada).
        target_lang: BCP-47 target language code (default: 'hi-IN' for Hindi).
        api_key:     Sarvam AI API key. Falls back to SARVAM_API_KEY env var.
    
    Returns:
        dict with keys:
            - 'source_text':      Original Kannada text.
            - 'translated_text':  Hindi translation.
            - 'method':           'sarvam_translate'.
            - 'translation_file': Path to saved JSON.
    
    Raises:
        ValueError:   If text is empty or API key is missing.
        RuntimeError: If the API call fails.
    """
    if not text or not text.strip():
        raise ValueError("Input text is empty. Cannot translate.")
    
    # Resolve API key from argument or environment variable
    api_key = api_key or os.environ.get("SARVAM_API_KEY")
    if not api_key:
        raise ValueError(
            "Sarvam AI API key is required. Provide via:\n"
            "  1. api_key argument, or\n"
            "  2. SARVAM_API_KEY environment variable\n"
            "Get your key at: https://dashboard.sarvam.ai/"
        )
    
    logger.info(f"Translating {len(text)} chars: {source_lang} → {target_lang}")
    logger.info(f"Source text: {text[:200]}{'...' if len(text) > 200 else ''}")
    
    # ── Call Sarvam Translate API ─────────────────────────────────────────
    # The API has a character limit per request (~900 chars).
    # For longer texts, we split into sentences and translate in batches.
    sentences = _split_sentences(text)
    
    translated_parts = []
    
    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
        
        logger.info(f"Translating sentence {i+1}/{len(sentences)}...")
        
        result = _call_sarvam_translate(
            text=sentence,
            source_lang=source_lang,
            target_lang=target_lang,
            api_key=api_key,
        )
        
        translated_parts.append(result)
    
    translated_text = " ".join(translated_parts)
    
    if not translated_text.strip():
        raise RuntimeError("Translation produced empty output.")
    
    logger.info(f"Translated text:\n  {translated_text}")
    
    # ── Save translation ─────────────────────────────────────────────────
    translation_data = {
        "source_language": source_lang,
        "target_language": target_lang,
        "source_text": text,
        "translated_text": translated_text,
        "method": "sarvam_translate",
    }
    
    translation_file = os.path.join(output_dir, "translation.json")
    with open(translation_file, "w", encoding="utf-8") as f:
        json.dump(translation_data, f, ensure_ascii=False, indent=2)
    
    # Also save plain Hindi text for easy inspection
    hindi_text_file = os.path.join(output_dir, "hindi_text.txt")
    with open(hindi_text_file, "w", encoding="utf-8") as f:
        f.write(translated_text)
    
    logger.info(f"Translation saved to: {translation_file}")
    
    return {
        "source_text": text,
        "translated_text": translated_text,
        "method": "sarvam_translate",
        "translation_file": translation_file,
    }


# ---------------------------------------------------------------------------
# Sarvam Translate API Call
# ---------------------------------------------------------------------------

def _call_sarvam_translate(
    text: str,
    source_lang: str,
    target_lang: str,
    api_key: str,
) -> str:
    """
    Make a single translation request to the Sarvam Translate API.
    
    Args:
        text:        Text to translate (should be within API character limit).
        source_lang: BCP-47 source language code.
        target_lang: BCP-47 target language code.
        api_key:     Sarvam AI API key.
    
    Returns:
        Translated text string.
    
    Raises:
        RuntimeError: If API call fails.
    """
    headers = {
        "Content-Type": "application/json",
        "api-subscription-key": api_key,
    }
    
    payload = {
        "input": text,
        "source_language_code": source_lang,
        "target_language_code": target_lang,
        "model": "mayura:v1",
        "mode": "formal",
        "enable_preprocessing": True,
    }
    
    try:
        response = requests.post(
            SARVAM_TRANSLATE_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Sarvam Translate API timed out after 30s.")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not connect to Sarvam Translate API. "
            "Check your internet connection."
        )
    
    if response.status_code != 200:
        error_detail = response.text[:500]
        raise RuntimeError(
            f"Sarvam Translate API error (HTTP {response.status_code}):\n{error_detail}"
        )
    
    result = response.json()
    translated = result.get("translated_text", "")
    
    if not translated:
        raise RuntimeError(
            f"Sarvam Translate returned empty output for: {text[:100]}"
        )
    
    return translated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using basic punctuation rules.
    
    Handles common Kannada/Devanagari sentence-ending marks:
    period (.), question mark (?), exclamation (!), and danda (।).
    """
    import re
    
    # Split on sentence-ending punctuation, keeping the delimiter
    parts = re.split(r'(?<=[.?!।])\s+', text.strip())
    
    # Filter out empty strings
    sentences = [s.strip() for s in parts if s.strip()]
    
    # If no sentence boundaries found, return the whole text as one "sentence"
    if not sentences:
        sentences = [text.strip()]
    
    return sentences
