# pipeline/transcribe.py
"""
Step 2–3: Transcribe Kannada audio using Sarvam AI's Saaras V3 API.

Uses the Saaras V3 speech-to-text model via REST API.
Requires a Sarvam AI API key (set via SARVAM_API_KEY environment variable).

API docs: https://docs.sarvam.ai/api-reference-docs/speech-to-text
"""

import json
import os
import requests
from typing import Optional

from pipeline.utils import setup_logger

logger = setup_logger("transcribe")

# Sarvam AI speech-to-text API endpoint
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


def transcribe_audio(
    audio_path: str,
    output_dir: str,
    language: str = "kn-IN",
    api_key: Optional[str] = None,
) -> dict:
    """
    Transcribe audio using Sarvam AI's Saaras V3 API.
    
    Args:
        audio_path: Path to the WAV audio file (16kHz mono recommended).
        output_dir: Directory to save transcript files.
        language:   BCP-47 language code (default: 'kn-IN' for Kannada).
        api_key:    Sarvam AI API key. Falls back to SARVAM_API_KEY env var.
    
    Returns:
        dict with keys:
            - 'text':            Full transcript string.
            - 'language':        Language code used.
            - 'transcript_file': Path to saved JSON transcript.
    
    Raises:
        FileNotFoundError: If audio file doesn't exist.
        ValueError:        If API key is not provided.
        RuntimeError:      If API call fails.
    """
    # ── Validate input ───────────────────────────────────────────────────
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Resolve API key from argument or environment variable
    api_key = api_key or os.environ.get("SARVAM_API_KEY")
    if not api_key:
        raise ValueError(
            "Sarvam AI API key is required. Provide via:\n"
            "  1. api_key argument, or\n"
            "  2. SARVAM_API_KEY environment variable\n"
            "Get your key at: https://dashboard.sarvam.ai/"
        )
    
    logger.info(f"Transcribing audio: {os.path.basename(audio_path)} (language={language})")
    logger.info("Using Sarvam AI Saaras V3 model...")
    
    # ── Call Sarvam AI Saaras V3 API ─────────────────────────────────────
    headers = {
        "api-subscription-key": api_key,
    }
    
    # Determine the MIME type based on file extension
    ext = os.path.splitext(audio_path)[1].lower()
    mime_map = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
    }
    mime_type = mime_map.get(ext, "audio/wav")
    
    # Build the multipart form data request
    with open(audio_path, "rb") as audio_file:
        files = {
            "file": (os.path.basename(audio_path), audio_file, mime_type),
        }
        data = {
            "model": "saaras:v3",
            "language_code": language,
            "mode": "transcribe",
            "with_timestamps": "true",
        }
        
        logger.info("Sending audio to Sarvam AI API...")
        
        try:
            response = requests.post(
                SARVAM_STT_URL,
                headers=headers,
                files=files,
                data=data,
                timeout=120,  # 2-minute timeout for audio processing
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                "Sarvam AI API request timed out after 120s. "
                "The audio file may be too long."
            )
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Could not connect to Sarvam AI API. "
                "Check your internet connection."
            )
    
    # ── Handle API response ──────────────────────────────────────────────
    if response.status_code != 200:
        error_detail = response.text[:500]
        raise RuntimeError(
            f"Sarvam AI API error (HTTP {response.status_code}):\n{error_detail}"
        )
    
    result = response.json()
    
    # Extract transcript text from response
    transcript_text = result.get("transcript", "")
    
    if not transcript_text:
        raise RuntimeError(
            "Sarvam AI returned an empty transcript. "
            "The audio may not contain clear speech."
        )
    
    # Extract language detection info
    detected_lang = result.get("language_code", language)
    lang_confidence = result.get("language_confidence", None)
    
    logger.info(f"Transcript: {transcript_text}")
    if detected_lang:
        confidence_str = f" (confidence: {lang_confidence:.1%})" if lang_confidence else ""
        logger.info(f"Detected language: {detected_lang}{confidence_str}")
    
    # ── Save transcript ──────────────────────────────────────────────────
    transcript_data = {
        "language": detected_lang,
        "language_confidence": lang_confidence,
        "text": transcript_text,
        "timestamps": result.get("timestamps", []),
        "api_response": result,  # Save full API response for debugging
    }
    
    transcript_file = os.path.join(output_dir, "transcript.json")
    with open(transcript_file, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, ensure_ascii=False, indent=2)
    
    # Also save a plain text version for easy manual editing
    text_file = os.path.join(output_dir, "transcript.txt")
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(transcript_text)
    
    logger.info(f"Transcript saved to: {transcript_file}")
    logger.info(
        f"Plain text transcript saved to: {text_file}\n"
        "  ⚠  Please review and correct the Kannada transcript before proceeding."
    )
    
    return {
        "text": transcript_text,
        "language": detected_lang,
        "transcript_file": transcript_file,
    }
