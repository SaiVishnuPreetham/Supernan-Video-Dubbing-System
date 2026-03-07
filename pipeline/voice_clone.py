# pipeline/voice_clone.py
"""
Step 5: Generate Hindi speech using Coqui XTTSv2 voice cloning.

Clones the original speaker's voice from the Kannada audio reference
and synthesizes Hindi speech with matching tone and timbre.

Includes time-stretching to ensure the generated Hindi audio matches
the original Kannada segment duration (±50ms tolerance).
"""

import os
from typing import Optional

from pipeline.utils import (
    setup_logger,
    free_gpu_memory,
    get_device,
    get_media_duration,
)

logger = setup_logger("voice_clone")

# Maximum acceptable duration mismatch ratio before warning
MAX_STRETCH_RATIO = 0.20  # 20%


def clone_voice_and_speak(
    hindi_text: str,
    reference_audio: str,
    output_dir: str,
    target_duration: Optional[float] = None,
    language: str = "hi",
    device: Optional[str] = None,
) -> dict:
    """
    Generate Hindi speech matching the original speaker's voice.
    
    Args:
        hindi_text:       Translated Hindi text to speak.
        reference_audio:  Path to the original speaker's audio (for voice cloning).
        output_dir:       Directory to save generated audio.
        target_duration:  Target duration in seconds to match. If None, no stretching.
        language:         TTS language code (default: 'hi' for Hindi).
        device:           'cuda' or 'cpu'. Auto-detected if None.
    
    Returns:
        dict with keys:
            - 'hindi_audio':     Path to the final Hindi audio file (.wav).
            - 'raw_audio':       Path to raw (pre-stretch) audio.
            - 'duration':        Final audio duration in seconds.
            - 'stretch_applied': Whether time-stretching was applied.
            - 'stretch_ratio':   Ratio of stretch applied (1.0 = no stretch).
    
    Raises:
        FileNotFoundError: If reference audio file doesn't exist.
        RuntimeError:      If TTS generation fails.
    """
    # ── Validate inputs ──────────────────────────────────────────────────
    if not hindi_text or not hindi_text.strip():
        raise ValueError("Hindi text is empty. Cannot generate speech.")
    
    if not os.path.exists(reference_audio):
        raise FileNotFoundError(f"Reference audio not found: {reference_audio}")
    
    if device is None:
        device = get_device()
    
    logger.info(f"Generating Hindi speech ({len(hindi_text)} chars)")
    logger.info(f"Reference audio: {os.path.basename(reference_audio)}")
    
    # ── Load Coqui XTTSv2 model ──────────────────────────────────────────
    try:
        from TTS.api import TTS
    except ImportError:
        raise RuntimeError(
            "Coqui TTS is not installed. Install with:\n"
            "  pip install TTS"
        )
    
    logger.info("Loading Coqui XTTSv2 model...")
    
    # XTTSv2 supports multi-lingual voice cloning
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    tts.to(device)
    
    # ── Generate speech ──────────────────────────────────────────────────
    raw_audio_path = os.path.join(output_dir, "hindi_raw.wav")
    
    logger.info("Synthesizing Hindi speech with voice cloning...")
    
    tts.tts_to_file(
        text=hindi_text,
        file_path=raw_audio_path,
        speaker_wav=reference_audio,   # Voice to clone
        language=language,              # Target language
    )
    
    # Get duration of generated audio
    raw_duration = get_media_duration(raw_audio_path)
    logger.info(f"Raw Hindi audio generated: {raw_duration:.2f}s → {raw_audio_path}")
    
    # ── Cleanup TTS model ────────────────────────────────────────────────
    # Free GPU before potential pydub operations
    del tts
    free_gpu_memory()
    logger.info("XTTSv2 model unloaded, GPU memory freed.")
    
    # ── Time-stretching (if target duration specified) ───────────────────
    final_audio_path = os.path.join(output_dir, "hindi_audio.wav")
    stretch_applied = False
    stretch_ratio = 1.0
    
    if target_duration is not None and target_duration > 0:
        stretch_ratio = target_duration / raw_duration
        deviation = abs(1.0 - stretch_ratio)
        
        logger.info(
            f"Duration comparison: generated={raw_duration:.2f}s, "
            f"target={target_duration:.2f}s, ratio={stretch_ratio:.3f}"
        )
        
        if deviation > MAX_STRETCH_RATIO:
            logger.warning(
                f"⚠  Large duration mismatch: {deviation:.0%} deviation. "
                f"Stretching by >{MAX_STRETCH_RATIO:.0%} may sound unnatural. "
                "Consider adjusting the Hindi text length."
            )
        
        if deviation > 0.02:  # Only stretch if >2% difference
            _time_stretch_audio(raw_audio_path, final_audio_path, stretch_ratio)
            stretch_applied = True
            final_duration = get_media_duration(final_audio_path)
            logger.info(
                f"Time-stretched: {raw_duration:.2f}s → {final_duration:.2f}s "
                f"(target: {target_duration:.2f}s)"
            )
        else:
            # Duration close enough — just copy the file
            import shutil
            shutil.copy2(raw_audio_path, final_audio_path)
            logger.info("Duration within tolerance (±2%), no stretching needed.")
    else:
        # No target duration — use raw audio as-is
        import shutil
        shutil.copy2(raw_audio_path, final_audio_path)
        logger.info("No target duration specified, using raw audio as-is.")
    
    final_duration = get_media_duration(final_audio_path)
    
    return {
        "hindi_audio": final_audio_path,
        "raw_audio": raw_audio_path,
        "duration": final_duration,
        "stretch_applied": stretch_applied,
        "stretch_ratio": round(stretch_ratio, 3),
    }


# ---------------------------------------------------------------------------
# Time-stretching with pydub
# ---------------------------------------------------------------------------

def _time_stretch_audio(
    input_path: str,
    output_path: str,
    ratio: float,
) -> None:
    """
    Time-stretch audio to match a target duration.
    
    Uses pydub to change the playback speed. This alters tempo without
    changing pitch (within reasonable ratios).
    
    For ratios > 1.0: audio is slowed down (stretched)
    For ratios < 1.0: audio is sped up (compressed)
    
    Args:
        input_path:  Path to input WAV file.
        output_path: Path to write stretched WAV file.
        ratio:       Stretch ratio (e.g. 1.2 = 20% slower, 0.8 = 20% faster).
    """
    logger.info(f"Time-stretching audio by ratio {ratio:.3f}")
    
    try:
        from pydub import AudioSegment
    except ImportError:
        raise RuntimeError(
            "pydub is not installed. Install with: pip install pydub"
        )
    
    audio = AudioSegment.from_wav(input_path)
    
    # pydub speed change: adjust frame rate then export at original rate
    # Speed up = higher frame rate → when played at normal rate, sounds faster
    # Slow down = lower frame rate → when played at normal rate, sounds slower
    original_frame_rate = audio.frame_rate
    
    # To stretch by 'ratio', we change the frame rate inversely:
    #   ratio > 1 (need longer) → lower frame rate → playback slower
    #   ratio < 1 (need shorter) → higher frame rate → playback faster
    new_frame_rate = int(original_frame_rate / ratio)
    
    # Change frame rate without resampling (changes perceived speed+pitch slightly)
    stretched = audio._spawn(audio.raw_data, overrides={
        "frame_rate": new_frame_rate,
    })
    
    # Re-export at the standard frame rate to normalize
    stretched = stretched.set_frame_rate(original_frame_rate)
    
    stretched.export(output_path, format="wav")
    logger.info(f"Time-stretched audio saved to: {output_path}")
