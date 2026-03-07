# pipeline/extract_clip.py
"""
Step 1: Extract a short clip from the source video using ffmpeg.

Extracts a segment (e.g., 0:15–0:30) from the full video without
re-encoding for speed. Also extracts the audio track as a separate WAV
file for downstream transcription.
"""

import os
from pipeline.utils import (
    setup_logger,
    run_ffmpeg,
    validate_video_file,
    get_media_duration,
    ensure_dir,
)

logger = setup_logger("extract_clip")


def extract_clip(
    input_video: str,
    output_dir: str,
    start_time: float = 15.0,
    end_time: float = 30.0,
) -> dict:
    """
    Extract a video clip and its audio from the source video.
    
    Args:
        input_video: Path to the full source video file.
        output_dir:  Directory to write output files into.
        start_time:  Start of the clip in seconds (default: 15.0).
        end_time:    End of the clip in seconds (default: 30.0).
    
    Returns:
        dict with keys:
            - 'clip_video': path to the extracted video clip (.mp4)
            - 'clip_audio': path to the extracted audio (.wav)
            - 'duration':   clip duration in seconds
    
    Raises:
        FileNotFoundError: If input video doesn't exist.
        ValueError:        If time range is invalid.
        RuntimeError:      If ffmpeg fails.
    """
    # ── Validate inputs ──────────────────────────────────────────────────
    input_video = validate_video_file(input_video)
    ensure_dir(output_dir)
    
    # Validate time range
    if start_time < 0:
        raise ValueError(f"start_time must be >= 0, got {start_time}")
    if end_time <= start_time:
        raise ValueError(
            f"end_time ({end_time}) must be greater than start_time ({start_time})"
        )
    
    duration = end_time - start_time
    if duration > 60:
        logger.warning(
            f"Clip duration is {duration}s — this is longer than the expected 15s. "
            "Processing may take longer and use more VRAM."
        )
    
    # Check that the source video is long enough
    source_duration = get_media_duration(input_video)
    if end_time > source_duration:
        raise ValueError(
            f"end_time ({end_time}s) exceeds video duration ({source_duration:.1f}s). "
            f"Choose a range within 0–{source_duration:.1f}s."
        )
    
    logger.info(
        f"Extracting clip: {start_time}s → {end_time}s "
        f"({duration}s) from {os.path.basename(input_video)}"
    )
    
    # ── Extract video clip ───────────────────────────────────────────────
    # Using -c copy avoids re-encoding — much faster.
    # If the clip has keyframe issues, we re-encode with a fast preset.
    clip_video_path = os.path.join(output_dir, "clip.mp4")
    
    run_ffmpeg(
        [
            "-y",                          # Overwrite output
            "-i", input_video,             # Input file
            "-ss", str(start_time),        # Start time
            "-to", str(end_time),          # End time
            "-c:v", "libx264",             # Re-encode video for exact cuts
            "-preset", "fast",             # Fast encoding preset
            "-crf", "18",                  # High quality (lower = better)
            "-c:a", "aac",                 # Re-encode audio too
            "-b:a", "192k",               # Audio bitrate
            clip_video_path,
        ],
        description=f"Extracting video clip ({start_time}s → {end_time}s)",
    )
    
    # ── Extract audio as WAV ─────────────────────────────────────────────
    # Whisper works best with 16kHz mono WAV
    clip_audio_path = os.path.join(output_dir, "clip_audio.wav")
    
    run_ffmpeg(
        [
            "-y",
            "-i", clip_video_path,         # Extract from the clip (not source)
            "-vn",                          # No video
            "-acodec", "pcm_s16le",         # 16-bit PCM WAV
            "-ar", "16000",                 # 16kHz sample rate (Whisper standard)
            "-ac", "1",                     # Mono channel
            clip_audio_path,
        ],
        description="Extracting audio track as 16kHz mono WAV",
    )
    
    # ── Verify outputs ───────────────────────────────────────────────────
    actual_duration = get_media_duration(clip_video_path)
    logger.info(f"Clip extracted: {actual_duration:.1f}s → {clip_video_path}")
    logger.info(f"Audio extracted: {clip_audio_path}")
    
    # Sanity check: clip duration should be within ±1s of expected
    if abs(actual_duration - duration) > 1.0:
        logger.warning(
            f"Clip duration mismatch: expected {duration:.1f}s, "
            f"got {actual_duration:.1f}s (possible keyframe issue)"
        )
    
    return {
        "clip_video": clip_video_path,
        "clip_audio": clip_audio_path,
        "duration": actual_duration,
    }
