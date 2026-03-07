# pipeline/utils.py
"""
Shared utilities for the Supernan dubbing pipeline.
Handles logging setup, GPU memory management, and ffmpeg operations.
"""

import logging
import os
import subprocess
import shutil
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a named logger with a consistent format.
    
    Args:
        name:  Logger name — typically the module name (e.g. 'transcribe').
        level: Logging level (default: INFO).
    
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding duplicate handlers if called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)-8s | %(name)-15s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# GPU Memory Management
# ---------------------------------------------------------------------------

def free_gpu_memory():
    """
    Aggressively free GPU memory between pipeline steps.
    
    On a Colab T4 (15GB), sequential model loading requires releasing VRAM
    before loading the next model.  This function:
      1. Calls torch.cuda.empty_cache() to release cached allocations.
      2. Triggers Python garbage collection to release dangling references.
    
    Safe to call even when CUDA is not available (it's a no-op).
    """
    import gc
    gc.collect()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            logger = setup_logger("gpu")
            logger.info(
                f"GPU memory after cleanup — allocated: {allocated:.0f} MB, "
                f"reserved: {reserved:.0f} MB"
            )
    except ImportError:
        # torch not installed yet — fine, nothing to free
        pass


def get_device() -> str:
    """
    Returns the best available device string ('cuda' or 'cpu').
    Logs a warning if falling back to CPU.
    """
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
            logger = setup_logger("gpu")
            logger.info(f"Using GPU: {name} ({vram:.1f} GB VRAM)")
            return "cuda"
    except ImportError:
        pass
    
    logger = setup_logger("gpu")
    logger.warning("CUDA not available — falling back to CPU (will be slow)")
    return "cpu"


# ---------------------------------------------------------------------------
# FFmpeg Helpers
# ---------------------------------------------------------------------------

def check_ffmpeg() -> str:
    """
    Verify that ffmpeg is installed and accessible on PATH.
    
    Returns:
        Path to the ffmpeg executable.
    
    Raises:
        RuntimeError: If ffmpeg is not found.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it with:\n"
            "  Colab:   !apt-get install -y ffmpeg\n"
            "  Ubuntu:  sudo apt-get install ffmpeg\n"
            "  Windows: choco install ffmpeg\n"
            "  macOS:   brew install ffmpeg"
        )
    return ffmpeg_path


def run_ffmpeg(args: list[str], description: str = "ffmpeg operation") -> str:
    """
    Execute an ffmpeg command with proper error handling and logging.
    
    Args:
        args:        List of ffmpeg arguments (without the leading 'ffmpeg').
        description: Human-readable label for logging.
    
    Returns:
        Combined stdout + stderr output.
    
    Raises:
        RuntimeError: If the command fails (non-zero exit code).
    """
    logger = setup_logger("ffmpeg")
    ffmpeg_path = check_ffmpeg()
    
    cmd = [ffmpeg_path] + args
    logger.info(f"Running: {description}")
    logger.debug(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5-minute timeout — generous for a 15s clip
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpeg timed out after 300s during: {description}")
    
    if result.returncode != 0:
        logger.error(f"ffmpeg failed:\n{result.stderr}")
        raise RuntimeError(
            f"ffmpeg failed during '{description}' (exit code {result.returncode}):\n"
            f"{result.stderr[:500]}"  # Truncate to avoid log spam
        )
    
    logger.info(f"Completed: {description}")
    return result.stdout + result.stderr


def get_media_duration(filepath: str) -> float:
    """
    Get the duration of a media file in seconds using ffprobe.
    
    Args:
        filepath: Path to the audio or video file.
    
    Returns:
        Duration in seconds (float).
    
    Raises:
        RuntimeError: If ffprobe fails or duration cannot be determined.
    """
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        raise RuntimeError("ffprobe not found. It is usually installed with ffmpeg.")
    
    cmd = [
        ffprobe_path,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        filepath,
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffprobe timed out reading duration of: {filepath}")
    
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {filepath}: {result.stderr}")
    
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        raise RuntimeError(f"Could not parse duration from ffprobe output: {result.stdout}")
    
    return duration


# ---------------------------------------------------------------------------
# File Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> Path:
    """Create a directory (and parents) if it doesn't exist. Returns the Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def validate_video_file(filepath: str) -> str:
    """
    Validate that a file exists and has a recognized video extension.
    
    Args:
        filepath: Path to the video file.
    
    Returns:
        Absolute path as a string.
    
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError:        If the extension is not a recognized video format.
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {filepath}")
    
    allowed_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}
    if path.suffix.lower() not in allowed_extensions:
        raise ValueError(
            f"Unsupported video format '{path.suffix}'. "
            f"Allowed: {', '.join(sorted(allowed_extensions))}"
        )
    
    return str(path.resolve())
