# pipeline/lip_sync.py
"""
Step 6a: Lip-sync video to Hindi audio using Wav2Lip.

Wav2Lip generates mouth movements that match the input audio, but it
produces a blurry lower-face region. This is intentional — the next step
(face_restore.py) uses CodeFormer to restore facial quality.

Wav2Lip requires:
  - A video with a visible face
  - An audio file to sync to
  - A pre-trained checkpoint (wav2lip_gan.pth for better quality)
"""

import os
import sys
import subprocess
from typing import Optional

from pipeline.utils import (
    setup_logger,
    free_gpu_memory,
    get_device,
    ensure_dir,
)

logger = setup_logger("lip_sync")

# Path to the Wav2Lip repo (cloned during setup)
WAV2LIP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Wav2Lip")


def lip_sync_video(
    video_path: str,
    audio_path: str,
    output_dir: str,
    wav2lip_dir: Optional[str] = None,
    checkpoint: str = "wav2lip_gan.pth",
    device: Optional[str] = None,
    resize_factor: int = 2,
    face_det_batch_size: int = 4,
    wav2lip_batch_size: int = 8,
) -> dict:
    """
    Apply Wav2Lip lip-sync to a video using the given audio.
    
    Args:
        video_path:          Path to the input video clip.
        audio_path:          Path to the Hindi audio to sync.
        output_dir:          Directory to save the output.
        wav2lip_dir:         Path to cloned Wav2Lip repository.
        checkpoint:          Wav2Lip model checkpoint filename.
        device:              'cuda' or 'cpu'.
        resize_factor:       Downscale factor for face detection (1=original).
        face_det_batch_size: Batch size for face detection.
        wav2lip_batch_size:  Batch size for Wav2Lip inference.
    
    Returns:
        dict with keys:
            - 'synced_video': Path to the lip-synced video.
            - 'checkpoint':   Checkpoint used.
    
    Raises:
        FileNotFoundError: If inputs or Wav2Lip checkpoint don't exist.
        RuntimeError:      If Wav2Lip inference fails.
    """
    # ── Validate inputs ──────────────────────────────────────────────────
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    if wav2lip_dir is None:
        wav2lip_dir = WAV2LIP_DIR
    
    if not os.path.isdir(wav2lip_dir):
        raise FileNotFoundError(
            f"Wav2Lip directory not found: {wav2lip_dir}\n"
            "Clone it with: git clone https://github.com/Rudrabha/Wav2Lip.git"
        )
    
    # Locate checkpoint
    checkpoint_path = os.path.join(wav2lip_dir, "checkpoints", checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Wav2Lip checkpoint not found: {checkpoint_path}\n"
            "Download wav2lip_gan.pth from:\n"
            "  https://github.com/Rudrabha/Wav2Lip#getting-the-weights\n"
            f"Place it in: {os.path.join(wav2lip_dir, 'checkpoints')}"
        )
    
    if device is None:
        device = get_device()
    
    ensure_dir(output_dir)
    output_video = os.path.join(output_dir, "wav2lip_output.mp4")
    
    logger.info("Running Wav2Lip lip-sync...")
    logger.info(f"  Video: {os.path.basename(video_path)}")
    logger.info(f"  Audio: {os.path.basename(audio_path)}")
    logger.info(f"  Checkpoint: {checkpoint}")
    
    # ── Run Wav2Lip inference ────────────────────────────────────────────
    # Wav2Lip is run as a subprocess (its own script with complex imports)
    inference_script = os.path.join(wav2lip_dir, "inference.py")
    
    if not os.path.exists(inference_script):
        raise FileNotFoundError(f"Wav2Lip inference.py not found at: {inference_script}")
    
    cmd = [
        sys.executable,                    # Use the same Python interpreter
        inference_script,
        "--checkpoint_path", checkpoint_path,
        "--face", video_path,
        "--audio", audio_path,
        "--outfile", output_video,
        "--resize_factor", str(resize_factor),
        "--face_det_batch_size", str(face_det_batch_size),
        "--wav2lip_batch_size", str(wav2lip_batch_size),
    ]
    
    # Add --nosmooth for sharper (but potentially jittery) results
    # cmd.append("--nosmooth")
    
    logger.info(f"Executing Wav2Lip inference...")
    logger.debug(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=wav2lip_dir,  # Run from Wav2Lip directory for relative imports
            capture_output=True,
            text=True,
            timeout=1800,  # 30-minute timeout for GPU inference
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "Wav2Lip inference timed out after 30 minutes. "
            "This may indicate a GPU memory issue."
        )
    
    # Log output for debugging
    if result.stdout:
        logger.debug(f"Wav2Lip stdout:\n{result.stdout[-1000:]}")
    if result.stderr:
        # Wav2Lip prints progress to stderr, so not all stderr is errors
        logger.debug(f"Wav2Lip stderr:\n{result.stderr[-1000:]}")
    
    if result.returncode != 0:
        raise RuntimeError(
            f"Wav2Lip failed (exit code {result.returncode}):\n"
            f"{result.stderr[-500:]}"
        )
    
    # ── Verify output ────────────────────────────────────────────────────
    if not os.path.exists(output_video):
        # Wav2Lip might save to a default location — check common paths
        default_output = os.path.join(wav2lip_dir, "results", "result_voice.mp4")
        if os.path.exists(default_output):
            import shutil
            shutil.move(default_output, output_video)
            logger.info(f"Moved Wav2Lip output from default location to: {output_video}")
        else:
            raise RuntimeError(
                "Wav2Lip did not produce an output file. "
                "Check the logs above for errors."
            )
    
    logger.info(f"Lip-synced video saved: {output_video}")
    logger.info(
        "⚠  Note: The output may have a blurry lower-face region. "
        "This will be fixed in Step 6b (CodeFormer face restoration)."
    )
    
    # ── Cleanup GPU ──────────────────────────────────────────────────────
    free_gpu_memory()
    
    return {
        "synced_video": output_video,
        "checkpoint": checkpoint,
    }
