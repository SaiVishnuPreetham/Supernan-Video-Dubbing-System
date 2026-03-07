# pipeline/face_restore.py
"""
Step 6b: Restore face quality using CodeFormer.

After Wav2Lip produces a lip-synced video with a blurry lower face,
CodeFormer processes each frame to restore photorealistic facial detail.

Pipeline: extract frames → detect faces → restore each face → reassemble video.

This step is what eliminates the "blurry mouth" quality penalty.
"""

import os
import sys
import subprocess
import shutil
import glob
from typing import Optional

from pipeline.utils import (
    setup_logger,
    free_gpu_memory,
    get_device,
    ensure_dir,
    run_ffmpeg,
    get_media_duration,
)

logger = setup_logger("face_restore")

# Path to the CodeFormer repo (cloned during setup)
CODEFORMER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CodeFormer")


def restore_faces(
    video_path: str,
    output_dir: str,
    codeformer_dir: Optional[str] = None,
    fidelity_weight: float = 0.7,
    upscale: int = 1,
    device: Optional[str] = None,
) -> dict:
    """
    Restore face quality in a Wav2Lip output video using CodeFormer.
    
    Process: video → frames → CodeFormer per-frame → reassembled video.
    
    Args:
        video_path:       Path to the Wav2Lip output video.
        output_dir:       Directory to save restored output.
        codeformer_dir:   Path to cloned CodeFormer repository.
        fidelity_weight:  Balance quality vs faithfulness (0.0=quality, 1.0=faithful).
                          Recommended: 0.5–0.7 for Wav2Lip restoration.
        upscale:          Upscaling factor (1=original resolution, 2=2x, etc.).
        device:           'cuda' or 'cpu'.
    
    Returns:
        dict with keys:
            - 'restored_video': Path to the face-restored video (.mp4).
            - 'fidelity_weight': Fidelity weight used.
            - 'frames_processed': Number of frames processed.
    
    Raises:
        FileNotFoundError: If inputs or CodeFormer aren't found.
        RuntimeError:      If face restoration fails.
    """
    # ── Validate inputs ──────────────────────────────────────────────────
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if codeformer_dir is None:
        codeformer_dir = CODEFORMER_DIR
    
    if not os.path.isdir(codeformer_dir):
        raise FileNotFoundError(
            f"CodeFormer directory not found: {codeformer_dir}\n"
            "Clone it with:\n"
            "  git clone https://github.com/sczhou/CodeFormer.git\n"
            "  cd CodeFormer && pip install -r requirements.txt\n"
            "  python basicsr/setup.py develop"
        )
    
    if device is None:
        device = get_device()
    
    ensure_dir(output_dir)
    
    logger.info("Starting CodeFormer face restoration...")
    logger.info(f"  Video: {os.path.basename(video_path)}")
    logger.info(f"  Fidelity weight: {fidelity_weight}")
    logger.info(f"  Upscale factor: {upscale}")
    
    # ── Step 1: Extract frames from video ────────────────────────────────
    frames_dir = os.path.join(output_dir, "frames_input")
    restored_frames_dir = os.path.join(output_dir, "frames_restored")
    ensure_dir(frames_dir)
    ensure_dir(restored_frames_dir)
    
    # Get video FPS for reassembly
    fps = _get_video_fps(video_path)
    logger.info(f"Video FPS: {fps}")
    
    # Extract all frames as PNG for quality preservation
    run_ffmpeg(
        [
            "-y",
            "-i", video_path,
            "-qscale:v", "1",      # Best quality
            "-qmin", "1",
            "-qmax", "1",
            "-vsync", "0",
            os.path.join(frames_dir, "frame_%06d.png"),
        ],
        description="Extracting frames from video",
    )
    
    # Count extracted frames
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    num_frames = len(frame_files)
    
    if num_frames == 0:
        raise RuntimeError("No frames extracted from video. The video may be corrupt.")
    
    logger.info(f"Extracted {num_frames} frames")
    
    # ── Step 2: Run CodeFormer on each frame ─────────────────────────────
    inference_script = os.path.join(codeformer_dir, "inference_codeformer.py")
    
    if not os.path.exists(inference_script):
        raise FileNotFoundError(
            f"CodeFormer inference script not found: {inference_script}"
        )
    
    cmd = [
        sys.executable,
        inference_script,
        "--input_path", frames_dir,
        "--output_path", restored_frames_dir,
        "--w", str(fidelity_weight),
        "--has_aligned",          # Faces are already cropped by Wav2Lip
        "--bg_upsampler", "none", # Skip background upsampling (not needed)
    ]
    
    if upscale > 1:
        cmd.extend(["--upscale", str(upscale)])
    
    logger.info(f"Running CodeFormer on {num_frames} frames...")
    logger.debug(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=codeformer_dir,
            capture_output=True,
            text=True,
            timeout=900,  # 15 min timeout
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "CodeFormer timed out after 15 minutes. "
            "Reduce video resolution or frame count."
        )
    
    if result.stdout:
        logger.debug(f"CodeFormer stdout:\n{result.stdout[-1000:]}")
    if result.stderr:
        logger.debug(f"CodeFormer stderr:\n{result.stderr[-1000:]}")
    
    if result.returncode != 0:
        raise RuntimeError(
            f"CodeFormer failed (exit code {result.returncode}):\n"
            f"{result.stderr[-500:]}"
        )
    
    # ── Step 3: Find restored frames ─────────────────────────────────────
    # CodeFormer saves to a subdirectory structure — find the actual output
    restored_dir = _find_restored_frames(restored_frames_dir)
    restored_files = sorted(glob.glob(os.path.join(restored_dir, "*.png")))
    
    if len(restored_files) == 0:
        raise RuntimeError(
            f"CodeFormer produced no output frames. "
            f"Check: {restored_frames_dir}"
        )
    
    logger.info(f"CodeFormer restored {len(restored_files)} frames")
    
    # ── Step 4: Reassemble video from restored frames ────────────────────
    restored_video = os.path.join(output_dir, "restored_video.mp4")
    
    # Determine the frame filename pattern in the restored directory
    first_file = os.path.basename(restored_files[0])
    
    run_ffmpeg(
        [
            "-y",
            "-framerate", str(fps),
            "-i", os.path.join(restored_dir, f"%06d.png")
            if first_file[0].isdigit()
            else os.path.join(restored_dir, f"frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",  # Compatibility
            "-crf", "18",           # High quality
            "-preset", "fast",
            restored_video,
        ],
        description="Reassembling video from restored frames",
    )
    
    logger.info(f"Restored video assembled: {restored_video}")
    
    # ── Cleanup temporary frame directories ──────────────────────────────
    try:
        shutil.rmtree(frames_dir)
        shutil.rmtree(restored_frames_dir)
        logger.info("Temporary frame directories cleaned up.")
    except Exception as e:
        logger.warning(f"Could not clean up temp frames: {e}")
    
    # ── Cleanup GPU ──────────────────────────────────────────────────────
    free_gpu_memory()
    
    return {
        "restored_video": restored_video,
        "fidelity_weight": fidelity_weight,
        "frames_processed": len(restored_files),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_video_fps(video_path: str) -> float:
    """Get the FPS of a video using ffprobe."""
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        raise RuntimeError("ffprobe not found (usually comes with ffmpeg)")
    
    cmd = [
        ffprobe_path,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    
    if result.returncode != 0:
        logger.warning(f"ffprobe failed, defaulting to 25 FPS: {result.stderr}")
        return 25.0
    
    # ffprobe returns FPS as a fraction like "30000/1001" or "25/1"
    fps_str = result.stdout.strip()
    try:
        if "/" in fps_str:
            num, den = fps_str.split("/")
            return float(num) / float(den)
        return float(fps_str)
    except (ValueError, ZeroDivisionError):
        logger.warning(f"Could not parse FPS '{fps_str}', defaulting to 25 FPS")
        return 25.0


def _find_restored_frames(base_dir: str) -> str:
    """
    Find the actual directory where CodeFormer saved restored frames.
    
    CodeFormer has a nested output structure:
      output_path/final_results/  or  output_path/restored_faces/
    """
    # Check common CodeFormer output subdirectories
    candidates = [
        os.path.join(base_dir, "final_results"),
        os.path.join(base_dir, "restored_faces"),
        os.path.join(base_dir, "restored_imgs"),
        base_dir,  # Fallback: frames directly in the output dir
    ]
    
    for candidate in candidates:
        if os.path.isdir(candidate):
            pngs = glob.glob(os.path.join(candidate, "*.png"))
            if pngs:
                logger.info(f"Found restored frames in: {candidate}")
                return candidate
    
    # Last resort: recursively find any directory with PNGs
    for root, dirs, files in os.walk(base_dir):
        pngs = [f for f in files if f.endswith(".png")]
        if pngs:
            logger.info(f"Found restored frames in: {root}")
            return root
    
    raise RuntimeError(
        f"Could not find restored frames in {base_dir}. "
        "CodeFormer may not have processed correctly."
    )
