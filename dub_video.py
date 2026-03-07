#!/usr/bin/env python3
# dub_video.py
"""
Supernan Video Dubbing Pipeline — Main Orchestrator
===================================================

Processes a Kannada video to produce a Hindi dubbed clip with:
  1. Clip extraction (ffmpeg)
  2. Audio transcription (Sarvam AI Saaras V3 API)
  3. Translation (Sarvam AI Translate API, Kannada → Hindi)
  4. Voice cloning (Coqui XTTSv2)
  5. Lip sync (Wav2Lip)
  6. Face restoration (CodeFormer)
  7. Final audio+video mux

Usage:
    python dub_video.py --input input/source.mp4 --start 15 --end 30 --target-lang hi

For Colab, use the defaults:
    python dub_video.py --input input/source.mp4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from pipeline.utils import (
    setup_logger,
    ensure_dir,
    validate_video_file,
    run_ffmpeg,
    get_media_duration,
    free_gpu_memory,
    check_ffmpeg,
)

logger = setup_logger("dub_video")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Supernan Video Dubbing Pipeline — Kannada → Hindi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the source video file (Kannada)",
    )
    parser.add_argument(
        "--start", "-s",
        type=float,
        default=15.0,
        help="Start time of the clip in seconds (default: 15.0)",
    )
    parser.add_argument(
        "--end", "-e",
        type=float,
        default=30.0,
        help="End time of the clip in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--target-lang",
        default="hi",
        help="Target language code (default: 'hi' for Hindi)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Output directory (default: 'output')",
    )
    parser.add_argument(
        "--sarvam-api-key",
        default=None,
        help="Sarvam AI API key (default: reads SARVAM_API_KEY env var)",
    )
    parser.add_argument(
        "--skip-transcribe",
        action="store_true",
        help="Skip transcription (use existing transcript.json in output dir)",
    )
    parser.add_argument(
        "--skip-translate",
        action="store_true",
        help="Skip translation (use existing hindi_text.txt in output dir)",
    )
    parser.add_argument(
        "--fidelity-weight",
        type=float,
        default=0.7,
        help="CodeFormer fidelity weight: 0.0=quality, 1.0=faithful (default: 0.7)",
    )
    
    return parser.parse_args()


def main():
    """Run the full dubbing pipeline."""
    args = parse_args()
    
    # ── Setup ────────────────────────────────────────────────────────────
    start_wall = time.time()
    output_dir = os.path.abspath(args.output_dir)
    temp_dir = os.path.join(output_dir, "temp")
    ensure_dir(output_dir)
    ensure_dir(temp_dir)
    
    logger.info("=" * 60)
    logger.info("  Supernan Video Dubbing Pipeline")
    logger.info("  Kannada → Hindi | Sarvam AI + Open Source")
    logger.info("=" * 60)
    logger.info(f"Input:  {args.input}")
    logger.info(f"Clip:   {args.start}s → {args.end}s")
    logger.info(f"Output: {output_dir}")
    
    # Preflight checks
    check_ffmpeg()
    logger.info("✓ ffmpeg found")
    
    # ────────────────────────────────────────────────────────────────────
    # STEP 1: Extract clip
    # ────────────────────────────────────────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("STEP 1/7: Extracting clip from source video")
    logger.info("─" * 50)
    
    from pipeline.extract_clip import extract_clip
    
    clip_result = extract_clip(
        input_video=args.input,
        output_dir=temp_dir,
        start_time=args.start,
        end_time=args.end,
    )
    
    clip_video = clip_result["clip_video"]
    clip_audio = clip_result["clip_audio"]
    clip_duration = clip_result["duration"]
    
    logger.info(f"✓ Clip extracted: {clip_duration:.1f}s")
    
    # ────────────────────────────────────────────────────────────────────
    # STEP 2-3: Transcribe audio
    # ────────────────────────────────────────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("STEP 2-3/7: Transcribing Kannada audio")
    logger.info("─" * 50)
    
    if args.skip_transcribe:
        # Load existing transcript
        transcript_file = os.path.join(temp_dir, "transcript.json")
        if not os.path.exists(transcript_file):
            logger.error(f"--skip-transcribe specified but {transcript_file} not found")
            sys.exit(1)
        with open(transcript_file, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)
        kannada_text = transcript_data["text"]
        logger.info(f"✓ Loaded existing transcript: {kannada_text[:100]}...")
    else:
        from pipeline.transcribe import transcribe_audio
        
        transcribe_result = transcribe_audio(
            audio_path=clip_audio,
            output_dir=temp_dir,
            language="kn-IN",
            api_key=args.sarvam_api_key,
        )
        kannada_text = transcribe_result["text"]
        logger.info(f"✓ Transcription complete (Sarvam Saaras V3): {kannada_text[:100]}...")
        logger.info(
            "  ⚠  Review temp/transcript.txt and correct if needed.\n"
            "  Re-run with --skip-transcribe to use corrected text."
        )
    
    # ────────────────────────────────────────────────────────────────────
    # STEP 4: Translate Kannada → Hindi
    # ────────────────────────────────────────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("STEP 4/7: Translating Kannada → Hindi")
    logger.info("─" * 50)
    
    if args.skip_translate:
        # Load existing Hindi text
        hindi_text_file = os.path.join(temp_dir, "hindi_text.txt")
        if not os.path.exists(hindi_text_file):
            logger.error(f"--skip-translate specified but {hindi_text_file} not found")
            sys.exit(1)
        with open(hindi_text_file, "r", encoding="utf-8") as f:
            hindi_text = f.read().strip()
        logger.info(f"✓ Loaded existing Hindi text: {hindi_text[:100]}...")
    else:
        from pipeline.translate import translate_text
        
        translate_result = translate_text(
            text=kannada_text,
            output_dir=temp_dir,
            api_key=args.sarvam_api_key,
        )
        hindi_text = translate_result["translated_text"]
        logger.info(f"✓ Translation complete (Sarvam Translate)")
    
    # ────────────────────────────────────────────────────────────────────
    # STEP 5: Voice cloning + Hindi speech generation
    # ────────────────────────────────────────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("STEP 5/7: Generating Hindi speech (voice cloning)")
    logger.info("─" * 50)
    
    from pipeline.voice_clone import clone_voice_and_speak
    
    voice_result = clone_voice_and_speak(
        hindi_text=hindi_text,
        reference_audio=clip_audio,
        output_dir=temp_dir,
        target_duration=clip_duration,
    )
    
    hindi_audio = voice_result["hindi_audio"]
    logger.info(
        f"✓ Hindi audio generated: {voice_result['duration']:.2f}s "
        f"(stretch: {'yes' if voice_result['stretch_applied'] else 'no'})"
    )
    
    # ────────────────────────────────────────────────────────────────────
    # STEP 6a: Lip-sync with Wav2Lip
    # ────────────────────────────────────────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("STEP 6a/7: Lip-syncing video to Hindi audio (Wav2Lip)")
    logger.info("─" * 50)
    
    from pipeline.lip_sync import lip_sync_video
    
    lip_result = lip_sync_video(
        video_path=clip_video,
        audio_path=hindi_audio,
        output_dir=temp_dir,
    )
    
    synced_video = lip_result["synced_video"]
    logger.info(f"✓ Lip-sync complete (blurry — will fix in 6b)")
    
    # ────────────────────────────────────────────────────────────────────
    # STEP 6b: Face restoration with CodeFormer
    # ────────────────────────────────────────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("STEP 6b/7: Restoring face quality (CodeFormer)")
    logger.info("─" * 50)
    
    from pipeline.face_restore import restore_faces
    
    face_result = restore_faces(
        video_path=synced_video,
        output_dir=temp_dir,
        fidelity_weight=args.fidelity_weight,
    )
    
    restored_video = face_result["restored_video"]
    logger.info(
        f"✓ Face restoration complete "
        f"({face_result['frames_processed']} frames, "
        f"fidelity={face_result['fidelity_weight']})"
    )
    
    # ────────────────────────────────────────────────────────────────────
    # STEP 7: Final mux — merge restored video + Hindi audio
    # ────────────────────────────────────────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("STEP 7/7: Final output — merging video + audio")
    logger.info("─" * 50)
    
    final_output = os.path.join(output_dir, "final_output.mp4")
    
    run_ffmpeg(
        [
            "-y",
            "-i", restored_video,        # Face-restored video (no audio)
            "-i", hindi_audio,            # Hindi audio track
            "-c:v", "copy",              # Don't re-encode video
            "-c:a", "aac",               # Encode audio as AAC
            "-b:a", "192k",
            "-shortest",                  # Match shortest stream length
            "-map", "0:v:0",             # Take video from first input
            "-map", "1:a:0",             # Take audio from second input
            final_output,
        ],
        description="Merging restored video with Hindi audio",
    )
    
    # ── Summary ──────────────────────────────────────────────────────────
    final_duration = get_media_duration(final_output)
    elapsed = time.time() - start_wall
    
    logger.info("\n" + "=" * 60)
    logger.info("  ✅  DUBBING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"  Output:   {final_output}")
    logger.info(f"  Duration: {final_duration:.1f}s")
    logger.info(f"  Time:     {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info(f"  Size:     {os.path.getsize(final_output) / 1024 / 1024:.1f} MB")
    logger.info("=" * 60)
    
    return final_output


if __name__ == "__main__":
    try:
        output = main()
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌  Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
