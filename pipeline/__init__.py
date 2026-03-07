# pipeline/__init__.py
"""
Supernan Video Dubbing Pipeline
================================
Modular ML pipeline for Kannada → Hindi video dubbing.

Pipeline steps:
  1. extract_clip   - Extract 15s clip from source video (ffmpeg)
  2. transcribe     - Transcribe Kannada audio (faster-whisper)
  3. translate      - Translate Kannada → Hindi (IndicTrans2 / googletrans)
  4. voice_clone    - Clone speaker voice + generate Hindi speech (Coqui XTTSv2)
  5. lip_sync       - Sync lip movements to Hindi audio (Wav2Lip)
  6. face_restore   - Restore face quality after lip sync (CodeFormer)
"""

__version__ = "0.1.0"
