# tests/test_pipeline.py
"""
Basic tests for the Supernan Video Dubbing Pipeline.

These tests validate:
  - Module imports and function signatures
  - Utility functions (no GPU required)
  - Input validation / error handling
  - Edge cases

Run with: python -m pytest tests/ -v
"""

import os
import sys
import json
import tempfile
import shutil
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =========================================================================
# Test: Pipeline Package Imports
# =========================================================================

class TestImports:
    """Verify all pipeline modules can be imported."""
    
    def test_import_utils(self):
        from pipeline import utils
        assert hasattr(utils, "setup_logger")
        assert hasattr(utils, "free_gpu_memory")
        assert hasattr(utils, "get_device")
        assert hasattr(utils, "run_ffmpeg")
    
    def test_import_extract_clip(self):
        from pipeline.extract_clip import extract_clip
        assert callable(extract_clip)
    
    def test_import_transcribe(self):
        from pipeline.transcribe import transcribe_audio
        assert callable(transcribe_audio)
    
    def test_import_translate(self):
        from pipeline.translate import translate_text
        assert callable(translate_text)
    
    def test_import_voice_clone(self):
        from pipeline.voice_clone import clone_voice_and_speak
        assert callable(clone_voice_and_speak)
    
    def test_import_lip_sync(self):
        from pipeline.lip_sync import lip_sync_video
        assert callable(lip_sync_video)
    
    def test_import_face_restore(self):
        from pipeline.face_restore import restore_faces
        assert callable(restore_faces)


# =========================================================================
# Test: Utility Functions
# =========================================================================

class TestUtils:
    """Test utility functions that don't require GPU or heavy dependencies."""
    
    def test_setup_logger(self):
        """Logger should return a configured logging.Logger instance."""
        from pipeline.utils import setup_logger
        log = setup_logger("test_module")
        assert log.name == "test_module"
        assert len(log.handlers) >= 1
    
    def test_setup_logger_no_duplicates(self):
        """Calling setup_logger twice shouldn't add duplicate handlers."""
        from pipeline.utils import setup_logger
        log1 = setup_logger("test_dedup")
        handler_count = len(log1.handlers)
        log2 = setup_logger("test_dedup")
        assert len(log2.handlers) == handler_count
    
    def test_ensure_dir_creates_directory(self):
        """ensure_dir should create a directory and parents."""
        from pipeline.utils import ensure_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "a", "b", "c")
            result = ensure_dir(new_dir)
            assert os.path.isdir(new_dir)
            assert str(result) == new_dir
    
    def test_ensure_dir_existing(self):
        """ensure_dir should not fail on existing directories."""
        from pipeline.utils import ensure_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ensure_dir(tmpdir)
            assert os.path.isdir(tmpdir)
    
    def test_validate_video_file_not_found(self):
        """Should raise FileNotFoundError for missing files."""
        from pipeline.utils import validate_video_file
        with pytest.raises(FileNotFoundError):
            validate_video_file("/nonexistent/video.mp4")
    
    def test_validate_video_file_bad_extension(self):
        """Should raise ValueError for unsupported extensions."""
        from pipeline.utils import validate_video_file
        path = os.path.join(tempfile.gettempdir(), "test_bad_ext.txt")
        with open(path, "w") as f:
            f.write("dummy")
        try:
            with pytest.raises(ValueError, match="Unsupported video format"):
                validate_video_file(path)
        finally:
            os.unlink(path)
    
    def test_validate_video_file_good_extension(self):
        """Should accept recognized video extensions."""
        from pipeline.utils import validate_video_file
        path = os.path.join(tempfile.gettempdir(), "test_good_ext.mp4")
        with open(path, "w") as f:
            f.write("dummy")
        try:
            result = validate_video_file(path)
            assert result.endswith(".mp4")
        finally:
            os.unlink(path)
    
    def test_free_gpu_memory_no_crash(self):
        """free_gpu_memory should not crash even without GPU."""
        from pipeline.utils import free_gpu_memory
        free_gpu_memory()  # Should be a silent no-op if no CUDA
    
    def test_check_ffmpeg(self):
        """ffmpeg should be found on PATH (or test gracefully skips)."""
        from pipeline.utils import check_ffmpeg
        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not installed")
        path = check_ffmpeg()
        assert "ffmpeg" in path.lower()


# =========================================================================
# Test: Extract Clip — Input Validation
# =========================================================================

class TestExtractClipValidation:
    """Test extract_clip input validation (no actual video processing)."""
    
    def test_missing_input_video(self):
        """Should raise FileNotFoundError for missing video."""
        from pipeline.extract_clip import extract_clip
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                extract_clip(
                    input_video="/nonexistent/video.mp4",
                    output_dir=tmpdir,
                )
    
    def test_invalid_time_range(self):
        """Should raise ValueError if end_time <= start_time."""
        from pipeline.extract_clip import extract_clip
        path = os.path.join(tempfile.gettempdir(), "test_time_range.mp4")
        with open(path, "w") as f:
            f.write("dummy")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with pytest.raises(ValueError, match="must be greater"):
                    extract_clip(
                        input_video=path,
                        output_dir=tmpdir,
                        start_time=30.0,
                        end_time=15.0,
                    )
        finally:
            os.unlink(path)
    
    def test_negative_start_time(self):
        """Should raise ValueError for negative start_time."""
        from pipeline.extract_clip import extract_clip
        path = os.path.join(tempfile.gettempdir(), "test_neg_start.mp4")
        with open(path, "w") as f:
            f.write("dummy")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with pytest.raises(ValueError, match="must be >= 0"):
                    extract_clip(
                        input_video=path,
                        output_dir=tmpdir,
                        start_time=-5.0,
                        end_time=10.0,
                    )
        finally:
            os.unlink(path)


# =========================================================================
# Test: Translate — Sentence Splitting
# =========================================================================

class TestTranslateHelpers:
    """Test translation helper functions."""
    
    def test_split_sentences_basic(self):
        """Should split on periods."""
        from pipeline.translate import _split_sentences
        result = _split_sentences("Hello world. This is a test.")
        assert len(result) == 2
    
    def test_split_sentences_danda(self):
        """Should split on Devanagari danda (।)."""
        from pipeline.translate import _split_sentences
        result = _split_sentences("यह पहला वाक्य है। यह दूसरा वाक्य है।")
        assert len(result) == 2
    
    def test_split_sentences_empty(self):
        """Should handle empty input."""
        from pipeline.translate import _split_sentences
        result = _split_sentences("")
        assert len(result) == 0 or result == [""]
    
    def test_split_sentences_no_delimiter(self):
        """Should return whole text if no sentence boundaries."""
        from pipeline.translate import _split_sentences
        result = _split_sentences("This is one continuous text without periods")
        assert len(result) == 1


# =========================================================================
# Test: Transcribe (Sarvam API) — Input Validation
# =========================================================================

class TestTranscribeValidation:
    """Test transcribe input validation (no API calls made)."""
    
    def test_missing_audio_file(self):
        """Should raise FileNotFoundError for missing audio."""
        from pipeline.transcribe import transcribe_audio
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                transcribe_audio(
                    audio_path="/nonexistent/audio.wav",
                    output_dir=tmpdir,
                    api_key="dummy_key",
                )
    
    def test_missing_api_key(self):
        """Should raise ValueError if no API key is provided."""
        from pipeline.transcribe import transcribe_audio
        # Temporarily unset env var if it exists
        old_key = os.environ.pop("SARVAM_API_KEY", None)
        try:
            path = os.path.join(tempfile.gettempdir(), "test_api_key.wav")
            with open(path, "w") as f:
                f.write("dummy")
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    with pytest.raises(ValueError, match="API key"):
                        transcribe_audio(
                            audio_path=path,
                            output_dir=tmpdir,
                            api_key=None,
                        )
            finally:
                os.unlink(path)
        finally:
            if old_key:
                os.environ["SARVAM_API_KEY"] = old_key


# =========================================================================
# Test: Translate (Sarvam API) — Input Validation
# =========================================================================

class TestTranslateValidation:
    """Test translate input validation (no API calls made)."""
    
    def test_empty_text(self):
        """Should raise ValueError for empty text."""
        from pipeline.translate import translate_text
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="empty"):
                translate_text(
                    text="",
                    output_dir=tmpdir,
                    api_key="dummy_key",
                )
    
    def test_missing_api_key(self):
        """Should raise ValueError if no API key is provided."""
        from pipeline.translate import translate_text
        old_key = os.environ.pop("SARVAM_API_KEY", None)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with pytest.raises(ValueError, match="API key"):
                    translate_text(
                        text="ಕನ್ನಡ ಪಠ್ಯ",
                        output_dir=tmpdir,
                        api_key=None,
                    )
        finally:
            if old_key:
                os.environ["SARVAM_API_KEY"] = old_key


# =========================================================================
# Test: Voice Clone — Input Validation
# =========================================================================

class TestVoiceCloneValidation:
    """Test voice_clone input validation."""
    
    def test_empty_hindi_text(self):
        """Should raise ValueError for empty text."""
        from pipeline.voice_clone import clone_voice_and_speak
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="empty"):
                clone_voice_and_speak(
                    hindi_text="",
                    reference_audio="/some/audio.wav",
                    output_dir=tmpdir,
                )
    
    def test_missing_reference_audio(self):
        """Should raise FileNotFoundError for missing reference."""
        from pipeline.voice_clone import clone_voice_and_speak
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                clone_voice_and_speak(
                    hindi_text="कुछ हिंदी टेक्स्ट",
                    reference_audio="/nonexistent/audio.wav",
                    output_dir=tmpdir,
                )


# =========================================================================
# Test: Lip Sync — Input Validation
# =========================================================================

class TestLipSyncValidation:
    """Test lip_sync input validation."""
    
    def test_missing_video(self):
        """Should raise FileNotFoundError for missing video."""
        from pipeline.lip_sync import lip_sync_video
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                lip_sync_video(
                    video_path="/nonexistent/video.mp4",
                    audio_path="/some/audio.wav",
                    output_dir=tmpdir,
                )
    
    def test_missing_audio(self):
        """Should raise FileNotFoundError for missing audio."""
        from pipeline.lip_sync import lip_sync_video
        path = os.path.join(tempfile.gettempdir(), "test_lip_audio.mp4")
        with open(path, "w") as f:
            f.write("dummy")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with pytest.raises(FileNotFoundError):
                    lip_sync_video(
                        video_path=path,
                        audio_path="/nonexistent/audio.wav",
                        output_dir=tmpdir,
                    )
        finally:
            os.unlink(path)


# =========================================================================
# Test: Face Restore — Input Validation
# =========================================================================

class TestFaceRestoreValidation:
    """Test face_restore input validation."""
    
    def test_missing_video(self):
        """Should raise FileNotFoundError for missing video."""
        from pipeline.face_restore import restore_faces
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                restore_faces(
                    video_path="/nonexistent/video.mp4",
                    output_dir=tmpdir,
                )
