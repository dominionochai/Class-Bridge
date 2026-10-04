from __future__ import annotations
import logging, os, tempfile
from pathlib import Path
from typing import Any
from fastapi import APIRouter, Request
from .utils import mark_loaded, mark_model_error, model_error_response, model_response, model_status, package_version, request_bytes
logger = logging.getLogger("classbridge.models")
router = APIRouter(tags=["captions"])
_WHISPER: Any = None
_STATUS = model_status("faster_whisper")
def _text(exc: BaseException) -> str: return str(exc) or exc.__class__.__name__
def _load_whisper() -> Any:
    global _WHISPER
    if _WHISPER is not None: return _WHISPER
    if _STATUS["status"] == "error": return None
    try:
        from faster_whisper import WhisperModel
        import faster_whisper
        configured = os.getenv("WHISPER_MODEL")
        path = Path(configured) if configured else Path(__file__).resolve().parents[1] / "models" / "faster-whisper-tiny"
        if not path.exists(): raise FileNotFoundError(f"faster-whisper model asset not found: {path}")
        _WHISPER = WhisperModel(str(path), device=os.getenv("WHISPER_DEVICE", "cpu"), compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"))
        mark_loaded(_STATUS, package_version(faster_whisper))
        return _WHISPER
    except Exception as exc:
        mark_model_error(_STATUS, _text(exc)); logger.warning("faster-whisper unavailable: %s", exc); return None
def model_status_info() -> dict[str, Any]: return dict(_STATUS)
def _error(error: str | BaseException) -> dict[str, Any]: return model_error_response(error, "captions")
@router.get("/captions")
async def get_captions() -> dict[str, Any]:
    if _load_whisper() is None: return _error(_STATUS.get("error", "faster-whisper model is unavailable"))
    return _error("audio upload is required")
@router.post("/captions")
async def captions(request: Request) -> dict[str, Any]:
    try: audio = await request_bytes(request, ("audio", "file"))
    except Exception as exc: return _error(_text(exc))
    if not audio: return _error("audio upload is required")
    model = _load_whisper()
    if model is None: return _error(_STATUS.get("error", "faster-whisper model is unavailable"))
    path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(audio); path = handle.name
        segments, info = model.transcribe(path)
        items, parts = [], []
        for segment in segments:
            text = str(getattr(segment, "text", "")).strip(); parts.append(text)
            items.append({"start": float(getattr(segment, "start", 0.0)), "end": float(getattr(segment, "end", 0.0)), "text": text})
        transcript = " ".join(p for p in parts if p).strip()
        from llm_bridge import smart_captions_with_gemini
        smart = smart_captions_with_gemini(transcript)
        if smart:
            return model_response("model", {"transcript": smart, "language": getattr(info, "language", None), "segments": items}, "POST /api/captions")
        return model_response("model", {"transcript": transcript, "language": getattr(info, "language", None), "segments": items}, "POST /api/captions")
    except Exception as exc: return _error(f"faster-whisper inference failed: {_text(exc)}")
    finally:
        if path:
            try: os.unlink(path)
            except OSError: pass
