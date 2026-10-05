'''Voice-bank enrollment with an honest,
audio-free fallback seam.'''
from __future__ import annotations
import base64, binascii, json, uuid
from pathlib import Path
from typing import Any
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
router = APIRouter(tags=['voicebank'])
VOICEBANK_DIR = Path(__file__).resolve().parents[1] / 'data' / 'voicebank'
def _error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse({'ok': False, 'error': message}, status_code=status_code)
async def _json(request: Request) -> dict[str, Any] | None:
    try: payload = await request.json()
    except Exception: return None
    return payload if isinstance(payload, dict) else None
@router.post('/voicebank/enroll', response_model=None)
async def enroll(request: Request) -> dict[str, str] | JSONResponse:
    payload = await _json(request)
    name = payload.get('name', '').strip() if payload else ''
    sample_b64 = payload.get('sample_b64', '').strip() if payload else ''
    if not name or not sample_b64: return _error('name and sample_b64 are required')
    try: sample = base64.b64decode(sample_b64, validate=True)
    except (binascii.Error, ValueError): return _error('sample_b64 must be valid base64')
    if not sample: return _error('sample_b64 must not be empty')
    enrollment_id = uuid.uuid4().hex
    VOICEBANK_DIR.mkdir(parents=True, exist_ok=True)
    (VOICEBANK_DIR / f'{enrollment_id}.json').write_text(json.dumps({'enrollment_id': enrollment_id, 'name': name, 'sample_bytes': len(sample)}), encoding='utf-8')
    return {'enrollment_id': enrollment_id}
@router.post('/voicebank/speak', response_model=None)
async def speak(request: Request) -> dict[str, str] | JSONResponse:
    payload = await _json(request)
    enrollment_id = payload.get('enrollment_id', '').strip() if payload else ''
    text = payload.get('text', '').strip() if payload else ''
    if not enrollment_id or not text: return _error('enrollment_id and text are required')
    if not (VOICEBANK_DIR / f'{enrollment_id}.json').exists(): return _error('enrollment_id was not found', 404)
    # A real voice-clone model (Coqui XTTS) plugs in at this seam; NEVER fake audio.
    return {'text': text, 'enrollment_id': enrollment_id, 'source': 'fallback'}
@router.get('/voicebank/{enrollment_id}', response_model=None)
async def status(enrollment_id: str) -> dict[str, Any] | JSONResponse:
    record = VOICEBANK_DIR / f'{enrollment_id}.json'
    if not record.exists(): return _error('enrollment_id was not found', 404)
    try: metadata = json.loads(record.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError): return _error('voice-bank record is unavailable', 500)
    return {'enrollment_id': enrollment_id, 'status': 'enrolled', 'name': metadata.get('name', '')}
