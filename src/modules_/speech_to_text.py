from pathlib import Path
import whisper
from ..utils.logger import get_logger

logger = get_logger(__name__)

def load_asr_model(model_name: str = "base"):
    logger.info(f"Loading ASR model: {model_name}")
    return whisper.load_model(model_name)

def transcribe_with_timestamps(model, wav_path: Path):
    logger.info(f"Transcribing {wav_path}")
    result = model.transcribe(str(wav_path), verbose=False)
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": seg["text"].strip()
        })
    logger.info(f"Transcription produced {len(segments)} segments")
    return segments
