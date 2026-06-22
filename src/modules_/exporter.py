import json
from pathlib import Path
from pydub import AudioSegment
from ..utils.logger import get_logger

logger = get_logger(__name__)

def export_audio(final_wav_path: Path, output_audio_path: Path, format: str):
    logger.info(f"Exporting final audio to {output_audio_path} ({format})")
    audio = AudioSegment.from_wav(final_wav_path)
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(output_audio_path, format=format)

def export_metadata(meta_path: Path, segments, topics):
    logger.info(f"Exporting metadata to {meta_path}")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "segments": segments,
        "topics": topics,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
