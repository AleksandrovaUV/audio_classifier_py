from pathlib import Path
from pydub import AudioSegment
from ..utils.logger import get_logger

logger = get_logger(__name__)

def simple_denoise(wav_path: Path) -> Path:
    logger.info(f"Applying simple denoise to {wav_path}")
    audio = AudioSegment.from_wav(wav_path)
    target_dBFS = -20.0
    change = target_dBFS - audio.dBFS
    audio = audio.apply_gain(change)
    out_path = wav_path.with_name(wav_path.stem + "_denoised.wav")
    audio.export(out_path, format="wav")
    logger.info(f"Denoised WAV saved to {out_path}")
    return out_path
