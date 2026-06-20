from pathlib import Path
from pydub import AudioSegment
from ..utils.audio import to_mono_16k
from ..utils.logger import get_logger

logger = get_logger(__name__)

def convert_to_wav_16k(input_path: Path, temp_dir: Path) -> Path:
    logger.info(f"Converting {input_path} to mono 16k WAV")
    audio = AudioSegment.from_file(input_path)
    audio = to_mono_16k(audio)
    temp_dir.mkdir(parents=True, exist_ok=True)
    out_path = temp_dir / (input_path.stem + "_16k.wav")
    audio.export(out_path, format="wav")
    logger.info(f"Temporary WAV saved to {out_path}")
    return out_path
