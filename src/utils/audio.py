from pydub import AudioSegment
import os
from pathlib import Path

def configure_ffmpeg():
    project_root = Path(__file__).resolve().parents[2]
    ffmpeg_bin = project_root / "ffmpeg" / "bin"

    if ffmpeg_bin.exists():
        os.environ["PATH"] += os.pathsep + str(ffmpeg_bin)


def load_audio(path: str) -> AudioSegment:
    return AudioSegment.from_file(path)

def save_audio(audio: AudioSegment, path: str, format: str):
    audio.export(path, format=format)

def to_mono_16k(audio: AudioSegment) -> AudioSegment:
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    return audio
