from pydub import AudioSegment

def load_audio(path: str) -> AudioSegment:
    return AudioSegment.from_file(path)

def save_audio(audio: AudioSegment, path: str, format: str):
    audio.export(path, format=format)

def to_mono_16k(audio: AudioSegment) -> AudioSegment:
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    return audio
