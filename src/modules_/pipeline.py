from pathlib import Path
import yaml

from src.utils.logger import get_logger
from src.utils.file import ensure_dir, build_output_paths
from src.modules_.converter import convert_to_wav_16k
from src.modules_.cleanser import simple_denoise
from src.modules_.speech_to_text import load_asr_model, transcribe_with_timestamps
from src.modules_.topic_analyzer import TopicAnalyzer
from src.modules_.exporter import export_audio, export_metadata
from src.utils.audio import configure_ffmpeg

configure_ffmpeg()


logger = get_logger(__name__)

class AudioPipeline:
    def __init__(self, config_path: Path):
        configure_ffmpeg()
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.temp_dir = Path(self.config["audio"]["temp_dir"])
        self.asr_model = load_asr_model(self.config["asr"]["model_name"])
        self.topic_analyzer = TopicAnalyzer()

    def run(self, input_path: Path, output_dir: Path, output_name: str | None = None):

        from pydub.utils import which
        print("FFmpeg path:", which("ffmpeg"))

        output_dir = ensure_dir(output_dir)

        # 1. Конвертация в WAV 16k
        wav_path = convert_to_wav_16k(input_path, self.temp_dir)

        # 2. Шумоподавление
        denoised_wav = simple_denoise(wav_path)

        # 3. ASR + таймкоды
        segments = transcribe_with_timestamps(self.asr_model, denoised_wav)

        # 4. Тематический анализ
        topics = self.topic_analyzer.analyze(segments)

        # 5. Экспорт
        audio_format = self.config["output"]["default_format"]
        audio_out, meta_out = build_output_paths(
            input_path=input_path,
            output_dir = ensure_dir(output_dir),
            output_name=output_name,
            ext=audio_format,
        )
        export_audio(denoised_wav, audio_out, audio_format)
        export_metadata(meta_out, segments, topics)

        logger.info(f"Done. Audio: {audio_out}, metadata: {meta_out}")
        return audio_out, meta_out