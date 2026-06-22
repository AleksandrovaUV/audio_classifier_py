from pathlib import Path
from rich.prompt import Prompt
from rich.console import Console
from src.modules_.pipeline import AudioPipeline

console = Console()

def run_cli():
    console.print("[bold cyan]Audio Processing Pipeline[/bold cyan]")

    input_path_str = Prompt.ask("Введите путь к входному аудио файлу")
    input_path = Path(input_path_str).expanduser().resolve()

    output_dir_str = Prompt.ask("Каталог для выходных файлов", default="examples/output")
    output_dir = Path(output_dir_str).expanduser().resolve()

    output_name = Prompt.ask("Имя выходного файла (без расширения, Enter — взять из исходного)", default="")

    config_path = Path("config.yaml")
    pipeline = AudioPipeline(config_path)

    if not output_name.strip():
        output_name = None

    audio_out, meta_out = pipeline.run(
        input_path=input_path,
        output_dir=output_dir,
        output_name=output_name,
    )

    console.print(f"[green]Готово![/green]")
    console.print(f"Аудио: {audio_out}")
    console.print(f"Метаданные: {meta_out}")
