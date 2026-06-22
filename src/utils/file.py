from pathlib import Path


def ensure_dir(path):
    p = Path(path)
    if p.suffix: 
        return p.parent
    p.mkdir(parents=True, exist_ok=True)
    return p


def validate_input_file(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Input file not found: {p}")
    return p

def build_output_paths(input_path: Path, output_dir: Path, output_name: str | None, ext: str):
    if output_name:
        base = output_name
    else:
        base = input_path.stem
    audio_out = output_dir / f"{base}.{ext}"
    meta_out = output_dir / f"{base}_meta.json"
    return audio_out, meta_out
