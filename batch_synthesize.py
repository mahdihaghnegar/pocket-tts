"""Batch-synthesize many input texts with many voice prompts.

Interactive console: lists .txt files from the input folder and .wav files
from the voice folder, lets you pick which ones (individually or all), then
runs ``training.farsi.synthesize`` for every input x voice combination.

Output naming: ``<output-dir>/<inputfile>_<sample>.wav``, e.g. input
``test.txt`` with voice ``prompt_hello.wav`` becomes
``output/test_prompt_hello.wav``.

Usage (interactive):
    python batch_synthesize.py

Usage (non-interactive, skip the menus):
    python batch_synthesize.py --input-file input/test.txt --voice voice_samples/prompt_hello.wav

All generation options mirror ``python -m training.farsi.synthesize --help``.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(pretty_exceptions_show_locals=False)


def _list_files(folder: Path, suffix: str) -> list[Path]:
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == suffix))


def _pick(files: list[Path], what: str) -> list[Path]:
    if not files:
        typer.echo(f"no files found for {what}.")
        raise typer.Exit(1)
    typer.echo(f"\n{what}:")
    for i, f in enumerate(files, 1):
        typer.echo(f"  {i}. {f.name}")
    raw = typer.prompt("select (e.g. 1,3 or all)", default="all").strip().lower()
    if raw in ("all", "a", "*"):
        return files
    chosen: list[Path] = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if part.isdigit() and 1 <= int(part) <= len(files):
            if files[int(part) - 1] not in chosen:
                chosen.append(files[int(part) - 1])
        else:
            typer.echo(f"ignoring invalid selection: {part!r}")
    if not chosen:
        typer.echo("nothing selected.")
        raise typer.Exit(1)
    return chosen


@app.command()
def main(
    config: str = typer.Option("./model_farsi_local.yaml", help="model config yaml"),
    input_dir: str = typer.Option("./input", help="folder with .txt input files"),
    voice_dir: str = typer.Option("./voice_samples", help="folder with voice .wav files"),
    output_dir: str = typer.Option("./output", help="folder for result wavs"),
    input_file: Optional[list[str]] = typer.Option(None, help="input .txt (repeatable); skips menu"),
    voice: Optional[list[str]] = typer.Option(None, help="voice .wav (repeatable); skips menu"),
    max_tokens: int = typer.Option(18),
    temperature: float = typer.Option(0.3),
    eos_threshold: float = typer.Option(-2.0),
    frames_after_eos: int = typer.Option(0),
    voice_sec: float = typer.Option(5.0),
    pause_sec: float = typer.Option(0.15),
    join_sec: float = typer.Option(0.15),
    pause_at_punct: bool = typer.Option(False),
    min_tokens: int = typer.Option(8),
    normalize_text: bool = typer.Option(True),
) -> None:
    input_path = Path(input_dir)
    voice_path = Path(voice_dir)
    out_path = Path(output_dir)
    if not input_path.is_dir():
        typer.echo(f"input folder not found: {input_path}")
        raise typer.Exit(1)
    if not voice_path.is_dir():
        typer.echo(f"voice folder not found: {voice_path}")
        raise typer.Exit(1)
    out_path.mkdir(parents=True, exist_ok=True)

    inputs = [Path(f) for f in input_file] if input_file else _pick(_list_files(input_path, ".txt"), "input files")
    voices = [Path(f) for f in voice] if voice else _pick(_list_files(voice_path, ".wav"), "voice samples")

    for f in [*inputs, *voices]:
        if not f.is_file():
            typer.echo(f"file not found: {f}")
            raise typer.Exit(1)

    combos = [(t, v) for t in inputs for v in voices]
    typer.echo(f"\n{len(combos)} combination(s): {len(inputs)} input(s) x {len(voices)} voice(s)")
    failed = 0
    for i, (text_f, voice_f) in enumerate(combos, 1):
        out = out_path / f"{text_f.stem}_{voice_f.stem}.wav"
        typer.echo(f"\n[{i}/{len(combos)}] {text_f.name} + {voice_f.name} -> {out}")
        cmd = [
            sys.executable, "-m", "training.farsi.synthesize",
            "--config", config,
            "--voice", str(voice_f),
            "--text-file", str(text_f),
            "--out", str(out),
            "--max-tokens", str(max_tokens),
            "--temperature", str(temperature),
            "--eos-threshold", str(eos_threshold),
            "--frames-after-eos", str(frames_after_eos),
            "--voice-sec", str(voice_sec),
            "--pause-sec", str(pause_sec),
            "--join-sec", str(join_sec),
            "--min-tokens", str(min_tokens),
            "--normalize-text" if normalize_text else "--no-normalize-text",
        ]
        if pause_at_punct:
            cmd.append("--pause-at-punct")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            typer.echo(f"FAILED: {text_f.name} + {voice_f.name}")
            failed += 1
    typer.echo(f"\ndone: {len(combos) - failed}/{len(combos)} succeeded, results in {out_path}")
    if failed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
