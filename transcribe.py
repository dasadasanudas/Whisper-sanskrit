"""
Whisper Sanskrit Transcription
================================
Transcribes Sanskrit audio files using OpenAI's Whisper speech-to-text model.

Usage:
    python transcribe.py <audio_file> [--model MODEL] [--language LANGUAGE]

Examples:
    python transcribe.py audio.mp3
    python transcribe.py audio.wav --model medium --language sa
    python transcribe.py audio.mp3 --model large
"""

import argparse
import os

import whisper


def transcribe_audio(
    audio_path: str,
    model_name: str = "base",
    language: str = "sa",
    task: str = "transcribe",
) -> dict:
    """
    Transcribe an audio file using OpenAI Whisper.

    Parameters
    ----------
    audio_path : str
        Path to the audio file (mp3, wav, m4a, ogg, flac, etc.).
    model_name : str
        Whisper model size to use. Options: tiny, base, small, medium, large,
        large-v2, large-v3. Larger models are more accurate but slower and
        require more memory.
    language : str
        ISO 639-1 language code. Use 'sa' for Sanskrit.
        Pass None to let Whisper auto-detect the language.
    task : str
        Either 'transcribe' (keep original language) or 'translate' (to English).

    Returns
    -------
    dict
        A dictionary containing:
        - 'text'     : Full transcription as a single string.
        - 'segments' : List of timed segment dicts with 'start', 'end', 'text'.
        - 'language' : Detected / specified language code.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"Loading Whisper model '{model_name}' …")
    model = whisper.load_model(model_name)

    print(f"Transcribing '{audio_path}' (language='{language}', task='{task}') …")
    result = model.transcribe(audio_path, language=language, task=task)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe Sanskrit audio using OpenAI Whisper."
    )
    parser.add_argument("audio", help="Path to the audio file to transcribe.")
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size (default: base).",
    )
    parser.add_argument(
        "--language",
        default="sa",
        help=(
            "Language code for the audio (default: 'sa' for Sanskrit). "
            "Pass 'auto' to let Whisper detect the language automatically."
        ),
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help=(
            "Task to perform: 'transcribe' keeps the original language, "
            "'translate' produces an English translation (default: transcribe)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the transcription text file.",
    )

    args = parser.parse_args()

    language = None if args.language.lower() == "auto" else args.language

    result = transcribe_audio(
        audio_path=args.audio,
        model_name=args.model,
        language=language,
        task=args.task,
    )

    print("\n=== Transcription ===")
    print(result["text"])

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result["text"])
        print(f"\nTranscription saved to '{args.output}'.")

    print("\n=== Timed Segments ===")
    for segment in result.get("segments", []):
        start = segment["start"]
        end = segment["end"]
        text = segment["text"].strip()
        print(f"[{start:6.2f}s -> {end:6.2f}s]  {text}")

    return result


if __name__ == "__main__":
    main()
