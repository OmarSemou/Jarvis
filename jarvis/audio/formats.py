"""Small deterministic PCM/WAV helpers used by local audio providers."""

from __future__ import annotations

import sys
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path


TARGET_SAMPLE_RATE = 16_000
PCM16_SAMPLE_WIDTH = 2


class AudioFormatError(ValueError):
    """Raised when captured audio cannot be converted safely."""


@dataclass(frozen=True, slots=True)
class WaveInfo:
    channels: int
    sample_width: int
    sample_rate: int
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.sample_rate if self.sample_rate else 0.0


def resample_pcm16_mono(data: bytes, source_rate: int, target_rate: int = TARGET_SAMPLE_RATE) -> bytes:
    """Linearly resample little-endian mono PCM16 with no external process."""

    if source_rate <= 0 or target_rate <= 0:
        raise AudioFormatError("sample rates must be positive")
    if len(data) % PCM16_SAMPLE_WIDTH:
        raise AudioFormatError("PCM16 data has an incomplete sample")
    if not data or source_rate == target_rate:
        return data

    samples = array("h")
    samples.frombytes(data)
    if sys.byteorder != "little":
        samples.byteswap()
    if len(samples) == 1:
        output_count = max(1, round(target_rate / source_rate))
        output = array("h", [samples[0]] * output_count)
    else:
        output_count = max(1, round(len(samples) * target_rate / source_rate))
        output = array("h")
        for output_index in range(output_count):
            source_position = output_index * source_rate / target_rate
            left = min(int(source_position), len(samples) - 1)
            right = min(left + 1, len(samples) - 1)
            fraction = source_position - left
            value = round(samples[left] + (samples[right] - samples[left]) * fraction)
            output.append(max(-32_768, min(32_767, value)))
    if sys.byteorder != "little":
        output.byteswap()
    return output.tobytes()


def write_pcm16_mono_wav(path: Path, pcm_data: bytes, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    if not pcm_data:
        raise AudioFormatError("recording is empty")
    if len(pcm_data) % PCM16_SAMPLE_WIDTH:
        raise AudioFormatError("PCM16 data has an incomplete sample")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(PCM16_SAMPLE_WIDTH)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)


def inspect_wav(path: Path) -> WaveInfo:
    try:
        with wave.open(str(path), "rb") as wav_file:
            return WaveInfo(
                wav_file.getnchannels(),
                wav_file.getsampwidth(),
                wav_file.getframerate(),
                wav_file.getnframes(),
            )
    except (OSError, EOFError, wave.Error) as exc:
        raise AudioFormatError(f"invalid WAV file: {exc}") from exc
