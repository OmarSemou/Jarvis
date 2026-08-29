"""Structured, monotonic timing for local voice interactions."""

from __future__ import annotations

from dataclasses import dataclass, fields
from statistics import fmean
from time import perf_counter
from typing import Callable


@dataclass(frozen=True, slots=True)
class VoiceLatencyMetrics:
    wake_to_speech_start: float | None = None
    utterance_duration: float | None = None
    end_detection_delay: float | None = None
    stt: float | None = None
    llm_tools: float | None = None
    tts: float | None = None
    playback_start: float | None = None
    total_response_start: float | None = None
    wake_to_playback_cancel: float | None = None
    stt_to_local_stop: float | None = None

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if value is not None and value < 0:
                raise ValueError(f"Latency field '{item.name}' cannot be negative")


Clock = Callable[[], float]


class VoiceLatencyTracker:
    def __init__(self, *, clock: Clock = perf_counter) -> None:
        self._clock = clock
        self._events: dict[str, float] = {}

    def mark(self, name: str, value: float | None = None) -> float:
        timestamp = self._clock() if value is None else float(value)
        if timestamp < 0:
            raise ValueError("Latency timestamps cannot be negative")
        self._events[name] = timestamp
        return timestamp

    def _elapsed(self, start: str, end: str) -> float | None:
        if start not in self._events or end not in self._events:
            return None
        elapsed = self._events[end] - self._events[start]
        if elapsed < 0:
            raise ValueError(f"Impossible timing: '{end}' occurred before '{start}'")
        return elapsed

    def metrics(self) -> VoiceLatencyMetrics:
        return VoiceLatencyMetrics(
            wake_to_speech_start=self._elapsed("wake", "speech_start"),
            utterance_duration=self._elapsed("speech_start", "speech_end"),
            end_detection_delay=self._elapsed("speech_end", "end_detected"),
            stt=self._elapsed("stt_start", "stt_end"),
            llm_tools=self._elapsed("llm_start", "llm_end"),
            tts=self._elapsed("tts_start", "tts_end"),
            playback_start=self._elapsed("playback_requested", "playback_started"),
            total_response_start=self._elapsed("speech_end", "playback_started"),
            wake_to_playback_cancel=self._elapsed("barge_wake", "playback_cancelled"),
            stt_to_local_stop=self._elapsed("stt_end", "local_stop_executed"),
        )


class LatencyHistory:
    def __init__(self) -> None:
        self._items: list[VoiceLatencyMetrics] = []

    @property
    def items(self) -> tuple[VoiceLatencyMetrics, ...]:
        return tuple(self._items)

    @property
    def last(self) -> VoiceLatencyMetrics | None:
        return self._items[-1] if self._items else None

    def add(self, metrics: VoiceLatencyMetrics) -> None:
        self._items.append(metrics)

    def averages(self) -> VoiceLatencyMetrics | None:
        if not self._items:
            return None
        values: dict[str, float | None] = {}
        for item in fields(VoiceLatencyMetrics):
            samples = [
                value
                for metric in self._items
                if (value := getattr(metric, item.name)) is not None
            ]
            values[item.name] = fmean(samples) if samples else None
        return VoiceLatencyMetrics(**values)


def format_latency(metrics: VoiceLatencyMetrics, *, label: str = "LATENCY") -> str:
    names = (
        ("wake-to-speech", metrics.wake_to_speech_start),
        ("utterance", metrics.utterance_duration),
        ("end-detect", metrics.end_detection_delay),
        ("STT", metrics.stt),
        ("LLM/tools", metrics.llm_tools),
        ("TTS", metrics.tts),
        ("playback-start", metrics.playback_start),
        ("total response start", metrics.total_response_start),
        ("wake-to-playback-cancel", metrics.wake_to_playback_cancel),
        ("STT-to-local-stop", metrics.stt_to_local_stop),
    )
    lines = [f"[{label}]"]
    lines.extend(f"{name}: {value:.2f}s" for name, value in names if value is not None)
    return "\n".join(lines)
