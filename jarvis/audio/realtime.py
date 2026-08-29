"""Continuous local microphone frames with no import-time device access."""

from __future__ import annotations

import math
import queue
from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from .devices import AudioDevice, MicrophoneDeviceService, MicrophoneError
from .formats import TARGET_SAMPLE_RATE, resample_pcm16_mono


class RealtimeAudioError(RuntimeError):
    """Raised for expected continuous-capture failures."""


class RealtimeAudioTimeoutError(RealtimeAudioError):
    """A transient absence of one realtime frame."""


@dataclass(frozen=True, slots=True)
class RealtimeAudioFrame:
    pcm16: bytes
    captured_at: float
    duration_seconds: float
    sample_rate: int = TARGET_SAMPLE_RATE
    sequence: int | None = None

    def __post_init__(self) -> None:
        if not self.pcm16 or len(self.pcm16) % 2:
            raise ValueError("realtime audio must contain complete PCM16 samples")
        if self.sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError("realtime voice frames must be 16 kHz")
        if self.duration_seconds <= 0 or self.captured_at < 0:
            raise ValueError("realtime frame timing must be non-negative")
        if self.sequence is not None and self.sequence < 0:
            raise ValueError("realtime frame sequence must be non-negative")


class RollingAudioFrameBuffer:
    """Strictly bounded, memory-only audio retained across a state handoff."""

    def __init__(self, *, max_duration_ms: int, frame_duration_ms: int) -> None:
        if max_duration_ms <= 0:
            raise ValueError("rolling audio duration must be positive")
        if frame_duration_ms <= 0:
            raise ValueError("rolling audio frame duration must be positive")
        self.max_duration_ms = max_duration_ms
        self.frame_duration_ms = frame_duration_ms
        self.max_frames = max(1, math.ceil(max_duration_ms / frame_duration_ms))
        self._frames: deque[RealtimeAudioFrame] = deque(maxlen=self.max_frames)

    def append(self, frame: RealtimeAudioFrame) -> None:
        self._frames.append(frame)

    def snapshot(self) -> tuple[RealtimeAudioFrame, ...]:
        return tuple(self._frames)

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def duration_ms(self) -> int:
        return round(sum(frame.duration_seconds for frame in self._frames) * 1_000)


class RealtimeAudioSource(Protocol):
    frame_duration_ms: int

    def start(self) -> None: ...

    def read(self, timeout_seconds: float = 1.0) -> RealtimeAudioFrame: ...

    def drain(self) -> int: ...

    def stop(self) -> None: ...


class SoundDeviceRealtimeInput:
    """Normalize one continuous sounddevice stream into fixed 16 kHz frames."""

    def __init__(
        self,
        devices: MicrophoneDeviceService,
        *,
        preferred_sample_rate: int | None = None,
        frame_duration_ms: int = 30,
        queue_seconds: float = 3.0,
    ) -> None:
        if not 10 <= frame_duration_ms <= 100:
            raise ValueError("frame_duration_ms must be from 10 to 100")
        if not 0.5 <= queue_seconds <= 10:
            raise ValueError("queue_seconds must be from 0.5 to 10")
        self.devices = devices
        self.preferred_sample_rate = preferred_sample_rate
        self.frame_duration_ms = frame_duration_ms
        queue_frames = max(2, round(queue_seconds * 1000 / frame_duration_ms))
        self._raw_queue: queue.Queue[tuple[bytes, float, int]] = queue.Queue(
            maxsize=queue_frames
        )
        self._normalized = bytearray()
        self._normalized_end_at: float | None = None
        self._normalized_sequence: int | None = None
        self._next_capture_sequence = 0
        self._stream: Any | None = None
        self._capture_rate: int | None = None
        self._statuses: list[str] = []

    @property
    def is_active(self) -> bool:
        return self._stream is not None

    @property
    def capture_sample_rate(self) -> int | None:
        return self._capture_rate

    def _select_rate(self, module: Any, device: AudioDevice) -> int:
        candidates = (
            self.preferred_sample_rate,
            TARGET_SAMPLE_RATE,
            device.default_sample_rate,
        )
        seen: set[int] = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            try:
                module.check_input_settings(
                    device=device.index,
                    samplerate=candidate,
                    channels=1,
                    dtype="int16",
                )
            except Exception:
                continue
            return candidate
        raise RealtimeAudioError(
            f"Microphone '{device.name}' has no supported mono PCM16 capture rate."
        )

    def start(self) -> None:
        if self.is_active:
            raise RealtimeAudioError("Continuous microphone capture is already active.")
        try:
            module = self.devices.backend()
            device = self.devices.selected_input()
            sample_rate = self._select_rate(module, device)
        except MicrophoneError as exc:
            raise RealtimeAudioError(str(exc)) from exc

        self._normalized.clear()
        self._normalized_end_at = None
        self._normalized_sequence = None
        self._next_capture_sequence = 0
        self._statuses.clear()
        while not self._raw_queue.empty():
            try:
                self._raw_queue.get_nowait()
            except queue.Empty:
                break

        def callback(indata: Any, _frames: int, _time_info: Any, status: Any) -> None:
            if status:
                self._statuses.append(str(status))
            sequence = self._next_capture_sequence
            self._next_capture_sequence += 1
            payload = (bytes(indata), perf_counter(), sequence)
            try:
                self._raw_queue.put_nowait(payload)
            except queue.Full:
                try:
                    self._raw_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._raw_queue.put_nowait(payload)
                except queue.Full:
                    pass

        stream = None
        try:
            blocksize = max(1, round(sample_rate * self.frame_duration_ms / 1000))
            stream = module.RawInputStream(
                samplerate=sample_rate,
                blocksize=blocksize,
                channels=1,
                dtype="int16",
                device=device.index,
                callback=callback,
            )
            stream.start()
        except Exception as exc:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            raise RealtimeAudioError(f"Could not start continuous microphone input: {exc}") from exc
        self._stream = stream
        self._capture_rate = sample_rate

    def read(self, timeout_seconds: float = 1.0) -> RealtimeAudioFrame:
        if not self.is_active or self._capture_rate is None:
            raise RealtimeAudioError("Continuous microphone capture is not active.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self._statuses:
            detail = "; ".join(dict.fromkeys(self._statuses))
            self._statuses.clear()
            raise RealtimeAudioError(f"Microphone stream reported an error: {detail}")
        target_samples = round(TARGET_SAMPLE_RATE * self.frame_duration_ms / 1000)
        target_bytes = target_samples * 2
        deadline = perf_counter() + timeout_seconds
        while len(self._normalized) < target_bytes:
            remaining = deadline - perf_counter()
            if remaining <= 0:
                raise RealtimeAudioTimeoutError("Timed out waiting for a microphone frame.")
            try:
                raw, captured_at, sequence = self._raw_queue.get(timeout=remaining)
            except queue.Empty as exc:
                raise RealtimeAudioTimeoutError(
                    "Timed out waiting for a microphone frame."
                ) from exc
            try:
                normalized = resample_pcm16_mono(raw, self._capture_rate)
            except Exception as exc:
                raise RealtimeAudioError(f"Could not normalize microphone audio: {exc}") from exc
            self._normalized.extend(normalized)
            self._normalized_end_at = captured_at
            self._normalized_sequence = sequence
        payload = bytes(self._normalized[:target_bytes])
        del self._normalized[:target_bytes]
        end_at = self._normalized_end_at or perf_counter()
        if self._normalized:
            end_at -= len(self._normalized) / (2 * TARGET_SAMPLE_RATE)
        sequence = self._normalized_sequence
        return RealtimeAudioFrame(
            payload,
            end_at,
            self.frame_duration_ms / 1000,
            sequence=sequence,
        )

    def stop(self) -> None:
        stream = self._stream
        self._stream = None
        self._capture_rate = None
        self._normalized.clear()
        self._normalized_end_at = None
        self._normalized_sequence = None
        self._statuses.clear()
        while not self._raw_queue.empty():
            try:
                self._raw_queue.get_nowait()
            except queue.Empty:
                break
        if stream is None:
            return
        try:
            stream.abort()
        except Exception:
            try:
                stream.stop()
            except Exception:
                pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    cancel = stop

    def drain(self) -> int:
        """Discard queued live frames, never persisting their contents."""

        discarded = 0
        self._normalized.clear()
        self._normalized_end_at = None
        self._normalized_sequence = None
        while not self._raw_queue.empty():
            try:
                self._raw_queue.get_nowait()
                discarded += 1
            except queue.Empty:
                break
        self._statuses.clear()
        return discarded
