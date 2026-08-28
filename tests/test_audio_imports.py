import subprocess
import sys


def test_audio_imports_do_not_probe_devices_or_start_subprocesses():
    code = r'''
import subprocess
import sys
import types

def forbidden(*args, **kwargs):
    raise AssertionError("import attempted an external side effect")

fake_sounddevice = types.ModuleType("sounddevice")
fake_sounddevice.query_devices = forbidden
fake_sounddevice.RawInputStream = forbidden
fake_sounddevice.RawOutputStream = forbidden
sys.modules["sounddevice"] = fake_sounddevice
subprocess.run = forbidden

import jarvis.audio
import jarvis.audio.devices
import jarvis.audio.recorder
import jarvis.audio.stt.base
import jarvis.audio.stt.whisper_cpp
import jarvis.audio.tts.base
import jarvis.audio.tts.kokoro
import jarvis.audio.tts.piper
import jarvis.audio.tts.playback
import jarvis.audio.tts.service
'''

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
