import subprocess
import sys


def test_importing_phase2_modules_does_not_construct_a_client_or_connect():
    code = """
import socket
import ollama

def forbidden(*args, **kwargs):
    raise AssertionError('import attempted an external side effect')

socket.create_connection = forbidden
ollama.Client = forbidden
import jarvis.cli
import jarvis.core.conversation
import jarvis.llm.ollama
import jarvis.personality.prompt
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
