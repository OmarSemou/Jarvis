# Notices and attribution

## Upstream project

Jarvis is derived from **Be More Agent**, originally created by brenpoly:

- Source: https://github.com/brenpoly/be-more-agent
- Upstream copyright: Copyright (c) 2026 brenpoly
- Upstream license: MIT

The upstream `LICENSE` file and Git history are preserved. Portions extracted
into the `jarvis` package remain derived from the upstream project where
applicable.

## Third-party components and unresolved asset provenance

Phase 1 does not install, redistribute, or newly integrate the future audio,
vision, LLM, or hardware stack. Existing upstream assets are intentionally
preserved while their terms are reviewed.

- **openWakeWord** code is Apache-2.0. Its included pre-trained wake-word
  models are documented by that project as CC BY-NC-SA 4.0. The tracked
  `wakeword.onnx` is retained from upstream and must not be assumed to be
  covered by this repository's MIT license.
  Source: https://github.com/dscripka/openWakeWord#license
- **Piper** engine code and individual Piper voice models/datasets have their
  own terms. The configured `en_GB-semaine-medium` voice is not tracked here;
  its model card identifies the source dataset as CC BY-NC-SA 4.0.
  Source: https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_GB/semaine/medium/MODEL_CARD
- The legacy setup script references a custom BMO-style voice. Its complete
  model, dataset, character, and redistribution provenance is not established
  by the repository's MIT license.
- Existing face PNGs and sound WAVs came from the upstream history, but no
  per-asset authorship/license manifest is present.

These items require explicit review before public redistribution or commercial
use. This notice documents the uncertainty; it does not claim to resolve it.

## Phase 2C1 local speech recognition

The explicit Windows setup script downloads, but this repository does not
redistribute, the official `whisper.cpp` v1.9.1 CPU binary and the multilingual
`ggml-small.bin` model from the upstream `ggerganov/whisper.cpp` model
repository. The engine and upstream model repository are MIT licensed. Their
downloaded files remain beneath ignored `data/` runtime storage.

- Engine: https://github.com/ggml-org/whisper.cpp
- Model repository: https://huggingface.co/ggerganov/whisper.cpp
