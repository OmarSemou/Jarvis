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

## Phase 2C2 local speech synthesis

The setup script downloads these components into ignored runtime storage; this
repository does not redistribute their model or voice files:

- **kokoro-onnx 0.6.1** is an MIT-licensed Python/ONNX wrapper maintained by
  thewh1teagle. The pinned stable wrapper model release is
  `model-files-v1.0`.
  Source: https://github.com/thewh1teagle/kokoro-onnx
- The wrapper's installed dependency chain includes **phonemizer 3.4.0**, which
  declares GPL-3.0. This is another distribution consideration distinct from
  the wrapper's own MIT license.
  Source: https://github.com/bootphon/phonemizer
- **Kokoro-82M** is licensed Apache-2.0 by its model publisher, Hexgrad. The
  Phase 2C2 benchmark uses the pinned wrapper model and bundle with the current
  official voice identifiers `am_fenrir`, `am_michael`, `am_puck`, and
  `bm_george`.
  Source: https://huggingface.co/hexgrad/Kokoro-82M
- **Open Home Foundation Piper 1.7.0** is GPL-3.0-or-later. That license is
  materially different from Jarvis's MIT license and must be reviewed before
  distributing or commercializing a combined application.
  Source: https://github.com/OHF-Voice/piper1-gpl
- **en_US-joe-medium** is pinned from the official Piper voice repository at
  tag `v1.0.0`; its model card identifies a CC0 source dataset.
  Source: https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/joe/medium/MODEL_CARD
- **en_US-john-medium** is pinned from the same immutable tag; its model card
  describes a US male voice trained from public-domain LibriVox recordings.
  Source: https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/john/medium/MODEL_CARD

Engine, model, and voice/dataset licensing are documented separately and must
not be treated as interchangeable. The Kokoro bundle exposes official voice
identifiers, but sufficiently detailed per-voice training-data provenance and
redistribution terms are not established here. Piper model cards document
dataset provenance, but downstream distribution implications still require
review. Generated benchmark WAVs are local evaluation artifacts, not assets
approved for redistribution.
