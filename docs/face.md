# Phase 2D face

Phase 2D adds a provider-neutral, observation-only face subsystem. The active
desktop prototype uses the repository's existing BMO artwork as-is. This is a
temporary visual prototype, not a claim that the eventual Jarvis design or
hardware face is BMO.

## What is active

`FaceController` accepts only small lifecycle, expression, gaze, and speech
generation events. It publishes immutable `FaceState` snapshots to a view.
It has no reference to `ConversationService`, the LLM, tool registry, robot
policy, `SafetySupervisor`, microphone, camera, transcript, or audio samples.
The Tk view is created only by `python -m jarvis face` or `face-demo`; imports
remain side-effect free. Background producers enqueue state and the Tk main
thread calls `pump()`.

Lifecycle mapping is deterministic: idle, wake-detected, listening,
processing, speaking, interrupted, error, and shutdown select their explicit
asset sequence. If a lifecycle sequence is unavailable, the constrained
expression mapping is used, then neutral idle. Shutdown and sleepy use the
available idle closed-eye frame. There are no direction-specific source
images; gaze is state for a future renderer and maps `user` to center.

Speaking events carry only an integer generation identity. Older stop/cancel
events cannot overwrite a newer speaking state. A voice state transition to
`SPEAKING` is emitted only after playback reports that its first audio has
started.

## Immutable prototype inventory

All files below are PNG files at 800x480. Windows reports `Format32bppArgb`,
but every sampled pixel has alpha 255, so the artwork is effectively opaque.
The files are never resized, rewritten, renamed, converted, or optimized by
the Phase 2D code. Hashes are recorded to make accidental mutation visible.

| exact path | SHA-256 | visual role |
| --- | --- | --- |
| `faces/capturing/capturing 01.png` | `5e130672aaafe5a3929202dc5bc39dadf9be2febb1e054df9bf42df98bd7e8c0` | wide alert eyes/open mouth; wake or excited fallback |
| `faces/error/error 01.png` | `a229f0ad0ccd173fd98eb91a18cde34dc76d0edc1dff3654054cb8407e00c673` | fractured concerned/error face |
| `faces/idle/idle 01.png` | `e9a72fcb29dd9fb7c2bd752484e2e7ff7ebb3f7ec6e8da00ab8eb5110c187ad7` | relaxed closed-eye neutral |
| `faces/listening/listen 01.png` | `161f63530f51fe256abf715f82c38831a4a8ab7fc8995290b74f5aa971b84975` | attentive open eyes and smile |
| `faces/listening/listen 02.png` | `e94cc26b01febc98862eccbfc8b6ce2e445ed1275a6ce1a703a714fc4514e81d` | alternate listening/blink frame |
| `faces/speaking/speaking 01.png` | `ed93d296e090129c338b94dc3fa400b892b16d1838f83deb67913a22e8400cf7` | speaking baseline |
| `faces/speaking/speaking 02.png` | `17ffa4b8470de2ac6a22b57cd25e28aa0287f032026163136fef8458aca265fc` | speaking open-mouth variant |
| `faces/speaking/speaking 03.png` | `df377320e2f40d276c839734b8a87836fa7a752bf23d74f83b556906e7da5c98` | speaking broad-smile variant |
| `faces/thinking/thinking 01.png` | `10b2fbfb8a80febda4e829ee5a8bfa58d3ffa5890e0dc38fdbb00ec579892a98` | thinking brow/smile |
| `faces/thinking/thinking 02.png` | `e1b2587f2a9a9b1c10f2780386850e20bf5618869a4d3d9e402e2dbbaaae7652` | thinking alternate eyes |
| `faces/thinking/thinking 03.png` | `6fbe75c5cb48b61dfc6fa5fedf902e6d7ffec9943708d529cf6e61def0dc1f3c` | thinking neutral mouth |
| `faces/thinking/thinking 04.png` | `052840df843a4ee2a7005b9c559bba9ae450deb993e456db01559cbdb7bf3bb1` | thinking alternate neutral |
| `faces/warmup/warmup 01.png` | `12f4fd054e817c246f94aaa64dad9e87f748b1caf6578914d635d283feb1416` | “Loading…” warmup card; gallery only |

The legacy `agent.py` loader also references these folders and creates
in-memory resized `ImageTk` frames. That compatibility GUI remains untouched;
the modular face does not import or call it. `be-more-agent.desktop` contains
an old absolute Raspberry Pi path and old `idle_0.png` naming and is not used
by the Windows CLI. `setup.sh` only creates the legacy directories.

## Provenance and limitations

The repository is derived from the MIT-licensed upstream Be More Agent project;
`LICENSE` and `NOTICE.md` remain authoritative. The individual artwork's
creator, source URL, and separate redistribution terms are not established by
the repository. It is therefore documented as private prototype artwork only;
do not publish or redistribute it until provenance is resolved. A later phase
must add a separately licensed Jarvis asset set and can replace the asset
provider without changing face state or controller contracts.

The simulator and face are not hardware-safety validation. Physical e-stop,
cliff/obstacle handling, and actuator control remain entirely below the face
boundary and must continue to work if the desktop, GUI, network, or LLM fails.

## Commands

```powershell
python -m jarvis face
python -m jarvis face-demo
python -m jarvis face-demo --gallery
python -m jarvis voice --face
```

`--fullscreen` is available on `face`, `face-demo`, and `voice --face`.
Escape or Ctrl+Q closes the view. No audio, vision, wake-word, web lookup,
persistent memory, ESP32, or physical robot feature is introduced here.
