# Jarvis safety policy

## Fundamental rule

**An LLM is never a safety authority.**

Model output is untrusted intent. It cannot control motor drivers, clear an
emergency stop, update sensor state, issue transport packets, or bypass policy.
All physical action must pass deterministic validation and the safety
supervisor before reaching a controller.

## Current logical guarantees

- `stop` and `stop_following` are always permitted.
- An emergency stop is latched once triggered.
- LLM and tool authorities cannot clear the latch or publish sensor state.
- Clearing the latch requires a local operator or hardware authority and a
  physical emergency-stop input explicitly reported as clear.
- A stale, missing, unknown, or failed heartbeat prevents motion.
- An unknown or failed controller/sensor state fails closed.
- An obstacle vetoes forward/follow movement.
- A cliff/drop signal vetoes base movement.
- Every denial includes a machine-readable reason and human-readable message.
- Approved physical movement receives only a short renewable lease.

Phase 1 contracts do not move hardware and do not implement serial or ESP32
communication.

## Phase 2B simulation policy

Phase 2B exposes only an explicit allowlist of high-level actions. Model output
cannot name arbitrary methods or supply actuator parameters. Registry
validation, batch policy, the safety supervisor, and the controller remain
deterministic and independent of the model.

The simulator starts with explicit synthetic clear/fresh/ready inputs so tools
can be exercised on a desktop. These are simulation facts, not sensor readings.
An e-stop blocks base and expressive motion, including wave, nod, head motion,
and pointing. `stop`, `stop_following`, and screen-only expression changes do
not create physical motion and remain permitted by the existing policy.

`/robot estop-reset` is a trusted local developer command. No reset, sensor
update, heartbeat change, obstacle override, or safety-disable tool is exposed
to the model. Resetting conversation history never resets the safety latch.

Simulation verifies software control flow only. It does not validate physical
stopping behavior, electrical isolation, motor drivers, sensor placement,
watchdogs, communication loss, or emergency-stop circuitry.

## Phase 2C3.1 local voice stop

An exact anchored voice grammar recognizes STOP after local Whisper
transcription and before the LLM. It cannot request general movement or mutate
safety state. The narrow integration translates only STOP to the existing
high-level `RobotIntent(STOP)`, then calls `SafeRobotController`; the safety
supervisor remains authoritative and the simulator is never called directly.
The route cannot clear a latched e-stop, and Qwen is not involved in deciding
whether STOP should execute.

This voice command is an additional convenience/safety route, not a physical
emergency stop. Speech recognition, the desktop process, or audio hardware may
fail. The future ESP32 watchdog, local motor-command timeout, and physical
e-stop/power-disable path remain mandatory.

## Future physical implementation

The host safety supervisor is only one layer. The ESP32 must independently
enforce communication watchdogs, motion limits, sensor stops, and lease expiry.
Loss of the desktop process, Wi-Fi, network connectivity, or LLM availability
must result in a stop.

A physical emergency-stop circuit must disable motor enable/power directly. It
must not depend on the LLM, desktop, Raspberry Pi, Wi-Fi, application process,
or ESP32 firmware successfully processing a command. Reset must be deliberate,
local, and must not automatically resume prior motion.
