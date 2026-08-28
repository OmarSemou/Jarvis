# Jarvis safety policy

## Fundamental rule

**An LLM is never a safety authority.**

Model output is untrusted intent. It cannot control motor drivers, clear an
emergency stop, update sensor state, issue transport packets, or bypass policy.
All physical action must pass deterministic validation and the safety
supervisor before reaching a controller.

## Phase 1 logical guarantees

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

## Future physical implementation

The host safety supervisor is only one layer. The ESP32 must independently
enforce communication watchdogs, motion limits, sensor stops, and lease expiry.
Loss of the desktop process, Wi-Fi, network connectivity, or LLM availability
must result in a stop.

A physical emergency-stop circuit must disable motor enable/power directly. It
must not depend on the LLM, desktop, Raspberry Pi, Wi-Fi, application process,
or ESP32 firmware successfully processing a command. Reset must be deliberate,
local, and must not automatically resume prior motion.

