# DBD Auto Skill Check — Context

## User Requirement
Make the DBD auto skill-check app (`run_app.bat`, `model_cpu_v4-1.onnx`, CPU/Balanced) consistently produce **great** skill checks for generator repair and healing, even with the Hyperfocus perk (which increases needle speed by up to +24% and triggers more skill checks).

## Environment
- Platform: Windows 10/11, Python 3.12 embedded
- Game: Dead by Daylight, capped at 70-90 FPS (user prefers 70-90 FPS)
- Monitor: 144 Hz
- Model: `models/model_cpu_v4-1.onnx` (ONNX, 1.7 MB, input [1,3,224,224], output [1,10])
- App version: V4.3 (compiled Cython, premium)
- CPU: 2 threads (forced for lowest inference latency ~5.4 ms)

## Game Mechanics
- Skill check zones: **Great zone (small, first) → Good zone (larger, after)**
- Needle speed: ~0.343°/ms, full rotation ~1.05 s
- Great zone: 11° wide (~32 ms at base speed)
- Good zone: 46° wide (~134 ms)
- At 90 FPS: great zone ~3 frames (~33 ms at base speed, ~26 ms at +24% Hyperfocus speed)
- Hyperfocus perk: +4% speed per token (max +24% at 6 tokens), loses tokens on good/miss

## App Architecture
- `app.cp312-win_amd64.pyd` — compiled 4.3 app (Cython)
- `great_patch.py` — Python patch (subclass + custom monitoring loop)
- `run_app.bat` — launcher (runs great_patch.py)
- `settings.json` — slider defaults (not reliable for this build)
- `great_patch_calib.json` — persisted adaptive timing calibration

## Key Technical Facts
- Model classes: 0=none, 1=repair-heal (great), 2=repair-heal (frontier), 3=repair-heal (out), 4-9=other skill types
- Class 2 ("frontier") appears before class 1 ("great") — used as early trigger
- Default frontier threshold 51% skips low-confidence class 2 (e.g., 32%) → late press → good
- Compiled loop is Cython (fast); Python replacement adds overhead but enables pipelining
- mss capture ~8.8 ms; inference ~5.4 ms at 2 threads; pipelined capture reduces frame age from ~16 ms to ~4 ms
- End-to-end latency: ~14 ms (capture 9 ms + inference 5 ms)

## Current Best Result
- 90 FPS, no Hyperfocus: 17/20 great (85%)
- 90 FPS, Hyperfocus (6 tokens): ~76% great (13/17), max 6 tokens reached
- Trigger: p3-mid (0.15–0.5 range), p2 >= adaptive threshold, pipelined capture, 2 threads
- Remaining goods are late by 1-2 ms (tight timing at +24% speed)

## Agent Instructions
**ALWAYS ask the user for clarification before modifying any files.** Do not make changes without confirming with the user first. Present options and let the user decide.
