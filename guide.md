# DBD Auto Skill Check — Build Guide

## What Was Done

### 1. `great_patch.py` (main patch)
Subclasses `app.AutoSkillCheckApp` and overrides `_read_run_configuration` to force:
- `frontier_threshold = 15` (down from 51)
- `frontier_delay = 0` ms (down from 5)
- `cpu_threads = 2` (fastest inference ~5.4 ms)
- Keeps user's main threshold and monitor selection

Replaces the monitoring loop with a custom Python version that:
- **Pipelined capture**: background thread grabs frames continuously, so inference always uses the freshest frame (reduces frame age from ~16 ms to ~4 ms)
- **Speed-adaptive threshold**: measures p3 ("out" probability) decline rate per frame; faster decline (Hyperfocus) → lower trigger threshold (8-15%)
- **p3-mid trigger**: fires when p3 is in (0.15, 0.5) range — between too-early (misses) and too-late (goods)
- **Adaptive cooldown**: 0.15-0.35 s, wakes early on new skill check (p3 > 0.6)
- **Persistent calibration**: saves measured delay to `great_patch_calib.json`
- **Diagnostic logging**: each hit logs probabilities, trigger type, decline rate, and delay

### 2. `run_app.bat` (updated)
Now launches `great_patch.py` instead of the original `import app; app.main()`.

### 3. `run_great.bat` (new)
Alias for `run_app.bat`.

### 4. `run_fast.bat` (new)
Compiled-mode launcher: keeps original Cython loop, only forces config values. For A/B testing.

### 5. `settings.json` (updated)
- `thresh_frontier`: 15 (was 51)
- `delay_frontier`: 0 (was 5)

### 6. `great_patch_calib.json` (new)
Persists adaptive timing calibration across sessions.

### 7. `great_patch_fast.py` (new)
Compiled-mode variant (no custom loop, just config forcing).

## Files Modified
- `run_app.bat` — changed to run `great_patch.py`
- `settings.json` — updated frontier threshold/delay

## Files Created
- `great_patch.py` — main patch
- `great_patch_fast.py` — compiled-mode variant
- `run_great.bat` — alias launcher
- `run_fast.bat` — compiled-mode launcher
- `great_patch_calib.json` — calibration persistence
- `context.md` — this context file
- `guide.md` — this guide file

## Key Iterations & Results

| Version | Trigger | Result (90 FPS) |
|---------|---------|-----------------|
| Original (compiled, 51% threshold) | argmax class 2 | All good (late) |
| Subclass only (25% threshold) | argmax class 2 | 1/9 great |
| + pipelined capture + 2 threads | argmax class 2 | 17/20 great (85%) |
| + rising-edge (0.10) | p2 rising | 0/9 great (misses) |
| + transition (p3<0.6) | p3 declining | 0/9 great (misses) |
| + p3-mid (0.15-0.5) | p3 mid-approach | 13/17 great (76%), max 6 tokens |
| + p3-mid (0.20-0.5) | p3 mid-approach | 11/15 great (worse) |
| + p3-mid (0.17-0.5) | p3 mid-approach | Worse (fewer tokens) |
| + p3-mid (0.25-0.35) | p3 mid-approach | 6/20 great (30%) |
| Reverted to 0.15-0.5 | p3 mid-approach | Best so far |

## Known Limitations
- At 90 FPS with Hyperfocus (+24% speed), great zone is ~26 ms wide; latency ~14 ms leaves ~12 ms margin
- ~76% great rate at best; remaining goods are late by 1-2 ms
- p3-based triggering is coarse; 1-2 ms precision requires needle-angle prediction (not yet implemented)
- GPU inference not available (no CUDA/TensorRT runtime on this system)

## Agent Instructions
**ALWAYS ask the user for clarification before modifying any files.** Do not make changes without confirming with the user first.
