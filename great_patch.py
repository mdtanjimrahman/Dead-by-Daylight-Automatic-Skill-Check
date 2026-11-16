"""Patch DBD-ASC community build to make repair/heal skill checks land as 'great'.

Speed-adaptive: measures needle speed from p3 decline rate and triggers earlier
for faster needles (Hyperfocus). Persists calibration to great_patch_calib.json.
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import copy
import json
import threading
import time

from dbd import app as app_module
from dbd.inference import monitoring as monitoring_module
from dbd.inference.AI_model import prediction_dict
from dbd.inference.events import make_event_emitter
from dbd.utils.configurations import RunConfiguration
from dbd.utils.directkeys import SPACE, PressKey, ReleaseKey
from dbd.utils.monitoring_mss import Monitoring_mss
from dbd.inference.monitoring import COMMUNITY_MODEL_PATH
from dbd.inference.AI_model import AIModel

CALIB_FILE = Path("great_patch_calib.json")
DEFAULT_DELAY = 40.0

try:
    _calib = json.loads(CALIB_FILE.read_text())
    _measured_delay = float(_calib.get("measured_delay", DEFAULT_DELAY))
except Exception:
    _measured_delay = DEFAULT_DELAY


def _model_name(path):
    try:
        return Path(path).name
    except Exception:
        return str(path)


def real_time_monitoring_great(pred_dict, monitor_id, emit_cb, stop_event=None):
    global _measured_delay

    try:
        monitoring = Monitoring_mss(monitor_id=monitor_id, crop_size=224)
    except Exception as e:
        emit_cb("log", f"Failed to create monitoring: {e}", "error")
        return
    try:
        ai_model = AIModel(str(COMMUNITY_MODEL_PATH), nb_cpu_threads=2, monitoring=monitoring)
    except Exception as e:
        emit_cb("log", f"Failed to load AI model: {e}", "error")
        return
    try:
        execution_provider = ai_model.check_provider().replace("ExecutionProvider", "")
    except Exception:
        execution_provider = "CPU"

    emit_cb(
        "startup",
        {
            "model_name": _model_name(COMMUNITY_MODEL_PATH),
            "device": execution_provider,
            "threads": 2,
            "monitoring": "MSS",
            "monitor_id": monitor_id,
            "threshold": int(pred_dict[1]["threshold"] * 100),
            "frontier_threshold": int(pred_dict[2]["threshold"] * 100),
            "frontier_delay": pred_dict[2]["hit_delay"],
        },
        None,
    )

    latest_frame = [None]
    capture_stop = threading.Event()

    def _capture_loop():
        while not capture_stop.is_set():
            try:
                latest_frame[0] = monitoring.get_frame_np()
            except Exception:
                pass

    cap_thread = threading.Thread(target=_capture_loop, daemon=True)
    cap_thread.start()

    t0 = time.monotonic()
    last_ui_update = 0.0
    nb_frames = 0

    t_approach = None
    p3_at_approach = None
    p3_prev = None
    decline_rate = 0.0
    triggered = False
    trigger_reason = None
    measured_delay = _measured_delay

    try:
        emit_cb("running", None, None)
        while stop_event is None or not stop_event.is_set():
            frame_np = latest_frame[0]
            if frame_np is None:
                time.sleep(0.001)
                continue

            nb_frames += 1
            pred, probs = ai_model.predict(frame_np)
            probs = list(probs)

            thr_main = pred_dict[1]["threshold"]
            thr_frontier = pred_dict[2]["threshold"]

            hit_class = None
            use_delay_ms = 0
            trigger_reason = None

            if p3_prev is not None:
                decline_rate = p3_prev - probs[3]
            p3_prev = probs[3]

            if (
                t_approach is None
                and probs[3] < 0.7
                and probs[3] > 0.1
                and probs[2] >= 0.005
            ):
                t_approach = time.monotonic()
                p3_at_approach = probs[3]

            if t_approach is not None:
                if probs[3] > 0.15 and p3_at_approach is not None and probs[3] > p3_at_approach * 0.5:
                    new_delay = (time.monotonic() - t_approach) * 1000.0
                    if 15 < new_delay < 200:
                        measured_delay = new_delay
                        _measured_delay = measured_delay
                        try:
                            CALIB_FILE.write_text(json.dumps({"measured_delay": measured_delay}))
                        except Exception:
                            pass
                    t_approach = None
                    p3_at_approach = None

            if decline_rate > 0.15:
                adaptive_threshold = 0.08
            elif decline_rate > 0.08:
                adaptive_threshold = 0.10
            elif decline_rate > 0.03:
                adaptive_threshold = 0.12
            else:
                adaptive_threshold = 0.15

            if t_approach is not None and not triggered:
                if 0.15 < probs[3] < 0.5 and decline_rate > 0.02:
                    hit_class = 2
                    use_delay_ms = 0
                    triggered = True
                    trigger_reason = "p3-mid"
                elif probs[2] >= adaptive_threshold and probs[2] >= probs[3]:
                    hit_class = 2
                    use_delay_ms = 0
                    triggered = True
                    trigger_reason = f"thresh-{adaptive_threshold:.2f}"

            if hit_class is None and pred_dict.get(pred, {}).get("hit") and probs[pred] >= pred_dict[pred]["threshold"]:
                hit_class = pred
                use_delay_ms = pred_dict[pred]["hit_delay"]
                trigger_reason = "standard"

            if hit_class is None and probs[1] >= thr_main:
                hit_class = 1
                use_delay_ms = 0
                trigger_reason = "great-fallback"

            now = time.monotonic()
            if now - last_ui_update >= 0.05:
                emit_cb(
                    "prediction",
                    {
                        "class_id": pred,
                        "description": pred_dict[pred]["desc"],
                        "confidence": int(probs[pred] * 100),
                        "scores": list(probs),
                    },
                    None,
                )
                last_ui_update = now

            if hit_class is not None and probs[hit_class] >= pred_dict[hit_class].get("threshold", 0):
                if use_delay_ms and use_delay_ms > 0:
                    time.sleep(use_delay_ms / 1000.0)
                try:
                    PressKey(SPACE)
                    time.sleep(0.002)
                    ReleaseKey(SPACE)
                except Exception as e:
                    emit_cb("log", f"Key press failed: {e}", "error")
                emit_cb(
                    "hit",
                    {"desc": pred_dict[hit_class]["desc"], "confidence": int(probs[hit_class] * 100)},
                    None,
                )
                emit_cb(
                    "log",
                    f"{pred_dict[hit_class]['desc']} hit at {int(probs[hit_class] * 100)}% "
                    f"(p1={probs[1]:.2f} p2={probs[2]:.2f} p3={probs[3]:.2f} "
                    f"trigger={trigger_reason or 'unknown'} decline={decline_rate:.3f} delay={measured_delay:.0f}ms)",
                    "info",
                )

                cooldown_min_end = time.monotonic() + 0.15
                cooldown_max_end = time.monotonic() + 0.35
                while time.monotonic() < cooldown_max_end:
                    if time.monotonic() >= cooldown_min_end:
                        frame = latest_frame[0]
                        if frame is not None:
                            try:
                                p = list(ai_model.predict(frame)[1])
                                if p[3] > 0.6:
                                    break
                            except Exception:
                                pass
                    time.sleep(0.02)

                t_approach = None
                p3_at_approach = None
                triggered = False
                t0 = time.monotonic()
                nb_frames = 0
                continue

            t_diff = time.monotonic() - t0
            if t_diff > 1.0:
                fps = round(nb_frames / t_diff, 1)
                emit_cb("fps", int(fps), None)
                t0 = time.monotonic()
                nb_frames = 0
    except KeyboardInterrupt:
        emit_cb("log", "Monitoring stopped by user.", "info")
    except Exception as e:
        emit_cb("log", f"Critical error during monitoring: {e}", "error")
        try:
            import traceback as _tb

            emit_cb("log", _tb.format_exc(), "error")
        except Exception:
            pass
    finally:
        try:
            capture_stop.set()
        except Exception:
            pass
        try:
            cap_thread.join(timeout=1)
        except Exception:
            pass
        try:
            emit_cb("fps", 0, None)
        except Exception:
            pass
        try:
            ai_model.cleanup()
        except Exception:
            pass


def run_monitoring_with_callbacks_great(*, configuration, event_queue, stop_event):
    pred_dict = copy.deepcopy(prediction_dict)
    for value in pred_dict.values():
        if value["hit"]:
            value["threshold"] = configuration.threshold / 100.0
    pred_dict[2]["threshold"] = 15 / 100.0
    pred_dict[2]["hit_delay"] = 0
    emit = make_event_emitter(event_queue)
    try:
        real_time_monitoring_great(
            pred_dict,
            monitor_id=configuration.monitor_id,
            emit_cb=emit,
            stop_event=stop_event,
        )
    except Exception as error:
        emit("log", f"Monitoring worker failed: {error}", "error")
    finally:
        emit("finished")


class GreatPatchApp(app_module.AutoSkillCheckApp):
    def _read_run_configuration(self):
        cfg = super()._read_run_configuration()
        if cfg is None:
            return None
        try:
            self._append_log(
                f"Speed-adaptive patch: threshold=15% threads=2 "
                f"(calibrated delay={_measured_delay:.0f}ms, main threshold={cfg.threshold}%)",
                "info",
            )
        except Exception:
            pass
        return RunConfiguration(
            monitor_id=cfg.monitor_id,
            threshold=cfg.threshold,
            frontier_threshold=15,
            frontier_delay=0,
        )


def apply_patch():
    """Apply the great-patch overrides to the community app modules."""
    monitoring_module.real_time_monitoring = real_time_monitoring_great
    monitoring_module.run_monitoring_with_callbacks = run_monitoring_with_callbacks_great
    app_module.real_time_monitoring = real_time_monitoring_great
    app_module.run_monitoring_with_callbacks = run_monitoring_with_callbacks_great
    app_module.AutoSkillCheckApp = GreatPatchApp


def main():
    apply_patch()
    app_module.main()


if __name__ == "__main__":
    main()
