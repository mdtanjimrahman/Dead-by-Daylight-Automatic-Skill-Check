"""Compiled-mode variant: keep original monitoring loop, only force config values."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from dbd import app as app_module
from dbd.app import AutoSkillCheckApp, RunConfiguration


class FastPatchApp(AutoSkillCheckApp):
    def _read_run_configuration(self):
        cfg = super()._read_run_configuration()
        if cfg is None:
            return None
        return RunConfiguration(
            monitor_id=cfg.monitor_id,
            threshold=cfg.threshold,
            frontier_threshold=15,
            frontier_delay=0,
        )


def apply_patch():
    app_module.AutoSkillCheckApp = FastPatchApp


def main():
    apply_patch()
    app_module.main()


if __name__ == "__main__":
    main()
