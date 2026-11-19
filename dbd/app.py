import ctypes
import json
import math
import sys
from time import monotonic
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from queue import Empty, Queue

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from threading import Event, Thread
from typing import Any

from dearpygui import dearpygui

from dbd.inference.AI_model import DEFAULT_FRONTIER_HIT_DELAY, DEFAULT_FRONTIER_THRESHOLD, DEFAULT_THRESHOLD, prediction_dict
from dbd.inference.events import EventQueue, WorkerEvent
from dbd.inference.monitoring import COMMUNITY_MODEL_PATH, run_monitoring_with_callbacks
from dbd.utils.configurations import RunConfiguration
from dbd.utils.monitoring_mss import Monitoring_mss

COMMUNITY_CAPTURE_NAME = "MSS"
DEFAULT_VIEWPORT_WIDTH = 1400
DEFAULT_VIEWPORT_HEIGHT = 950
MIN_VIEWPORT_WIDTH = 360
MIN_VIEWPORT_HEIGHT = 420
UI_FONT_SIZE = 16
STRONG_FONT_SIZE = 15
HEADING_FONT_SIZE = 23
DISPLAY_FONT_SIZE = 29
FONTS_FOLDER = Path("fonts").resolve()
INFERENCE_SCORE_GROUPS = (
    ("none", "None", (0,)),
    ("repair_heal", "Repair-heal", (1, 2, 3)),
    ("full_white", "Full white", (4, 5)),
    ("full_black", "Full black", (6, 7)),
    ("wiggle", "Wiggle", (8, 9)),
)


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Optional UI presets loaded before a run is configured."""

    monitor_id: int | None = None
    thresh: int = DEFAULT_THRESHOLD
    thresh_frontier: int = DEFAULT_FRONTIER_THRESHOLD
    delay_frontier: int = DEFAULT_FRONTIER_HIT_DELAY
    viewport_width: int = DEFAULT_VIEWPORT_WIDTH
    viewport_height: int = DEFAULT_VIEWPORT_HEIGHT


def _safe_score(score: Any) -> float:
    """Return a finite model score clamped to the progress bar range."""
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        return 0.0
    return min(max(score_value, 0.0), 1.0) if math.isfinite(score_value) else 0.0


def _score_at(scores: Any, class_id: int) -> float:
    try:
        return _safe_score(scores[class_id])
    except (IndexError, TypeError):
        return 0.0


def get_inference_score_values(scores: Any) -> dict[str, float]:
    """Group the ten model probabilities into the five displayed score bars."""
    return {key: round(min(sum(_score_at(scores, class_id) for class_id in class_ids), 1.0), 10) for key, _label, class_ids in INFERENCE_SCORE_GROUPS}


def get_monitor_choices() -> list[tuple[str, int]]:
    """Get monitor labels and ids from the fixed MSS capture backend."""
    try:
        return list(Monitoring_mss.get_monitors_info())
    except Exception:
        return []


def load_settings(config_path: Path = Path("settings.json")) -> AppSettings:
    """Load known settings from JSON, falling back to defaults."""
    defaults = AppSettings()

    if config_path.exists():
        try:
            with config_path.open() as f:
                loaded_settings = json.load(f)
            if isinstance(loaded_settings, dict):
                known_names = {field.name for field in fields(AppSettings)}
                known_settings = {key: value for key, value in loaded_settings.items() if key in known_names}
                return AppSettings(**known_settings)
        except (json.JSONDecodeError, OSError):
            pass

    return defaults


class AutoSkillCheckApp:
    """Dear PyGui application for configuring and displaying skill-check monitoring."""

    # --- adaptive layout -------------------------------------------------
    PADDING = 16
    SPACING = 10
    STACK_BREAKPOINT = 1000  # below this viewport width the panels stack vertically
    FONT_TIERS = (0.85, 1.0, 1.2, 1.4)  # crisp pre-rasterised font sizes (multiples of the base size)
    SETTINGS_MIN_WIDTH = 400
    SETTINGS_MAX_WIDTH = 580
    METRIC_CARD_MIN_WIDTH = 270  # two cards sit side by side only if both can be at least this wide
    METRIC_CARD_CONTENT_HEIGHT = 180  # enough room for all inference metrics without inner scrolling
    METRIC_CARD_FIXED_HEIGHT = 150  # padding/spacing that does not scale with the font
    LOG_MIN_HEIGHT = 200
    ACTION_BUTTON_WIDTH_MIN = 205
    ACTION_BUTTON_HEIGHT = 46
    FPS_BAR_HEIGHT = 20
    SCORE_BAR_HEIGHT = 16
    MAX_EVENTS_PER_POLL = 100
    TELEMETRY_EVENT_TYPES = frozenset({"prediction", "fps"})
    ORDERED_EVENT_TYPES = frozenset({"startup", "running", "hit", "log", "finished"})

    WINDOW_TAG = "main_window"
    HEADER_TAG = "header_panel"
    PANELS_TAG = "panels_group"
    SETTINGS_PANEL_TAG = "settings_panel"
    LIVE_PANEL_TAG = "live_panel"
    CARDS_GROUP_TAG = "metric_cards_group"
    CONFIG_PANEL_TAG = "configuration_panel"
    ACTION_GROUP_TAG = "action_buttons_group"
    ACTION_SECTION_TAG = "action_buttons_section"
    TITLE_TAG = "app_title"
    HEADER_LICENSE_TEXT_TAG = "header_license_text"
    COMMUNITY_DETAILS_TEXT_TAG = "community_details_text"
    COMMUNITY_LINK_TEXT_TAG = "community_link_text"
    SETUP_TITLE_TAG = "setup_title"
    LIVE_TITLE_TAG = "live_title"
    STATUS_DOT_TAG = "status_dot"
    MONITOR_TAG = "monitor"
    THRESHOLD_TAG = "threshold"
    FRONTIER_THRESHOLD_TAG = "frontier_threshold"
    FRONTIER_DELAY_TAG = "frontier_delay"
    RUN_TAG = "run_button"
    STOP_TAG = "stop_button"
    STATUS_TAG = "status"
    STATUS_WAITING = "WAITING"
    STATUS_STARTING = "STARTING"
    STATUS_RUNNING = "RUNNING"
    STATUS_STOPPING = "STOPPING"
    STATUS_STOPPED = "STOPPED"
    DETECTION_TAG = "detection"
    INFERENCE_PANEL_TAG = "inference_panel"
    DETECTION_PANEL_TAG = "detection_panel"
    SKILL_CHECK_COUNT_TAG = "skill_check_count"
    FPS_TAG = "fps"
    FPS_MARK_TAG = "fps_mark"
    FPS_BAR_TAG = "fps_bar"
    LOG_TAG = "event_log"
    SIGNAL_LOGO_TAG = "signal_logo"
    DETECTION_LOGO_TAG = "detection_logo"
    CONFIG_MODEL_TAG = "configuration_model"
    CONFIG_DEVICE_TAG = "configuration_device"
    CONFIG_CAPTURE_TAG = "configuration_capture"
    CONFIG_THRESH_TAG = "configuration_thresh"
    EMPTY_VALUE = "-"

    # --- crimson "mission control" palette ---------------------------------
    TEXT = (248, 241, 243, 255)
    MUTED = (180, 160, 167, 255)
    ACCENT = (255, 62, 96, 255)
    ACCENT_HOVER = (255, 112, 138, 255)
    ACCENT_DIM = (160, 28, 56, 255)
    INFO = (88, 214, 222, 255)
    POSITIVE = (72, 214, 130, 255)
    WARNING = (247, 190, 74, 255)
    ORANGE = (255, 150, 84, 255)
    NEGATIVE = (255, 84, 76, 255)

    WINDOW_BG = (8, 6, 10, 255)
    PANEL_BG = (15, 10, 14, 255)
    CARD_BG = (23, 15, 21, 255)
    FRAME_BG = (36, 25, 32, 255)
    BORDER = (62, 24, 36, 255)
    BORDER_STRONG = (118, 36, 58, 255)

    def __init__(
        self,
        dearpygui: Any,
        presets: AppSettings | None = None,
    ) -> None:
        self.dpg = dearpygui
        self.presets = presets or AppSettings()
        self.heading_font: Any = None
        self.fonts: dict[float, dict[str, Any]] = {}
        self.role_items: dict[str, list[Any]] = {"strong": [], "heading": [], "display": []}
        self.item_themes: dict[str, Any] = {}
        self.progress_themes: dict[tuple[int, int, int, int], Any] = {}
        self.ui_scale = 1.0
        self.scaled_heights: dict[str, int] = {}
        self.viewport_size: tuple[int, int] | None = None
        self.action_layout: tuple[int, float] | None = None
        self._fitted_log_height: int | None = None
        self._last_viewport_save_at = 0.0
        self._last_saved_viewport_size: tuple[int, int] | None = None
        self.worker: Thread | None = None
        self.event_queue: EventQueue | None = None
        self.stop_event: Event | None = None
        self.worker_finished = False
        self.skill_check_count = 0

    def _preset(self, name: str, default: Any = None) -> Any:
        value = getattr(self.presets, name, None)
        return default if value is None else value

    def _build_fonts(self, fonts_folder: Path = FONTS_FOLDER) -> None:
        """Load the bundled UI font (regular + bold faces) at several crisp sizes."""
        regular_path = fonts_folder / "CascadiaCode" / "CaskaydiaCoveNerdFontMono-Regular.ttf"
        bold_path = fonts_folder / "CascadiaCode" / "CaskaydiaCoveNerdFontMono-Bold.ttf"
        if not regular_path.is_file():
            return

        dpg = self.dpg
        bold_or_regular = bold_path if bold_path.is_file() else regular_path
        with dpg.font_registry():
            for tier in self.FONT_TIERS:
                self.fonts[tier] = {
                    "ui": dpg.add_font(str(regular_path), round(UI_FONT_SIZE * tier)),
                    "strong": dpg.add_font(str(bold_or_regular), round(STRONG_FONT_SIZE * tier)),
                    "heading": dpg.add_font(str(bold_or_regular), round(HEADING_FONT_SIZE * tier)),
                    "display": dpg.add_font(str(bold_or_regular), round(DISPLAY_FONT_SIZE * tier)),
                }
        self.heading_font = self.fonts[1.0]["heading"]
        dpg.bind_font(self.fonts[1.0]["ui"])

    def _build_theme(self) -> None:
        dpg = self.dpg
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvAll):
                for color, value in (
                    (dpg.mvThemeCol_WindowBg, self.WINDOW_BG),
                    (dpg.mvThemeCol_ChildBg, self.PANEL_BG),
                    (dpg.mvThemeCol_PopupBg, (19, 12, 17, 250)),
                    (dpg.mvThemeCol_FrameBg, self.FRAME_BG),
                    (dpg.mvThemeCol_FrameBgHovered, (52, 33, 43, 255)),
                    (dpg.mvThemeCol_FrameBgActive, (70, 38, 52, 255)),
                    (dpg.mvThemeCol_Text, self.TEXT),
                    (dpg.mvThemeCol_TextDisabled, self.MUTED),
                    (dpg.mvThemeCol_Border, self.BORDER),
                    (dpg.mvThemeCol_Separator, (52, 22, 32, 255)),
                    (dpg.mvThemeCol_Button, (52, 30, 40, 255)),
                    (dpg.mvThemeCol_ButtonHovered, (96, 40, 58, 255)),
                    (dpg.mvThemeCol_ButtonActive, (138, 36, 62, 255)),
                    (dpg.mvThemeCol_Header, (38, 22, 31, 255)),
                    (dpg.mvThemeCol_HeaderHovered, (70, 30, 44, 255)),
                    (dpg.mvThemeCol_HeaderActive, (104, 34, 54, 255)),
                    (dpg.mvThemeCol_CheckMark, self.ACCENT),
                    (dpg.mvThemeCol_SliderGrab, self.ACCENT),
                    (dpg.mvThemeCol_SliderGrabActive, self.ACCENT_HOVER),
                    (dpg.mvThemeCol_PlotHistogram, self.ACCENT),
                    (dpg.mvThemeCol_TextSelectedBg, (160, 28, 56, 130)),
                    (dpg.mvThemeCol_ScrollbarBg, (0, 0, 0, 0)),
                    (dpg.mvThemeCol_ScrollbarGrab, (84, 40, 56, 255)),
                    (dpg.mvThemeCol_ScrollbarGrabHovered, (130, 52, 74, 255)),
                    (dpg.mvThemeCol_ScrollbarGrabActive, self.ACCENT),
                ):
                    dpg.add_theme_color(color, value)
                # Padding/spacing values feed the adaptive layout maths: keep PADDING / SPACING exact.
                for style, values in (
                    (dpg.mvStyleVar_WindowRounding, (14,)),
                    (dpg.mvStyleVar_ChildRounding, (14,)),
                    (dpg.mvStyleVar_PopupRounding, (12,)),
                    (dpg.mvStyleVar_FrameRounding, (10,)),
                    (dpg.mvStyleVar_GrabRounding, (12,)),
                    (dpg.mvStyleVar_ScrollbarRounding, (12,)),
                    (dpg.mvStyleVar_ScrollbarSize, (8,)),
                    (dpg.mvStyleVar_GrabMinSize, (18,)),
                    (dpg.mvStyleVar_WindowPadding, (self.PADDING, self.PADDING - 3)),
                    (dpg.mvStyleVar_FramePadding, (12, 8)),
                    (dpg.mvStyleVar_ItemSpacing, (self.SPACING, 7)),
                    (dpg.mvStyleVar_ItemInnerSpacing, (8, 5)),
                    (dpg.mvStyleVar_WindowBorderSize, (0,)),
                    (dpg.mvStyleVar_ChildBorderSize, (1,)),
                    (dpg.mvStyleVar_PopupBorderSize, (1,)),
                    (dpg.mvStyleVar_FrameBorderSize, (0,)),
                ):
                    dpg.add_theme_style(style, *values)
        dpg.bind_theme(theme)

        def child_theme(background: tuple[int, int, int, int], border: tuple[int, int, int, int], rounding: int) -> Any:
            with dpg.theme() as item_theme, dpg.theme_component(dpg.mvChildWindow):
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, background)
                dpg.add_theme_color(dpg.mvThemeCol_Border, border)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, rounding)
            return item_theme

        def button_theme(
            normal: tuple[int, int, int, int],
            hovered: tuple[int, int, int, int],
            active: tuple[int, int, int, int],
            border: tuple[int, int, int, int] | None = None,
            text: tuple[int, int, int, int] | None = None,
        ) -> Any:
            with dpg.theme() as item_theme:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, normal)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, hovered)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, active)
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 12)
                    if text is not None:
                        dpg.add_theme_color(dpg.mvThemeCol_Text, text)
                    if border is not None:
                        dpg.add_theme_color(dpg.mvThemeCol_Border, border)
                        dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
                with dpg.theme_component(dpg.mvButton, enabled_state=False):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (21, 14, 19, 255))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (21, 14, 19, 255))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (21, 14, 19, 255))
                    dpg.add_theme_color(dpg.mvThemeCol_Text, (116, 96, 103, 255))
                    dpg.add_theme_color(dpg.mvThemeCol_Border, (44, 24, 32, 255))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
            return item_theme

        def flat_theme(color: tuple[int, int, int, int]) -> Any:
            """Non-interactive, hairline-thin colour strip used for the section rails."""
            with dpg.theme() as item_theme:
                for enabled in (True, False):
                    with dpg.theme_component(dpg.mvButton, enabled_state=enabled):
                        for button_color in (dpg.mvThemeCol_Button, dpg.mvThemeCol_ButtonHovered, dpg.mvThemeCol_ButtonActive):
                            dpg.add_theme_color(button_color, color)
                        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 2)
                        dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
            return item_theme

        with dpg.theme() as help_theme, dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (52, 24, 36, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, self.ACCENT_DIM)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, self.ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_Text, self.ACCENT_HOVER)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 12)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 7, 1)

        self.item_themes = {
            "panel": child_theme(self.PANEL_BG, self.BORDER, 16),
            "header": child_theme((19, 11, 16, 255), self.BORDER_STRONG, 16),
            "card": child_theme(self.CARD_BG, (82, 30, 46, 255), 14),
            "log": child_theme((11, 7, 10, 255), (46, 21, 30, 255), 12),
            "primary_button": button_theme(
                (222, 34, 72, 255), (255, 72, 106, 255), (170, 22, 54, 255), border=(255, 128, 152, 255), text=(255, 255, 255, 255)
            ),
            "secondary_button": button_theme(
                (30, 19, 26, 255), (66, 26, 40, 255), (98, 28, 48, 255), border=(176, 46, 74, 255), text=(255, 160, 180, 255)
            ),
            "help": help_theme,
            "rail_bright": flat_theme(self.ACCENT),
            "rail_dim": flat_theme((66, 26, 38, 255)),
        }

        self.progress_themes = {}
        for color in (self.NEGATIVE, self.ORANGE, self.WARNING, self.POSITIVE, self.ACCENT, self.INFO, self.MUTED):
            with dpg.theme() as progress_theme, dpg.theme_component(dpg.mvProgressBar):
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, color)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (38, 26, 35, 255))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 7)
            self.progress_themes[color] = progress_theme

    def _style(self, item: Any, key: str) -> None:
        """Bind one of the prepared item themes (no-op when themes were not built)."""
        theme = self.item_themes.get(key)
        if theme is not None:
            self.dpg.bind_item_theme(item, theme)

    def _set_status(self, status: str) -> None:
        status_color = {
            self.STATUS_WAITING: self.WARNING,
            self.STATUS_STARTING: self.INFO,
            self.STATUS_RUNNING: self.POSITIVE,
            self.STATUS_STOPPING: self.WARNING,
            self.STATUS_STOPPED: self.NEGATIVE,
        }.get(status, self.MUTED)
        self.dpg.set_value(self.STATUS_TAG, status)
        self.dpg.configure_item(self.STATUS_TAG, color=status_color)
        self.dpg.configure_item(self.STATUS_DOT_TAG, color=status_color)
        self._update_buttons()

    def _set_monitor_choices(self, preferred_monitor_id: int | None = None) -> None:
        choices = get_monitor_choices()
        monitor_ids = dict(choices)
        labels = list(monitor_ids)
        self.dpg.configure_item(self.MONITOR_TAG, items=labels, enabled=bool(labels), user_data=monitor_ids)
        if labels:
            selected_label = next(
                (label for label, monitor_id in choices if monitor_id == preferred_monitor_id),
                labels[0],
            )
            self.dpg.set_value(self.MONITOR_TAG, selected_label)
            self._set_status(self.STATUS_WAITING)
            if preferred_monitor_id is not None and monitor_ids[selected_label] != preferred_monitor_id:
                self._append_log(f"Preset monitor {preferred_monitor_id} was not found; using {selected_label}.", "warning")
        else:
            self.dpg.set_value(self.MONITOR_TAG, "No monitors detected")
            self._set_status(self.STATUS_WAITING)

    def _scroll_log_to_bottom(self, *_args: Any) -> None:
        set_y_scroll = getattr(self.dpg, "set_y_scroll", None)
        if callable(set_y_scroll):
            set_y_scroll(self.LOG_TAG, -1)

    def _append_log(self, message: str, level: str = "info") -> None:
        icons = {"success": "✓", "warning": "!", "error": "×", "info": "•"}
        colors = {"success": self.POSITIVE, "warning": self.WARNING, "error": self.NEGATIVE, "info": self.MUTED}
        level = level if level in colors else "info"
        line = f"{icons[level]}  {message}"

        does_item_exist = getattr(self.dpg, "does_item_exist", None)
        add_text = getattr(self.dpg, "add_text", None)
        if callable(does_item_exist) and does_item_exist(self.LOG_TAG) and callable(add_text):
            add_text(
                line,
                parent=self.LOG_TAG,
                color=colors[level],
                wrap=0,
            )
            self._scroll_log_to_bottom()
        else:
            self.dpg.set_value(self.LOG_TAG, line)

    def _update_buttons(self) -> None:
        status = self.dpg.get_value(self.STATUS_TAG)
        self.dpg.configure_item(
            self.RUN_TAG,
            enabled=status in {self.STATUS_WAITING, self.STATUS_STOPPED},
        )
        self.dpg.configure_item(
            self.STOP_TAG,
            enabled=status in {self.STATUS_STARTING, self.STATUS_RUNNING},
        )

    def _start_worker(
        self,
        target: Callable[..., None],
        worker_kwargs: dict[str, Any],
        started_message: str,
    ) -> bool:
        """Start one background worker with fresh synchronization objects."""
        self._reap_worker()
        if self.worker is not None:
            if self.worker_finished:
                self._append_log("Previous monitoring worker is still shutting down; please retry.", "info")
                self._set_status(self.STATUS_STOPPED)
                return False
            self._append_log("Monitoring is already running.", "error")
            if self.worker.is_alive():
                self._set_status(self.STATUS_RUNNING)
            return False

        event_queue: EventQueue = Queue()
        stop_event = Event()
        worker = Thread(
            target=target,
            kwargs={
                **worker_kwargs,
                "event_queue": event_queue,
                "stop_event": stop_event,
            },
            daemon=True,
        )

        self.worker_finished = False
        self._set_status(self.STATUS_STARTING)
        try:
            worker.start()
        except RuntimeError as error:
            self._append_log(f"Failed to start background worker: {error}", "error")
            self._set_status(self.STATUS_WAITING)
            return False

        self.event_queue = event_queue
        self.stop_event = stop_event
        self.worker = worker
        self._append_log(started_message, "success")
        return True

    def _request_worker_stop(self) -> bool:
        """Signal the active worker without blocking the render thread."""
        if self.worker is None or not self.worker.is_alive() or self.stop_event is None:
            return False
        self.stop_event.set()
        return True

    def _reap_worker(self) -> None:
        """Forget a completed worker after all of its events are consumed."""
        if self.worker is None or self.worker.is_alive():
            return
        if self.event_queue is not None and not self.event_queue.empty():
            return
        stopped_unexpectedly = not self.worker_finished
        self.worker = None
        self.event_queue = None
        self.stop_event = None
        if stopped_unexpectedly:
            self.worker_finished = True
            self._set_status(self.STATUS_STOPPED)
            self._append_log("Monitoring worker stopped unexpectedly.", "error")

    def _shutdown_worker(self) -> None:
        """Stop and briefly join the worker while the application closes."""
        if self.worker is None:
            return
        if self.worker.is_alive() and self.stop_event is not None:
            self.stop_event.set()
        self.worker.join(timeout=5)
        if not self.worker.is_alive():
            self.worker = None
            self.event_queue = None
            self.stop_event = None

    def _read_run_configuration(self) -> RunConfiguration | None:
        """Validate the selected controls and return one run configuration."""
        if not COMMUNITY_MODEL_PATH.is_file():
            self._append_log(f"Community AI model not found: {COMMUNITY_MODEL_PATH}", "error")
            self._set_status(self.STATUS_WAITING)
            return None

        monitor_label = self.dpg.get_value(self.MONITOR_TAG)
        monitor_ids = self.dpg.get_item_user_data(self.MONITOR_TAG) or {}
        monitor_id = monitor_ids.get(monitor_label)
        if monitor_id is None:
            self._append_log("Select an available monitor before starting.", "error")
            self._set_status(self.STATUS_WAITING)
            return None

        return RunConfiguration(
            monitor_id=monitor_id,
            threshold=int(self.dpg.get_value(self.THRESHOLD_TAG)),
            frontier_threshold=int(self.dpg.get_value(self.FRONTIER_THRESHOLD_TAG)),
            frontier_delay=int(self.dpg.get_value(self.FRONTIER_DELAY_TAG)),
        )

    def _run(self, *_args: Any) -> None:
        if self.dpg.get_value(self.STATUS_TAG) not in {self.STATUS_WAITING, self.STATUS_STOPPED}:
            return

        configuration = self._read_run_configuration()
        if configuration is None:
            return

        if self._start_worker(
            run_monitoring_with_callbacks,
            {"configuration": configuration},
            "Monitoring started.",
        ):
            self.skill_check_count = 0
            self.dpg.set_value(self.SKILL_CHECK_COUNT_TAG, "0")

    def _stop(self, *_args: Any) -> None:
        status = self.dpg.get_value(self.STATUS_TAG)
        if status == self.STATUS_STOPPING:
            return
        if status not in {self.STATUS_STARTING, self.STATUS_RUNNING}:
            return

        self._set_status(self.STATUS_STOPPING)
        if self._request_worker_stop():
            self._append_log("Monitoring stop requested.", "info")
        else:
            self._append_log("Monitoring worker has already stopped; waiting for final events.", "info")

    def _update_fps_display(self, fps: Any) -> None:
        try:
            fps_value = float(fps)
        except (TypeError, ValueError):
            fps_value = 0.0
        if not math.isfinite(fps_value) or fps_value < 0:
            fps_value = 0.0

        fps_color = next(
            (
                color
                for limit, color in (
                    (40, self.NEGATIVE),
                    (60, self.ORANGE),
                    (80, self.WARNING),
                    (110, self.POSITIVE),
                )
                if fps_value < limit
            ),
            self.INFO,
        )
        self.dpg.set_value(self.FPS_TAG, f"{fps_value:.0f} FPS")
        self.dpg.set_value(self.FPS_BAR_TAG, min(fps_value, 120) / 120)
        self.dpg.configure_item(self.FPS_BAR_TAG, overlay=f"{fps_value:.0f} FPS")
        self.dpg.configure_item(self.FPS_TAG, color=fps_color)
        self.dpg.configure_item(self.FPS_MARK_TAG, color=fps_color)

        fps_theme = self.progress_themes.get(fps_color)
        bind_item_theme = getattr(self.dpg, "bind_item_theme", None)
        if fps_theme is not None and callable(bind_item_theme):
            bind_item_theme(self.FPS_BAR_TAG, fps_theme)

    def _prediction_color(self, description: Any) -> tuple[int, int, int, int]:
        description = str(description).lower()
        if description == "none":
            return self.MUTED
        if "out" in description:
            return self.NEGATIVE
        if "great" in description or "frontier" in description:
            return self.POSITIVE
        return self.MUTED

    def _update_inference_score_display(self, score_data: Any) -> None:
        scores = score_data.get("scores") if isinstance(score_data, dict) else None
        values = get_inference_score_values(scores) if scores is not None else {key: 0.0 for key, _label, _class_ids in INFERENCE_SCORE_GROUPS}

        bind_item_theme = getattr(self.dpg, "bind_item_theme", None)
        for key, _label, class_ids in INFERENCE_SCORE_GROUPS:
            value = values[key]
            color = self.MUTED
            if scores is not None:
                best_class_id = max(class_ids, key=lambda candidate: _score_at(scores, candidate))
                color = self._prediction_color(prediction_dict[best_class_id]["desc"])
            bar_tag = f"inference_score_{key}"
            self.dpg.set_value(bar_tag, value)
            self.dpg.configure_item(bar_tag, overlay=f"{value:.0%}")
            self.dpg.configure_item(f"inference_score_label_{key}", color=color)
            score_theme = self.progress_themes.get(color)
            if score_theme is not None and callable(bind_item_theme):
                bind_item_theme(bar_tag, score_theme)

    def _handle_startup(self, event: WorkerEvent) -> None:
        config = event.get("data")
        if not isinstance(config, dict):
            return
        if self.dpg.get_value(self.STATUS_TAG) not in {self.STATUS_RUNNING, self.STATUS_STOPPING, self.STATUS_STOPPED}:
            self._set_status(self.STATUS_STARTING)

        device = config.get("device", self.EMPTY_VALUE)
        if config.get("threads") is not None:
            device = f"{device} · {config['threads']} threads"
        monitoring = str(config.get("monitoring", self.EMPTY_VALUE)).upper()
        capture = f"{monitoring} · Monitor {config.get('monitor_id', self.EMPTY_VALUE)}"
        thresholds = (
            f"Default: {config.get('threshold', self.EMPTY_VALUE)}% · "
            f"Gens: {config.get('frontier_threshold', self.EMPTY_VALUE)}% + "
            f"{config.get('frontier_delay', self.EMPTY_VALUE)} ms"
        )
        configuration = {
            self.CONFIG_MODEL_TAG: str(config.get("model_name", self.EMPTY_VALUE)),
            self.CONFIG_DEVICE_TAG: str(device),
            self.CONFIG_CAPTURE_TAG: capture,
            self.CONFIG_THRESH_TAG: thresholds,
        }
        for tag, value in configuration.items():
            self.dpg.set_value(tag, value)
            self.dpg.configure_item(tag, color=self.TEXT)

    def _handle_prediction(self, event: WorkerEvent) -> None:
        data = event.get("data")
        if not isinstance(data, dict):
            return
        scores = data.get("scores")
        if not isinstance(scores, (list, tuple)) or len(scores) < 10:
            return
        prediction = str(data.get("description", data.get("desc", self.EMPTY_VALUE)))
        self.dpg.set_value(self.DETECTION_TAG, prediction)
        self.dpg.configure_item(self.DETECTION_TAG, color=self._prediction_color(prediction))
        self._update_inference_score_display(data)
        self._set_status(self.STATUS_RUNNING)

    def _handle_hit(self, event: WorkerEvent) -> None:
        hit = event.get("data")
        if not isinstance(hit, dict):
            return
        desc = str(hit.get("desc", "Skill check"))
        confidence = hit.get("confidence", 0)
        self.skill_check_count += 1
        self.dpg.set_value(self.SKILL_CHECK_COUNT_TAG, str(self.skill_check_count))
        self._append_log(f"{desc.upper()} hit at {confidence}% confidence", "info")

    def _handle_finished(self, _event: WorkerEvent) -> None:
        was_stopping = self.dpg.get_value(self.STATUS_TAG) == self.STATUS_STOPPING
        self.worker_finished = True
        self._set_status(self.STATUS_STOPPED)
        if was_stopping:
            self._append_log("Monitoring stopped.", "success")

    def _dispatch_event(self, event: WorkerEvent) -> None:
        event_type = event.get("type")
        if not isinstance(event_type, str):
            return
        if self.worker_finished and event_type in self.TELEMETRY_EVENT_TYPES:
            return
        if event_type == "startup":
            self._handle_startup(event)
        elif event_type == "running":
            self._set_status(self.STATUS_RUNNING)
        elif event_type == "prediction":
            self._handle_prediction(event)
        elif event_type == "fps":
            self._update_fps_display(event.get("data"))
        elif event_type == "hit":
            self._handle_hit(event)
        elif event_type == "finished":
            self._handle_finished(event)
        elif event_type == "log":
            self._append_log(str(event.get("data")), str(event.get("level", "info")))

    def _drain_events(self) -> list[WorkerEvent]:
        """Read and coalesce a bounded batch of worker events."""
        if self.event_queue is None:
            return []

        drained: list[WorkerEvent] = []
        pending_telemetry: dict[str, WorkerEvent] = {}

        def flush_telemetry() -> None:
            for event_type in ("prediction", "fps"):
                event = pending_telemetry.pop(event_type, None)
                if event is not None:
                    drained.append(event)

        for _event_count in range(self.MAX_EVENTS_PER_POLL):
            try:
                event = self.event_queue.get_nowait()
            except (Empty, EOFError, OSError):
                break
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if not isinstance(event_type, str):
                continue
            if event_type in self.TELEMETRY_EVENT_TYPES:
                pending_telemetry[event_type] = event
            elif event_type in self.ORDERED_EVENT_TYPES:
                flush_telemetry()
                drained.append(event)

        flush_telemetry()
        return drained

    def _poll_events(self, *_args: Any) -> None:
        for event in self._drain_events():
            self._dispatch_event(event)
        self._reap_worker()

    def _add_help_marker(self, message: str) -> None:
        dpg = self.dpg
        help_button = dpg.add_button(label="?", small=True)
        self._style(help_button, "help")
        with dpg.tooltip(help_button):
            dpg.add_text(message, wrap=360)

    def _add_text(self, text: str, role: str | None = None, **kwargs: Any) -> Any:
        """Add a text item; ``role`` registers it for the bold strong / heading / display fonts."""
        item = self.dpg.add_text(text, **kwargs)
        if role is not None:
            self.role_items[role].append(item)
        return item

    def _add_rail(self, tab_width: int = 44) -> None:
        """Thin HUD-style underline: a short bright tab followed by a dim line that fills the width."""
        dpg = self.dpg
        with dpg.group(horizontal=True, horizontal_spacing=0):
            tab = dpg.add_button(label="", width=tab_width, height=3, enabled=False)
            line = dpg.add_button(label="", width=-1, height=3, enabled=False)
        self._style(tab, "rail_bright")
        self._style(line, "rail_dim")

    def _add_section_title(self, text: str) -> None:
        with self.dpg.group(horizontal=True, horizontal_spacing=7):
            self._add_text("◆", "strong", color=self.ACCENT)
            self._add_text(text, "strong", color=self.TEXT)
        self._add_rail()

    def _add_labeled_slider(
        self,
        label: str,
        limits: tuple[int, int],
        default: int,
        tag: str,
        help_text: str,
    ) -> None:
        minimum, maximum = limits
        with self.dpg.group(horizontal=True):
            self.dpg.add_text(label, color=self.TEXT)
            self._add_help_marker(help_text)
        self.dpg.add_slider_int(
            min_value=minimum,
            max_value=maximum,
            default_value=default,
            tag=tag,
            width=-1,
        )

    def _build_header(self) -> None:
        dpg = self.dpg
        with dpg.child_window(tag=self.HEADER_TAG, width=-1, auto_resize_y=True, border=True, horizontal_scrollbar=False):
            with dpg.group(horizontal=True, horizontal_spacing=9):
                self._add_text("◢", "heading", color=self.ACCENT)
                self._add_text("DBD AUTO SKILL CHECK — COMMUNITY", "heading", tag=self.TITLE_TAG, color=self.TEXT, wrap=0)
            self._add_rail(tab_width=64)
            with dpg.collapsing_header(label="Community information & license", default_open=False):
                dpg.add_text(
                    "Reuse or redistribution: credit https://github.com/Manuteaa/dbd_autoSkillCheck. "
                    "Follow GPLv3: include the license and provide corresponding source.",
                    tag=self.HEADER_LICENSE_TEXT_TAG,
                    color=self.MUTED,
                    wrap=420,
                )
        self._style(self.HEADER_TAG, "header")

    def _build_capture_settings(self) -> None:
        dpg = self.dpg
        dpg.add_spacer(height=6)
        self._add_section_title("SCREEN CAPTURE")
        with dpg.group(horizontal=True):
            self._add_text("SOURCE", "strong", color=self.MUTED)
            self._add_help_marker(
                "The community edition uses MSS to capture a 224x224 crop from the center of the selected display."
                "\n\nOn Linux, launch the application from an X11/XWayland session where DISPLAY is set.",
            )
            dpg.add_spacer(width=6)
            self._add_text(COMMUNITY_CAPTURE_NAME, "strong", color=self.TEXT)
        self._add_text("DISPLAY", "strong", color=self.MUTED)
        dpg.add_combo(items=[], tag=self.MONITOR_TAG, width=-1)

    def _build_additional_ai_settings(self) -> None:
        dpg = self.dpg
        dpg.add_spacer(height=6)
        with dpg.collapsing_header(label="Additional AI Settings", default_open=False):
            for label, minimum, maximum, default, tag, help_text in (
                (
                    "Default SC confidence threshold (%)",
                    51,
                    100,
                    self._preset("thresh", DEFAULT_THRESHOLD),
                    self.THRESHOLD_TAG,
                    "Minimum confidence required before automatically hitting skill checks other than repair/heal. "
                    "Increase the value if the AI activates too early.",
                ),
                (
                    "Repair-heal SC confidence threshold (%)",
                    51,
                    100,
                    self._preset("thresh_frontier", DEFAULT_FRONTIER_THRESHOLD),
                    self.FRONTIER_THRESHOLD_TAG,
                    "Minimum confidence required before automatically hitting repair/heal skill checks. Increase the value if the AI activates too early.",
                ),
                (
                    "Repair-heal SC hit delay (ms)",
                    0,
                    50,
                    self._preset("delay_frontier", DEFAULT_FRONTIER_HIT_DELAY),
                    self.FRONTIER_DELAY_TAG,
                    "Forced wait time before hitting SPACE (repair/heal skill checks only). Increase the value if the AI hits too early.",
                ),
            ):
                self._add_labeled_slider(label, (minimum, maximum), default, tag, help_text)

    def _build_action_buttons(self) -> None:
        dpg = self.dpg
        # Keep divider, spacing, and buttons together so whole section can move
        # above LIVE MONITOR in compact stacked layout.
        with dpg.group(tag=self.ACTION_SECTION_TAG, horizontal=False):
            dpg.add_spacer(height=6)
            dpg.add_separator()
            dpg.add_spacer(height=2)
            with dpg.group(tag=self.ACTION_GROUP_TAG, horizontal=True):
                for label, tag, callback, enabled, style in (
                    ("▶  START MONITORING", self.RUN_TAG, self._run, True, "primary_button"),
                    ("■  STOP MONITORING", self.STOP_TAG, self._stop, False, "secondary_button"),
                ):
                    dpg.add_button(
                        label=label,
                        tag=tag,
                        callback=callback,
                        width=-1,
                        height=self.ACTION_BUTTON_HEIGHT,
                        enabled=enabled,
                    )
                    self._style(tag, style)
        self.scaled_heights[self.RUN_TAG] = self.ACTION_BUTTON_HEIGHT
        self.scaled_heights[self.STOP_TAG] = self.ACTION_BUTTON_HEIGHT

    def _build_settings_panel(self) -> None:
        dpg = self.dpg
        with dpg.child_window(
            tag=self.SETTINGS_PANEL_TAG,
            width=self.SETTINGS_MAX_WIDTH,
            height=-1,
            border=True,
            horizontal_scrollbar=False,
        ):
            with dpg.group(horizontal=True):
                self._add_text("SETTINGS", "heading", tag=self.SETUP_TITLE_TAG, color=self.TEXT)
                self._add_help_marker(
                    "Default values are loaded from settings.json when the application starts.\n"
                    "Edit that file before launch to keep your preferred settings. ",
                )
            self._add_rail(tab_width=56)
            with dpg.collapsing_header(label="Community edition details", default_open=False):
                dpg.add_text(
                    "This is the limited community version of DBD Auto Skill Check. "
                    "Complete version available through Discord after accepting fair-use agreement.",
                    tag=self.COMMUNITY_DETAILS_TEXT_TAG,
                    color=self.MUTED,
                    wrap=360,
                )
                dpg.add_text(
                    "https://discord.gg/3mewehHHpZ",
                    tag=self.COMMUNITY_LINK_TEXT_TAG,
                    color=self.ACCENT_HOVER,
                    wrap=360,
                )
            dpg.add_spacer(height=4)
            self._build_capture_settings()
            self._build_additional_ai_settings()
            self._build_action_buttons()
        self._style(self.SETTINGS_PANEL_TAG, "panel")

    def _build_inference_card(self) -> None:
        dpg = self.dpg
        with dpg.child_window(
            tag=self.INFERENCE_PANEL_TAG,
            width=-1,
            height=0,
            auto_resize_y=True,
            border=True,
            horizontal_scrollbar=False,
        ):
            with dpg.group(horizontal=True, horizontal_spacing=7):
                self._add_text("◆", "strong", color=self.ACCENT)
                self._add_text("AI INFERENCE PERFORMANCE", "strong", tag=self.SIGNAL_LOGO_TAG, color=self.TEXT, wrap=0)
            self._add_rail()
            with dpg.table(
                header_row=False,
                policy=dpg.mvTable_SizingStretchProp,
                width=-1,
                borders_innerH=False,
                borders_outerH=False,
                borders_innerV=False,
                borders_outerV=False,
            ):
                dpg.add_table_column()
                dpg.add_table_column()
                with dpg.table_row():
                    with dpg.group(horizontal=True, horizontal_spacing=7):
                        dpg.add_text("●", tag=self.STATUS_DOT_TAG, color=self.WARNING)
                        self._add_text("STATUS", "strong", color=self.MUTED)
                    self._add_text("SKILL COUNT", "strong", tag="skill_check_count_label", color=self.ACCENT_HOVER)
                with dpg.table_row():
                    self._add_text(self.STATUS_WAITING, "display", tag=self.STATUS_TAG, color=self.WARNING)
                    self._add_text("0", "display", tag=self.SKILL_CHECK_COUNT_TAG, color=self.TEXT)
            with dpg.group(horizontal=True, horizontal_spacing=7):
                dpg.add_text("↯", tag=self.FPS_MARK_TAG, color=self.INFO)
                self._add_text("CURRENT INFERENCE FPS", "strong", color=self.MUTED)
            self._add_text("0 FPS", "display", tag=self.FPS_TAG, color=self.INFO)
            dpg.add_progress_bar(default_value=0.0, tag=self.FPS_BAR_TAG, width=-1, height=self.FPS_BAR_HEIGHT, overlay="0 FPS")
        self._style(self.INFERENCE_PANEL_TAG, "card")
        self.scaled_heights[self.FPS_BAR_TAG] = self.FPS_BAR_HEIGHT

    def _build_detection_card(self) -> None:
        dpg = self.dpg
        with dpg.child_window(
            tag=self.DETECTION_PANEL_TAG,
            width=-1,
            height=0,
            auto_resize_y=True,
            border=True,
            horizontal_scrollbar=False,
        ):
            with dpg.group(horizontal=True, horizontal_spacing=7):
                self._add_text("◆", "strong", color=self.ACCENT)
                self._add_text("AI DETECTION", "strong", tag=self.DETECTION_LOGO_TAG, color=self.TEXT)
                self._add_help_marker(
                    "For best results, run the game and AI at 120 FPS or higher and set your monitor to at least 120 Hz.\n\n"
                    "Fine-tune the AI settings if needed. If you have trouble, visit the Discord server for help.",
                )
            self._add_rail()
            dpg.add_spacer(height=2)
            self._add_text(self.EMPTY_VALUE, "heading", tag=self.DETECTION_TAG, color=self.MUTED, wrap=0)
            dpg.add_spacer(height=2)
            with dpg.table(
                header_row=False,
                policy=dpg.mvTable_SizingStretchProp,
                width=-1,
                inner_width=8,
                borders_innerH=False,
                borders_outerH=False,
                borders_innerV=False,
                borders_outerV=False,
            ):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=120)
                dpg.add_table_column()
                for key, label, _class_ids in INFERENCE_SCORE_GROUPS:
                    with dpg.table_row():
                        self._add_text(label, "strong", tag=f"inference_score_label_{key}", color=self.MUTED)
                        dpg.add_progress_bar(
                            default_value=0.0,
                            tag=f"inference_score_{key}",
                            width=-1,
                            height=self.SCORE_BAR_HEIGHT,
                            overlay="0%",
                        )
                    self.scaled_heights[f"inference_score_{key}"] = self.SCORE_BAR_HEIGHT
        self._style(self.DETECTION_PANEL_TAG, "card")

    def _build_configuration_card(self) -> None:
        dpg = self.dpg
        self._add_section_title("ACTIVE CONFIGURATION")
        with (
            dpg.child_window(tag=self.CONFIG_PANEL_TAG, width=-1, auto_resize_y=True, border=True, horizontal_scrollbar=False),
            dpg.table(
                header_row=False,
                policy=dpg.mvTable_SizingStretchSame,
                width=-1,
                inner_width=8,
                borders_innerH=False,
                borders_outerH=False,
                borders_innerV=False,
                borders_outerV=False,
            ),
        ):
            dpg.add_table_column()
            dpg.add_table_column()
            for row in (
                (("◈ MODEL", self.CONFIG_MODEL_TAG), ("⚙ DEVICE", self.CONFIG_DEVICE_TAG)),
                (("▣ CAPTURE", self.CONFIG_CAPTURE_TAG), ("◌ AI THRESHOLDS", self.CONFIG_THRESH_TAG)),
            ):
                with dpg.table_row():
                    for title, tag in row:
                        with dpg.group():
                            self._add_text(title, "strong", color=self.ACCENT_HOVER)
                            self._add_text(self.EMPTY_VALUE, "strong", tag=tag, color=self.TEXT, wrap=0)
        self._style(self.CONFIG_PANEL_TAG, "card")

    def _build_activity_log(self) -> None:
        self._add_section_title("RECENT ACTIVITIES")
        with self.dpg.child_window(tag=self.LOG_TAG, height=-1, border=True, horizontal_scrollbar=False):
            pass
        self._style(self.LOG_TAG, "log")

    def _build_live_panel(self) -> None:
        dpg = self.dpg
        with dpg.child_window(tag=self.LIVE_PANEL_TAG, width=-1, height=-1, border=True, horizontal_scrollbar=False):
            self._add_text("LIVE MONITOR", "heading", tag=self.LIVE_TITLE_TAG, color=self.TEXT)
            self._add_rail(tab_width=56)
            with dpg.group(tag=self.CARDS_GROUP_TAG, horizontal=True):
                self._build_inference_card()
                self._build_detection_card()
            dpg.add_spacer(height=4)
            self._build_configuration_card()
            dpg.add_spacer(height=2)
            self._build_activity_log()
        self._style(self.LIVE_PANEL_TAG, "panel")

    def _bind_heading_fonts(self) -> None:
        fonts = self.fonts.get(self.ui_scale)
        if not fonts:
            return
        for role, items in self.role_items.items():
            font = fonts.get(role)
            if font is None:
                continue
            for item in items:
                self.dpg.bind_item_font(item, font)

    def _build_window(self) -> None:
        with self.dpg.window(tag=self.WINDOW_TAG, label="DBD ASC", no_collapse=True):
            self._build_header()
            with self.dpg.group(tag=self.PANELS_TAG, horizontal=True):
                self._build_settings_panel()
                self._build_live_panel()

        self._bind_heading_fonts()
        self._set_monitor_choices(preferred_monitor_id=self._preset("monitor_id"))
        self._update_fps_display(0)
        self._update_inference_score_display(None)

    # ------------------------------------------------------------------
    # Adaptive layout
    # ------------------------------------------------------------------
    def _set_ui_scale(self, scale: float) -> None:
        """Switch to one of the pre-rasterised font tiers and rescale fixed-height widgets."""
        dpg = self.dpg
        self.ui_scale = scale
        fonts = self.fonts.get(scale)
        if fonts:
            dpg.bind_font(fonts["ui"])
            self._bind_heading_fonts()
        else:
            dpg.set_global_font_scale(scale)
        for tag, base_height in self.scaled_heights.items():
            dpg.configure_item(tag, height=round(base_height * scale))

    def _metric_card_height(self, scale: float) -> int:
        return round(self.METRIC_CARD_CONTENT_HEIGHT * scale + self.METRIC_CARD_FIXED_HEIGHT)

    def _apply_layout(self, width: int, height: int) -> None:
        """Pick one- or two-column layout and size every panel for the current viewport."""
        dpg = self.dpg
        stacked = width < self.STACK_BREAKPOINT
        basis = width / 900 if stacked else min(width / DEFAULT_VIEWPORT_WIDTH, height / 880)
        scale = min(self.FONT_TIERS, key=lambda tier: abs(tier - basis))
        if scale != self.ui_scale or self.viewport_size is None:
            self._set_ui_scale(scale)

        dpg.configure_item(self.PANELS_TAG, horizontal=not stacked)
        # On compact windows, put live monitoring first so key information stays visible.
        if stacked:
            dpg.move_item(self.LIVE_PANEL_TAG, parent=self.PANELS_TAG, before=self.SETTINGS_PANEL_TAG)
            # Put monitoring controls immediately above LIVE MONITOR heading.
            dpg.move_item(self.ACTION_SECTION_TAG, parent=self.LIVE_PANEL_TAG, before=self.LIVE_TITLE_TAG)
        else:
            dpg.move_item(self.SETTINGS_PANEL_TAG, parent=self.PANELS_TAG, before=self.LIVE_PANEL_TAG)
            # Restore controls to bottom of Settings in desktop two-column layout.
            dpg.move_item(self.ACTION_SECTION_TAG, parent=self.SETTINGS_PANEL_TAG)
        log_height = round(self.LOG_MIN_HEIGHT * scale)
        if stacked:
            for panel in (self.SETTINGS_PANEL_TAG, self.LIVE_PANEL_TAG):
                dpg.configure_item(panel, width=-1, auto_resize_y=True, height=0)
            dpg.configure_item(self.LOG_TAG, height=max(110, round(log_height * 0.65)))
            live_inner = width - 4 * self.PADDING
        else:
            settings_width = round(min(max(width * 0.36, self.SETTINGS_MIN_WIDTH * scale), self.SETTINGS_MAX_WIDTH * scale))
            dpg.configure_item(self.SETTINGS_PANEL_TAG, width=settings_width, auto_resize_y=False, height=-1)
            dpg.configure_item(self.LIVE_PANEL_TAG, width=-1, auto_resize_y=False, height=-1)
            # Never use "-1" (remaining space) here: the cards above can use all of it and
            # leave the log collapsed. Start from a guaranteed minimum; _fit_log_height()
            # grows it to fill spare room on tall windows.
            dpg.configure_item(self.LOG_TAG, height=log_height)
            self._fitted_log_height = None
            live_inner = width - settings_width - 4 * self.PADDING - self.SPACING

        side_by_side = live_inner >= (2 * self.METRIC_CARD_MIN_WIDTH + self.SPACING) * scale
        dpg.configure_item(self.CARDS_GROUP_TAG, horizontal=side_by_side)
        if side_by_side:
            # Equal fixed heights keep card bottoms aligned in wide layouts.
            # Sized to the tallest card (AI DETECTION) so no empty space is left at the
            # bottom and no inner vertical scrollbar appears.
            card_height = max(self._metric_card_height(scale), round(330 * scale))
            dpg.configure_item(
                self.INFERENCE_PANEL_TAG,
                width=round((live_inner - self.SPACING) / 2),
                auto_resize_y=False,
                height=card_height,
            )
            dpg.configure_item(
                self.DETECTION_PANEL_TAG,
                width=-1,
                auto_resize_y=False,
                height=card_height,
            )
        else:
            # Narrow layouts retain natural content height and stack vertically.
            for card in (self.INFERENCE_PANEL_TAG, self.DETECTION_PANEL_TAG):
                dpg.configure_item(card, width=-1, auto_resize_y=True, height=0)
        self.action_layout = None  # re-fit the Start/Stop buttons against the new panel width

    def _fit_action_buttons(self) -> None:
        """Place Start/Stop side by side when they fit, otherwise stack them full-width."""
        dpg = self.dpg
        available = dpg.get_item_rect_size(self.MONITOR_TAG)[0]
        if available <= 0:
            return
        side_by_side = (available - self.SPACING) / 2 >= self.ACTION_BUTTON_WIDTH_MIN * self.ui_scale
        layout = (round(available), side_by_side)
        if layout == self.action_layout:
            return
        self.action_layout = layout
        dpg.configure_item(self.ACTION_GROUP_TAG, horizontal=side_by_side)
        if side_by_side:
            half = int((available - self.SPACING) // 2)
            dpg.configure_item(self.RUN_TAG, width=half)
            dpg.configure_item(self.STOP_TAG, width=-1)
        else:
            dpg.configure_item(self.RUN_TAG, width=-1)
            dpg.configure_item(self.STOP_TAG, width=-1)

    def _fit_explanatory_text(self) -> None:
        """Wrap long explanatory copy to available panel width, never to 1 pixel."""
        dpg = self.dpg
        for parent_tag, text_tags, inset in (
            (self.HEADER_TAG, (self.HEADER_LICENSE_TEXT_TAG,), 36),
            (
                self.SETTINGS_PANEL_TAG,
                (self.COMMUNITY_DETAILS_TEXT_TAG, self.COMMUNITY_LINK_TEXT_TAG),
                34,
            ),
        ):
            try:
                parent_width = dpg.get_item_rect_size(parent_tag)[0]
            except (SystemError, RuntimeError, TypeError):
                continue
            if parent_width <= 0:
                continue
            wrap_width = max(140, int(parent_width - inset))
            for text_tag in text_tags:
                if dpg.does_item_exist(text_tag):
                    dpg.configure_item(text_tag, wrap=wrap_width)

    def _fit_log_height(self) -> None:
        """Wide layout: stretch the activity log to the bottom of the Live Monitor panel when there is
        spare room; otherwise keep its minimum height (the panel then scrolls, so the log stays readable)."""
        dpg = self.dpg
        if self.viewport_size is None or self.viewport_size[0] < self.STACK_BREAKPOINT:
            return
        try:
            panel_height = dpg.get_item_rect_size(self.LIVE_PANEL_TAG)[1]
            log_top = dpg.get_item_pos(self.LOG_TAG)[1]  # relative to the panel's content, independent of scrolling
        except (KeyError, SystemError, RuntimeError, TypeError):
            return
        if panel_height <= 0 or log_top <= 0:
            return
        minimum = round(self.LOG_MIN_HEIGHT * self.ui_scale)
        spare = panel_height - log_top - (self.PADDING - 3) - 1  # leave the panel's bottom padding
        target = max(minimum, int(spare))
        if self._fitted_log_height is None or abs(target - self._fitted_log_height) > 1:
            self._fitted_log_height = target
            dpg.configure_item(self.LOG_TAG, height=target)

    def _update_layout(self) -> None:
        dpg = self.dpg
        size = (dpg.get_viewport_client_width(), dpg.get_viewport_client_height())
        if size != self.viewport_size and size[0] > 0 and size[1] > 0:
            self._apply_layout(*size)
            self.viewport_size = size
            self._fit_explanatory_text()
            now = monotonic()
            if now - self._last_viewport_save_at >= 1.0:
                self._save_viewport_size()
                self._last_viewport_save_at = now
        self._fit_action_buttons()
        self._fit_log_height()

    def _save_viewport_size(self) -> None:
        """Persist last window dimensions without overwriting other settings."""
        try:
            width = int(self.dpg.get_viewport_width())
            height = int(self.dpg.get_viewport_height())
            if width < MIN_VIEWPORT_WIDTH or height < MIN_VIEWPORT_HEIGHT:
                return
            config_path = Path("settings.json")
            try:
                with config_path.open(encoding="utf-8") as config_file:
                    saved = json.load(config_file)
                if not isinstance(saved, dict):
                    saved = {}
            except (json.JSONDecodeError, OSError):
                saved = {}
            saved["viewport_width"] = width
            saved["viewport_height"] = height
            temporary_path = config_path.with_suffix(".json.tmp")
            temporary_path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
            temporary_path.replace(config_path)
            self._last_saved_viewport_size = (width, height)
        except (OSError, TypeError, ValueError):
            # Window-size persistence must never prevent application shutdown.
            pass

    def _icon_kwargs(self) -> dict[str, str]:
        """Window/taskbar icon (Dead by Daylight logo) if dbd_icon.ico sits next to this file."""
        icon_path = Path(__file__).resolve().with_name("dbd_icon.ico")
        if icon_path.is_file():
            return {"small_icon": str(icon_path), "large_icon": str(icon_path)}
        return {}

    def _apply_dark_title_bar(self) -> None:
        """Make the native Windows title bar black instead of white (cosmetic; failures are ignored)."""
        if sys.platform != "win32":
            return
        try:
            user32 = ctypes.windll.user32
            dwmapi = ctypes.windll.dwmapi
            hwnd = user32.FindWindowW(None, "DBD ASC COMMUNITY")
            if not hwnd:
                return

            def set_attribute(attribute: int, value: int) -> None:
                data = ctypes.c_int(value)
                dwmapi.DwmSetWindowAttribute(hwnd, attribute, ctypes.byref(data), ctypes.sizeof(data))

            def colorref(color: tuple[int, int, int, int]) -> int:
                return color[0] | (color[1] << 8) | (color[2] << 16)  # COLORREF is 0x00BBGGRR

            set_attribute(20, 1)  # DWMWA_USE_IMMERSIVE_DARK_MODE (Windows 10 20H1+ / 11)
            set_attribute(19, 1)  # same flag on older Windows 10 builds
            set_attribute(35, colorref(self.WINDOW_BG))  # DWMWA_CAPTION_COLOR (Windows 11)
            set_attribute(36, colorref(self.TEXT))  # DWMWA_TEXT_COLOR (Windows 11)
            set_attribute(34, colorref(self.WINDOW_BG))  # DWMWA_BORDER_COLOR (Windows 11)
        except (AttributeError, OSError):
            pass

    def run(self, *_args: Any) -> None:
        dpg = self.dpg
        dpg.create_context()
        try:
            self._build_fonts()
            self._build_theme()
            self._build_window()
            try:
                saved_width = int(self._preset("viewport_width", DEFAULT_VIEWPORT_WIDTH))
            except (TypeError, ValueError):
                saved_width = DEFAULT_VIEWPORT_WIDTH
            try:
                saved_height = int(self._preset("viewport_height", DEFAULT_VIEWPORT_HEIGHT))
            except (TypeError, ValueError):
                saved_height = DEFAULT_VIEWPORT_HEIGHT
            dpg.create_viewport(
                title="DBD ASC COMMUNITY",
                width=max(MIN_VIEWPORT_WIDTH, saved_width),
                height=max(MIN_VIEWPORT_HEIGHT, saved_height),
                min_width=MIN_VIEWPORT_WIDTH,
                min_height=MIN_VIEWPORT_HEIGHT,
                resizable=True,
                **self._icon_kwargs(),
            )
            dpg.setup_dearpygui()
            dpg.show_viewport()
            self._apply_dark_title_bar()
            dpg.set_primary_window(self.WINDOW_TAG, True)
            while dpg.is_dearpygui_running():
                self._update_layout()
                self._poll_events()
                dpg.render_dearpygui_frame()
        finally:
            self._save_viewport_size()
            self._shutdown_worker()
            dpg.destroy_context()


def main() -> None:
    """Start the Dear PyGui desktop application."""
    presets = load_settings()
    AutoSkillCheckApp(dearpygui, presets=presets).run()


if __name__ == "__main__":
    main()
