from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageEnhance, ImageTk
from tkinter import filedialog, messagebox, ttk

try:  # mss is opcional; usado para captura de tela
    import mss
except Exception:  # pragma: no cover - fallback simples
    mss = None

from .config import DEFAULT_TARGET_FPS, DEFAULT_VEHICLE_MODEL, DEFAULT_PLATE_MODEL, MODELS_DIR, VIDEO_SOURCES
from .data_structures import VehicleTrack
from .device import get_device
from .model_manager import ModelManager
from .pipeline import Logger, Pipeline

Region = Tuple[int, int, int, int]


@dataclass
class PlateInfo:
    image: Optional[np.ndarray]
    text: str
    plate_type: str
    yolo_confidence: float
    ocr_confidence: float
    timestamp: str
    vehicle_type: str = "Desconhecido"


class ScreenRegionSelector:
    def __init__(self, parent: tk.Tk, callback: Callable[[Optional[Region]], None]) -> None:
        self.parent = parent
        self.callback = callback
        self.windows: List[tk.Toplevel] = []
        self.start_x = 0
        self.start_y = 0
        self.canvas_start_x = 0
        self.canvas_start_y = 0
        self.is_selecting = False
        self.active_canvas: Optional[tk.Canvas] = None
        self.monitor_info: List[Dict[str, int]] = []
        self._create_overlay()

    def _get_all_monitors(self) -> List[Dict[str, int]]:
        monitors: List[Dict[str, int]] = []
        if mss is not None:
            try:
                with mss.mss() as sct:
                    for idx, mon in enumerate(sct.monitors[1:], start=1):
                        monitors.append(
                            {
                                "index": idx,
                                "left": mon["left"],
                                "top": mon["top"],
                                "width": mon["width"],
                                "height": mon["height"],
                            }
                        )
            except Exception:
                monitors = []
        if not monitors:
            monitors.append(
                {
                    "index": 1,
                    "left": 0,
                    "top": 0,
                    "width": self.parent.winfo_screenwidth(),
                    "height": self.parent.winfo_screenheight(),
                }
            )
        return monitors

    def _capture_screenshot(self, monitor: Dict[str, int]) -> Optional[Image.Image]:
        if mss is None:
            return None
        try:
            with mss.mss() as sct:
                mon = {
                    "left": monitor["left"],
                    "top": monitor["top"],
                    "width": monitor["width"],
                    "height": monitor["height"],
                }
                screenshot = sct.grab(mon)
                return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        except Exception:
            return None

    def _create_overlay(self) -> None:
        self.monitor_info = self._get_all_monitors()
        for mon in self.monitor_info:
            window = tk.Toplevel(self.parent)
            window.title(f"Selecionar Região - Monitor {mon['index']}")
            window.geometry(f"{mon['width']}x{mon['height']}+{mon['left']}+{mon['top']}")
            window.overrideredirect(True)
            window.attributes("-topmost", True)
            window.configure(bg="black")

            canvas = tk.Canvas(window, width=mon["width"], height=mon["height"], highlightthickness=0, bg="black")
            canvas.pack(fill="both", expand=True)

            screenshot = self._capture_screenshot(mon)
            if screenshot is not None:
                enhancer = ImageEnhance.Brightness(screenshot)
                darkened = enhancer.enhance(0.5)
                photo = ImageTk.PhotoImage(darkened)
                canvas.create_image(0, 0, anchor="nw", image=photo, tags="bg")
                canvas.photo = photo
                canvas.original_screenshot = screenshot

            canvas.monitor = mon  # type: ignore[attr-defined]
            canvas.create_text(
                mon["width"] // 2,
                50,
                text="Clique e arraste para selecionar\nESC para cancelar",
                font=("Arial", 18, "bold"),
                fill="white",
            )
            canvas.create_text(
                mon["width"] // 2,
                mon["height"] - 30,
                text=f"Monitor {mon['index']}: {mon['width']}x{mon['height']} @ ({mon['left']}, {mon['top']})",
                font=("Arial", 12),
                fill="#00ff00",
            )

            canvas.bind("<Button-1>", self._on_press)
            canvas.bind("<B1-Motion>", self._on_drag)
            canvas.bind("<ButtonRelease-1>", self._on_release)
            window.bind("<Escape>", self._on_cancel)

            self.windows.append(window)

    def _on_press(self, event: tk.Event) -> None:
        self.active_canvas = event.widget  # type: ignore[assignment]
        self.is_selecting = True
        canvas = self.active_canvas
        canvas.delete("selection")
        canvas.delete("size_label")
        self.start_x = event.x + canvas.monitor["left"]  # type: ignore[index]
        self.start_y = event.y + canvas.monitor["top"]
        self.canvas_start_x = event.x
        self.canvas_start_y = event.y

    def _on_drag(self, event: tk.Event) -> None:
        if not self.is_selecting or not self.active_canvas:
            return
        canvas = self.active_canvas
        canvas.delete("selection")
        canvas.delete("size_label")
        x1, y1 = self.canvas_start_x, self.canvas_start_y
        x2, y2 = event.x, event.y
        canvas.create_rectangle(x1, y1, x2, y2, outline="#00ff00", width=3, tags="selection")
        canvas.create_rectangle(x1, y1, x2, y2, outline="", fill="white", stipple="gray25", tags="selection")
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        canvas.create_text(
            mid_x,
            mid_y,
            text=f"{width} x {height}",
            font=("Arial", 14, "bold"),
            fill="#00ff00",
            tags="size_label",
        )

    def _on_release(self, event: tk.Event) -> None:
        if not self.is_selecting or not self.active_canvas:
            return
        canvas = self.active_canvas
        self.is_selecting = False
        abs_x1 = min(self.start_x, event.x + canvas.monitor["left"])  # type: ignore[index]
        abs_y1 = min(self.start_y, event.y + canvas.monitor["top"])
        abs_x2 = max(self.start_x, event.x + canvas.monitor["left"])
        abs_y2 = max(self.start_y, event.y + canvas.monitor["top"])
        width = abs_x2 - abs_x1
        height = abs_y2 - abs_y1
        if width > 50 and height > 50:
            region: Region = (abs_x1, abs_y1, width, height)
            self._close_all()
            self.callback(region)
            return
        canvas.delete("selection")
        canvas.delete("size_label")
        canvas.create_text(
            event.x,
            event.y - 30,
            text="Mínimo 50x50",
            font=("Arial", 12, "bold"),
            fill="red",
            tags="warning",
        )
        canvas.after(1500, lambda: canvas.delete("warning"))

    def _on_cancel(self, _event: Optional[tk.Event] = None) -> None:
        self._close_all()
        self.callback(None)

    def _close_all(self) -> None:
        for window in list(self.windows):
            try:
                window.destroy()
            except Exception:
                pass
        self.windows.clear()


class PlatePanel(ttk.Frame):
    def __init__(self, parent: tk.Widget, plate_info: PlateInfo, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.plate_info = plate_info
        self.expanded = False
        self.configure(relief="solid", borderwidth=1, padding=4)
        self._create_widgets()

    def _create_widgets(self) -> None:
        header = ttk.Frame(self)
        header.pack(fill="x")
        if self.plate_info.image is not None and self.plate_info.image.size > 0:
            preview = self._build_thumbnail(self.plate_info.image)
            if preview is not None:
                img_label = ttk.Label(header, image=preview)
                img_label.image = preview
                img_label.pack(side="left", padx=4)
        info = ttk.Frame(header)
        info.pack(side="left", fill="both", expand=True)
        ttk.Label(info, text=self.plate_info.text, font=("Courier", 12, "bold")).pack(anchor="w")
        ttk.Label(
            info,
            text=f"{self.plate_info.plate_type} • {self.plate_info.timestamp}",
            font=("Arial", 8),
            foreground="#555",
        ).pack(anchor="w")
        self.toggle_btn = ttk.Button(header, text="▼", width=3, command=self._toggle)
        self.toggle_btn.pack(side="right")
        self.detail = ttk.Frame(self)

    def _build_thumbnail(self, image: np.ndarray) -> Optional[ImageTk.PhotoImage]:
        try:
            plate_img = image.copy()
            if len(plate_img.shape) == 2:
                plate_img = cv2.cvtColor(plate_img, cv2.COLOR_GRAY2RGB)
            else:
                plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
            h, w = plate_img.shape[:2]
            scale = 80 / max(w, 1)
            new_w = int(w * scale)
            new_h = max(1, int(h * scale))
            resized = cv2.resize(plate_img, (new_w, new_h))
            return ImageTk.PhotoImage(Image.fromarray(resized))
        except Exception:
            return None

    def _toggle(self) -> None:
        self.expanded = not self.expanded
        if self.expanded:
            self._render_details()
            self.detail.pack(fill="x", pady=4)
            self.toggle_btn.configure(text="▲")
            return
        self.detail.pack_forget()
        self.toggle_btn.configure(text="▼")

    def _render_details(self) -> None:
        for child in self.detail.winfo_children():
            child.destroy()
        ttk.Label(self.detail, text=f"Veículo: {self.plate_info.vehicle_type}", font=("Arial", 9)).pack(anchor="w")
        ttk.Label(
            self.detail,
            text=f"YOLO {self.plate_info.yolo_confidence:.0f}% • OCR {self.plate_info.ocr_confidence:.0f}%",
            font=("Arial", 9),
        ).pack(anchor="w")


class DetectionSidePanel(ttk.Frame):
    def __init__(self, parent: tk.Widget, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.plate_panels: List[PlatePanel] = []
        self.max_plates = 60
        self.on_clear: Optional[Callable[[], None]] = None
        self._create_widgets()

    def _create_widgets(self) -> None:
        title = ttk.Frame(self)
        title.pack(fill="x", pady=(0, 2))
        ttk.Label(title, text="Placas Detectadas", font=("Arial", 11, "bold")).pack(side="left", padx=6)
        ttk.Button(title, text="Limpar", width=7, command=self.clear).pack(side="right", padx=6)
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(container, width=260, highlightthickness=0, bg="#f9f9f9")
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.scrollable = ttk.Frame(self.canvas)
        self.scrollable.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.create_window((0, 0), window=self.scrollable, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.canvas.bind("<Enter>", lambda _: self.canvas.bind_all("<MouseWheel>", self._on_mousewheel))
        self.canvas.bind("<Leave>", lambda _: self.canvas.unbind_all("<MouseWheel>"))

    def _on_mousewheel(self, event: tk.Event) -> None:
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def add_plate(self, plate_info: PlateInfo) -> None:
        while len(self.plate_panels) >= self.max_plates:
            panel = self.plate_panels.pop(0)
            panel.destroy()
        panel = PlatePanel(self.scrollable, plate_info)
        panel.pack(fill="x", padx=4, pady=2)
        self.plate_panels.append(panel)
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)

    def clear(self) -> None:
        for panel in self.plate_panels:
            panel.destroy()
        self.plate_panels.clear()
        if self.on_clear:
            self.on_clear()


class VideoDisplay(ttk.Frame):
    def __init__(self, parent: tk.Widget, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, bg="#1c1c1c", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.current_image = None
        self.fps_label = None
        self.bind("<Configure>", lambda _e: None)

    def update_frame(self, frame: np.ndarray, fps: float) -> None:
        if frame is None:
            return
        try:
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            if canvas_w < 10 or canvas_h < 10:
                return
            h, w = frame.shape[:2]
            scale = min(canvas_w / max(w, 1), canvas_h / max(h, 1))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            resized = cv2.resize(frame, (new_w, new_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            image = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.canvas.delete("all")
            x = (canvas_w - new_w) // 2
            y = (canvas_h - new_h) // 2
            self.canvas.create_image(x, y, anchor="nw", image=image, tags="video")
            self.current_image = image
            self.canvas.create_text(
                20,
                20,
                anchor="nw",
                text=f"FPS: {fps:.1f}",
                fill="#00ff00",
                font=("Arial", 12, "bold"),
            )
        except Exception:
            pass


class CompactControlPanel(ttk.Frame):
    def __init__(self, parent: tk.Widget, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.on_start: Optional[Callable[[], None]] = None
        self.on_stop: Optional[Callable[[], None]] = None
        self.on_model_change: Optional[Callable[[str, str], None]] = None
        self.on_source_change: Optional[Callable[[str, Optional[str]], None]] = None
        self.on_fps_change: Optional[Callable[[int], None]] = None
        self.on_screen_select: Optional[Callable[[], None]] = None
        self.on_ocr_toggle: Optional[Callable[[bool], None]] = None
        self.on_gpu_toggle: Optional[Callable[[bool], None]] = None
        self.on_logs_toggle: Optional[Callable[[bool], None]] = None
        self.on_overlay_toggle: Optional[Callable[[bool], None]] = None
        self.target_fps = tk.IntVar(value=DEFAULT_TARGET_FPS)
        self.vehicle_model = tk.StringVar()
        self.plate_model = tk.StringVar()
        self.selected_source = tk.StringVar(value="Webcam 0")
        self.selected_file = ""
        self.screen_region: Optional[Region] = None
        self.ocr_enabled = tk.BooleanVar(value=True)
        self.gpu_enabled = tk.BooleanVar(value=True)
        self.logs_enabled = tk.BooleanVar(value=False)
        self.overlay_enabled = tk.BooleanVar(value=True)
        self._create_widgets()

    def _create_widgets(self) -> None:
        for col in range(4):
            self.columnconfigure(col, weight=1)
        row = 0
        button_frame = ttk.Frame(self)
        button_frame.grid(row=row, column=0, sticky="w", padx=4, pady=2)
        ttk.Button(button_frame, text="▶ Iniciar", width=10, command=self._start).pack(side="left", padx=2)
        ttk.Button(button_frame, text="⏹ Parar", width=10, command=self._stop).pack(side="left", padx=2)

        source_frame = ttk.Frame(self)
        source_frame.grid(row=row, column=1, padx=4, pady=2, sticky="ew")
        ttk.Label(source_frame, text="Fonte:").pack(side="left")
        values = list(VIDEO_SOURCES.keys()) + ["Arquivo", "Tela"]
        self.source_combo = ttk.Combobox(source_frame, state="readonly", values=values, textvariable=self.selected_source, width=16)
        self.source_combo.pack(side="left", padx=4)
        self.source_combo.bind("<<ComboboxSelected>>", lambda _e: self._source_changed())
        ttk.Button(source_frame, text="…", width=3, command=self._select_source_aux).pack(side="left")

        fps_frame = ttk.Frame(self)
        fps_frame.grid(row=row, column=2, padx=4, pady=2, sticky="e")
        ttk.Button(fps_frame, text="-", width=3, command=lambda: self._change_fps(-5)).pack(side="left")
        self.fps_display = ttk.Label(fps_frame, text=str(self.target_fps.get()), font=("Arial", 11, "bold"), width=4, anchor="center")
        self.fps_display.pack(side="left", padx=4)
        ttk.Button(fps_frame, text="+", width=3, command=lambda: self._change_fps(5)).pack(side="left")

        toggle_frame = ttk.Frame(self)
        toggle_frame.grid(row=row, column=3, padx=4, pady=2, sticky="e")
        ttk.Checkbutton(toggle_frame, text="📋 Logs", variable=self.logs_enabled, command=self._toggle_logs).pack(side="right", padx=4)
        ttk.Checkbutton(toggle_frame, text="🖼️ Overlay", variable=self.overlay_enabled, command=self._toggle_overlay).pack(side="right", padx=4)
        ttk.Checkbutton(toggle_frame, text="🔤 OCR", variable=self.ocr_enabled, command=self._toggle_ocr).pack(side="right", padx=4)
        ttk.Checkbutton(toggle_frame, text="🎮 GPU", variable=self.gpu_enabled, command=self._toggle_gpu).pack(side="right")

        row = 1
        ttk.Label(self, text="Modelo veículo:", font=("Arial", 8)).grid(row=row, column=0, sticky="w", padx=5)
        self.vehicle_combo = ttk.Combobox(self, textvariable=self.vehicle_model, state="readonly")
        self.vehicle_combo.grid(row=row, column=1, sticky="ew", padx=4)
        self.vehicle_combo.bind("<<ComboboxSelected>>", lambda _e: self._models_changed())
        ttk.Label(self, text="Modelo placa:", font=("Arial", 8)).grid(row=row, column=2, sticky="w", padx=5)
        self.plate_combo = ttk.Combobox(self, textvariable=self.plate_model, state="readonly")
        self.plate_combo.grid(row=row, column=3, sticky="ew", padx=4)
        self.plate_combo.bind("<<ComboboxSelected>>", lambda _e: self._models_changed())

        row = 2
        self.model_status = ttk.Label(self, text="Modelos não carregados", font=("Arial", 8), foreground="#666")
        self.model_status.grid(row=row, column=0, columnspan=4, sticky="w", padx=5, pady=(2, 0))

    def _start(self) -> None:
        if self.on_start:
            self.on_start()

    def _stop(self) -> None:
        if self.on_stop:
            self.on_stop()

    def _source_changed(self) -> None:
        source = self.selected_source.get()
        if source == "Tela" and self.on_screen_select:
            # Apenas abre o seletor de região, on_source_change será chamado após seleção
            self.on_screen_select()
            return
        elif source == "Arquivo":
            self._select_file()
            return
        if self.on_source_change:
            self.on_source_change(source, self.selected_file)

    def _select_source_aux(self) -> None:
        source = self.selected_source.get()
        if source == "Tela" and self.on_screen_select:
            self.on_screen_select()
        elif source == "Arquivo":
            self._select_file()

    def _select_file(self) -> None:
        filetypes = [("Vídeos", "*.mp4 *.avi *.mkv *.mov *.wmv"), ("Todos", "*.*")]
        filename = filedialog.askopenfilename(title="Selecionar vídeo", filetypes=filetypes)
        if filename:
            self.selected_file = filename
            if self.on_source_change:
                self.on_source_change("Arquivo", filename)

    def _change_fps(self, delta: int) -> None:
        new_value = max(1, min(60, self.target_fps.get() + delta))
        self.target_fps.set(new_value)
        self.fps_display.configure(text=str(new_value))
        if self.on_fps_change:
            self.on_fps_change(new_value)

    def _models_changed(self) -> None:
        if self.on_model_change:
            self.on_model_change(self.vehicle_model.get(), self.plate_model.get())

    def _toggle_ocr(self) -> None:
        if self.on_ocr_toggle:
            self.on_ocr_toggle(self.ocr_enabled.get())

    def _toggle_gpu(self) -> None:
        if self.on_gpu_toggle:
            self.on_gpu_toggle(self.gpu_enabled.get())

    def _toggle_logs(self) -> None:
        if self.on_logs_toggle:
            self.on_logs_toggle(self.logs_enabled.get())

    def _toggle_overlay(self) -> None:
        if self.on_overlay_toggle:
            self.on_overlay_toggle(self.overlay_enabled.get())

    def set_models(self, models: List[str]) -> None:
        self.vehicle_combo.configure(values=models)
        self.plate_combo.configure(values=models)
        if models and not self.vehicle_model.get():
            self.vehicle_model.set(models[0])
        if models and not self.plate_model.get():
            self.plate_model.set(models[-1])
        self._models_changed()

    def set_screen_region(self, region: Optional[Region]) -> None:
        self.screen_region = region
        if region:
            self.selected_source.set("Tela")
            if self.on_source_change:
                self.on_source_change("Tela", None)

    def get_selected_source(self) -> Tuple[str, Optional[str]]:
        source = self.selected_source.get()
        if source == "Arquivo":
            return source, self.selected_file or None
        return source, None

    def get_target_fps(self) -> int:
        return self.target_fps.get()

    def update_model_status(self, vehicle_model: str, plate_model: str) -> None:
        self.model_status.configure(text=f"Veículo: {vehicle_model or 'n/d'} • Placa: {plate_model or 'n/d'}")


class StatusBar(ttk.Frame):
    def __init__(self, parent: tk.Widget, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.status_label = ttk.Label(self, text="● Pronto", font=("Arial", 9))
        self.status_label.pack(side="left", padx=6)
        ttk.Separator(self, orient="vertical").pack(side="left", fill="y", pady=4)
        self.stats_label = ttk.Label(self, text="Detecções: 0 | 00:00:00", font=("Arial", 9))
        self.stats_label.pack(side="left", padx=6)
        self.device_label = ttk.Label(self, text="Dispositivo: --", font=("Arial", 9))
        self.device_label.pack(side="right", padx=6)

    def update_status(self, text: str) -> None:
        self.status_label.configure(text=f"● {text}")

    def update_stats(self, detections: int, duration: str) -> None:
        self.stats_label.configure(text=f"Detecções: {detections} | {duration}")

    def update_device(self, device_mode: str) -> None:
        color = "#0a7a0a" if device_mode.upper() == "GPU" else "#666666"
        self.device_label.configure(text=f"Dispositivo: {device_mode}", foreground=color)


class MainWindow(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Sistema de Detecção de Placas Brasileiras")
        self.geometry("1200x700")
        self.minsize(980, 560)
        self._setup_style()
        self.on_start: Optional[Callable[[], None]] = None
        self.on_stop: Optional[Callable[[], None]] = None
        self.on_model_change: Optional[Callable[[str, str], None]] = None
        self.on_source_change: Optional[Callable[[str, Optional[str]], None]] = None
        self.on_fps_change: Optional[Callable[[int], None]] = None
        self.on_screen_select: Optional[Callable[[], None]] = None
        self.on_ocr_toggle: Optional[Callable[[bool], None]] = None
        self.on_gpu_toggle: Optional[Callable[[bool], None]] = None
        self.on_logs_toggle: Optional[Callable[[bool], None]] = None
        self.on_overlay_toggle: Optional[Callable[[bool], None]] = None
        self._create_widgets()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background="#f3f3f3")
        style.configure("TLabel", background="#f3f3f3")

    def _create_widgets(self) -> None:
        main_container = ttk.Frame(self)
        main_container.pack(fill="both", expand=True, padx=6, pady=6)
        self.control_panel = CompactControlPanel(main_container)
        self.control_panel.pack(fill="x")
        ttk.Separator(main_container, orient="horizontal").pack(fill="x", pady=4)
        self.paned = ttk.PanedWindow(main_container, orient="horizontal")
        self.paned.pack(fill="both", expand=True)
        self.video_display = VideoDisplay(self.paned)
        self.paned.add(self.video_display, weight=3)
        self.side_panel = DetectionSidePanel(self.paned)
        self.paned.add(self.side_panel, weight=1)
        self.status_bar = StatusBar(self)
        self.status_bar.pack(fill="x", side="bottom")
        self._wire_callbacks()

    def _wire_callbacks(self) -> None:
        self.control_panel.on_start = lambda: self.on_start and self.on_start()
        self.control_panel.on_stop = lambda: self.on_stop and self.on_stop()
        self.control_panel.on_model_change = lambda v, p: self.on_model_change and self.on_model_change(v, p)
        self.control_panel.on_source_change = lambda s, f: self.on_source_change and self.on_source_change(s, f)
        self.control_panel.on_fps_change = lambda fps: self.on_fps_change and self.on_fps_change(fps)
        self.control_panel.on_screen_select = lambda: self.on_screen_select and self.on_screen_select()
        self.control_panel.on_ocr_toggle = lambda enabled: self.on_ocr_toggle and self.on_ocr_toggle(enabled)
        self.control_panel.on_gpu_toggle = lambda enabled: self.on_gpu_toggle and self.on_gpu_toggle(enabled)
        self.control_panel.on_logs_toggle = lambda enabled: self.on_logs_toggle and self.on_logs_toggle(enabled)
        self.control_panel.on_overlay_toggle = lambda enabled: self.on_overlay_toggle and self.on_overlay_toggle(enabled)

    def _on_close(self) -> None:
        if self.on_stop:
            self.on_stop()
        self.destroy()

    def open_screen_selector(self, callback: Callable[[Optional[Region]], None]) -> None:
        ScreenRegionSelector(self, callback)

    def update_frame(self, frame: np.ndarray, fps: float) -> None:
        self.video_display.update_frame(frame, fps)

    def add_detection(self, plate_info: PlateInfo) -> None:
        self.side_panel.add_plate(plate_info)

    def clear_detections(self) -> None:
        self.side_panel.clear()

    def set_models(self, models: List[str]) -> None:
        self.control_panel.set_models(models)

    def update_model_status(self, vehicle_model: str, plate_model: str) -> None:
        self.control_panel.update_model_status(vehicle_model, plate_model)

    def update_status(self, text: str) -> None:
        self.status_bar.update_status(text)

    def update_stats(self, detections: int, duration: str) -> None:
        self.status_bar.update_stats(detections, duration)

    def update_device_status(self, mode: str) -> None:
        self.status_bar.update_device(mode)

    def get_target_fps(self) -> int:
        return self.control_panel.get_target_fps()

    def get_selected_source(self) -> Tuple[str, Optional[str]]:
        return self.control_panel.get_selected_source()


class LicensePlateApp:
    def __init__(self) -> None:
        self.window = MainWindow()
        self.model_manager = ModelManager(MODELS_DIR)
        self.logger = Logger()
        self.pipeline = Pipeline(self.model_manager, logger=self.logger)
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.capture_mode: Optional[str] = None
        self.screen_region: Optional[Region] = None
        self.running = False
        self.session_start: Optional[float] = None
        self.detections_count = 0
        self._last_loop_ts = 0.0
        self._prev_frame_time = time.time()
        self._track_history: Dict[int, str] = {}
        self._ocr_enabled = True
        self._logs_enabled = False
        self._overlay_enabled = True
        self._device = get_device()
        self.window.update_device_status("GPU" if self._device.type == "cuda" else "CPU")
        self._bind_callbacks()
        self._populate_models()

    def _bind_callbacks(self) -> None:
        self.window.on_start = self.start
        self.window.on_stop = self.stop
        self.window.on_model_change = self._on_model_change
        self.window.on_source_change = self._on_source_change
        self.window.on_fps_change = self.pipeline.set_target_fps
        self.window.on_screen_select = self._request_screen_region
        self.window.on_ocr_toggle = self._toggle_ocr
        self.window.on_gpu_toggle = self._toggle_gpu
        self.window.on_logs_toggle = self._toggle_logs
        self.window.on_overlay_toggle = self._toggle_overlay
        self.window.side_panel.on_clear = self._on_clear_detections

    def _populate_models(self) -> None:
        models = self.model_manager.list_models()
        self.window.set_models(models)
        if models:
            threading.Thread(target=self._load_initial_models, args=(models,), daemon=True).start()
        else:
            self.window.update_status("Nenhum modelo encontrado em /models")

    def _load_initial_models(self, models: List[str]) -> None:
        # Usar modelos padrão se disponíveis, senão usar primeiro/último da lista
        vehicle_model = DEFAULT_VEHICLE_MODEL if DEFAULT_VEHICLE_MODEL in models else models[0]
        plate_model = DEFAULT_PLATE_MODEL if DEFAULT_PLATE_MODEL in models else models[-1]
        
        # Atualizar seleção nos combos
        self.window.control_panel.vehicle_model.set(vehicle_model)
        self.window.control_panel.plate_model.set(plate_model)
        
        self._load_vehicle_model(vehicle_model)
        self._load_plate_model(plate_model)
        self.window.after(0, lambda: self.window.update_model_status(vehicle_model, plate_model))

    def _on_model_change(self, vehicle_model: str, plate_model: str) -> None:
        if vehicle_model:
            threading.Thread(target=self._load_vehicle_model, args=(vehicle_model,), daemon=True).start()
        if plate_model:
            threading.Thread(target=self._load_plate_model, args=(plate_model,), daemon=True).start()

    def _load_vehicle_model(self, model_name: str) -> None:
        try:
            self._set_status(f"Carregando modelo de veículos: {model_name}")
            self.model_manager.set_vehicle_model(model_name)
            self.window.after(0, lambda: self.window.update_model_status(model_name, self.model_manager.plate_model_path or ""))
            self._set_status("Modelo de veículos pronto")
        except Exception as exc:
            self._set_status(f"Erro veículo: {exc}")

    def _load_plate_model(self, model_name: str) -> None:
        try:
            self._set_status(f"Carregando modelo de placas: {model_name}")
            self.model_manager.set_plate_model(model_name)
            self.window.after(0, lambda: self.window.update_model_status(self.model_manager.vehicle_model_path or "", model_name))
            self._set_status("Modelo de placas pronto")
        except Exception as exc:
            self._set_status(f"Erro placa: {exc}")

    def _on_source_change(self, source: str, payload: Optional[str]) -> None:
        if source == "Tela":
            # Região já foi selecionada, apenas atualiza status
            if self.screen_region:
                self.window.update_status("Região da tela configurada")
        elif source == "Arquivo" and payload:
            self.window.update_status(f"Arquivo selecionado: {Path(payload).name}")

    def _request_screen_region(self) -> None:
        self.window.iconify()

        def _after_select(region: Optional[Region]) -> None:
            self.window.deiconify()
            if region:
                self.screen_region = region
                self._set_status("Região da tela configurada")
            else:
                self._set_status("Seleção de tela cancelada")

        self.window.after(200, lambda: self.window.open_screen_selector(_after_select))

    def _toggle_ocr(self, enabled: bool) -> None:
        self._ocr_enabled = enabled
        self.window.update_status(f"OCR {'ativado' if enabled else 'desativado'}")

    def _toggle_gpu(self, enabled: bool) -> None:
        mode = "GPU" if enabled and self._device.type == "cuda" else "CPU"
        self.window.update_device_status(mode)

    def _toggle_logs(self, enabled: bool) -> None:
        self._logs_enabled = enabled
        self.window.update_status(f"Logs no terminal {'ativados' if enabled else 'desativados'}")

    def _toggle_overlay(self, enabled: bool) -> None:
        self._overlay_enabled = enabled
        self.window.update_status(f"Overlay no vídeo {'ativado' if enabled else 'desativado'}")

    def _on_clear_detections(self) -> None:
        """Limpa o histórico de tracks para permitir novas detecções."""
        self._track_history.clear()
        self.detections_count = 0
        self._update_stats()

    def start(self) -> None:
        if self.running:
            return
        if not self.model_manager.ready():
            messagebox.showwarning("Modelos", "Carregue os modelos de veículo e placa antes de iniciar.")
            return
        source, payload = self.window.get_selected_source()
        self.window.clear_detections()
        self._track_history.clear()
        self.detections_count = 0
        self.session_start = time.time()
        self._prev_frame_time = time.time()
        self.capture_mode = None
        if source in VIDEO_SOURCES:
            idx = VIDEO_SOURCES[source]
            self.video_capture = cv2.VideoCapture(idx)
            self.capture_mode = "camera"
        elif source == "Arquivo":
            if not payload:
                messagebox.showinfo("Fonte", "Selecione um arquivo de vídeo")
                return
            self.video_capture = cv2.VideoCapture(payload)
            self.capture_mode = "file"
        elif source == "Tela":
            if not self.screen_region:
                messagebox.showinfo("Fonte", "Selecione uma região da tela")
                return
            if mss is None:
                messagebox.showwarning("Captura de tela", "Instale o pacote 'mss' para usar a captura de tela.")
                return
            self.capture_mode = "screen"
        else:
            messagebox.showerror("Fonte", f"Fonte desconhecida: {source}")
            return
        if self.capture_mode in {"camera", "file"} and (not self.video_capture or not self.video_capture.isOpened()):
            messagebox.showerror("Captura", "Não foi possível abrir a fonte de vídeo.")
            return
        self.running = True
        self._set_status("Capturando...")
        self._schedule_loop()

    def stop(self) -> None:
        self.running = False
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        self._set_status("Parado")

    def _schedule_loop(self) -> None:
        if self.running:
            self.window.after(5, self._loop)

    def _loop(self) -> None:
        if not self.running:
            return
        now = time.time()
        min_interval = 1.0 / max(1, self.window.get_target_fps())
        if now - self._last_loop_ts < min_interval:
            self._schedule_loop()
            return
        self._last_loop_ts = now
        frame = self._read_frame()
        if frame is None:
            self.stop()
            return
        processed_frame, tracks = self.pipeline.process_frame(frame)
        decorated = self._draw_tracks(processed_frame, tracks) if self._overlay_enabled else processed_frame
        elapsed = now - self._prev_frame_time
        self._prev_frame_time = now
        fps = 1.0 / elapsed if elapsed > 0 else 0.0
        self.window.update_frame(decorated, fps)
        self._update_detections(tracks)
        self._update_stats()
        self._schedule_loop()

    def _read_frame(self) -> Optional[np.ndarray]:
        if self.capture_mode in {"camera", "file"}:
            if not self.video_capture:
                return None
            ret, frame = self.video_capture.read()
            if not ret:
                return None
            return frame
        if self.capture_mode == "screen" and self.screen_region and mss is not None:
            x, y, w, h = self.screen_region
            with mss.mss() as sct:
                region = {"left": x, "top": y, "width": w, "height": h}
                img = sct.grab(region)
                frame = np.array(img)
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return None

    def _draw_tracks(self, frame: np.ndarray, tracks: List[VehicleTrack]) -> np.ndarray:
        result = frame.copy()
        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            color = tuple(int(c) for c in track.color)
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            if track.reading and track.reading.text:
                label = f"{track.reading.text} ({track.reading.ocr_confidence*100:.0f}%)"
                cv2.rectangle(result, (x1, y1 - 22), (x1 + 160, y1), color, -1)
                cv2.putText(result, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            # Trail desabilitado para evitar poluição visual
            # for idx in range(1, len(track.trail)):
            #     cv2.line(result, track.trail[idx - 1], track.trail[idx], color, 2)
        return result

    def _update_detections(self, tracks: List[VehicleTrack]) -> None:
        for track in tracks:
            reading = track.reading
            if not reading.text:
                continue
            
            # Validar placa antes de adicionar
            if not self._is_valid_plate(reading.text, reading.detection_confidence, reading.ocr_confidence):
                continue
            
            # Evitar duplicatas do mesmo track
            history_key = self._track_history.get(track.track_id)
            if history_key == reading.text:
                continue
            self._track_history[track.track_id] = reading.text
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Log no terminal se ativado
            if self._logs_enabled:
                print(f"[{timestamp}] Placa: {reading.text} | Tipo: {reading.plate_type or 'N/A'} | "
                      f"Veículo: {track.label} | YOLO: {reading.detection_confidence*100:.0f}% | "
                      f"OCR: {reading.ocr_confidence*100:.0f}%")
            
            # Adicionar na interface se OCR estiver ativado
            if not self._ocr_enabled:
                continue
            
            plate_info = PlateInfo(
                image=reading.crop,
                text=reading.text,
                plate_type=reading.plate_type or "Desconhecida",
                yolo_confidence=reading.detection_confidence * 100,
                ocr_confidence=reading.ocr_confidence * 100,
                timestamp=timestamp,
                vehicle_type=track.label,
            )
            self.detections_count += 1
            self.window.add_detection(plate_info)

    def _is_valid_plate(self, text: str, yolo_conf: float, ocr_conf: float) -> bool:
        """Valida se a placa atende aos critérios mínimos."""
        from .config import MIN_PLATE_CONFIDENCE, MIN_OCR_CONFIDENCE, MIN_PLATE_LENGTH, MAX_PLATE_LENGTH
        
        # Verificar confianças mínimas
        if yolo_conf < MIN_PLATE_CONFIDENCE:
            return False
        if ocr_conf < MIN_OCR_CONFIDENCE:
            return False
        
        # Verificar tamanho da placa
        if len(text) < MIN_PLATE_LENGTH or len(text) > MAX_PLATE_LENGTH:
            return False
        
        # Verificar se contém apenas caracteres válidos (letras e números)
        if not text.replace('-', '').replace(' ', '').isalnum():
            return False
        
        return True

    def _update_stats(self) -> None:
        if not self.session_start:
            return
        elapsed = int(time.time() - self.session_start)
        duration = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        self.window.update_stats(self.detections_count, duration)

    def _set_status(self, text: str) -> None:
        self.window.after(0, lambda: self.window.update_status(text))

    def on_close(self) -> None:
        self.stop()
        self.pipeline.close()

    def run(self) -> None:
        self.window.mainloop()


__all__ = ["LicensePlateApp"]
