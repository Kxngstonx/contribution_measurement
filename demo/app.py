"""Tk application: main / loading / result screens."""

from __future__ import annotations

import argparse
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from . import backend, config, texts
from .ktext import KButton, KLabel, KOptionMenu, KRadiobutton, KStepper

BG = "#f4f4f4"
PANEL = "#e9ecef"


def _frame(master, bg=BG, **kw):
    return tk.Frame(master, bg=bg, **kw)


class MainScreen(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=BG)
        self.app = app
        KLabel(self, texts.TITLE, size=26, bold=True).pack(anchor="w", padx=28, pady=(20, 10))

        body = _frame(self)
        body.pack(fill="x", padx=28)
        left = _frame(body)
        left.pack(side="left", fill="both", expand=True)
        right = _frame(body)
        right.pack(side="right", fill="y", padx=(16, 0))

        wrap = 760
        KLabel(left, texts.INTRO_FL, size=15, wrap_px=wrap).pack(anchor="w", pady=(0, 10))
        KLabel(left, texts.INTRO_WHY, size=15, wrap_px=wrap).pack(anchor="w", pady=(0, 16))
        KLabel(left, texts.METHOD_PARZEN_TITLE, size=17, bold=True).pack(anchor="w")
        KLabel(left, texts.METHOD_PARZEN, size=15, wrap_px=wrap).pack(anchor="w", pady=(4, 14))
        KLabel(left, texts.METHOD_TRAC_TITLE, size=17, bold=True).pack(anchor="w")
        KLabel(left, texts.METHOD_TRAC, size=15, wrap_px=wrap).pack(anchor="w", pady=(4, 14))

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        from .plots import schematic_figure

        figure = schematic_figure(figsize=(6.2, 4.0))
        figure.patch.set_facecolor(BG)
        self._schematic = FigureCanvasTkAgg(figure, master=right)
        self._schematic.draw()
        self._schematic.get_tk_widget().pack(anchor="n")

        # The settings panel takes all the space left under the text and the
        # schematic; its rows are spread over that height.
        panel = _frame(self, bg=PANEL, padx=36, pady=22)
        panel.pack(fill="both", expand=True, padx=28, pady=(14, 14))
        for column in range(2):
            panel.grid_columnconfigure(column, weight=1, uniform="pick")
        for row in (2, 3, 4):
            panel.grid_rowconfigure(row, weight=1)
        KLabel(panel, texts.CONTROLS_TITLE, size=20, bold=True).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        KLabel(panel, texts.SETTING_HINT, size=14, color="#555555").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(0, 8))

        # One dataset picker and, per dataset, one pre-built setting picker
        # stacked in the same cell (KOptionMenu has a fixed value list); the
        # picker of the chosen dataset is raised on top.
        cell = _frame(panel, bg=PANEL)
        cell.grid(row=2, column=0, sticky="ew", pady=6)
        KLabel(cell, texts.DATASET_LABEL, size=16, bold=True).pack(anchor="w", pady=(0, 6))
        self.dataset_title = tk.StringVar(value=config.DATASET_TITLES[config.DEFAULT_DATASET])
        KOptionMenu(cell, self.dataset_title,
                    [config.DATASET_TITLES[d] for d in config.DATASETS],
                    size=16, command=self._dataset_changed).pack(fill="x")

        cell = _frame(panel, bg=PANEL)
        cell.grid(row=2, column=1, sticky="ew", padx=(24, 0), pady=6)
        KLabel(cell, texts.SETTING_LABEL, size=16, bold=True).pack(anchor="w", pady=(0, 6))
        holder = _frame(cell, bg=PANEL)
        holder.pack(fill="x")
        holder.grid_columnconfigure(0, weight=1)
        self.setting_vars, self.setting_menus = {}, {}
        for dataset in config.DATASETS:
            var = tk.StringVar(value=config.DEFAULT_CHOICE[dataset])
            menu = KOptionMenu(holder, var, backend.labels_for(dataset), size=16)
            menu.grid(row=0, column=0, sticky="ew")
            self.setting_vars[dataset], self.setting_menus[dataset] = var, menu

        mode_row = _frame(panel, bg=PANEL)
        mode_row.grid(row=3, column=0, columnspan=2, sticky="w", pady=6)
        KLabel(mode_row, texts.MODE_LABEL, size=16, bold=True).pack(side="left", padx=(0, 24))
        self.mode = tk.StringVar(value="replay")
        KRadiobutton(mode_row, texts.MODE_REPLAY, variable=self.mode, value="replay",
                     size=16, command=self._refresh).pack(side="left")
        KRadiobutton(mode_row, texts.MODE_REAL, variable=self.mode, value="real",
                     size=16, command=self._refresh).pack(side="left", padx=(36, 0))

        rounds_row = _frame(panel, bg=PANEL)
        rounds_row.grid(row=4, column=0, columnspan=2, sticky="w", pady=6)
        KLabel(rounds_row, texts.ROUNDS_LABEL, size=16, bold=True).pack(side="left", padx=(0, 24))
        self.rounds = tk.StringVar(value=str(config.DEFAULT_TRAC_ROUNDS))
        self.rounds_box = KStepper(rounds_row, self.rounds, from_=1, to=config.MAX_TRAC_ROUNDS,
                                   size=16, command=self._refresh)
        self.rounds_box.pack(side="left")
        self.rounds_hint = KLabel(rounds_row, "", size=14, color="#555555")
        self.rounds_hint.pack(side="left", padx=(24, 0))

        if app.device != "cuda":
            KLabel(panel, texts.DEVICE_CPU_WARNING, size=14, color="#b02a37").grid(
                row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))

        buttons = _frame(self)
        buttons.pack(pady=(0, 22))
        self.btn_parzen = KButton(buttons, texts.BTN_PARZEN, size=18,
                                  command=lambda: app.start_job("parzen"))
        self.btn_parzen.pack(side="left", padx=20)
        self.btn_trac = KButton(buttons, texts.BTN_TRAC, size=18,
                                command=lambda: app.start_job("trac"))
        self.btn_trac.pack(side="left", padx=20)
        self._dataset_changed()

    _TITLE_TO_DATASET = {title: dataset for dataset, title in config.DATASET_TITLES.items()}

    def dataset(self) -> str:
        return self._TITLE_TO_DATASET[self.dataset_title.get()]

    def _dataset_changed(self):
        self.setting_menus[self.dataset()].tkraise()
        self._refresh()

    def _refresh(self):
        dataset = self.dataset()
        rounds = self.rounds_value()
        self.rounds_hint.set(texts.ROUNDS_HINT.format(
            rounds=rounds, minutes=config.estimated_trac_seconds(rounds, dataset) / 60,
            full_rounds=config.FULL_TRAC_ROUNDS[dataset],
            full=config.full_trac_seconds(dataset) / 60))
        self.rounds_box.set_enabled(self.mode.get() == "real")

    def choice(self) -> tuple[str, str]:
        """(dataset, setting label) currently picked."""

        dataset = self.dataset()
        return dataset, self.setting_vars[dataset].get()

    def rounds_value(self) -> int:
        try:
            return max(1, min(config.MAX_TRAC_ROUNDS, int(self.rounds.get())))
        except ValueError:
            return config.DEFAULT_TRAC_ROUNDS

    def set_busy(self, busy: bool):
        state = "disabled" if busy else "normal"
        self.btn_parzen.configure(state=state)
        self.btn_trac.configure(state=state)


class LoadingScreen(tk.Frame):
    LOG_LINES = 12

    def __init__(self, master, app):
        super().__init__(master, bg=BG)
        self.app = app
        self.title = KLabel(self, "", size=24, bold=True)
        self.title.pack(anchor="w", padx=28, pady=(28, 6))
        self.subtitle = KLabel(self, "", size=14, color="#555555")
        self.subtitle.pack(anchor="w", padx=28, pady=(0, 20))

        row = _frame(self)
        row.pack(fill="x", padx=28, pady=10)
        self.name = KLabel(row, "", size=16, bold=True)
        self.name.pack(side="left", padx=(0, 16))
        self.bar = ttk.Progressbar(row, mode="determinate", length=720)
        self.bar.pack(side="left", padx=(0, 16))
        self.status = KLabel(row, texts.STATUS_WAITING, size=14)
        self.status.pack(side="left")

        self.log_lines = []
        self.log = KLabel(self, "", size=13, color="#333333", wrap_px=1300)
        self.log.pack(anchor="w", padx=28, pady=(24, 10))
        self.cancel_button = KButton(self, texts.BTN_CANCEL, command=self._cancel)
        self.cancel_button.pack(pady=12)

    def reset(self, method: str, mode: str, rounds: int, dataset: str, label: str):
        self.title.set(texts.LOADING_TITLE[method])
        subtitle = texts.LOADING_MODE[mode]
        if method == "trac" and mode == "real":
            subtitle += f"  ·  {rounds}라운드"
        self.subtitle.set(subtitle)
        self.name.set(f"{config.DATASET_TITLES[dataset]} {label}")
        self.bar["value"] = 0
        self.bar["maximum"] = 1
        self.status.set(texts.STATUS_WAITING)
        self.log_lines = []
        self.log.set("")
        self.cancel_button.configure(state="normal")

    def _cancel(self):
        self.app.cancel.set()
        self.cancel_button.configure(state="disabled")
        if self.bar["value"] < self.bar["maximum"]:
            self.status.set(texts.STATUS_CANCELLING)

    def _append_log(self, text: str):
        self.log_lines.append(text)
        self.log.set("\n".join(self.log_lines[-self.LOG_LINES:]))

    def handle(self, event: dict):
        kind = event["type"]
        if kind == "dataset_start":
            self.name.set(f"{config.DATASET_TITLES[event['dataset']]} {event['label']}")
            self.bar["maximum"] = max(1, int(event["max"]))
            self.bar["value"] = 0
        elif kind == "progress":
            self.bar["maximum"] = max(1, int(event["max"]))
            self.bar["value"] = event["value"]
            if event.get("text"):
                self.status.set(event["text"])
        elif kind == "log":
            self._append_log(event["text"])
        elif kind == "result":
            self.bar["value"] = self.bar["maximum"]
            self.status.set(texts.STATUS_DONE.format(pcc=event["result"].pcc))


class ResultScreen(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=BG)
        self.app = app
        self.title = KLabel(self, "", size=24, bold=True)
        self.title.pack(anchor="w", padx=28, pady=(18, 4))
        self.header = KLabel(self, "", size=14, color="#333333")
        self.header.pack(anchor="w", padx=28, pady=(0, 8))
        self.plots = _frame(self)
        self.plots.pack(fill="both", expand=True, padx=12)
        self._canvases = []
        KButton(self, texts.BTN_BACK, command=lambda: app.show("main")).pack(pady=10)

    def show(self, method: str, result):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        from .plots import bar_figure, format_seconds, scatter_figure, time_figure

        self.title.set(texts.RESULT_TITLE[method])
        extra = ""
        if result.rounds is not None:
            extra += texts.RESULT_ROUNDS.format(rounds=result.rounds)
        lines = [texts.RESULT_LINE.format(
            title=config.DATASET_TITLES[result.dataset], label=result.label,
            pcc=result.pcc, source=texts.RESULT_SOURCE[result.source], extra=extra)]
        if result.timing and result.gt_seconds is not None:
            lines.append(texts.RESULT_TIME.format(
                ours=format_seconds(result.timing["total_seconds"]),
                gt=format_seconds(result.gt_seconds)))
        self.header.set("\n".join(lines))

        for canvas in self._canvases:
            canvas.get_tk_widget().destroy()
        self._canvases = []
        for frame in getattr(self, "_rows", []):
            frame.destroy()
        name = texts.METHOD_NAMES[method]
        top = _frame(self.plots)
        top.pack(fill="both", expand=True)
        bottom = _frame(self.plots)
        bottom.pack(fill="both", expand=True)
        self._rows = [top, bottom]
        placements = (
            (bar_figure([result], name, (7.2, 3.4)), top, "left"),
            (scatter_figure([result], name, (7.2, 3.4)), top, "left"),
            (time_figure(result, name, (14.5, 3.2)), bottom, "top"),
        )
        for figure, master, side in placements:
            figure.patch.set_facecolor(BG)
            canvas = FigureCanvasTkAgg(figure, master=master)
            canvas.draw()
            canvas.get_tk_widget().pack(side=side, fill="both", expand=True, padx=2, pady=2)
            self._canvases.append(canvas)


class App(tk.Tk):
    POLL_MS = 100

    def __init__(self, device: str | None = None):
        super().__init__()
        from .plots import setup_fonts

        setup_fonts()
        # The window title is drawn by the window manager with Tk's own font,
        # which cannot show Hangul here, so it stays in English.
        self.title(texts.WINDOW_TITLE)
        # Fit the screen (WSLg reports the Windows desktop size); the result
        # figures scale with the window, so a larger screen just gets bigger plots.
        width = min(1500, self.winfo_screenwidth() - 16)
        height = min(1000, self.winfo_screenheight() - 90)
        self.geometry(f"{width}x{height}")
        self.minsize(1180, 760)
        self.configure(bg=BG)
        self.device = device or config.detect_device()
        self.events = None
        self.cancel = threading.Event()
        self.result = None
        self.method = None
        self.mode = None

        container = _frame(self)
        container.pack(fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.screens = {
            "main": MainScreen(container, self),
            "loading": LoadingScreen(container, self),
            "result": ResultScreen(container, self),
        }
        for screen in self.screens.values():
            screen.grid(row=0, column=0, sticky="nsew")
        self.show("main")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def show(self, name: str):
        self.screens[name].tkraise()

    def start_job(self, method: str):
        main = self.screens["main"]
        self.method = method
        self.mode = main.mode.get()
        rounds = main.rounds_value()
        dataset, label = main.choice()
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.result = None
        main.set_busy(True)
        loading = self.screens["loading"]
        loading.reset(method, self.mode, rounds, dataset, label)
        self.show("loading")
        threading.Thread(
            target=backend.run_job,
            args=(method, self.mode, dataset, label, rounds, self.device, self.events, self.cancel),
            daemon=True,
        ).start()
        self.after(self.POLL_MS, self._poll)

    def _poll(self):
        loading = self.screens["loading"]
        finished = None
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind = event["type"]
            if kind == "result":
                self.result = event["result"]
            if kind in ("done", "cancelled", "error"):
                finished = event
                break
            loading.handle(event)
        if finished is None:
            self.after(self.POLL_MS, self._poll)
            return
        self.screens["main"].set_busy(False)
        if finished["type"] == "done":
            self.screens["result"].show(self.method, self.result)
            self.show("result")
        elif finished["type"] == "error":
            messagebox.showerror(texts.ERROR_TITLE, texts.ERROR_BODY.format(text=finished["text"]))
            self.show("main")
        else:
            self.show("main")

    def _on_close(self):
        self.cancel.set()
        self.destroy()


def export(out_dir, method: str, mode: str, rounds: int, device: str,
           dataset: str, label: str) -> list:
    """Headless: run the job synchronously and save the figures as PNG."""

    from .plots import format_seconds, save_figures, setup_fonts

    setup_fonts()
    methods = ("parzen", "trac") if method == "both" else (method,)
    saved = []
    for one in methods:
        events = queue.Queue()
        backend.run_job(one, mode, dataset, label, rounds, device, events, threading.Event())
        result, final = None, None
        while not events.empty():
            event = events.get()
            if event["type"] == "result":
                result = event["result"]
            elif event["type"] in ("done", "cancelled", "error"):
                final = event
            elif event["type"] == "log":
                print(f"[{event.get('dataset', '-')}] {event['text']}", flush=True)
        if final is None or final["type"] != "done" or result is None:
            raise RuntimeError(final["text"] if final and final["type"] == "error" else "job did not finish")
        saved += save_figures(result, one, out_dir)
        timing = (result.timing or {}).get("total_seconds")
        ours = format_seconds(timing) if timing is not None else "?"
        gt = format_seconds(result.gt_seconds) if result.gt_seconds is not None else "?"
        print(f"{one} {result.dataset} {result.label}: PCC {result.pcc:.3f} ({result.source})"
              f"  time {ours} vs Ground Truth {gt}", flush=True)
    for path in saved:
        print(f"saved -> {path}", flush=True)
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description="Parzen-Shapley / TRAC contribution demo")
    parser.add_argument("--export", default=None, help="save figures to this directory instead of opening a window")
    parser.add_argument("--method", default="both", choices=("both", "parzen", "trac"))
    parser.add_argument("--mode", default="replay", choices=("replay", "real"))
    parser.add_argument("--dataset", default=config.DEFAULT_DATASET, choices=config.DATASETS,
                        help="dataset for --export (default: %(default)s)")
    parser.add_argument("--setting", default=None,
                        help="contamination setting for --export: mislabel-k or noniid-k "
                             "(default: the dataset's default)")
    parser.add_argument("--rounds", type=int, default=config.DEFAULT_TRAC_ROUNDS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--download-data", action="store_true",
                        help="download MNIST / Fashion-MNIST / CIFAR-10 into data/ and exit")
    args = parser.parse_args(argv)
    if args.download_data:
        backend.download_datasets()
        return
    device = args.device or config.detect_device()
    if args.export:
        label = args.setting or config.DEFAULT_CHOICE[args.dataset]
        if label not in backend.labels_for(args.dataset):
            parser.error(f"--setting for {args.dataset} must be one of: "
                         + ", ".join(backend.labels_for(args.dataset)))
        export(args.export, args.method, args.mode, args.rounds, device, args.dataset, label)
        return
    App(device=device).mainloop()
