"""Korean text in Tk, rendered through PIL.

The Tk bundled with the conda environment is built without Xft and knows only
the `fixed` font, so it draws every Hangul syllable as a `\\uXXXX` escape.
These widgets render the text with a TrueType font into an image instead.
"""

from __future__ import annotations

import tkinter as tk
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont, ImageTk

from . import config


@lru_cache(maxsize=None)
def _font(size: int, bold: bool):
    path = config.korean_font_path(bold)
    if path is None:
        return ImageFont.load_default()
    return ImageFont.truetype(path, size)


def _wrap(text: str, font, wrap_px: int) -> list[str]:
    """Greedy per-character wrapping (Hangul rarely has spaces to break on)."""

    lines = []
    for paragraph in text.split("\n"):
        if wrap_px <= 0 or font.getlength(paragraph) <= wrap_px:
            lines.append(paragraph)
            continue
        current = ""
        for char in paragraph:
            candidate = current + char
            if current and font.getlength(candidate) > wrap_px:
                lines.append(current.rstrip())
                current = char.lstrip()
            else:
                current = candidate
        lines.append(current)
    return lines


def render_image(
    text: str,
    *,
    size: int = 12,
    bold: bool = False,
    color: str = "#202020",
    bg=(244, 244, 244),
    wrap_px: int = 0,
    justify: str = "left",
    padding=2,
    line_spacing: int = 4,
) -> Image.Image:
    """`padding` is an int or an (x, y) pair of pixels around the text. Tk
    ignores padX/padY for image buttons, so widget padding lives here."""

    font = _font(size, bold)
    pad_x, pad_y = (padding, padding) if isinstance(padding, int) else padding
    lines = _wrap(text, font, wrap_px) or [""]
    joined = "\n".join(lines)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    left, top, right, bottom = probe.multiline_textbbox(
        (0, 0), joined, font=font, spacing=line_spacing
    )
    width = max(1, right - left + 2 * pad_x)
    height = max(1, bottom - top + 2 * pad_y)
    image = Image.new("RGB", (width, height), bg)
    ImageDraw.Draw(image).multiline_text(
        (pad_x - left, pad_y - top),
        joined,
        font=font,
        fill=color,
        spacing=line_spacing,
        align=justify,
    )
    return image


def widget_bg(widget: tk.Misc) -> tuple[int, int, int]:
    r, g, b = widget.winfo_rgb(widget.cget("background"))
    return (r >> 8, g >> 8, b >> 8)


def _inherit_bg(master: tk.Misc, kw: dict) -> dict:
    """Default a widget's background to its parent's so the rendered text
    image blends in (tk's default #d9d9d9 would show as a grey block)."""

    if "bg" not in kw and "background" not in kw:
        try:
            kw["bg"] = master.cget("background")
        except tk.TclError:
            pass
    return kw


class KLabel(tk.Label):
    """`tk.Label` whose text is a PIL-rendered image; `.set(text)` re-renders."""

    def __init__(self, master, text: str = "", *, size=14, bold=False,
                 color="#202020", wrap_px=0, justify="left", padding=2, **kw):
        super().__init__(master, **_inherit_bg(master, kw))
        self._style = dict(size=size, bold=bold, color=color,
                           wrap_px=wrap_px, justify=justify, padding=padding)
        self._img = None
        self.set(text)

    def set(self, text: str) -> None:
        image = render_image(text, bg=widget_bg(self), **self._style)
        self._img = ImageTk.PhotoImage(image)
        self.configure(image=self._img)


class KButton(tk.Button):
    """`tk.Button` with a PIL-rendered Korean caption."""

    def __init__(self, master, text: str, command=None, *, size=16, bold=True,
                 color="#202020", padx=28, pady=12, **kw):
        super().__init__(master, command=command, **kw)
        image = render_image(text, size=size, bold=bold, color=color,
                             bg=widget_bg(self), padding=(padx, pady))
        self._img = ImageTk.PhotoImage(image)
        self.configure(image=self._img)


class KRadiobutton(tk.Radiobutton):
    """`tk.Radiobutton` with a PIL-rendered Korean caption."""

    def __init__(self, master, text: str, *, size=14, **kw):
        kw = _inherit_bg(master, kw)
        kw.setdefault("activebackground", kw.get("bg"))
        super().__init__(master, **kw)
        image = render_image(text, size=size, bg=widget_bg(self), padding=(6, 6))
        self._img = ImageTk.PhotoImage(image)
        self.configure(image=self._img)


class KOptionMenu(tk.Menubutton):
    """Drop-down selector whose current value and menu entries are PIL-rendered
    images (ttk.Combobox would draw them with Tk's tiny bitmap font)."""

    def __init__(self, master, variable, values, *, size=16, command=None, **kw):
        kw.setdefault("bg", "#ffffff")
        kw.setdefault("activebackground", "#e2e6ea")
        kw.setdefault("indicatoron", True)
        super().__init__(master, relief="raised", bd=2, anchor="w", **kw)
        self._variable = variable
        self._command = command
        bg = widget_bg(self)
        self._images = {
            value: ImageTk.PhotoImage(render_image(value, size=size, bg=bg, padding=(14, 10)))
            for value in values
        }
        menu = tk.Menu(self, tearoff=0)
        menu_bg = widget_bg(menu)
        self._menu_images = {
            value: ImageTk.PhotoImage(render_image(value, size=size, bg=menu_bg, padding=(10, 6)))
            for value in values
        }
        for value in values:
            menu.add_radiobutton(image=self._menu_images[value], variable=variable,
                                 value=value, command=self._changed)
        self._menu = menu
        self.configure(menu=menu)
        self._changed(notify=False)

    def _changed(self, notify=True):
        self.configure(image=self._images[self._variable.get()])
        if notify and self._command is not None:
            self._command()


class KStepper(tk.Frame):
    """Integer chooser: [-] value [+] with PIL-rendered digits (a tk Spinbox
    would show them in the bitmap font)."""

    def __init__(self, master, variable, *, from_, to, size=16, command=None, **kw):
        super().__init__(master, **_inherit_bg(master, kw))
        self._variable = variable
        self._from, self._to = int(from_), int(to)
        self._command = command
        self._minus = KButton(self, "-", size=size, padx=16, pady=4,
                              command=lambda: self._step(-1))
        self._minus.pack(side="left")
        self._value = KLabel(self, "", size=size + 2, bold=True, bg="#ffffff",
                             relief="sunken", bd=1, padding=(18, 8))
        self._value.pack(side="left", padx=6)
        self._plus = KButton(self, "+", size=size, padx=16, pady=4,
                             command=lambda: self._step(1))
        self._plus.pack(side="left")
        self._refresh()

    def value(self) -> int:
        try:
            return max(self._from, min(self._to, int(self._variable.get())))
        except (ValueError, tk.TclError):
            return self._from

    def _step(self, delta: int):
        self._variable.set(str(max(self._from, min(self._to, self.value() + delta))))
        self._refresh()
        if self._command is not None:
            self._command()

    def _refresh(self):
        self._value.set(str(self.value()))

    def set_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self._minus.configure(state=state)
        self._plus.configure(state=state)
        self._value.configure(fg="#202020" if enabled else "#999999")
