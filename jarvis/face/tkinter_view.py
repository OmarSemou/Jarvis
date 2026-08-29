"""Tkinter renderer for the BMO prototype assets.

Tk is imported and a window is created only when a caller explicitly starts a
view.  The controller's ``pump`` method must run on this same main thread.
"""

from __future__ import annotations

import base64
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .assets import FaceAsset, FaceAssetSet, default_bmo_asset_set
from .controller import FaceController
from .state import FaceActivity, FaceState


class TkinterFaceView:
    def __init__(
        self,
        controller: FaceController,
        assets: FaceAssetSet | None = None,
        *,
        width: int = 800,
        height: int = 480,
        fullscreen: bool = False,
        title: str = "Jarvis Face — BMO prototype",
        gallery: bool = False,
        on_close: Callable[[], None] | None = None,
        on_ready: Callable[["TkinterFaceView"], None] | None = None,
    ) -> None:
        if width < 160 or height < 120:
            raise ValueError("face window is too small")
        self.controller = controller
        self.assets = assets or controller.assets or default_bmo_asset_set()
        self.width = width
        self.height = height
        self.fullscreen = fullscreen
        self.title = title
        self.gallery = gallery
        self.on_close = on_close
        self.on_ready = on_ready
        self._close_notified = False
        self._root: Any | None = None
        self._canvas: Any | None = None
        self._photo_cache: dict[tuple[Path, int, int], Any] = {}
        self._image_ref: Any | None = None
        self._rendered: tuple[FaceState, FaceAsset] | None = None
        self._last_draw_key: tuple[Any, ...] | None = None
        self._after_id: Any | None = None
        self._close_requested = threading.Event()
        self.controller.bind_view(self)

    @property
    def root(self) -> Any | None:
        return self._root

    def _create(self) -> None:
        if self._root is not None:
            return
        import tkinter as tk

        root = tk.Tk()
        root.title(self.title)
        root.configure(bg="#111111")
        root.minsize(320, 192)
        if self.fullscreen:
            root.attributes("-fullscreen", True)
        canvas = tk.Canvas(root, bg="#111111", highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        root.bind("<Escape>", lambda _event: self.close())
        root.bind("<Control-q>", lambda _event: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Configure>", lambda _event: self._draw())
        self._root = root
        self._canvas = canvas
        if self.on_ready is not None:
            self.on_ready(self)

    def _photo(self, asset: FaceAsset, target_width: int, target_height: int) -> Any:
        import tkinter as tk

        # Use Tcl's base64 data form instead of a filename. This avoids a
        # Windows/Tcl path-escaping edge case for the original spaced names,
        # while keeping the source files read-only.
        source = tk.PhotoImage(data=base64.b64encode(asset.path.read_bytes()))
        source_width, source_height = source.width(), source.height()
        scale = min(target_width / source_width, target_height / source_height)
        if scale < 1:
            factor = max(1, int(1 / scale))
            image = source.subsample(factor, factor)
        elif scale > 1:
            factor = max(1, int(scale))
            image = source.zoom(factor, factor)
        else:
            image = source
        # The original files are already 800x480 and opaque. Tk's integer
        # scaling is deliberately conservative; the canvas supplies the
        # letterbox so aspect ratio is never distorted.
        return image

    def _draw(self) -> None:
        if self._root is None or self._canvas is None or self._rendered is None:
            return
        state, asset = self._rendered
        width = max(1, int(self._canvas.winfo_width()))
        height = max(1, int(self._canvas.winfo_height()))
        key = (asset.path, width, height, state.activity, state.expression, state.gaze)
        if key == self._last_draw_key:
            return
        cache_key = (asset.path, width, height)
        image = self._photo_cache.get(cache_key)
        if image is None:
            image = self._photo(asset, width, height)
            self._photo_cache[cache_key] = image
        self._image_ref = image
        self._canvas.delete("all")
        self._canvas.create_image(width // 2, height // 2, image=image, anchor="center")
        if self.gallery:
            self._canvas.create_text(
                12,
                height - 12,
                text=asset.key,
                anchor="sw",
                fill="#ffffff",
                font=("Segoe UI", 10),
            )
        self._last_draw_key = key

    def render(self, state: FaceState, asset: FaceAsset) -> None:
        self._rendered = (state, asset)
        self._draw()

    def _tick(self) -> None:
        if self._root is None:
            return
        if self._close_requested.is_set():
            self.close()
            return
        self.controller.pump()
        if self.gallery and self._rendered is not None and self.assets.assets:
            gallery_asset = self.assets.assets[int(time.monotonic()) % len(self.assets.assets)]
            state, current = self._rendered
            if gallery_asset.path != current.path:
                self._rendered = (state, gallery_asset)
                self._last_draw_key = None
                self._draw()
            self._after_id = self._root.after(33, self._tick)
            return
        # Frame selection is deterministic and remains in the GUI thread.
        if self._rendered is not None:
            state, current = self._rendered
            frames = self.assets.frames_for(state)
            if len(frames) > 1:
                interval = 0.12 if state.activity is FaceActivity.SPEAKING else 0.5
                frame = frames[int(time.monotonic() / interval) % len(frames)]
                if frame.path != current.path:
                    self._rendered = (state, frame)
                    self._last_draw_key = None
                    self._draw()
        self._after_id = self._root.after(33, self._tick)

    def run(self) -> None:
        self._create()
        assert self._root is not None
        self._tick()
        self._root.mainloop()

    start = run

    def request_close(self) -> None:
        """Request closure from a worker; actual Tk calls happen in ``_tick``."""

        self._close_requested.set()

    def close(self) -> None:
        if self._root is None:
            self._close_requested.set()
            if self.on_close is not None and not self._close_notified:
                self._close_notified = True
                try:
                    self.on_close()
                except Exception:
                    pass
            return
        if self._after_id is not None:
            try:
                self._root.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        root, self._root = self._root, None
        root.destroy()
        if self.on_close is not None and not self._close_notified:
            self._close_notified = True
            try:
                self.on_close()
            except Exception:
                pass
