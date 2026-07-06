#!/usr/bin/env python3
"""Abidin's Game of Death v2.3 — Collatz Cellular Automaton.

Tkinter + Python 3 cellular automaton.

This version fixes the first v2 issue: the rule no longer explodes into one
large saturated block. Each cell still carries an integer Collatz state, but the
next generation uses a bounded Collatz-neighborhood rule with density gates.
Colors show Collatz stopping time with selectable log or percentile color scaling.

Rules v2.1
----------
Dead cells have state 1. Live cells have state >= 2.

For each cell:
    1. Take the 8 neighbors.
    2. Collatz-step each live neighbor.
    3. Sum those transformed neighbor values: S.
    4. Add a small self-inertia term for already-live cells.
    5. Candidate = CollatzStep(S + inertia), then fold into a bounded range.
    6. Density gates decide birth/survival/death:
         - isolation and overcrowding die
         - births require 2-3 live neighbors and a Collatz residue trigger
         - survival requires 2-5 live neighbors and avoids a death residue
    7. Color = Collatz stopping time of the bounded candidate state.

Keyboard:
    Space       start/stop
    Right       step once
    R           randomize
    C           clear
    S           seed pattern
    G           toggle grid
    W           toggle wrap
    P           cycle palette
    E           export PNG
    + / -       speed up / slow down
    ?           help
"""
from __future__ import annotations

import math
import random
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageTk

APP_TITLE = "Abidin's Game of Death v2.3 — Collatz Cellular Automaton (Percentile Rule)"
DEAD = 1
DEFAULT_STATE_MOD = 9973
DEFAULT_STOPPING_CAP = 1000
COLOR_LOG_MAX = 4.0  # log10 scale: 1..10,000; prevents yellow/white dominance

SIZE_OPTIONS = {
    "120 x 70": (120, 70),
    "160 x 90": (160, 90),
    "200 x 110": (200, 110),
    "240 x 135": (240, 135),
}
CELL_OPTIONS = [5, 6, 7, 8, 9, 10, 12]
SPEED_OPTIONS = [40, 80, 120, 180, 240, 360, 500]
PALETTES = ["Collatz Fire", "Purple Gold", "Death Neon", "Ice Furnace", "Grayscale"]
COLOR_MODES = ["Percentile color", "Log scale"]
SEED_PATTERNS = ["Abidin Spiral", "R-pentomino", "Collatz Cross", "Prime Sparks", "Nebula"]

BG = "#0b0b0b"
PANEL = "#151515"
PANEL_2 = "#202020"
FG = "#f2f2f2"
MUTED = "#b7b7b7"
GREEN = "#57e86a"
GRID = "#1b1b1b"
GRID_STRONG = "#2a2a2a"

PALETTE_STOPS = {
    "Collatz Fire": ["#05030c", "#21005a", "#6500a8", "#d91578", "#ff4d35", "#ffb62e", "#fff3a3"],
    "Purple Gold": ["#05020d", "#29104a", "#69247c", "#da498d", "#fac67a", "#fff8c9"],
    "Death Neon": ["#020204", "#150050", "#3f0071", "#fb2576", "#ff7b00", "#faff00"],
    "Ice Furnace": ["#000814", "#001d3d", "#0077b6", "#90e0ef", "#ffb703", "#fb5607"],
    "Grayscale": ["#050505", "#222222", "#555555", "#999999", "#dddddd", "#ffffff"],
}


@dataclass
class Metrics:
    generation: int = 0
    alive: int = 0
    births: int = 0
    deaths: int = 0
    avg_state: float = 0.0
    max_state: int = 1
    step_ms: float = 0.0
    life_threshold: float = 0.0
    life_percentile: float = 35.0


def collatz_step_int(n: int) -> int:
    if n <= 1:
        return 1
    return n // 2 if n % 2 == 0 else 3 * n + 1


def collatz_step_array(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64)
    return np.where(a <= 1, 1, np.where((a % 2) == 0, a // 2, 3 * a + 1))


def stopping_time_array(values: np.ndarray, cap: int = DEFAULT_STOPPING_CAP) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    n = values.copy()
    out = np.zeros(values.shape, dtype=np.int16)
    active = n > 1
    for step in range(1, cap + 1):
        if not active.any():
            break
        n_active = n[active]
        n[active] = np.where((n_active % 2) == 0, n_active // 2, 3 * n_active + 1)
        newly_done = active & (n == 1)
        out[newly_done] = step
        active = active & (n != 1)
    out[active] = cap
    return out


class AbidinAutomaton:
    def __init__(self, cols: int = 160, rows: int = 90, state_mod: int = DEFAULT_STATE_MOD):
        self.cols = cols
        self.rows = rows
        self.state_mod = state_mod
        self.states = np.ones((rows, cols), dtype=np.int64)
        self.metrics = Metrics()

    def resize(self, cols: int, rows: int) -> None:
        self.cols = cols
        self.rows = rows
        self.states = np.ones((rows, cols), dtype=np.int64)
        self.metrics = Metrics()

    def clear(self) -> None:
        self.states.fill(DEAD)
        self.metrics = Metrics()

    def randomize(self, density: float = 0.12, min_state: int = 2, max_state: int = 1600) -> None:
        mask = np.random.random((self.rows, self.cols)) < density
        # Heavy-tailed but bounded initial states: many small sparks, a few large hot cells.
        small = np.random.randint(min_state, 180, size=(self.rows, self.cols), dtype=np.int64)
        large = np.random.randint(180, max_state + 1, size=(self.rows, self.cols), dtype=np.int64)
        choose_large = np.random.random((self.rows, self.cols)) < 0.12
        values = np.where(choose_large, large, small)
        self.states = np.where(mask, values, DEAD).astype(np.int64)
        self.metrics.generation = 0
        self._refresh_alive_metrics(0.0, births=int(mask.sum()), deaths=0)

    def seed_pattern(self, name: str) -> None:
        self.clear()
        cx, cy = self.cols // 2, self.rows // 2

        def put(x: int, y: int, value: int) -> None:
            if 0 <= x < self.cols and 0 <= y < self.rows:
                self.states[y, x] = max(2, int(value))

        if name == "R-pentomino":
            pts = [(0, -1), (1, -1), (-1, 0), (0, 0), (0, 1)]
            vals = [27, 7, 19, 871, 97]
            for (dx, dy), v in zip(pts, vals):
                put(cx + dx, cy + dy, v)

        elif name == "Collatz Cross":
            vals = [3, 5, 7, 11, 27, 31, 41, 63, 97, 129, 255]
            for i, v in enumerate(vals):
                d = i - len(vals) // 2
                put(cx + d, cy, v)
                put(cx, cy + d, v * 2 + 1)

        elif name == "Prime Sparks":
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
            for _ in range(max(350, (self.cols * self.rows) // 35)):
                x = random.randrange(self.cols)
                y = random.randrange(self.rows)
                put(x, y, random.choice(primes) * random.choice([1, 3, 9, 27]))

        elif name == "Nebula":
            # Several sparse hot clouds, closer to the intended screenshot-like flow.
            for _ in range(10):
                bx = random.randrange(self.cols)
                by = random.randrange(self.rows)
                radius = random.randint(5, 15)
                for yy in range(by - radius, by + radius + 1):
                    for xx in range(bx - radius, bx + radius + 1):
                        if 0 <= xx < self.cols and 0 <= yy < self.rows:
                            dist = math.hypot(xx - bx, yy - by) / max(radius, 1)
                            p = max(0.0, 0.26 * (1.0 - dist))
                            if random.random() < p:
                                put(xx, yy, random.choice([7, 11, 19, 27, 31, 63, 97, 127, 255, 511, 871]))

        else:  # Abidin Spiral: sparse spiral, not a dense square.
            for t in range(650):
                angle = t * 0.31
                r = 0.026 * t
                x = int(cx + math.cos(angle) * r * self.cols / 7)
                y = int(cy + math.sin(angle) * r * self.rows / 7)
                if random.random() < 0.74:
                    value = 2 + ((t * 3 + 1) % 997)
                    put(x, y, value)
                if t % 17 == 0:
                    put(x + 1, y, 3 * t + 7)
                if t % 29 == 0:
                    put(x, y + 1, 2 * t + 11)

        self.metrics.generation = 0
        self._refresh_alive_metrics(0.0, births=int(np.sum(self.states > DEAD)), deaths=0)

    def toggle_cell(self, col: int, row: int) -> None:
        if not (0 <= col < self.cols and 0 <= row < self.rows):
            return
        if self.states[row, col] > DEAD:
            self.states[row, col] = DEAD
        else:
            self.states[row, col] = random.choice([7, 11, 19, 27, 31, 63, 97, 127, 255, 871])
        self._refresh_alive_metrics(self.metrics.step_ms, 0, 0)

    def step(self, wrap: bool = False, life_percentile: float = 35.0) -> Metrics:
        """Advance one generation using a Collatz percentile live/death rule.

        The important v2.2 idea is simple: every cell receives a candidate integer
        state from its 8-neighbor Collatz field. Then we compute a percentile
        threshold over the active field. Candidate states above that threshold live;
        states at or below it die. The toolbar slider/spinbox controls that
        percentile interactively.
        """
        t0 = time.perf_counter()
        old = self.states
        alive_old = old > DEAD

        neighbor_live_count = self._neighbor_sum(alive_old.astype(np.int64), wrap=wrap)
        neighbor_collatz_sum = self._neighbor_sum(np.where(alive_old, collatz_step_array(old), 0), wrap=wrap)

        # Self-inertia keeps moving fronts coherent without allowing one solid block
        # to take over the whole grid.
        self_inertia = np.where(alive_old, collatz_step_array(old) // 5, 0)
        raw = neighbor_collatz_sum + self_inertia
        candidate_raw = collatz_step_array(raw)

        # Fold to a bounded range so the percentile threshold is stable.
        folded = 2 + (candidate_raw % self.state_mod)
        stop = stopping_time_array(folded, cap=DEFAULT_STOPPING_CAP)

        active_field = (neighbor_live_count > 0) | alive_old
        pct = float(np.clip(life_percentile, 1.0, 99.0))
        if np.any(active_field):
            threshold = float(np.percentile(folded[active_field], pct))
        else:
            threshold = 2.0

        # Main rule requested by the user:
        #     candidate <= percentile threshold -> death
        #     candidate > percentile threshold  -> live
        percentile_gate = folded > threshold

        # Local density gates preserve cellular-automaton structure. Without these,
        # the percentile rule alone becomes noisy static. These gates make flow.
        birth_density = np.isin(neighbor_live_count, [2, 3, 4])
        survival_density = (neighbor_live_count >= 1) & (neighbor_live_count <= 5)

        # Soft Collatz residue gates add texture but do not override the percentile
        # rule. Higher percentile values create a harsher/deadlier universe.
        birth_residue = np.isin((candidate_raw + stop) % 13, [1, 5, 8])
        survival_residue = ~np.isin((candidate_raw + neighbor_live_count) % 17, [0, 11, 13])
        not_too_hot = stop < 260

        birth = (~alive_old) & birth_density & percentile_gate & birth_residue & not_too_hot
        survive = alive_old & survival_density & percentile_gate & survival_residue & not_too_hot

        # Rare sparks keep sparse seeds alive, but the percentile rule still controls
        # the dominant behavior.
        spark = (np.random.random(old.shape) < 0.00020) & (neighbor_live_count == 0)

        alive_new = birth | survive | spark
        new = np.where(alive_new, folded, DEAD).astype(np.int64)
        if spark.any():
            new[spark] = np.random.choice([7, 11, 19, 27, 31, 63, 97, 127, 255], size=int(spark.sum()))

        births = int(np.sum(alive_new & ~alive_old))
        deaths = int(np.sum(~alive_new & alive_old))
        self.states = new
        self.metrics.generation += 1
        self._refresh_alive_metrics((time.perf_counter() - t0) * 1000.0, births, deaths)
        self.metrics.life_threshold = threshold
        self.metrics.life_percentile = pct
        return self.metrics

    def _neighbor_sum(self, arr: np.ndarray, wrap: bool) -> np.ndarray:
        if wrap:
            total = np.zeros_like(arr, dtype=np.int64)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx or dy:
                        total += np.roll(np.roll(arr, dy, axis=0), dx, axis=1)
            return total
        padded = np.pad(arr, ((1, 1), (1, 1)), mode="constant", constant_values=0)
        return (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
            + padded[1:-1, :-2] + padded[1:-1, 2:]
            + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        )

    def _refresh_alive_metrics(self, step_ms: float, births: int, deaths: int) -> None:
        alive = self.states[self.states > DEAD]
        self.metrics.alive = int(alive.size)
        self.metrics.births = births
        self.metrics.deaths = deaths
        self.metrics.avg_state = float(alive.mean()) if alive.size else 0.0
        self.metrics.max_state = int(alive.max()) if alive.size else DEAD
        self.metrics.step_ms = step_ms


def interpolate_hex(stops: list[str], x: np.ndarray) -> np.ndarray:
    rgb_stops = np.array([tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in stops], dtype=np.float32)
    x = np.clip(x, 0.0, 1.0)
    scaled = x * (len(stops) - 1)
    idx = np.floor(scaled).astype(np.int32)
    idx2 = np.clip(idx + 1, 0, len(stops) - 1)
    frac = (scaled - idx)[..., None]
    idx = np.clip(idx, 0, len(stops) - 1)
    return (rgb_stops[idx] * (1 - frac) + rgb_stops[idx2] * frac).astype(np.uint8)


def states_to_image(states: np.ndarray, cell_px: int, palette: str, show_grid: bool, color_mode: str = "Percentile color") -> Image.Image:
    """Render states by Collatz stopping time.

    Color modes:
      - Percentile color: ranks active stopping times by percentile, so every
        generation uses the full palette. This gives richer visuals and avoids
        the near-monochrome problem when most stopping times are similar.
      - Log scale: fixed base-10 scale, useful for comparing colors across
        generations with the same meaning.
    """
    alive = states > DEAD
    stop = stopping_time_array(states, cap=DEFAULT_STOPPING_CAP).astype(np.float32)

    x = np.zeros_like(stop, dtype=np.float32)
    if alive.any():
        alive_stop = np.maximum(stop[alive], 1.0)
        if color_mode == "Log scale":
            x[alive] = np.clip(np.log10(alive_stop) / COLOR_LOG_MAX, 0.0, 1.0)
        else:
            # Percentile/rank color scaling. This is intentionally visual rather
            # than absolute: it spreads the current generation over the palette.
            # Stable sort makes ties deterministic; average rank is not needed
            # here because slight color variation is visually useful.
            order = np.argsort(alive_stop, kind="mergesort")
            ranks = np.empty_like(order, dtype=np.float32)
            if alive_stop.size == 1:
                ranks[order] = 0.5
            else:
                ranks[order] = np.linspace(0.05, 1.0, alive_stop.size, dtype=np.float32)
            # Gentle gamma lifts mid/high values into orange/yellow without making
            # the whole field white.
            x[alive] = np.power(np.clip(ranks, 0.0, 1.0), 0.85)

    colors = interpolate_hex(PALETTE_STOPS.get(palette, PALETTE_STOPS["Collatz Fire"]), x)
    colors[~alive] = np.array([4, 4, 5], dtype=np.uint8)
    img_arr = np.repeat(np.repeat(colors, cell_px, axis=0), cell_px, axis=1)
    img = Image.fromarray(img_arr, mode="RGB")

    if show_grid and cell_px >= 4:
        draw = ImageDraw.Draw(img)
        w, h = img.size
        for xpx in range(0, w, cell_px):
            color = GRID_STRONG if (xpx // cell_px) % 10 == 0 else GRID
            draw.line([(xpx, 0), (xpx, h)], fill=color)
        for ypx in range(0, h, cell_px):
            color = GRID_STRONG if (ypx // cell_px) % 10 == 0 else GRID
            draw.line([(0, ypx), (w, ypx)], fill=color)
    return img

class AbidinsGameApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.configure(bg=BG)
        self.root.minsize(1180, 720)

        self.running = False
        self.after_id: str | None = None
        self.last_image: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.legend_photo: ImageTk.PhotoImage | None = None
        self.display_cell_px = 8

        self.size_var = tk.StringVar(value="160 x 90")
        cols, rows = SIZE_OPTIONS[self.size_var.get()]
        self.engine = AbidinAutomaton(cols=cols, rows=rows)

        self.cell_var = tk.IntVar(value=8)
        self.speed_var = tk.IntVar(value=120)
        self.palette_var = tk.StringVar(value="Collatz Fire")
        self.color_mode_var = tk.StringVar(value="Percentile color")
        self.pattern_var = tk.StringVar(value="Nebula")
        self.grid_var = tk.BooleanVar(value=True)
        self.wrap_var = tk.BooleanVar(value=False)
        self.density_var = tk.DoubleVar(value=0.12)
        self.life_percentile_var = tk.DoubleVar(value=35.0)

        self._build_theme()
        self._build_layout()
        self._bind_keys()
        self.engine.seed_pattern(self.pattern_var.get())
        self.root.after(100, self._render)

    def _build_theme(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("Panel.TLabel", background=PANEL, foreground=FG)
        style.configure("Green.TLabel", background=PANEL, foreground=GREEN)
        style.configure("TButton", background=PANEL_2, foreground=FG, borderwidth=1, padding=(10, 5))
        style.map("TButton", background=[("active", "#303030")])
        style.configure("TCheckbutton", background=PANEL, foreground=FG)
        style.configure(
            "Readable.TCombobox",
            fieldbackground="#2a2a2a",
            background="#2a2a2a",
            foreground="#ffffff",
            arrowcolor="#ffffff",
            selectbackground="#3a3a3a",
            selectforeground="#ffffff",
            bordercolor="#5a5a5a",
            lightcolor="#5a5a5a",
            darkcolor="#1a1a1a",
        )
        style.map(
            "Readable.TCombobox",
            fieldbackground=[("readonly", "#2a2a2a"), ("disabled", "#2a2a2a")],
            foreground=[("readonly", "#ffffff"), ("disabled", "#d0d0d0")],
            background=[("readonly", "#2a2a2a"), ("active", "#353535")],
            selectforeground=[("readonly", "#ffffff")],
            selectbackground=[("readonly", "#3a3a3a")],
        )

    def _build_layout(self) -> None:
        # Two toolbar rows. Earlier builds put every control in one row; on
        # normal laptop widths the Life % control could be pushed off-screen.
        toolbar_outer = ttk.Frame(self.root, style="Panel.TFrame", padding=(8, 7, 8, 5))
        toolbar_outer.pack(side=tk.TOP, fill=tk.X)

        toolbar = ttk.Frame(toolbar_outer, style="Panel.TFrame")
        toolbar.pack(side=tk.TOP, fill=tk.X)
        toolbar2 = ttk.Frame(toolbar_outer, style="Panel.TFrame")
        toolbar2.pack(side=tk.TOP, fill=tk.X, pady=(6, 0))

        for text, cmd in [("Start", self.start), ("Stop", self.stop), ("Step", self.step_once), ("Clear", self.clear), ("Random", self.randomize)]:
            self._button(toolbar, text, cmd).pack(side=tk.LEFT, padx=4)

        ttk.Label(toolbar, text="Seed:", style="Panel.TLabel").pack(side=tk.LEFT, padx=(12, 4))
        self._combo(toolbar, self.pattern_var, SEED_PATTERNS, 14, lambda _=None: self.seed()).pack(side=tk.LEFT)
        ttk.Label(toolbar, text="Size:", style="Panel.TLabel").pack(side=tk.LEFT, padx=(14, 4))
        self._combo(toolbar, self.size_var, list(SIZE_OPTIONS), 10, lambda _=None: self.resize_grid()).pack(side=tk.LEFT)
        ttk.Label(toolbar, text="Cell:", style="Panel.TLabel").pack(side=tk.LEFT, padx=(14, 4))
        self._combo(toolbar, self.cell_var, CELL_OPTIONS, 6, lambda _=None: self._render()).pack(side=tk.LEFT)
        ttk.Label(toolbar, text="Speed:", style="Panel.TLabel").pack(side=tk.LEFT, padx=(14, 4))
        self._combo(toolbar, self.speed_var, SPEED_OPTIONS, 7).pack(side=tk.LEFT)
        ttk.Label(toolbar, text="ms", style="Panel.TLabel").pack(side=tk.LEFT, padx=(3, 0))
        ttk.Label(toolbar, text="Palette:", style="Panel.TLabel").pack(side=tk.LEFT, padx=(14, 4))
        self._combo(toolbar, self.palette_var, PALETTES, 13, lambda _=None: self._render()).pack(side=tk.LEFT)
        ttk.Label(toolbar, text="Color:", style="Panel.TLabel").pack(side=tk.LEFT, padx=(12, 4))
        self._combo(toolbar, self.color_mode_var, COLOR_MODES, 14, lambda _=None: self._render()).pack(side=tk.LEFT)
        ttk.Checkbutton(toolbar, text="Grid", variable=self.grid_var, command=self._render).pack(side=tk.LEFT, padx=(8, 4))
        ttk.Checkbutton(toolbar, text="Wrap", variable=self.wrap_var).pack(side=tk.LEFT, padx=4)
        self._button(toolbar, "?", self.show_help).pack(side=tk.RIGHT, padx=4)

        # Percentile controls are intentionally on the second row so they are
        # always visible. This is the main v2.3 experiment control.
        ttk.Label(toolbar2, text="Life percentile:", style="Panel.TLabel").pack(side=tk.LEFT, padx=(4, 6))
        self.life_spin = tk.Spinbox(
            toolbar2, from_=1, to=99, increment=1, width=5,
            textvariable=self.life_percentile_var, command=self._render,
            bg="#252525", fg="#ffffff", insertbackground="#ffffff",
            buttonbackground="#3a3a3a", highlightbackground="#666666",
            relief="solid", bd=1, justify="right", format="%.0f"
        )
        self.life_spin.pack(side=tk.LEFT)
        self.life_scale = tk.Scale(
            toolbar2, from_=1, to=99, orient=tk.HORIZONTAL, length=260,
            variable=self.life_percentile_var, command=lambda _=None: self._render(),
            bg=PANEL, fg=FG, troughcolor="#2a2a2a", activebackground="#3a3a3a",
            highlightthickness=0, showvalue=False, sliderlength=18
        )
        self.life_scale.pack(side=tk.LEFT, padx=(8, 12))
        ttk.Label(
            toolbar2,
            text="lower = denser / higher = deadlier | candidate <= threshold dies",
            style="Panel.TLabel"
        ).pack(side=tk.LEFT, padx=(4, 0))

        main = ttk.Frame(self.root, style="TFrame")
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(main, bg="#050505", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_click)
        self.canvas.bind("<Configure>", lambda _e: self._render())

        side_outer = ttk.Frame(main, style="Panel.TFrame", width=310)
        side_outer.pack(side=tk.RIGHT, fill=tk.Y)
        side_outer.pack_propagate(False)

        # Scrollable side panel, so the explanation always fits on smaller screens.
        side_canvas = tk.Canvas(side_outer, bg=PANEL, highlightthickness=0, width=292)
        side_scroll = ttk.Scrollbar(side_outer, orient="vertical", command=side_canvas.yview)
        self.side_canvas = side_canvas
        self.side = ttk.Frame(side_canvas, style="Panel.TFrame", padding=12)
        self.side.bind("<Configure>", lambda e: side_canvas.configure(scrollregion=side_canvas.bbox("all")))
        side_canvas.create_window((0, 0), window=self.side, anchor="nw")
        side_canvas.configure(yscrollcommand=side_scroll.set)
        side_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        side_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        side_canvas.bind("<Enter>", lambda _e: side_canvas.bind_all("<MouseWheel>", self._on_side_mousewheel))
        side_canvas.bind("<Leave>", lambda _e: side_canvas.unbind_all("<MouseWheel>"))

        self.legend_canvas = tk.Canvas(self.side, width=258, height=232, bg=PANEL, highlightthickness=1, highlightbackground="#303030")
        self.legend_canvas.pack(fill=tk.X, pady=(0, 12))

        stats = ttk.Frame(self.side, style="Panel.TFrame", padding=10)
        stats.pack(fill=tk.X, pady=(0, 12))
        self.stat_labels: dict[str, ttk.Label] = {}
        for label in ["Generation", "Alive Cells", "Avg State", "Max State", "Births", "Deaths", "Life %", "Threshold", "Step"]:
            row = ttk.Frame(stats, style="Panel.TFrame")
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"{label}:", width=11, style="Panel.TLabel").pack(side=tk.LEFT)
            value = ttk.Label(row, text="-", style="Green.TLabel")
            value.pack(side=tk.RIGHT)
            self.stat_labels[label] = value

        rules_text = (
            "Rules v2.3\n\n"
            "For every cell, the 8-neighbor Collatz field creates a candidate integer. "
            "The Life percentile slider sets the death/live cutoff.\n\n"
            "candidate <= threshold: dead\n"
            "candidate > threshold: can live\n\n"
            "Local density and Collatz residue gates keep the world from turning into pure noise. "
            "Color is stopping time: dark = short path to 1, bright = long path to 1. Percentile color spreads each generation over the full palette; Log scale keeps fixed absolute meaning.\n\n"
            "Wrap ON makes the grid toroidal: left/right and top/bottom edges connect. "
            "Wrap OFF treats outside-grid neighbors as dead."
        )
        self.rules_label = ttk.Label(self.side, text=rules_text, style="Panel.TLabel", justify=tk.LEFT, wraplength=255)
        self.rules_label.pack(fill=tk.X, pady=(0, 12), anchor="w")
        self._button(self.side, "Export PNG", self.export_png).pack(fill=tk.X, pady=(4, 0))

        footer = ttk.Frame(self.root, style="Panel.TFrame", padding=(12, 8))
        footer.pack(side=tk.BOTTOM, fill=tk.X)
        shortcuts = "Space: start/stop    Right: step    R: random    C: clear    S: seed    +/-: speed    G: grid    W: wrap    P: palette    E: export    ?: help"
        ttk.Label(footer, text=shortcuts, style="Panel.TLabel").pack(side=tk.LEFT)

    def _button(self, parent, text: str, command: Callable) -> ttk.Button:
        return ttk.Button(parent, text=text, command=command)

    def _combo(self, parent, var, values, width: int, command: Callable | None = None) -> tk.OptionMenu:
        """Readable dark dropdown.

        ttk.Combobox is unreliable on Windows dark themes: readonly fields can
        stay white/grey and become unreadable. A styled tk.OptionMenu is simpler
        and visibly consistent.
        """
        menu = tk.OptionMenu(parent, var, *values, command=(command if command else None))
        menu.configure(
            bg="#252525",
            fg="#ffffff",
            activebackground="#3a3a3a",
            activeforeground="#ffffff",
            highlightthickness=1,
            highlightbackground="#5a5a5a",
            relief="solid",
            bd=1,
            width=width,
            padx=6,
        )
        menu["menu"].configure(
            bg="#202020",
            fg="#ffffff",
            activebackground="#3a5f9f",
            activeforeground="#ffffff",
            tearoff=False,
        )
        return menu

    def _on_side_mousewheel(self, event) -> None:
        # Windows/macOS mouse wheel scrolling for the right explanation panel.
        delta = -1 * int(event.delta / 120) if event.delta else 0
        self.side_canvas.yview_scroll(delta, "units")

    def _bind_keys(self) -> None:
        self.root.bind("<space>", lambda e: self.toggle_running())
        self.root.bind("<Right>", lambda e: self.step_once())
        for key in ("r", "R"):
            self.root.bind(key, lambda e: self.randomize())
        for key in ("c", "C"):
            self.root.bind(key, lambda e: self.clear())
        for key in ("s", "S"):
            self.root.bind(key, lambda e: self.seed())
        for key in ("g", "G"):
            self.root.bind(key, lambda e: self.toggle_grid())
        for key in ("w", "W"):
            self.root.bind(key, lambda e: self.toggle_wrap())
        for key in ("p", "P"):
            self.root.bind(key, lambda e: self.cycle_palette())
        for key in ("e", "E"):
            self.root.bind(key, lambda e: self.export_png())
        self.root.bind("+", lambda e: self.change_speed(-40))
        self.root.bind("=", lambda e: self.change_speed(-40))
        self.root.bind("-", lambda e: self.change_speed(40))
        self.root.bind("?", lambda e: self.show_help())

    def _on_canvas_click(self, event) -> None:
        col = int((event.x - self.image_offset_x) // max(1, self.display_cell_px))
        row = int((event.y - self.image_offset_y) // max(1, self.display_cell_px))
        self.engine.toggle_cell(col, row)
        self._render()

    def start(self) -> None:
        if not self.running:
            self.running = True
            self._loop()

    def stop(self) -> None:
        self.running = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def toggle_running(self) -> None:
        self.stop() if self.running else self.start()

    def _loop(self) -> None:
        if not self.running:
            return
        self.step_once()
        self.after_id = self.root.after(int(self.speed_var.get()), self._loop)

    def step_once(self) -> None:
        self.engine.step(wrap=self.wrap_var.get(), life_percentile=float(self.life_percentile_var.get()))
        self._render()

    def clear(self) -> None:
        self.stop()
        self.engine.clear()
        self._render()

    def randomize(self) -> None:
        self.stop()
        self.engine.randomize(density=float(self.density_var.get()))
        self._render()

    def seed(self) -> None:
        self.stop()
        self.engine.seed_pattern(self.pattern_var.get())
        self._render()

    def resize_grid(self) -> None:
        self.stop()
        cols, rows = SIZE_OPTIONS[self.size_var.get()]
        self.engine.resize(cols, rows)
        self.engine.seed_pattern(self.pattern_var.get())
        self._render()

    def toggle_grid(self) -> None:
        self.grid_var.set(not self.grid_var.get())
        self._render()

    def toggle_wrap(self) -> None:
        self.wrap_var.set(not self.wrap_var.get())

    def cycle_palette(self) -> None:
        idx = PALETTES.index(self.palette_var.get())
        self.palette_var.set(PALETTES[(idx + 1) % len(PALETTES)])
        self._render()

    def change_speed(self, delta: int) -> None:
        self.speed_var.set(max(20, min(1000, int(self.speed_var.get()) + delta)))

    def _effective_cell_px(self) -> int:
        requested = int(self.cell_var.get())
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        if cw < 100 or ch < 100:
            return requested
        fit = max(2, min((cw - 8) // self.engine.cols, (ch - 8) // self.engine.rows))
        return max(2, min(requested, fit))

    def _render(self) -> None:
        self.display_cell_px = self._effective_cell_px()
        img = states_to_image(
            self.engine.states,
            self.display_cell_px,
            self.palette_var.get(),
            self.grid_var.get(),
            self.color_mode_var.get(),
        )
        self.last_image = img
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        self.image_offset_x = max(0, (cw - img.width) // 2)
        self.image_offset_y = max(0, (ch - img.height) // 2)
        self.canvas.create_image(self.image_offset_x, self.image_offset_y, image=self.photo, anchor="nw")
        self._update_stats()
        self._draw_legend()

    def _update_stats(self) -> None:
        m = self.engine.metrics
        self.stat_labels["Generation"].config(text=f"{m.generation:,}")
        self.stat_labels["Alive Cells"].config(text=f"{m.alive:,}")
        self.stat_labels["Avg State"].config(text=f"{m.avg_state:,.2f}")
        self.stat_labels["Max State"].config(text=f"{m.max_state:,}")
        self.stat_labels["Births"].config(text=f"{m.births:,}")
        self.stat_labels["Deaths"].config(text=f"{m.deaths:,}")
        self.stat_labels["Life %"].config(text=f"{m.life_percentile:.0f}")
        self.stat_labels["Threshold"].config(text=f"{m.life_threshold:,.0f}")
        self.stat_labels["Step"].config(text=f"{m.step_ms:.1f} ms")

    def _draw_legend(self) -> None:
        c = self.legend_canvas
        c.delete("all")
        c.create_text(129, 20, text="Stopping Time (steps)", fill=FG, font=("Segoe UI", 10, "bold"))
        scale_name = "log10 scale" if self.color_mode_var.get() == "Log scale" else "percentile color"
        c.create_text(129, 39, text=scale_name, fill=MUTED, font=("Segoe UI", 9))
        stops = PALETTE_STOPS.get(self.palette_var.get(), PALETTE_STOPS["Collatz Fire"])
        h, w = 128, 24
        x0, y0 = 96, 62
        # Top of legend is x=1.0, bottom is x=0.0.
        vals = np.linspace(1, 0, h).reshape(h, 1)
        colors = interpolate_hex(stops, vals)
        grad = np.repeat(colors, w, axis=1)
        img = Image.fromarray(grad, mode="RGB")
        self.legend_photo = ImageTk.PhotoImage(img)
        c.create_image(x0, y0, image=self.legend_photo, anchor="nw")
        if self.color_mode_var.get() == "Log scale":
            ticks = [
                ("1000+", y0 + 0),
                ("100", y0 + 32),
                ("10", y0 + 64),
                ("3", y0 + 94),
                ("1", y0 + 126),
            ]
        else:
            ticks = [
                ("p100", y0 + 0),
                ("p75", y0 + 32),
                ("p50", y0 + 64),
                ("p25", y0 + 96),
                ("p0", y0 + 126),
            ]
        for txt, yy in ticks:
            c.create_text(x0 + 38, yy + 3, text=txt, fill=FG, anchor="w", font=("Segoe UI", 10))
        c.create_text(129, 210, text="white/yellow = long stopping time", fill=MUTED, font=("Segoe UI", 8))

    def export_png(self) -> None:
        if self.last_image is None:
            return
        default = f"abidins_game_of_death_v2_gen_{self.engine.metrics.generation}.png"
        path = filedialog.asksaveasfilename(title="Export PNG", defaultextension=".png", initialfile=default, filetypes=[("PNG image", "*.png")])
        if path:
            self.last_image.save(path, optimize=True)

    def show_help(self) -> None:
        messagebox.showinfo(
            "Abidin's Game of Death v2.3",
            "This version uses a Collatz percentile rule.\n\n"
            "Every cell receives a candidate integer state from the 8-neighbor "
            "Collatz field. The Life % control computes that percentile over the "
            "active field. Candidate <= threshold dies; candidate > threshold can live.\n\n"
            "Lower Life % values create denser worlds. Higher values create harsher, "
            "sparser worlds. Local density and residue gates preserve flow.\n\n"
            "Color mode: Percentile color ranks active stopping times within the current generation and uses the full palette. Log scale uses a fixed log10 scale for absolute comparison across generations.\n\n"
            "Wrap ON connects the left/right and top/bottom edges, making the grid toroidal. Wrap OFF treats outside-grid neighbors as dead.",
        )

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    AbidinsGameApp().run()
