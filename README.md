![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-green.svg)
![License](https://img.shields.io/badge/License-Apache--2.0-orange.svg)

# Abidin's Game of Death

> A Collatz-inspired cellular automaton where life and death are governed by integer dynamics rather than fixed Conway rules.

---

## Overview

Abidin's Game of Death is an experimental two-dimensional cellular automaton inspired by:

- Conway's Game of Life
- The Collatz Conjecture (3n + 1)
- Cellular automata
- Integer dynamics
- Emergent systems

Unlike Conway's Game of Life, this automaton does **not** use simple neighbor counts.

Every cell carries an integer state.

During each generation, neighboring cells interact through Collatz transformations, producing new candidate states. Survival is then determined dynamically using percentile thresholds.

The result is an evolving system that often produces organic-looking structures, clusters and wave fronts while remaining mathematically driven.

---

# Features

- Interactive Tkinter GUI
- Real-time simulation
- Collatz-based evolution rules
- Integer-valued cell states
- Multiple seed patterns
- Adjustable simulation speed
- Grid on/off
- Toroidal wrapping
- Multiple color palettes
- Percentile survival rule
- Percentile color mapping
- PNG export
- Live statistics panel
- Stopping-time visualization

---

# Core Idea

Instead of counting neighbors like Conway:

```
2-3 -> survive
3   -> birth
```

this project computes a new integer state.

For every cell:

1. Collect 8 neighboring cells
2. Apply one Collatz step to every live neighbor
3. Sum transformed values
4. Generate a candidate state
5. Compute the selected percentile over the active field
6. Compare the candidate against that threshold

```
candidate <= threshold
        ↓
      Death

candidate > threshold
        ↓
      Alive
```

This makes the automaton adaptive.

The survival threshold changes automatically as the system evolves.

---

# Color Modes

## Percentile Color

Maps stopping times according to their percentile.

Advantages:

- Uses the entire palette
- Better visual contrast
- Adapts automatically
- Suitable for exploration

---

## Log Scale

Maps stopping time using a logarithmic scale.

Advantages:

- Preserves absolute differences
- Stable across generations
- Useful for analysis

---

# Controls

| Control | Description |
|----------|-------------|
| Start | Start simulation |
| Stop | Stop simulation |
| Step | Single generation |
| Random | Generate random world |
| Seed | Load predefined pattern |
| Life Percentile | Survival threshold |
| Palette | Select color palette |
| Color Mode | Percentile or Log scale |
| Grid | Toggle grid |
| Wrap | Toroidal world |
| Export | Save PNG |

---

# Seed Patterns

- Abidin Spiral
- R-pentomino
- Collatz Cross
- Prime Sparks
- Nebula

---

# Statistics

The right panel displays:

- Generation
- Alive cells
- Births
- Deaths
- Average state
- Maximum state
- Life percentile
- Threshold
- Simulation speed

---

# Why?

The purpose of this project is **not** to reproduce Conway's Game of Life.

Instead, it explores what happens when

- integer dynamics,
- Collatz transformations,
- adaptive thresholds,
- and cellular automata

are combined into a single simulation.

The resulting behavior is intentionally different from Conway's original universe.

---

# Technologies

- Python 3
- Tkinter
- NumPy
- Pillow

---

# Future Ideas

- GPU acceleration
- 3D automata
- Additional integer rules
- Prime-number universes
- Multi-state automata
- Custom rule editor
- GIF / MP4 recording