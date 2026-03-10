# twxtools

**twxtools** is an interactive image and video sequence viewer built on top of [matplotlib](https://matplotlib.org/). It provides a convenient `twx()` function for quickly displaying, comparing, and navigating through images and video sequences in Python.

## Features

- Display single images or compare multiple images side-by-side
- Navigate video sequences frame-by-frame with keyboard shortcuts
- Automatic intensity range detection (with optional percentile clipping)
- Support for grayscale and RGB data
- Flexible data layout modes: `HW`, `HWC`, `CHW`, `NHW`, `NHWC`, `NCHW`
- Interactive colormap switching
- Accepts NumPy arrays, lists, tuples, or dicts

## Installation

```bash
pip install twxtools
```

## Quick Start

```python
import numpy as np
from twxtools import twx

# Display a single grayscale image
img = np.random.rand(256, 256)
twx(img)

# Compare two images
img1 = np.random.rand(256, 256)
img2 = np.random.rand(256, 256)
twx([img1, img2])

# Display a grayscale video sequence (N x H x W)
video = np.random.rand(30, 256, 256)
twx(video)

# Use a dict to label images automatically
twx({'original': img1, 'processed': img2})
```

## API Reference

### `twx(data, dataRange=[], cmap='gray', mode=None, titles=None)`

**Parameters:**

| Parameter   | Type                        | Description |
|-------------|-----------------------------|-------------|
| `data`      | `ndarray`, `list`, `tuple`, or `dict` | Image(s) or video sequence(s) to display. Each item can be a NumPy array of up to 4 dimensions. |
| `dataRange` | `list`                      | Intensity range(s) for display. See details below. |
| `cmap`      | `str`                       | Colormap for grayscale data (default: `'gray'`). Ignored for RGB data. |
| `mode`      | `str` or `None`             | Dimension layout of the input data. Auto-detected if `None`. |
| `titles`    | `list` of `str` or `None`   | Labels for each input. Auto-generated as `'1'`, `'2'`, … if not provided. When `data` is a dict, keys are used as titles. |

#### `dataRange` options

- `[]` — use `[min(data), max(data)]`
- `[P]` — use `[percentile(data, P), percentile(data, 100-P)]`
- `[low, high]` — use the given fixed range

#### `mode` options

| Mode   | Shape           | Description                                 |
|--------|-----------------|---------------------------------------------|
| `'HW'`   | `H × W`         | Grayscale image                             |
| `'HWC'`  | `H × W × C`     | RGB image (channels last)                   |
| `'CHW'`  | `C × H × W`     | RGB image (channels first)                  |
| `'NHW'`  | `N × H × W`     | Grayscale video sequence                    |
| `'NHWC'` | `N × H × W × C` | RGB video sequence (channels last)          |
| `'NCHW'` | `N × C × H × W` | RGB video sequence (channels first)         |

If `mode` is not provided, it is auto-detected from the array shape.

## Keyboard Controls

| Key         | Action                                   |
|-------------|------------------------------------------|
| `1`–`9`, `0` | Switch to the corresponding input (1–10) |
| `↑` / `↓`  | Previous / next frame in a video         |
| `a`         | Toggle colorbar                          |
| `m`         | Cycle through colormaps                  |

## Requirements

- Python ≥ 3.8
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)

## License

[MIT](LICENSE)
