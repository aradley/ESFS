# Backend Configuration

ESFS supports three computational backends for the ES matrix calculations:

| Backend | Hardware | Package |
|---------|----------|---------|
| CPU | Any | NumPy + Numba |
| CUDA | NVIDIA GPU | CuPy |
| MLX | Apple Silicon GPU (M1/M2/M3+) | MLX/Metal |

On import, ESFS auto-detects and selects the best available backend (CUDA → MLX → CPU) and prints a banner showing what is active. The ES-CCF and ES-FMG algorithms always run on CPU regardless of the active backend.

---

## Auto-detection

```python
import esfs
# Banner is printed automatically showing the active backend and precision
```

The detection priority is:
1. **CUDA/CuPy** — if CuPy is installed and a CUDA GPU is available
2. **MLX/Metal** — if MLX is installed and Apple Silicon is detected
3. **CPU (NumPy/Numba)** — fallback

---

## `use_cpu`

Force the CPU backend (NumPy/Numba).

```python
esfs.use_cpu()
```

No parameters. Updates all module-level backend references immediately.

---

## `use_gpu`

Force the GPU backend. Tries CUDA/CuPy first; if unavailable, falls back to MLX, then CPU.

```python
esfs.use_gpu()
```

No parameters. A warning is printed if CuPy is installed but CUDA libraries are not accessible (e.g. CUDA module not loaded on HPC).

---

## `use_mlx`

Force the MLX/Metal backend for Apple Silicon GPUs.

```python
esfs.use_mlx()
```

No parameters. Falls back to CPU with a warning if MLX is not available.

> **Note:** MLX does not support float64 precision. If you need float64, use the CPU backend.

---

## `configure`

Full backend configuration — set device and precision in one call.

```python
esfs.configure(gpu=True, upcast=False)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `gpu` | `bool` | — | If `True`, try to use GPU (same priority as `use_gpu()`). If `False`, force CPU. |
| `upcast` | `bool` | `False` | If `True`, use float64 precision. Default is float32 (recommended for memory efficiency). |
| `verbose` | `bool` | `True` | If `True`, print the backend banner after configuration. |

### Returns

`None`

### Notes

- Default precision is **float32**, which is sufficient for ES metric values (which are bounded between -1 and 1) and uses half the memory of float64.
- For float64 precision on Apple Silicon, you must use the CPU backend: `esfs.configure(gpu=False, upcast=True)`.

### Examples

```python
import esfs

# Force CPU with float32 (default precision)
esfs.configure(gpu=False)

# Force CPU with float64
esfs.configure(gpu=False, upcast=True)

# Auto-detect GPU with float32
esfs.configure(gpu=True)

# Auto-detect GPU with float64 (not supported on MLX)
esfs.configure(gpu=True, upcast=True)
```

---

## `get_backend_info`

Return a string describing the current backend and precision.

```python
info = esfs.get_backend_info()
print(info)
# e.g. "Backend: CUDA/CuPy (NVIDIA GPU), Precision: float32"
```

### Returns

`str` — a human-readable summary of the active backend and precision.
