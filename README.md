# Acuity

**Learned Perceptual Quality Assessment for Triangle Mesh LOD Selection**

![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue)
![Vulkan](https://img.shields.io/badge/Vulkan-1.4-red)
![Python](https://img.shields.io/badge/Python-3.13-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

Acuity is a real-time rendering system that replaces distance-based Level of Detail (LOD) selection with a small neural network that predicts **perceived visual quality**. It was built end to end for an M.Tech dissertation: a custom C++17/Vulkan renderer, a from-scratch Quadric Error Metric (QEM) simplifier, an automated dataset pipeline, a PyTorch model, and ONNX Runtime inference inside the render loop.

> **Research question:** Can a learned perceptual quality model choose mesh LODs better than a geometric-error heuristic, in real time and at negligible cost?

<!-- Add a short demo GIF here: docs/screenshots/demo.gif -->
![Acuity demo](docs/screenshots/acuity.gif)

---

## Key Results

| Metric | Geometric baseline | Acuity (learned) |
|---|---|---|
| SRCC vs. perceptual quality score | 0.7145 | **0.9224** |
| MSE | 0.210392 | **0.001224** (171x lower) |
| R² | -79.3 | **0.92+** |
| Inference time per mesh (ONNX Runtime) | n/a | **0.0062 ms** (161x under a 1 ms budget) |
| FPS overhead vs. baseline | n/a | **0% measurable** |

- All three LOD methods (Geometric, Perceptual, Oracle) sustained 3,500+ FPS across the four test scenes (Bunny, Dragon, Sponza, San Miguel), well above the 60 FPS target.
- Memory savings are **scene-dependent**. For example, Sponza showed roughly 50% triangle reduction under LOD switching, while in some scenes the perceptual selector matched the oracle's triangle count to preserve quality. See the thesis for the per-scene breakdown.

---

## How It Works

```
Meshes ──► QEM simplification ──► 5 LOD levels (100%, 50%, 25%, 12.5%, 6.25%)
                                        │
                                        ▼
                  Batch renderer (14 meshes × 5 LODs × 40 camera poses = 2,800 images)
                                        │
                                        ▼
                  Image-quality model (TOPIQ) ──► perceptual quality labels
                                        │
                                        ▼
        38-D features (geometric + perceptual + view-dependent) ──► MLP (~25K params)
                                        │
                                        ▼
                         Export to ONNX ──► ONNX Runtime in C++
                                        │
                                        ▼
        PerceptualLODSelector (spatial-hash cache + hysteresis) in the Vulkan render loop
```

1. **LOD generation:** Quadric Error Metrics (Garland & Heckbert, 1997), implemented from scratch in C++.
2. **Dataset:** an automated pipeline renders every mesh/LOD/pose combination and annotates it with an image-quality model as a perceptual proxy.
3. **Features:** 38 dimensions across three groups: geometric, perceptual (curvature, saliency, normal variation), and view-dependent (distance, angle, screen coverage).
4. **Model:** a lightweight MLP trained in PyTorch, exported to ONNX.
5. **Runtime selection:** the C++ renderer extracts the same features, runs ONNX inference, and picks a LOD. Predictions are cached with a spatial hash and stabilised with hysteresis to avoid LOD popping.
6. **Evaluation:** an automated benchmark suite compares Geometric, Perceptual, and Oracle selection across four scenes.

---

## Features

- Custom Vulkan renderer with orbit camera, mesh loading (Assimp), and a Dear ImGui debug overlay
- From-scratch QEM edge-collapse simplifier and 5-level LOD generator
- Three switchable LOD methods at runtime: **Geometric**, **Perceptual**, **Oracle**
- LOD colour-coding visualisation
- Automated benchmark runner (scenes × methods × repetitions) with JSON/CSV output
- Reproducible Python pipeline for dataset generation, training, evaluation, and ablation studies

---

## Tech Stack

| Area | Technology |
|---|---|
| Rendering | C++17, Vulkan 1.4, GLSL, GLFW, GLM, Assimp, Dear ImGui |
| ML training | Python 3.13, PyTorch, scikit-learn |
| Dataset generation | PyVista, Open3D, TOPIQ (image quality model) |
| Deployment | ONNX (opset 14), ONNX Runtime |
| Build | CMake, vcpkg, MSVC (Visual Studio 2026) |
| Analysis | NumPy, pandas, SciPy, Matplotlib |

---

## System Requirements

**Minimum:** Windows 10/11 (64-bit), GTX 1060 / RX 580 (Vulkan 1.3), 8 GB RAM, 50 GB storage (datasets)
**Recommended:** Windows 11, RTX 3060+ / RX 6700+ (Vulkan 1.4), 16 GB+ RAM, 100 GB SSD

> Developed and tested on Windows. Linux support has not been verified.

---

## Build Instructions

### Prerequisites

1. **Visual Studio 2026 Community** with "Desktop development with C++" (MSVC, CMake tools, Windows SDK)
2. **CMake 3.31+**, added to PATH
3. **Vulkan SDK 1.4.341.1** with validation layers
4. **Python 3.13**, added to PATH
5. **Git**
6. **vcpkg** (installed at `C:\vcpkg` in the examples below)

### 1. Clone

```bash
git clone https://github.com/prthm412/acuity.git
cd acuity
```

### 2. C++ dependencies

```bash
cd C:\vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install glfw3:x64-windows glm:x64-windows assimp:x64-windows
.\vcpkg install "imgui[glfw-binding,vulkan-binding]:x64-windows"
```

ONNX Runtime is also required; see `CMakeLists.txt` for how it is located.

### 3. Python environment

```bash
cd acuity
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r python/requirements.txt
```

### 4. Configure and build

```bash
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```

### 5. Run

```bash
# Interactive viewer
.\Release\Acuity.exe <path-to-mesh>

# Automated benchmark
.\Release\Acuity.exe --benchmark
```

**Controls:** left mouse = orbit, middle mouse = pan, scroll = zoom, `Esc` = quit.
Use the ImGui panel to switch between Geometric / Perceptual / Oracle LOD selection.

---

## Reproducing the ML Pipeline

```bash
# 1. Render the dataset (14 meshes × 5 LODs × 40 poses)
python python/dataset/generate_dataset.py

# 2. Extract the 38-D features
python python/dataset/extract_features.py

# 3. Train the model
python python/training/train.py

# 4. Export to ONNX
python python/training/export_onnx.py

# 5. Analysis and ablation studies
python python/evaluation/<script-name>.py
```

> Script names reflect the repository layout; check each folder for options and arguments.

---

## Project Structure

```
acuity/
├── src/
│   ├── renderer/           # Vulkan renderer
│   ├── lod/                # QEM simplifier, LOD generator, feature extraction, LOD selectors
│   ├── ml/                 # ONNX Runtime inference wrapper
│   └── benchmark/          # Benchmark runner
├── python/
│   ├── dataset/            # Dataset generation and feature extraction
│   ├── training/           # Model, training, ONNX export
│   └── evaluation/         # Analysis and ablation studies
├── assets/
│   ├── shaders/            # GLSL shaders
│   └── models/             # 3D mesh files
├── data/
│   ├── raw/                # Downloaded datasets (not tracked)
│   ├── processed/          # Generated dataset, features, splits
│   └── models/             # Trained PyTorch and ONNX models
├── docs/                   # Notes, experiment reports, screenshots
├── results/                # Benchmarks, figures, LaTeX tables
├── diagrams/               # Class diagrams and architecture documentation
└── build/                  # CMake output (gitignored)
```

---

## Limitations

- Quality labels come from an image-quality model used as a **perceptual proxy**, not from a human user study.
- The training set covers 14 meshes; generalisation to very different geometry (e.g. thin or highly organic structures) is untested.
- Memory savings vary by scene rather than following a single flat percentage.
- Tested on Windows with Vulkan only.

## Future Work

- Human user study to validate the proxy labels
- Larger and more varied mesh set; texture- and material-aware features
- Integration into an existing engine, and temporal/animation-aware selection

---

## Documentation

- Thesis and presentation: `docs/` *(add links if public)*
- Experiment reports: `docs/experiments/`
- Research notes and literature review: `docs/notes/`

## Citation

```bibtex
@mastersthesis{acuity2026,
  title  = {Learned Perceptual Quality Assessment for Triangle Mesh LOD Selection},
  author = {Mathur, Prathmesh},
  year   = {2026},
  school = {Jaypee Institute of Information Technology}
}
```

## License

Released under the [MIT License](LICENSE).

## Author

**Prathmesh Mathur**
M.Tech, Jaypee Institute of Information Technology

- GitHub: [@prthm412](https://github.com/prthm412)
- LinkedIn: [Prathmesh Mathur](https://www.linkedin.com/in/prthmmthr/)
- [Portfolio](https://prthm.vercel.app)

## Acknowledgements

Garland & Heckbert (QEM), Q-Bench, TOPIQ, and the Stanford and McGuire mesh repositories.