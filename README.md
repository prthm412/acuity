# Acuity

**Learned Perceptual Quality Assessment for Triangle Mesh LOD Selection**

A research project implementing ML-based perceptual quality assessment for real-time Level of Detail (LOD) selection in 3D mesh rendering.

## Overview

Reduces memory usage in real-time 3D rendering by using a learned perceptual quality model to intelligently select mesh LOD levels, maintaining visual quality while optimizing resources.
This project aims to reduce memory usage in real-time 3D rendering by using a learned perceptual quality model to intelligently select mesh LOD levels, maintaining visual quality while optimizing resource usage.

**Key Innovation**: Building a system to apply learned image quality assessment to real-time triangle mesh LOD selection.


## Tech Stack

- **Rendering**: C++17, Vulkan
- **Machine Learning**: Python, PyTorch, ONNX Runtime
- **Build System**: CMake
- **Dependencies**: GLFW, GLM, Assimp, ImGui

### Detailed Stack
### C++ Rendering Engine

- **Graphics API**: Vulkan 1.4.341.1
- **Window/Input**: GLFW 3.4+
- **Math**: GLM 1.0+
- **Mesh Loading**: Assimp 5.4+
- **UI**: Dear ImGui 1.91+
- **Build**: CMake 3.31.11
- **Compiler**: MSVC 19.4+ (Visual Studio 2026)

### Machine Learning Pipeline

- **Framework**: PyTorch 2.5+
- **Runtime**: Python 3.13.1
- **Export**: ONNX 1.17+
- **Inference**: ONNX Runtime 1.20+
- **Dataset**: Q-Bench (ICLR 2024)


## System Requirements

### Minimum

- **OS**: Windows 10/11 (64-bit)
- **GPU**: NVIDIA GTX 1060 / AMD RX 580 (Vulkan 1.3 support)
- **RAM**: 8 GB
- **Storage**: 50 GB (for datasets)

### Recommended

- **OS**: Windows 11
- **GPU**: NVIDIA RTX 3060+ / AMD RX 6700+ (Vulkan 1.4 support)
- **RAM**: 16 GB+
- **Storage**: 100 GB SSD


## Build Instructions

### Prerequisites

Install these tools in order:

1. **Visual Studio 2026 Community**

   - Download: https://visualstudio.microsoft.com/
   - Select: "Desktop development with C++"
   - Include: MSVC, CMake tools, Windows 11 SDK, vcpkg
2. **CMake 3.31.11**

   - Download: https://cmake.org/download/
   - Add to PATH during installation
3. **Vulkan SDK 1.4.341.1**

   - Download: https://vulkan.lunarg.com/
   - Install with validation layers
4. **Python 3.13.1**

   - Download: https://www.python.org/downloads/
   - Check "Add Python to PATH"
5. **Git**

   - Verify: `git --version`


### Installation

#### Step 1: Clone Repository

```bash
git clone https://github.com/orthm412/acuity.git
cd acuity
git checkout develop
```

#### Step 2: Install C++ Dependencies (vcpkg)

```bash
# Navigate to vcpkg installation (or install it)
cd C:\vcpkg

# Bootstrap (first time only)
.\bootstrap-vcpkg.bat

# Install dependencies
.\vcpkg install glfw3:x64-windows
.\vcpkg install glm:x64-windows
.\vcpkg install assimp:x64-windows
.\vcpkg install imgui[glfw-binding,vulkan-binding]:x64-windows

# Linux users: replace :x64-windows with :x64-linux
```

#### Step 3: Setup Python Environment

```bash
cd acuity

# Create virtual environment
python -m venv venv

# Activate
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r python/requirements.txt
```

#### Step 4: Configure CMake

```bash
mkdir build
cd build

# Configure (set VCPKG_ROOT to your vcpkg path)
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake

# Or if VCPKG_ROOT environment variable is set:
cmake ..
```

#### Step 5: Build

```bash
# Windows (Visual Studio)
cmake --build . --config Release

# Linux (Make)
make -j$(nproc)
```

#### Step 6: Run

```bash
# From build directory
./Release/Acuity.exe  # Windows
# ./Acuity  # Linux
```

Expected output:

```
=== Acuity Build Test ===
✓ GLFW initialized
✓ Vulkan available (version: 1.4)
✓ GLM working (test vector: 1, 2, 3)

✓ All systems operational!
Environment setup complete. Ready for Step 1.3!
```

## Project Structure

```
acuity/
├── src/                    # C++ source code
│   ├── renderer/           # Vulkan renderer
│   ├── lod/                # LOD system
│   ├── ml/                 # ONNX Runtime inference wrapper
│   └── benchmark/          # Benchmark suite
├── python/                 # ML training pipeline
│   ├── dataset/            # Dataset generation & feature extraction
│   ├── training/           # Model training & evaluation
│   └── evaluation/         # Results analysis & ablation studies
├── assets/
│   ├── shaders/            # GLSL shaders
│   └── models/             # 3D mesh files
├── data/
│   ├── raw/                # Downloaded datasets
│   ├── processed/          # Generated dataset, features, splits
│   └── models/             # Trained model files
├── docs/
│   ├── screenshots/        # Renderer screenshots
│   ├── experiments/        # Experiment logs & baseline reports
│   └── notes/              # Research notes & literature review
├── results/                # Benchmark results, figures, LaTeX tables
├── diagrams/               # Class diagrams & architecture documentation
└── build/                  # CMake build output (gitignored)
```

## Development Status

![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![Phase](https://img.shields.io/badge/Phase-2%20of%204(5)-blue)

### Currently working on

🚧 **In Development** - Phase 2: Data Generation


- [ ] Phase 2.2: Rendering Pipeline
- [ ] Phase 2.3: Quality Annotation

### Completed

- [x] Phase 1.1: Literature review
- [x] Phase 1.2: Environment setup (in progress)
- [x] Phase 1.3: Basic Vulkan renderer
- [x] Phase 1.4: Baseline LOD system
- [x] Phase 2.1: Dataset


## License

![License](https://img.shields.io/badge/License-MIT-green)

## Author

Prathmesh Mathur
M.Tech Research Project

- GitHub: [@prthm412](https://github.com/prthm412)
- LinkedIn: [Prathmesh Mathur](https://www.linkedin.com/in/prthmmthr/)


Computer Graphics & Machine Learning


**Research Question**: Can we reduce memory usage in real-time rendering by 20-30% using learned perceptual quality assessment while maintaining visual fidelity?


## Citation

```bibtex
@mastersthesis{acuity2026,
  title={Learned Perceptual Quality Assessment for Triangle Mesh LOD Selection},
  author={Prathmesh Mathur},
  year={2026},
  school={Jaypee Institute of Information Technology}
}
```

---