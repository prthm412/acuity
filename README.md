# Acuity

**Learned Perceptual Quality Assessment for Triangle Mesh LOD Selection**

A research project implementing ML-based perceptual quality assessment for real-time Level of Detail (LOD) selection in 3D mesh rendering.

## Overview

Reduces memory usage in real-time 3D rendering by using a learned perceptual quality model to intelligently select mesh LOD levels, maintaining visual quality while optimizing resources.

**Key Innovation**: First system to apply learned image quality assessment to real-time triangle mesh LOD selection.

---

## Tech Stack

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

---

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

---

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

---

### Installation

#### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/acuity.git
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

---

## Project Structure

```
acuity/
├── src/           # C++ source code
│   ├── core/      # Application, window, timer
│   ├── renderer/  # Vulkan renderer
│   ├── mesh/      # Mesh loading & processing
│   ├── lod/       # LOD generation & selection
│   ├── ml/        # ONNX inference
│   └── ui/        # ImGui debug UI
├── python/        # ML training pipeline
│   ├── dataset/   # Data generation
│   ├── training/  # Model training
│   └── analysis/  # Results analysis
├── shaders/       # GLSL shaders
├── assets/        # 3D models, textures
├── data/          # Datasets, trained models
├── docs/          # Documentation
└── results/       # Benchmarks, figures
```

---

## Development Status

**Current Phase**: Phase 1 - Foundation
**Current Step**: Step 1.2 - Environment Setup

### Completed

- ✅ Phase 0: GitHub repository setup
- ✅ Phase 1.1: Literature review
- 🔄 Phase 1.2: Environment setup (in progress)

### Next Steps

- [ ] Phase 1.3: Basic Vulkan renderer
- [ ] Phase 1.4: Baseline LOD system

---

## Technology References

### Core Papers

- **Quadric Error Metrics**: Garland & Heckbert (SIGGRAPH 1997)
- **Q-Bench Dataset**: Wu et al. (ICLR 2024)
- **LPIPS Metric**: Zhang et al. (CVPR 2018)

### Graphics APIs

- **Vulkan**: Khronos Group specification 1.4
- **GLSL**: OpenGL Shading Language 4.60

### ML Frameworks

- **PyTorch**: Meta AI (Facebook Research)
- **ONNX**: Open Neural Network Exchange (Microsoft, Facebook, AWS)

---

## License

TBD (MIT or Apache 2.0 for open source release)

---

## Citation

```bibtex
@mastersthesis{acuity2026,
  title={Learned Perceptual Quality Assessment for Triangle Mesh LOD Selection},
  author={Prathmesh Mathur},
  year={2026},
  school={Jaypee Institute of Information Technology}
}
```

**Learned Perceptual Quality Assessment for Triangle Mesh LOD Selection**

A research project implementing machine learning-based perceptual quality assessment for real-time Level of Detail (LOD) selection in 3D mesh rendering.

## Project Overview

This project aims to reduce memory usage in real-time 3D rendering by using a learned perceptual quality model to intelligently select mesh LOD levels, maintaining visual quality while optimizing resource usage.

## Status

🚧 **In Development** - Phase 1: Foundation

## Tech Stack (tentative as the project progresses)

- **Rendering**: C++17, Vulkan
- **Machine Learning**: Python, PyTorch, ONNX Runtime
- **Build System**: CMake
- **Dependencies**: GLFW, GLM, Assimp, ImGui

## Getting Started

Documentation will be added as the project develops.

## License

TBD

## Author

Computer Graphics & Machine Learning

Prathmesh Mathur
M.Tech Research Project

- GitHub: [@prthm412](https://github.com/prthm412)
- LinkedIn: [Prathmesh Mathur](https://www.linkedin.com/in/prthmmthr/)

**Research Question**: Can we reduce memory usage in real-time rendering by 20-30% using learned perceptual quality assessment while maintaining visual fidelity?
