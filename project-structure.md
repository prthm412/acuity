acuity/
│
├── README.md
├── LICENSE
├── .gitignore
├── CMakeLists.txt
│
├── docs/
│   ├── literature/              # Research papers (PDFs)
│   ├── notes/                   # Reading notes, summaries
│   ├── architecture/            # System diagrams
│   ├── experiments/             # Experiment logs
│   └── thesis/                  # Thesis LaTeX files
│       ├── chapters/
│       ├── figures/
│       ├── tables/
│       └── main.tex
│
├── src/
│   ├── main.cpp
│   │
│   ├── core/
│   │   ├── Application.h/cpp
│   │   ├── Window.h/cpp
│   │   └── Timer.h/cpp
│   │
│   ├── renderer/
│   │   ├── VulkanRenderer.h/cpp
│   │   ├── VulkanContext.h/cpp
│   │   ├── Pipeline.h/cpp
│   │   ├── Swapchain.h/cpp
│   │   ├── Camera.h/cpp
│   │   └── Scene.h/cpp
│   │
│   ├── mesh/
│   │   ├── Mesh.h/cpp
│   │   ├── MeshLoader.h/cpp
│   │   ├── Vertex.h
│   │   └── MeshBuffer.h/cpp
│   │
│   ├── lod/
│   │   ├── LODGenerator.h/cpp
│   │   ├── QuadricSimplifier.h/cpp
│   │   ├── LODHierarchy.h/cpp
│   │   ├── LODSelector.h/cpp
│   │   ├── GeometricLODSelector.h/cpp
│   │   ├── PerceptualLODSelector.h/cpp
│   │   └── FeatureExtractor.h/cpp
│   │
│   ├── ml/
│   │   ├── ONNXInference.h/cpp
│   │   ├── ModelLoader.h/cpp
│   │   └── FeatureVector.h
│   │
│   ├── benchmark/
│   │   ├── BenchmarkRunner.h/cpp
│   │   ├── PerformanceStats.h/cpp
│   │   └── Comparison.h/cpp
│   │
│   └── ui/
│       ├── DebugUI.h/cpp
│       └── Visualization.h/cpp
│
├── shaders/
│   ├── basic.vert
│   ├── basic.frag
│   ├── lod_blend.vert
│   └── lod_blend.frag
│
├── python/
│   ├── requirements.txt
│   │
│   ├── dataset/
│   │   ├── generate_dataset.py
│   │   ├── render_mesh.py
│   │   ├── extract_features.py
│   │   └── annotate_quality.py
│   │
│   ├── training/
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── losses.py
│   │   └── dataset_loader.py
│   │
│   ├── analysis/
│   │   ├── analyze_results.py
│   │   ├── visualize.py
│   │   ├── statistical_tests.py
│   │   └── generate_figures.py
│   │
│   └── utils/
│       ├── metrics.py
│       ├── quality_assessment.py
│       └── helpers.py
│
├── assets/
│   ├── models/                  # Test meshes
│   │   ├── simple/
│   │   │   ├── bunny.obj
│   │   │   ├── sphere.obj
│   │   │   └── teapot.obj
│   │   └── complex/
│   │       ├── dragon.obj
│   │       ├── buddha.obj
│   │       └── sponza/
│   │
│   ├── textures/
│   └── shaders_compiled/        # SPIR-V bytecode
│
├── data/
│   ├── raw/
│   │   ├── qbench/              # Downloaded Q-Bench dataset
│   │   ├── kadid/               # Downloaded KADID-10k
│   │   └── meshes/              # Source meshes
│   │
│   ├── processed/
│   │   ├── rendered_images/     # Generated LOD images
│   │   ├── features/            # Extracted features
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   │
│   └── models/                  # Trained ML models
│       ├── best_model.pth
│       ├── best_model.onnx
│       └── training_logs/
│
├── tests/
│   ├── unit/
│   │   ├── test_quadric.cpp
│   │   ├── test_lod_generator.cpp
│   │   └── test_feature_extraction.cpp
│   │
│   └── integration/
│       ├── test_pipeline.cpp
│       └── test_renderer.cpp
│
├── scripts/
│   ├── setup_env.sh/bat
│   ├── download_datasets.sh
│   ├── build.sh/bat
│   ├── run_benchmarks.sh
│   └── generate_results.sh
│
├── results/
│   ├── benchmarks/
│   │   ├── geometric_method/
│   │   ├── perceptual_method/
│   │   └── comparison/
│   │
│   ├── figures/                 # Paper figures
│   │   ├── fig1_overview.pdf
│   │   ├── fig2_results.pdf
│   │   └── ...
│   │
│   ├── tables/                  # Paper tables
│   │   ├── table1_performance.tex
│   │   └── ...
│   │
│   └── videos/                  # Demo videos
│       ├── side_by_side.mp4
│       └── presentation.mp4
│
├── external/                    # Third-party libraries (submodules)
│   ├── glfw/
│   ├── glm/
│   ├── assimp/
│   ├── imgui/
│   └── onnxruntime/
│
└── build/                       # Generated build files (gitignored)
    ├── Debug/
    ├── Release/
    └── ...
