---

# **PROJECT: Learned Perceptual Quality Assessment for Triangle Mesh LOD Selection**

**Duration:** 10-12 months

---

## **PHASE 1: FOUNDATION**

**Duration:** 8-10 weeks

**Goal:** Establish theoretical understanding, set up development environment, and implement baseline LOD system.

---

### **Step 1.1: Literature Review**

**Goals:**
- Understand existing LOD techniques and perceptual quality assessment
- Identify research gap clearly
- Build strong theoretical foundation

**Tasks:**
1. Read foundational papers (Garland 1997, Corsini 2013)
2. Read recent work (Q-Bench 2024, LOD papers 2022-2024)
3. Create annotated bibliography (30-50 papers)
4. Write literature survey document (5 pages)
5. Create comparison table of existing methods
6. Document research gap clearly

**Deliverables:**
- `docs/literature/` folder with paper PDFs
- `docs/notes/literature_review.md`
- Research gap statement
- Draft thesis introduction

---

### **Step 1.2: Environment Setup**

**Goals:**
- Install all development tools
- Configure build system
- Verify everything compiles

**Tasks:**
1. Install base tools (Visual Studio/GCC, CMake, Git, Vulkan SDK, Python)
2. Install C++ dependencies via vcpkg (GLFW, GLM, Assimp, ImGui, ONNX Runtime)
3. Setup Python environment (PyTorch, ONNX, scikit-learn, OpenCV)
4. Create project directory structure
5. Initialize Git repository
6. Configure CMakeLists.txt
7. Write initial README.md
8. First successful build

**Deliverables:**
- Working build environment
- `CMakeLists.txt` configured
- `README.md` with build instructions
- All dependencies installed

---

### **Step 1.3: Basic Renderer**

**Goals:**
- Implement minimal Vulkan renderer
- Load and display 3D meshes
- Add interactive camera controls

**Tasks:**
1. Implement Vulkan initialization (instance, device, swapchain)
2. Create render pass and graphics pipeline
3. Implement mesh loading with Assimp
4. Upload mesh data to GPU buffers
5. Implement basic vertex and fragment shaders
6. Add orbit camera (mouse controls)
7. Add FPS counter and basic UI
8. Test with simple meshes (bunny, sphere)

**Deliverables:**
- Working Vulkan renderer
- Can display meshes at 60 FPS
- Smooth camera controls
- `src/renderer/` directory populated

---

### **Step 1.4: Baseline LOD System**

**Goals:**
- Implement Quadric Error Metric algorithm
- Generate LOD hierarchies
- Implement distance-based LOD selection

**Tasks:**
1. Implement Quadric Matrix data structure
2. Implement edge collapse algorithm
3. Implement Quadric Error simplification
4. Create LOD hierarchy generator (5 levels: 100%, 50%, 25%, 12.5%, 6.25%)
5. Implement distance-based LOD selector
6. Integrate with renderer
7. Add LOD visualization (color coding)
8. Profile baseline performance
9. Document baseline metrics

**Deliverables:**
- `src/lod/QuadricSimplifier.cpp` working
- `src/lod/LODGenerator.cpp` working
- `src/lod/GeometricLODSelector.cpp` working
- Generated LOD hierarchies for test meshes
- Baseline performance report (2-3 pages)

---

## **PHASE 2: DATA GENERATION**

**Duration:** 6-8 weeks

**Goal:** Create training dataset by rendering meshes at various LOD levels and annotating with perceptual quality scores.

---

### **Step 2.1: Dataset Download**

**Goals:**
- Download image quality assessment datasets
- Collect diverse 3D meshes
- Organize data properly

**Tasks:**
1. Download Q-Bench dataset (~50GB)
2. Download KADID-10k dataset (~10GB)
3. Explore dataset structure and annotations
4. Collect 15-20 diverse meshes (Stanford, McGuire, Sketchfab)
5. Organize in `data/raw/` directory
6. Document dataset statistics
7. Verify file integrity

**Deliverables:**
- `data/raw/qbench/` populated
- `data/raw/kadid/` populated
- `assets/models/` with 15-20 meshes
- `docs/experiments/dataset_statistics.md`

---

### **Step 2.2: Rendering Pipeline**

**Goals:**
- Automate mesh rendering at different LODs
- Generate training images from multiple viewpoints
- Create reproducible rendering workflow

**Tasks:**
1. Implement batch renderer in C++
2. Create camera pose generation (8 angles × 5 distances = 40 poses)
3. Write Python script to generate render configurations
4. Render all meshes × LOD levels × camera poses
5. Save images systematically (organized naming)
6. Verify rendering quality
7. Document rendering parameters

**Expected Output:**
- ~3,000 rendered images (15 meshes × 5 LODs × 40 poses)
- Time: 4-6 hours automated

**Deliverables:**
- `data/processed/rendered_images/` directory (3000+ images)
- Batch rendering system working
- `python/dataset/generate_dataset.py` script
- Rendering log file

---

### **Step 2.3: Quality Annotation**

**Goals:**
- Use Q-Bench model to assign quality scores
- Create labeled dataset
- Verify annotation quality

**Tasks:**
1. Load pre-trained Q-Bench model
2. Score all rendered images
3. Parse filename metadata (mesh, LOD, distance, angle)
4. Create CSV with image paths and scores
5. Analyze score distributions
6. Check for outliers
7. Document annotation process

**Deliverables:**
- `data/processed/dataset_raw.csv`
- Quality score statistics report
- Annotation verification report

---

### **Step 2.4: Feature Extraction**

**Goals:**
- Extract geometric features from meshes
- Extract perceptual features
- Extract view-dependent features

**Tasks:**
1. Implement geometric feature extraction (face count, area, volume, edge length)
2. Implement perceptual features (curvature, saliency, normal variation)
3. Implement view features (distance, angle, screen coverage)
4. Process all dataset samples
5. Combine features with quality scores
6. Verify feature distributions
7. Document all features and their meanings

**Total Features:** 38 dimensions

**Deliverables:**
- `data/processed/dataset_with_features.csv`
- `python/dataset/extract_features.py`
- Feature documentation
- Feature correlation analysis

---

### **Step 2.5: Dataset Finalization**

**Goals:**
- Split into train/val/test sets
- Normalize features
- Save in efficient format

**Tasks:**
1. Split by mesh (not images) to prevent data leakage (70/15/15 split)
2. Compute feature statistics from training set
3. Normalize all features using training statistics
4. Save CSVs for each split
5. Save HDF5 format for fast loading
6. Save feature scaler for deployment
7. Verify splits are balanced
8. Document dataset statistics

**Deliverables:**
- `data/processed/train.csv` (~2,100 samples)
- `data/processed/val.csv` (~450 samples)
- `data/processed/test.csv` (~450 samples)
- `data/processed/dataset.h5`
- `data/processed/feature_scaler.pkl`

---

## **PHASE 3: MODEL TRAINING**

**Duration:** 4-6 weeks

**Goal:** Design, train, and evaluate neural network for perceptual quality prediction.

---

### **Step 3.1: Model Architecture**

**Goals:**
- Design neural network architecture
- Implement model in PyTorch
- Test forward pass

**Tasks:**
1. Design MLP architecture (input: 38D → hidden layers → output: 1D)
2. Implement LODPerceptionNet class
3. Implement custom loss function (quality + ranking + correlation)
4. Test model forward pass
5. Count parameters
6. Verify gradient flow
7. Document architecture decisions

**Model Size:** ~25,000 parameters (lightweight)

**Deliverables:**
- `python/training/model.py`
- Architecture diagram
- Model tested and working

---

### **Step 3.2: Training Loop**

**Goals:**
- Implement complete training pipeline
- Add experiment tracking
- Train model to convergence

**Tasks:**
1. Implement PyTorch Dataset class
2. Implement training loop with validation
3. Setup Weights & Biases for experiment tracking
4. Implement checkpointing
5. Add early stopping
6. Train for 100 epochs
7. Monitor loss curves
8. Save best model

**Training Time:** 1-2 hours on GPU

**Expected Performance:** Validation SRCC > 0.75

**Deliverables:**
- `python/training/train.py`
- `data/models/best_model.pth`
- Training logs and curves (W&B dashboard)
- Training report

---

### **Step 3.3: Model Evaluation**

**Goals:**
- Evaluate on test set
- Compare with baselines
- Analyze errors

**Tasks:**
1. Evaluate on test set (compute SRCC, PLCC, MSE, MAE, R²)
2. Generate prediction vs ground truth plots
3. Analyze error distribution
4. Identify failure cases
5. Compare with geometric error baseline
6. Compute improvement metrics
7. Create evaluation report

**Target Metrics:**
- SRCC > 0.75
- Better than geometric baseline

**Deliverables:**
- `results/test_predictions.csv`
- `results/test_metrics.txt`
- Evaluation plots (3-4 figures)
- Error analysis report

---

### **Step 3.4: Model Export**

**Goals:**
- Export model to ONNX format
- Verify numerical accuracy
- Benchmark inference speed

**Tasks:**
1. Export PyTorch model to ONNX
2. Verify ONNX vs PyTorch outputs match
3. Benchmark ONNX inference time
4. Optimize ONNX graph
5. Test on CPU and GPU
6. Document export process

**Target:** <1ms inference time

**Deliverables:**
- `data/models/lod_perception.onnx`
- ONNX verification report
- Inference benchmark results

---

## **PHASE 4: SYSTEM INTEGRATION**

**Duration:** 8-10 weeks

**Goal:** Integrate trained model into C++ renderer for real-time LOD selection.

---

### **Step 4.1: ONNX Integration (C++)**

**Goals:**
- Load ONNX model in C++
- Implement inference wrapper
- Verify predictions match Python

**Tasks:**
1. Setup ONNX Runtime in C++
2. Implement PerceptionModel class
3. Implement single and batch inference
4. Write integration tests
5. Verify C++ predictions match Python
6. Benchmark C++ inference
7. Document API

**Deliverables:**
- `src/ml/ONNXInference.h/cpp`
- Integration tests passing
- C++ inference <1ms

---

### **Step 4.2: Feature Extraction (C++)**

**Goals:**
- Implement feature extraction in C++
- Match Python implementation exactly
- Optimize for real-time

**Tasks:**
1. Implement FeatureExtractor class
2. Extract geometric features (10 features)
3. Extract perceptual features (15 features)
4. Extract view-dependent features (13 features)
5. Verify matches Python implementation
6. Optimize performance
7. Add unit tests

**Deliverables:**
- `src/lod/FeatureExtractor.h/cpp`
- Unit tests passing
- Verified against Python

---

### **Step 4.3: Perceptual LOD Selector**

**Goals:**
- Implement LOD selection using perception model
- Add caching for performance
- Support memory budget constraints

**Tasks:**
1. Implement PerceptualLODSelector class
2. Integrate with PerceptionModel
3. Implement caching mechanism (spatial hash)
4. Support batch selection
5. Handle memory budget constraints
6. Add temporal stability
7. Test thoroughly

**Deliverables:**
- `src/lod/PerceptualLODSelector.h/cpp`
- Caching working correctly
- Memory constraints working

---

### **Step 4.4: Renderer Integration**

**Goals:**
- Use perceptual selector in render loop
- Support multiple LOD methods
- Add debug visualization

**Tasks:**
1. Integrate perceptual selector into Scene class
2. Support switching between methods (Geometric/Perceptual/Oracle)
3. Update render loop to use selected LODs
4. Implement LOD visualization (color coding)
5. Add ImGui debug UI
6. Display performance stats
7. Test all methods work

**Deliverables:**
- Perceptual LOD selector integrated
- Method switching working
- Debug UI functional
- Real-time rendering at 60 FPS

---

### **Step 4.5: Performance Optimization**

**Goals:**
- Profile system bottlenecks
- Optimize hot paths
- Achieve target performance

**Tasks:**
1. Profile with NVIDIA Nsight Graphics
2. Identify bottlenecks
3. Optimize caching strategy
4. Implement frustum culling
5. Optimize batch inference
6. Add multi-threading if needed
7. Verify performance targets met

**Target:** <1ms LOD selection overhead

**Deliverables:**
- Profiling report
- Optimized code
- Performance benchmark showing <1ms overhead

---

## **PHASE 5: EVALUATION & BENCHMARKING**

**Duration:** 6-8 weeks

**Goal:** Comprehensive evaluation comparing perceptual method against baselines.

---

### **Step 5.1: Benchmark Suite**

**Goals:**
- Create automated benchmark system
- Test multiple scenes and methods
- Collect comprehensive metrics

**Tasks:**
1. Implement BenchmarkRunner class
2. Define benchmark configurations (scene, method, duration, camera path)
3. Create predefined camera paths for consistency
4. Implement performance metric collection (FPS, memory, triangles, LOD distribution)
5. Add screenshot/video recording
6. Run benchmarks on 4 scenes × 3 methods × 3 repetitions
7. Save results as JSON and CSV

**Benchmark Matrix:**
- Scenes: Bunny, Dragon, Sponza, San Miguel
- Methods: Geometric, Perceptual, Oracle
- Duration: 60 seconds each
- Total runs: 36

**Deliverables:**
- `src/benchmark/BenchmarkRunner.h/cpp`
- Benchmark results for all combinations
- `results/benchmarks/` directory with JSON files
- Automated benchmark script

---

### **Step 5.2: Quality Assessment**

**Goals:**
- Compare rendered image quality across methods
- Compute objective quality metrics
- Generate visual comparisons

**Tasks:**
1. Collect screenshots from all benchmark runs
2. Use Oracle as reference (ground truth)
3. Compute PSNR, SSIM, LPIPS for each method vs Oracle
4. Analyze quality metrics over time
5. Create side-by-side visual comparisons
6. Generate quality comparison charts
7. Document quality findings

**Deliverables:**
- `results/quality_metrics.csv`
- Visual comparison images
- Quality analysis report

---

### **Step 5.3: Statistical Analysis**

**Goals:**
- Aggregate all results
- Perform hypothesis testing
- Generate publication-ready figures

**Tasks:**
1. Load and combine all benchmark results
2. Compute descriptive statistics
3. Perform t-tests (memory usage, FPS comparison)
4. Compute effect sizes (Cohen's d)
5. Generate 6-8 publication figures (bar charts, scatter plots, box plots)
6. Generate LaTeX tables for paper
7. Write statistical analysis report

**Key Findings to Demonstrate:**
- Memory savings (target: 20-30%)
- No FPS degradation
- Quality maintained (SSIM/LPIPS comparable)
- Statistical significance (p < 0.05)

**Deliverables:**
- `results/figures/comprehensive_results.pdf`
- `results/tables/*.tex` (LaTeX tables)
- Statistical analysis report
- All figures for paper

---

### **Step 5.4: Ablation Studies**

**Goals:**
- Test feature importance
- Validate architectural choices
- Document what matters most

**Tasks:**
1. Train models with feature groups removed (geometric, perceptual, view)
2. Train models with individual features removed
3. Test different architectures (hidden layer sizes, dropout rates)
4. Compare all ablated models to baseline
5. Rank features by importance
6. Generate ablation study table
7. Document insights

**Ablation Experiments:**
- Without geometric features
- Without perceptual features  
- Without view-dependent features
- Without curvature features
- Different network architectures

**Deliverables:**
- `results/ablation_results.csv`
- `results/tables/ablation_table.tex`
- Feature importance ranking
- Ablation study report

---

## **PHASE 6: DOCUMENTATION & PUBLICATION**

**Duration:** 6-8 weeks

**Goal:** Document project, write papers, prepare for defense and publication.

---

### **Step 6.1: Code Documentation**

**Goals:**
- Document all code comprehensively
- Create usage guides
- Generate API documentation

**Tasks:**
1. Add Doxygen comments to all C++ classes and functions
2. Generate Doxygen HTML documentation
3. Write comprehensive README.md
4. Write ARCHITECTURE.md explaining system design
5. Write USAGE.md with examples
6. Create architecture diagrams (draw.io)
7. Add inline code comments
8. Write tutorial examples

**Deliverables:**
- `README.md` (comprehensive)
- `docs/ARCHITECTURE.md`
- `docs/USAGE.md`
- API documentation (HTML)
- Architecture diagrams

---

### **Step 6.2: Conference Paper Writing**

**Goals:**
- Write 20-25 page conference paper
- Create all figures and tables
- Prepare for submission

**Tasks:**

**Week 1:**
1. Write abstract (200 words)
2. Write introduction (2 pages)
3. Write related work (3 pages)
4. Create system overview figure

**Week 2:**
5. Write methodology (6 pages)
6. Create method figures (architecture, pipeline, features)
7. Write experimental setup

**Week 3:**
8. Write results section (6 pages)
9. Insert all figures and tables
10. Write discussion (2 pages)
11. Write conclusion

**Week 4:**
12. Revise entire paper
13. Proofread carefully
14. Format for venue
15. Get supervisor feedback
16. Final polish

**Paper Structure:**
- Abstract
- Introduction (motivation, problem, solution, contributions)
- Related Work (simplification, perception, ML in graphics)
- Method (dataset, features, model, system)
- Evaluation (setup, results, comparisons, ablations)
- Discussion (insights, limitations)
- Conclusion

**Deliverables:**
- `paper.pdf` (20-25 pages)
- All figures (8-10 figures)
- All tables (5-7 tables)
- Supplementary materials

---

### **Step 6.3: M.Tech Thesis Writing**

**Goals:**
- Write comprehensive thesis (100-120 pages)
- Document entire project in detail
- Prepare for defense

**Tasks:**

**Week 1:**
1. Write Chapter 1: Introduction (10 pages)
2. Write Chapter 2: Literature Review (20 pages)

**Week 2:**
3. Write Chapter 3: Methodology Part 1 (15 pages - baseline, dataset)
4. Write Chapter 3: Methodology Part 2 (15 pages - model, system)

**Week 3:**
5. Write Chapter 4: Results Part 1 (15 pages - benchmarks, quality)
6. Write Chapter 4: Results Part 2 (15 pages - analysis, ablations)

**Week 4:**
7. Write Chapter 5: Conclusion (10 pages)
8. Write Appendices (20 pages - algorithms, code, additional results)
9. Add all references

**Week 5:**
10. Revise entire thesis
11. Format properly
12. Proofread
13. Get supervisor approval
14. Final print-ready version

**Thesis Structure:**
- Chapter 1: Introduction (10 pages)
- Chapter 2: Literature Review (20 pages)
- Chapter 3: Methodology (30 pages)
- Chapter 4: Experiments and Results (30 pages)
- Chapter 5: Conclusion and Future Work (10 pages)
- References (8-10 pages)
- Appendices (20 pages)

**Deliverables:**
- `thesis.pdf` (100-120 pages)
- Defense presentation (30 slides)
- Defense notes

---

### **Step 6.4: Open Source Release**

**Goals:**
- Prepare repository for public release
- Make project accessible
- Build community

**Tasks:**
1. Clean repository (remove sensitive data)
2. Add LICENSE file (MIT or Apache 2.0)
3. Add CONTRIBUTING.md
4. Add CODE_OF_CONDUCT.md
5. Improve README with badges, screenshots
6. Create GitHub release (v1.0.0)
7. Upload pre-trained model
8. Create sample dataset
9. Build Windows binaries (optional)
10. Create project website/landing page
11. Write blog post about project
12. Share on social media (Reddit, Twitter)

**Deliverables:**
- Public GitHub repository
- v1.0.0 release with assets
- Project website
- Blog post
- Demo video (3-5 minutes)

---

# **📊 DELIVERABLES SUMMARY**

## **Technical Artifacts:**
- Working C++/Vulkan LOD system
- Trained perception model (PyTorch + ONNX)
- Training dataset (~3,000 samples)
- Benchmark suite with results
- Complete codebase (~10,000 lines C++, ~2,000 lines Python)

## **Research Outputs:**
- Literature review document
- Comprehensive experimental results
- Statistical analysis
- Ablation studies

## **Documentation:**
- API documentation (Doxygen)
- User guides
- Architecture documentation
- Tutorial examples

## **Academic Publications:**
- Conference paper (20-25 pages) submitted to CGF/I3D/HPG
- M.Tech thesis (100-120 pages)
- Defense presentation
- Supplementary materials

## **Public Release:**
- Open-source GitHub repository
- Pre-trained models
- Demo video
- Project website

---

# **🎯 SUCCESS CRITERIA**

## **Minimum (Must Achieve):**
- ✅ Working LOD system using perception model
- ✅ Real-time performance (60 FPS)
- ✅ Measurable improvement (>10% memory savings)
- ✅ Thesis completed and defended
- ✅ Code released

## **Target (Expected):**
- ✅ 20-30% memory reduction vs geometric
- ✅ Quality correlation SRCC > 0.75
- ✅ Paper accepted at Tier 2-3 venue
- ✅ Comprehensive documentation
- ✅ Open source with 50+ stars

## **Stretch (Bonus):**
- ✅ 30%+ memory savings
- ✅ SRCC > 0.85
- ✅ User study validation
- ✅ Tier 1 venue acceptance (SIGGRAPH/TOG)
- ✅ 100+ GitHub stars

---

# **⏱️ TIME ESTIMATES**

| Phase | Duration | Key Milestones |
|-------|----------|----------------|
| Phase 1: Foundation | 8-10 weeks | Baseline LOD system working |
| Phase 2: Data Generation | 6-8 weeks | 3,000 training samples created |
| Phase 3: Model Training | 4-6 weeks | Model trained (SRCC > 0.75) |
| Phase 4: Integration | 8-10 weeks | Real-time system working |
| Phase 5: Evaluation | 6-8 weeks | All benchmarks complete |
| Phase 6: Documentation | 6-8 weeks | Paper + thesis written |
| **Total** | **38-50 weeks** | **10-12 months** |

---

# **🚀 GETTING STARTED**

**Start with Phase 1, Step 1.1: Literature Review**

**First Week Tasks:**
1. Download and read Garland & Heckbert 1997 paper
2. Download and read Corsini et al. 2013 survey
3. Download and read Q-Bench paper (ICLR 2024)
4. Create reading notes document
5. Set up development environment (VS Code, CMake, Vulkan SDK)
