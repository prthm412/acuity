# Dataset Statistics 

**Project:** Acuity — Learned Perceptual Quality Assessment for Triangle Mesh LOD Selection
**Phase:** Data Generation
**Step:** Dataset Download
**Last Updated:** May 2026

---

## Image Quality Assessment Datasets

### Q-Bench (Primary Annotation Source)

| Property | Value |
|---|---|
| Paper | Wu et al., ICLR 2024 (Spotlight) — Paper #18 in bibliography |
| Dev split | 1,495 items |
| Test split | 1,495 items |
| Total annotations | 2,990 |
| Local path | `data/raw/qbench/` |
| Files | `llvisionqa_qbench_dev.json`, `llvisionqa_qbench_test.json` |
| Size on disk | ~512 KB (annotation JSONs only) |
| Usage in Acuity | Pre-trained Q-Bench model used in Step 2.3 to score rendered mesh images |

### KADID-10k (Secondary Cross-Validation)

| Property | Value |
|---|---|
| Source | http://database.mmsp-kn.de/kadid-10k-database.html |
| Images | 10,125 (81 reference × 25 distortion types × 5 severity levels) |
| Annotations | DMOS (difference mean opinion scores) |
| Local path | `data/raw/kadid/` |
| Files | `dmos.csv` |
| Size on disk | ~3.2 KB (metadata CSV only) |
| Usage in Acuity | Cross-validates that Q-Bench scores span meaningful 0–1 range |

---

## 3D Mesh Assets

### Summary

| Property | Value |
|---|---|
| Total meshes | 20 |
| Local path | `assets/models/` |
| Manifest | `assets/models/mesh_manifest.csv` (generated in Task 5) |

### Mesh Inventory

| # | File | Source | Category | Approx. Size |
|---|---|---|---|---|
| 1 | stanford-bunny.obj | Stanford / Jacobson | Organic, classic benchmark | 2,352 KB |
| 2 | armadillo.obj | Stanford / Jacobson | Organic, classic benchmark | 4,527 KB |
| 3 | lucy.obj | Stanford / Jacobson | Organic, high detail | — |
| 4 | happy.obj | Stanford / Jacobson | Organic, statue | — |
| 5 | xyzrgb_dragon.obj | Stanford / Jacobson | Organic, high poly | 11,541 KB |
| 6 | max-planck.obj | Stanford / Jacobson | Human bust | 4,467 KB |
| 7 | horse.obj | Stanford / Jacobson | Animal, organic | — |
| 8 | cow.obj | Jacobson | Animal, organic | — |
| 9 | fandisk.obj | Jacobson | Hard surface, sharp features | — |
| 10 | spot.obj | Jacobson / K. Crane | Smooth organic animal | 323 KB |
| 11 | cheburashka.obj | Jacobson | Smooth cartoon character | 413 KB |
| 12 | suzanne.obj | Blender / Jacobson | Standard test mesh | 48 KB |
| 13 | teapot.obj | Jacobson | Classic primitive | 206 KB |
| 14 | woody.obj | Jacobson | Character figure | 40 KB |
| 15 | sponza.obj | McGuire Archive | Architectural scene | 5,480 KB |
| 16 | rungholt.obj | McGuire Archive | Dense urban scene | 2,69,215 KB |
| 17 | CornellBox-Original.obj | McGuire Archive | Rendering test scene | 3 KB |
| 18 | AmericanRockSaltMinePinkHalite.obj | Sketchfab CC0 | Rock, irregular hard surface | 15,134 KB |
| 19 | PRI_TyrannosaurusRexSkull.obj | Sketchfab CC0 | Organic, complex curvature | 42,982 KB |
| 20 | Krzeslo_0_LR.obj | Sketchfab CC0 | Furniture, hard surface | 9,375 KB |

### Diversity Coverage

| Category | Meshes |
|---|---|
| Organic / character | bunny, armadillo, lucy, happy, dragon, cheburashka, woody, horse, cow |
| Human / bust | max-planck |
| Animal | spot, horse, cow |
| Hard surface / mechanical | fandisk, teapot, chair |
| Architectural / scene | sponza, rungholt, CornellBox |
| Natural / irregular | rock salt mine |
| Paleontological | T-Rex skull |

---

## Expected Output After Step 2.2

| Property | Value |
|---|---|
| Meshes used | 20 |
| LOD levels per mesh | 5 (100%, 50%, 25%, 12.5%, 6.25%) |
| Camera poses per mesh×LOD | 40 (8 angles × 5 distances) |
| Total rendered images | ~4,000 |
| Annotation model | Q-Bench pre-trained (Step 2.3) |
| Feature dimensions | 38 (Step 2.4) |
| Train / Val / Test split | 70 / 15 / 15 percent by mesh |

---

## Citations

- Wu et al., "Q-Bench: A Benchmark for General-Purpose Foundation Models on Low-level Vision", ICLR 2024
- Lin et al., "KADID-10k: A Large-scale Artificially Distorted IQA Database", QoMEX 2019
- Alec Jacobson et al., *Common 3D Test Models*, https://github.com/alecjacobson/common-3d-test-models
- Stanford 3D Scanning Repository, http://graphics.stanford.edu/data/3Dscanrep/
- Morgan McGuire, *Computer Graphics Archive*, https://casual-effects.com/data, July 2017