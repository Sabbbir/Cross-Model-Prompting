<div align="center">

<br/>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/-%F0%9F%A7%A0%20GUIDED%20UNSUPERVISED%20SEGMENTATION-1a1a2e?style=for-the-badge&labelColor=16213e" />
  <img src="https://img.shields.io/badge/-%F0%9F%A7%A0%20GUIDED%20UNSUPERVISED%20SEGMENTATION-f0f4ff?style=for-the-badge&labelColor=dde8ff" />
</picture>

# Guided Unsupervised Segmentation via Cross-Model Prompting

### *Can the mistakes of one AI teach another to see?*

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](https://opensource.org/licenses/MIT)
[![Thesis](https://img.shields.io/badge/Thesis-B.Sc.%20CSE-8b5cf6?style=flat-square&logo=readthedocs&logoColor=white)](./Thesis_Sabbir_Ahmed.pdf)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-f59e0b?style=flat-square)]()
[![BraTS](https://img.shields.io/badge/Dataset-BraTS%20Challenge-06b6d4?style=flat-square)]()

<br/>

<table>
<tr>
<td align="center"><b>👤 Author</b></td>
<td align="center"><b>🏛️ Institution</b></td>
<td align="center"><b>🎓 Supervisor</b></td>
<td align="center"><b>📅 Year</b></td>
</tr>
<tr>
<td align="center">Sabbir Ahmed</td>
<td align="center">Jashore University of Science<br/>and Technology, Bangladesh</td>
<td align="center">Dr. A F M Shahab Uddin<br/><sub>Asst. Professor, Dept. of CSE</sub></td>
<td align="center">May 2025</td>
</tr>
</table>

<br/>

> **B.Sc. Thesis** · Department of Computer Science and Engineering · JUST

<br/>

---

</div>

📄 **[Read the Full Thesis (PDF)](./Thesis_Sabbir_Ahmed.pdf)** &nbsp;·&nbsp; 🎞️ **[Defense Slides](./Final%20Defense.pdf)**

## 📌 Table of Contents

- [Overview](#-overview)
- [The Core Insight](#-the-core-insight)
- [Pipeline Architecture](#-pipeline-architecture)
- [Dataset](#-dataset)
- [Phase 1 — Zero-Shot OneFormer Evaluation](#-phase-1--zero-shot-oneformer-evaluation)
- [Phase 2 — Guided SegGPT Inference](#-phase-2--guided-seggpt-inference)
- [Quantitative Results](#-quantitative-results)
- [Qualitative Results](#-qualitative-results)
- [Why This Matters](#-why-this-matters)
- [Limitations](#%EF%B8%8F-limitations)
- [Future Directions](#-future-directions)
- [Repository Structure](#-repository-structure)
- [Setup and Reproduction](#-setup-and-reproduction)
- [Hardware](#-hardware)
- [Citation](#-citation)
- [Contact](#-contact)

---

## 🔭 Overview

Medical image segmentation — identifying and delineating structures such as brain tumors within MRI scans — has historically depended on large, expertly annotated datasets. These annotations require trained radiologists, are expensive to produce, and remain critically scarce, particularly in low-resource clinical environments.

This project investigates a fundamentally different question:

> **Instead of acquiring more labels, what signal can be extracted from the structured failures of models that have never seen medical data?**

We demonstrate that **cross-domain misclassifications** — instances where a generalist model assigns a semantically incorrect but spatially coherent label to a specialized-domain region — can be deliberately repurposed as visual prompts for a second segmentation model. The result is a fully unsupervised, annotation-free pipeline capable of producing non-trivial tumor segmentation with zero target-domain training.

<br/>

<div align="center">

| Property | Detail |
|---|---|
| **Task** | Brain tumor segmentation on MRI |
| **Paradigm** | Unsupervised / zero-shot / prompt-based |
| **Target-domain annotations used** | ❌ None (evaluation only) |
| **Domain adaptation training** | ❌ None |
| **Inference-time only** | ✅ Yes |
| **Best reported Mean Dice** | **0.316** |

</div>

---

## 💡 The Core Insight

<div align="center">

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   "A brain tumor, viewed by a natural-image model, looks like             │
│    a sports ball. Semantically absurd. Spatially informative."           │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

</div>

When **OneFormer** (trained on COCO natural-image categories) was applied directly to BraTS MRI slices, it produced overwhelmingly incorrect labels — walls, furniture, food. This was expected.

However, within ~23,000 generated masks across 1,500 image slices, **12 instances** emerged where OneFormer had labeled the confirmed tumor region as a **"sports ball."** This is a semantic failure. But geometrically, it encodes a genuine insight: brain tumors in T1-weighted MRI often appear as bright, roughly circular, high-contrast blobs — sharing low-level shape statistics with the sports ball category.

We then posed the question: **Can this spatially correct but semantically absurd mask serve as a visual prompt for a second model?**

Using **SegGPT** — a visually prompted segmentation model — we fed it the "sports ball" mask as a reference, asking it to identify similar visual regions in unseen MRI slices. The results were striking.

<br/>

<div align="center">
  <img src="results/figures/sports_ball_verification/sports_ball_detection.png" width="420" alt="Sports Ball Detection"/>
  <br/>
  <sub><i>A confirmed tumor region, misclassified as "sports ball" by OneFormer. Semantically wrong. Spatially right.</i></sub>
</div>

<br/>

<div align="center">
  <img src="results/figures/sports_ball_verification/sports_ball_verification_a.png" width="300" alt="Verification A"/>
  &nbsp;&nbsp;
  <img src="results/figures/sports_ball_verification/sports_ball_verification_b.png" width="300" alt="Verification B"/>
  <br/>
  <sub><i>Overlay verification: OneFormer "sports ball" predictions vs. ground-truth tumor masks.</i></sub>
</div>

---

## 🏗️ Pipeline Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                     CROSS-MODEL PROMPTING PIPELINE                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   ┌─────────────────┐                                                        ║
║   │  BraTS MRI Data │  (1,500 slices — no annotations used during inference) ║
║   └────────┬────────┘                                                        ║
║            │                                                                 ║
║            ▼                                                                 ║
║   ┌──────────────────────────────────────────────┐                           ║
║   │           PHASE 1: OneFormer (COCO)          │                           ║
║   │   Zero-Shot Inference on Medical Images      │                           ║
║   │   → ~23,000 masks generated                  │                           ║
║   │   → Filter for "sports ball" label           │                           ║
║   │   → 12 spatially informative masks found     │                           ║
║   └──────────────────────┬───────────────────────┘                           ║
║                          │                                                   ║
║              Rare "Weak Signal" Masks (n=3 selected as prompts)             ║
║                          │                                                   ║
║                          ▼                                                   ║
║   ┌──────────────────────────────────────────────┐                           ║
║   │        PHASE 2: SegGPT (Visual Prompting)    │                           ║
║   │   Reference: {sports ball image, mask pair}  │                           ║
║   │   Target: Remaining BraTS slices             │                           ║
║   │   → Regions visually similar to prompt       │                           ║
║   │     are segmented in target images           │                           ║
║   └──────────────────────┬───────────────────────┘                           ║
║                          │                                                   ║
║                          ▼                                                   ║
║   ┌──────────────────────────────────────────────┐                           ║
║   │     POST-HOC EVALUATION (GT masks only here) │                           ║
║   │   Dice Score / IoU vs. Ground Truth          │                           ║
║   │   Best prompt (1112.jpg): Mean Dice = 0.316  │                           ║
║   └──────────────────────────────────────────────┘                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

<div align="center">
  <img src="results/figures/sports_ball_verification/seggpt_prompting_mechanism.png" width="780" alt="SegGPT Prompting Mechanism"/>
  <br/>
  <sub><i>The full SegGPT prompting mechanism: a "sports ball" mask guides visual similarity search across unseen MRI slices.</i></sub>
</div>

---

## 📦 Dataset

We use the **Brain Tumor Image Dataset: Semantic Segmentation**, derived from the BraTS (Brain Tumor Segmentation) Challenge. It consists of T1-weighted, contrast-enhanced MRI slices with associated COCO-format annotations.

- **Total slices used:** 1,500
- **Ground truth masks:** Binary (tumor vs. background), derived from COCO annotations
- **Usage of GT during inference:** ❌ None — ground truth was reserved **exclusively** for post-hoc evaluation

<br/>

<div align="center">

| MRI Slice | Ground Truth Mask |
|:---------:|:-----------------:|
| <img src="results/figures/dataset/mri_slice_17.jpg" width="160"/> | <img src="results/figures/dataset/gt_mask_17.png" width="160"/> |
| <img src="results/figures/dataset/mri_slice_28.jpg" width="160"/> | <img src="results/figures/dataset/gt_mask_28.png" width="160"/> |
| <img src="results/figures/dataset/mri_slice_105.jpg" width="160"/> | <img src="results/figures/dataset/gt_mask_105.png" width="160"/> |

<sub><i>Representative MRI slices alongside their binary ground truth masks (used only at evaluation).</i></sub>

</div>

---

## 🔬 Phase 1 — Zero-Shot OneFormer Evaluation

Three pre-trained **OneFormer** checkpoints were applied directly to MRI slices with no adaptation:

| Checkpoint | Training Domain |
|---|---|
| `oneformer_coco_swin_large` | MS-COCO (80 categories, everyday objects) |
| `oneformer_ade20k_swin_large` | ADE20K (150 categories, scenes) |
| `oneformer_cityscapes_swin_large` | Cityscapes (urban street scenes) |

As expected, all three models failed to produce medically meaningful segmentations. The COCO checkpoint generated the richest failure set — approximately 23,000 individual instance masks across the full slice set — and was the source of the 12 "sports ball" coincidences.

<br/>

<div align="center">
  <img src="results/figures/oneformer_failures/oneformer_failures_grid.png" width="800" alt="OneFormer Failure Grid"/>
  <br/>
  <sub><i>A representative grid of OneFormer zero-shot predictions on BraTS MRI slices. Tumor regions labeled as walls, food, furniture.</i></sub>
</div>

<br/>

> 💬 **Key observation:** Failure is not uniform. Within the noise of 23,000 wrong masks, 12 were geometrically correct for reasons rooted in shared low-level visual statistics between tumor morphology and natural-image categories.

---

## 🎯 Phase 2 — Guided SegGPT Inference

**SegGPT** is an in-context visual segmentation model: given a (reference image, reference mask) pair, it identifies and segments visually similar regions in a target image.

We selected **3 of the 12 "sports ball" masks** as visual prompts and ran SegGPT over the full BraTS slice set.

#### Prompt Selection

| Prompt ID | Source Image | Resulting Mask | Performance |
|:---------:|:------------:|:--------------:|:-----------:|
| **1112.jpg** | <img src="results/figures/seggpt_qualitative/best_prompt_source_image.jpg" width="160"/> | <img src="results/figures/seggpt_qualitative/best_prompt_mask.png" width="160"/> | 🟢 **Best** — Mean Dice 0.316 |
| **2829.jpg** | *(mid)* | *(mid)* | 🟡 **Mid** |
| **281.jpg** | <img src="results/figures/seggpt_qualitative/worst_prompt_source_image.jpg" width="160"/> | <img src="results/figures/seggpt_qualitative/worst_prompt_mask.png" width="160"/> | 🔴 **Worst** — Mean Dice 0.120 |

> The sharp performance gap between prompts reveals that the **quality and representativeness of the weak-signal mask** is the dominant factor in downstream segmentation quality — not the SegGPT model itself.

---

## 📊 Quantitative Results

<div align="center">

### Summary Table

| Prompt | Mean Dice | Median Dice | Mean IoU |
|--------|:---------:|:-----------:|:--------:|
| **1112.jpg (Best)** | **0.316** | **0.253** | — |
| 2829.jpg (Mid) | — | — | — |
| 281.jpg (Worst) | 0.120 | 0.000 | — |

<br/>

### Dice Score Distributions

<img src="results/quantitative/dice_distribution_best.png" width="32%"/>
<img src="results/quantitative/dice_distribution_worst.png" width="32%"/>
<img src="results/quantitative/dice_distribution_combined.png" width="32%"/>

<sub><i>Left: Best prompt Dice distribution &nbsp;|&nbsp; Centre: Worst prompt &nbsp;|&nbsp; Right: Combined overlay</i></sub>

<br/><br/>

### IoU vs. Dice — Boxplot Comparison

<img src="results/quantitative/boxplot_best.png" width="45%"/>
&nbsp;&nbsp;
<img src="results/quantitative/boxplot_worst.png" width="45%"/>

<sub><i>Box plots showing metric spread across all target slices for the best (left) and worst (right) prompt configurations.</i></sub>

</div>

<br/>

> **Context:** A Dice score of 0.316 is modest relative to supervised segmentation benchmarks (typically 0.85–0.95 for BraTS). However, it represents a substantial non-trivial signal produced by a pipeline with **zero target-domain annotations, zero training, and zero domain adaptation**. The baseline — raw OneFormer — scores effectively zero on the same metric.

---

## 🖼️ Qualitative Results

<div align="center">

### OneFormer Failure Analysis

<img src="results/figures/oneformer_failures/oneformer_failures_grid.png" width="800"/>

### Sports Ball as Spatial Signal

<img src="results/figures/sports_ball_verification/sports_ball_detection.png" width="800"/>

<img src="results/figures/sports_ball_verification/sports_ball_verification_a.png" width="800"/>

&nbsp;
<img src="results/figures/sports_ball_verification/sports_ball_verification_b.png" width="800"/>

### SegGPT Guided Outputs

<img src="results/figures/sports_ball_verification/seggpt_prompting_mechanism.png" width="800"/>

</div>

---

## 🌍 Why This Matters

This work is not a production-ready segmentation system. It is a **proof-of-concept for an unexplored mechanism**: the deliberate repurposing of cross-domain model misclassifications as visual prompts for downstream models.

Most of the field treats model failure outputs as noise to be discarded. This work treats a specific, rare class of failure — spatially coherent, semantically incorrect misclassification — as **latent geometric signal**.

<div align="center">

| Traditional Approach | This Work |
|---|---|
| Collect expert annotations | Use model failures as signal |
| Fine-tune on target domain | Zero training, inference only |
| High cost, slow iteration | No labeling cost |
| Requires domain expertise | Requires only failure analysis |

</div>

If this mechanism generalises, it could provide pathways into:

- **Medical imaging** — tumors, lesions, organs
- **Satellite/aerial imagery** — structures, land cover, anomalies
- **Industrial inspection** — defects, components
- **Any domain** where labeled data is genuinely scarce

The only resource required is careful analysis of what generalist foundation models get *wrong* — and understanding why.

---

## ⚠️ Limitations

This is an **undergraduate research thesis**. The following limitations are acknowledged transparently:

| Limitation | Description |
|---|---|
| **Signal rarity** | The "sports ball" signal appeared in only 12 of 1,500 slices (~0.8%). This makes the specific pipeline impractical as a general-purpose system. |
| **Prompt sensitivity** | Performance varied significantly (Dice 0.316 vs 0.120) across prompts. Automated prompt selection is an open problem. |
| **Absolute performance** | Dice/IoU scores are substantially below supervised benchmarks. |
| **Mechanism opacity** | The precise feature-level reason why certain misclassifications produce spatially accurate masks is not yet formally characterised. |
| **Dataset scope** | Experiments conducted on a single dataset (BraTS), single modality (T1 MRI), single anatomy (brain). Generalisation is unverified. |

---

## 🔭 Future Directions

The questions this work opens are, in some ways, more interesting than the answers it provides:

- **Signal discovery:** Can more frequent and less serendipitous "weak signals" be systematically identified from generalist models applied to specialized domains?
- **Automated prompt selection:** Can prompt quality be predicted — or ranked — before running downstream inference?
- **Cross-modal generalisation:** Does the mechanism transfer across CT, ultrasound, fundoscopy, and histopathology?
- **Cross-anatomy generalisation:** Liver, lung, prostate — does it hold beyond brain tumors?
- **Formal characterisation:** Can we formally explain, using feature-space analysis, why certain category confusions preserve spatial structure?
- **Scaling the weak signal:** Can weak signals be combined or ensembled to reduce variance across prompts?

These questions point directly toward the research directions that motivated pursuing doctoral study.

---

## 📂 Repository Structure

```text
cross-model-prompting-for-medical-segmentation/
│
├── README.md                           ← You are here
├── Thesis_Sabbir_Ahmed.pdf             ← Full B.Sc. thesis document
├── Final Defense.pdf                   ← Slides presented at final defense
├── requirements.txt                    ← Python package dependencies
├── environment.yml                     ← Reproducible Conda environment
│
├── src/
│   ├── phase1_oneformer/
│   │   ├── run_inference.py            ← Zero-shot OneFormer inference on BraTS slices
│   │   └── extract_sports_ball.py      ← Filters and saves "sports ball" masks from results
│   │
│   ├── phase2_seggpt/
│   │   ├── run_prompted_inference.py   ← SegGPT inference with prompt image-mask pairs
│   │   └── evaluate.py                 ← Computes Dice and IoU against ground truth
│   │
│   └── utils/
│       ├── preprocess_dataset.py       ← Parses COCO annotations, generates binary GT masks
│       ├── visualize_results.py        ← Overlay figures, distribution plots, grid views
│       └── metrics.py                  ← Dice and IoU metric implementations
│
├── data/                               ← (Not included — see setup instructions)
│   ├── brats_kaggle/                   ← Raw BraTS dataset download destination
│   └── processed/
│       ├── images/                     ← Pre-processed MRI slices (JPEG)
│       └── masks/                      ← Binary ground truth masks (PNG)
│
└── results/
    ├── figures/
    │   ├── dataset/                    ← Sample MRI slices and GT masks
    │   ├── oneformer_failures/         ← Failure grids and qualitative outputs
    │   ├── seggpt_qualitative/         ← SegGPT output overlays and comparisons
    │   └── sports_ball_verification/   ← Sports ball mask verification and prompting figures
    └── quantitative/
        ├── dice_distribution_best.png
        ├── dice_distribution_worst.png
        ├── dice_distribution_combined.png
        ├── boxplot_best.png
        └── boxplot_worst.png
```

> **Note on code quality:** This is an experimental undergraduate implementation. Scripts are written for research exploration and reproducibility, not production deployment. Refer to the [thesis PDF](./Thesis_Sabbir_Ahmed.pdf) for detailed methodology, derivations, and discussion.

---

## 🛠️ Setup and Reproduction

### Prerequisites

| Dependency | Version |
|---|---|
| Python | 3.9+ |
| PyTorch | 2.5.1 |
| Detectron2 | 0.6 |
| Transformers (HuggingFace) | 4.51.3 |
| OpenCV | latest |
| Matplotlib / Seaborn / NumPy | latest |

### Installation

```bash
# Clone the repository
git clone https://github.com/Sabbbir/cross-model-prompting-for-medical-segmentation
cd cross-model-prompting-for-medical-segmentation

# Create and activate environment
conda env create -f environment.yml
conda activate cross-model-seg
```

### Step-by-Step Reproduction

#### 1. Download and Preprocess the Dataset

Download the BraTS dataset from Kaggle into `data/brats_kaggle/`, then:

```bash
python src/utils/preprocess_dataset.py \
    --data_dir data/brats_kaggle/ \
    --output_dir data/processed/
```

This parses COCO annotations and generates binary ground truth masks alongside the pre-processed MRI slices.

#### 2. Phase 1 — OneFormer Zero-Shot Inference

```bash
# Run zero-shot inference with the COCO checkpoint
python src/phase1_oneformer/run_inference.py \
    --model coco \
    --data_dir data/processed/images/

# Extract and save "sports ball" coincidence masks
python src/phase1_oneformer/extract_sports_ball.py \
    --results_dir results/oneformer/
```

#### 3. Phase 2 — SegGPT Prompted Inference

```bash
# Run SegGPT using the best-performing prompt (1112.jpg)
python src/phase2_seggpt/run_prompted_inference.py \
    --prompt_image data/processed/images/1112.jpg \
    --prompt_mask results/oneformer/sports_ball_masks/1112_mask.png \
    --target_dir data/processed/images/ \
    --output_dir results/seggpt/
```

#### 4. Evaluate Against Ground Truth

```bash
python src/phase2_seggpt/evaluate.py \
    --predictions_dir results/seggpt/ \
    --gt_dir data/processed/masks/
```

Outputs per-slice Dice and IoU scores, aggregated statistics, and distribution plots.

---

## 💻 Hardware

All experiments were conducted at the **VMI Lab, Jashore University of Science and Technology**.

| Component | Specification |
|---|---|
| GPU | Dual NVIDIA GeForce RTX 2070 Super |
| System RAM | 32 GB |
| OS | Ubuntu (Linux) |


## 📬 Contact

<div align="center">

**Sabbir Ahmed**
B.Sc. in Computer Science and Engineering


📧 [sabbir.ahmed.cse19.just@gmail.com](mailto:sabbir.ahmed.cse19.just@gmail.com)

🏫 Jashore University of Science and Technology, Bangladesh

🔗 [GitHub](https://github.com/Sabbbir)

<br/>

---



</div>
