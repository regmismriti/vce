
# Rethinking Evaluation for Video Capsule Endoscopy: Validation Protocol, Source-Video Diversity, and Cross-Dataset Shift

**Authors:** Smriti Regmi and Debesh Jha

**Affiliation:** Biomedical Perception & Intelligence Lab, Department of Computer Science, University of South Dakota, USA


---

## Overview

Video Capsule Endoscopy (VCE) is a non-invasive imaging technique used to examine the gastrointestinal (GI) tract and identify abnormalities such as ulcers, erosions, angiectasia, and bleeding.

Although deep learning has demonstrated promising performance in automated VCE classification, reliable generalization remains challenging due to:

- Severe class imbalance.
- High visual similarity between different gastrointestinal findings.
- Significant intra-class variation.
- Limited independent source-video diversity.
- Strong correlations between frames originating from the same video.
- Distribution shifts between different VCE datasets.

This study investigates how validation strategies, training-source diversity, and cross-dataset shift influence the performance and reliability of deep learning models for VCE classification.

We introduce **DMSNet (Deformable Multi-Scale Edge-Attention Network)** and benchmark it against 18 representative CNN, transformer, and pretrained vision architectures.

Our experiments demonstrate that model performance and relative rankings vary substantially across evaluation protocols, emphasizing the importance of source-aware validation and external evaluation when assessing generalization.

---

## Key Contributions

### 1. Comprehensive Architecture Benchmarking

We evaluate 19 architectures, including DMSNet, under three evaluation settings:

- Random frame-level validation.
- Source-video-disjoint validation.
- Cross-dataset evaluation on CV2024.

The results demonstrate that no single architecture achieves the highest Matthews Correlation Coefficient (MCC) across all three evaluation settings.

### 2. Source-Video Diversity Analysis

We investigate the effect of increasing the number of independent source videos available during training.

Under a per-class cap of 200 training frames, increasing the maximum source-video coverage from four to eight improves MCC for 17 of 19 architectures while keeping the total training-set size fixed at 2,010 frames.

### 3. Cross-Dataset Generalization

Models trained on Kvasir-Capsule are evaluated on the external CV2024 dataset without additional fine-tuning.

The experiments reveal substantial performance degradation across architectures, indicating that strong in-domain performance does not necessarily translate to reliable external generalization.

### 4. DMSNet Architecture

We introduce a classification network that combines multi-scale feature fusion, deformable convolution, edge-guided attention, and sparse feature selection to learn complementary spatial and semantic representations of gastrointestinal abnormalities.

---

## Proposed Architecture: DMSNet

DMSNet is built on an EfficientNetV2-S backbone and consists of six major components.

### 1. EfficientNetV2-S Backbone

Extracts hierarchical feature representations from four stages of the backbone.

Shallow features capture fine-grained spatial and boundary information, while deeper features encode higher-level semantic representations.

### 2. Multi-Scale Feature Fusion (MSFF)

Aggregates features from multiple backbone stages by:

- Projecting feature maps to a common channel dimension.
- Aligning spatial resolutions through bilinear interpolation.
- Concatenating the aligned feature representations.
- Refining the combined features using convolutional blocks.

This enables the network to integrate local spatial details with high-level semantic information.

### 3. Deformable Lesion Refinement (DLR)

Uses deformable convolution to adapt the sampling locations of convolutional kernels to irregular gastrointestinal structures.

A residual connection combines the refined representation with the original fused features.

### 4. Edge-Guided Attention (EGA)

Incorporates explicit boundary information through Sobel-based edge extraction.

The resulting edge information is processed by a lightweight attention branch to generate a spatial attention map that emphasizes boundary-aware regions.

### 5. Sparse Feature Selection (SFS)

Combines global average pooling and global max pooling to construct a feature representation.

A gating network learns a feature-selection mask, encouraging the suppression of less relevant feature dimensions.

A sparsity penalty is incorporated into the training objective to regularize the gate activations.

### 6. Classification Head

The selected feature representation is passed through a linear classifier to generate predictions for the 11 gastrointestinal categories.

---

## Datasets

### Kvasir-Capsule

Kvasir-Capsule is the primary dataset used for training and in-domain evaluation.

The original dataset contains 47,238 labeled frames across 14 categories.

For this study, we construct an 11-class classification task by excluding Blood-hematin, Ampulla of Vater, and Polyp, while retaining Blood-fresh as Blood.

The resulting categories are:

1. Normal
2. Pylorus
3. Ileo-cecal Valve
4. Angiectasia
5. Blood
6. Erosion
7. Erythematous
8. Foreign Bodies
9. Lymphangiectasia
10. Reduced Mucosal View
11. Ulcer

Source-video identifiers are extracted from frame filenames to support group-aware dataset partitioning.

### CV2024

CV2024 is used for external evaluation of models trained on Kvasir-Capsule.

Cross-dataset evaluation is restricted to eight semantically corresponding categories:

- Angiectasia / Angioectasia
- Blood / Bleeding
- Erosion
- Erythematous / Erythema
- Foreign Bodies / Foreign Body
- Lymphangiectasia
- Normal
- Ulcer

The three Kvasir-Capsule categories that are absent from the shared label space—Pylorus, Ileo-cecal Valve, and Reduced Mucosal View—are masked from the prediction space during external evaluation.

Models are evaluated on CV2024 without additional fine-tuning.

---

## Experimental Setup

### Training Configuration

The following training configuration is used for the architecture benchmarking experiments.

| Parameter | Configuration |
|---|---|
| Framework | PyTorch |
| Input resolution | 224 × 224 |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Maximum epochs | 100 |
| Learning rate scheduler | CosineAnnealingLR |
| Loss function | Unweighted focal loss |
| Focal loss gamma | 2 |
| Early stopping criterion | Validation MCC |
| Random seeds | 41, 42, 43 |

DMSNet additionally incorporates a mean gate-activation sparsity penalty with a regularization coefficient of 1e-3.

A separate class-weighted focal-loss experiment is conducted for DMSNet, including its component-wise ablation study.

All baseline architectures are fully fine-tuned except DINOv3, Endo-FM, and ViT-L/16 (MAE), which use a linear-probing protocol with frozen pretrained backbones.

### Data Augmentation

Training augmentation includes:

- Random horizontal flipping.
- Random vertical flipping.
- Random rotation of up to 15 degrees.
- Color jitter with brightness and contrast adjustments of 0.2.

Input images are resized to 224 × 224 pixels.

### Evaluation Metrics

Four classification metrics are reported:

- Matthews Correlation Coefficient (MCC).
- Macro Recall.
- Macro F1-score.
- Overall Accuracy.

MCC is used as the primary evaluation metric because the classification task exhibits severe class imbalance.

Results are reported as mean ± standard deviation across three random seeds.

---

## Evaluation Protocols

### 1. Random Frame-Level Validation

Frames from the official Kvasir-Capsule training pool are randomly partitioned into training and validation subsets.

Because splitting is performed at the frame level, visually correlated frames from the same source video may appear in both subsets.

### 2. Source-Video-Disjoint Validation

All frames originating from a particular source video are assigned to a single partition to prevent source-video overlap between training and validation.

Group-aware partitioning uses source-video identifiers and StratifiedGroupKFold.

The resulting split contains 15 training videos and four validation videos with zero overlap.

Categories with insufficient independent source videos are retained entirely in training.

Consequently, validation MCC is used for checkpoint selection rather than as a complete 11-class performance estimate.

Models selected under both validation strategies are subsequently evaluated on the same official Kvasir-Capsule test set.

**Important:** The official Kvasir-Capsule training and test pools are not fully source-video-disjoint. The source-video-disjoint protocol described here applies to the training and validation partition.

### 3. Source-Video Diversity Experiment

To investigate the influence of independent training-source diversity, the maximum number of sampled source videos per class is varied.

The experiment uses:

- Maximum source videos per class: k = 1, 2, 4, 8.
- Maximum training frames per class: 200.
- Frame sampling distributed as evenly as possible across selected videos.

The resulting training-set sizes are:

| Maximum source videos (k) | Training frames |
|---|---:|
| 1 | 1,434 |
| 2 | 1,723 |
| 4 | 2,010 |
| 8 | 2,010 |

The comparison between k = 4 and k = 8 provides the clearest controlled assessment of source-video diversity because both conditions contain the same total number of training frames.

### 4. Cross-Dataset Evaluation

Models trained on Kvasir-Capsule using random frame-level validation are evaluated on CV2024 without additional fine-tuning.

The evaluation is restricted to the eight shared gastrointestinal categories to investigate generalization under dataset shift.

---

## Experimental Results

### 1. In-Domain Benchmark

The following table summarizes representative results from the common unweighted focal-loss benchmark.

All reported MCC values are measured on the official Kvasir-Capsule test set after checkpoint selection using the indicated validation strategy.

| Model | Random Validation MCC | Source-Video-Disjoint Validation MCC |
|---|---:|---:|
| DMSNet | 0.427 ± 0.022 | 0.405 ± 0.025 |
| MaxViT | 0.398 ± 0.024 | 0.452 ± 0.026 |
| Swin-V1 | 0.417 ± 0.019 | 0.365 ± 0.046 |
| EfficientNet-B7 | 0.397 ± 0.027 | 0.426 ± 0.049 |
| ConvNeXt-Tiny | 0.396 ± 0.018 | 0.390 ± 0.029 |
| DINOv3 | 0.407 ± 0.002 | 0.307 ± 0.047 |

**Key observations:**

- DMSNet achieves the highest MCC of 0.427 under random frame-level validation using the common unweighted focal-loss protocol.
- MaxViT achieves the highest MCC of 0.452 under source-video-disjoint validation.
- Model performance and relative rankings change substantially between validation strategies.
- The results highlight the importance of evaluation protocol when comparing VCE classification architectures.

In a separate class-weighted focal-loss experiment, DMSNet achieves an MCC of 0.444 ± 0.028 under random frame-level validation.

This weighted-loss result is reported separately from the common unweighted focal-loss architecture comparison.

### 2. Effect of Source-Video Diversity

The following table reports representative MCC results under different maximum source-video coverage settings.

| Model | k = 1 | k = 2 | k = 4 | k = 8 |
|---|---:|---:|---:|---:|
| DMSNet | 0.126 | 0.190 | 0.211 | 0.242 |
| MaxViT | 0.163 | 0.179 | 0.245 | 0.311 |
| ConvNeXtV2 | 0.171 | 0.201 | 0.261 | 0.276 |
| ViT-B/16 | 0.142 | 0.195 | 0.256 | 0.297 |
| DINOv3 | 0.067 | 0.050 | 0.107 | 0.172 |

Values represent mean MCC across three random seeds.

When the training-set size is fixed at 2,010 frames, increasing the maximum source-video coverage from four to eight improves MCC for 17 of 19 architectures.

The average MCC improvement across architectures is 0.0449.

A paired one-sided Wilcoxon signed-rank test confirms a statistically significant positive shift across architectures (W = 187.0, p = 9.99 × 10⁻⁶).

These findings demonstrate that independent source-video diversity is an important consideration for VCE classification, beyond the total number of training frames.

### 3. Cross-Dataset Generalization

The following table reports representative results on the external CV2024 dataset.

| Model | MCC |
|---|---:|
| ConvNeXt-Tiny | 0.207 ± 0.003 |
| MaxViT | 0.195 ± 0.024 |
| DMSNet | 0.192 ± 0.063 |
| ConvNeXtV2 | 0.191 ± 0.038 |
| DINOv3 | 0.167 ± 0.012 |

ConvNeXt-Tiny achieves the highest external MCC of 0.207.

All evaluated architectures exhibit substantial performance degradation under cross-dataset shift relative to their in-domain evaluation.

These results indicate that strong performance on Kvasir-Capsule does not necessarily translate into comparable performance on an external VCE dataset.

### 4. DMSNet Ablation Study

A leave-one-module-out ablation study is conducted using class-weighted focal loss.

| Configuration | MCC |
|---|---:|
| Full DMSNet | 0.444 ± 0.028 |
| Without DLR | 0.430 ± 0.021 |
| Without MSFF | 0.422 ± 0.009 |
| Without SFS | 0.410 ± 0.034 |
| Without EGA | 0.375 ± 0.057 |

Removing any of the four components decreases mean MCC relative to the full DMSNet architecture.

The largest reduction in MCC is observed when Edge-Guided Attention is removed.

### 5. Computational Efficiency

DMSNet is evaluated at an input resolution of 224 × 224 on an NVIDIA Tesla V100 GPU.

| Metric | DMSNet |
|---|---:|
| Parameters | 23.78 M |
| Computational cost | 10.34 GFLOPs |
| Throughput | 553 frames/s |
| Latency | 1.809 ms/image |

These measurements characterize the computational requirements of DMSNet for frame-level VCE classification.

---

## Main Findings

The experimental results support the following conclusions:

**Evaluation protocol matters.** Relative model performance varies substantially depending on whether random frame-level validation, source-video-disjoint validation, or cross-dataset evaluation is used.

**Independent source-video diversity matters.** Increasing the number of independent source videos can improve classification performance even when the total training-set size remains unchanged.

**In-domain accuracy is not sufficient.** Models demonstrating strong performance on Kvasir-Capsule can experience substantial performance degradation when evaluated on CV2024.

**Model architecture alone does not determine generalization.** Validation strategy, source-video composition, class imbalance, and dataset shift must also be considered when interpreting experimental results.

---

## Limitations

Several limitations should be considered when interpreting the results:

- Source-video availability is highly imbalanced across gastrointestinal categories.
- Some categories contain too few source videos to contribute to source-video-disjoint validation.
- The official Kvasir-Capsule training and test partitions are not fully source-video-disjoint.
- The lower-k source-video diversity experiments differ in both frame count and source-video coverage.
- External evaluation is restricted to the eight categories shared by Kvasir-Capsule and CV2024.
- The study focuses on frame-level classification and does not incorporate temporal information from complete VCE examinations.

Future research should investigate source-aware training on larger multi-center datasets and incorporate temporal modeling to improve generalization across complete VCE examinations.

---
