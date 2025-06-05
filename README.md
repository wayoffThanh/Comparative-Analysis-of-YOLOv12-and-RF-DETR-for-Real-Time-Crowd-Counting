# Comparative-Analysis-of-YOLOv12-and-RF-DETR-for-Real-Time-Crowd-Counting



## Project Overview

This project conducts a comprehensive comparative analysis between two state-of-the-art deep learning models—YOLOv12 and RF-DETR—for the task of real-time crowd counting. The primary objective is to assess their effectiveness based on detection accuracy, inference speed, and resilience under varying levels of crowd density and challenging environmental conditions. Both models were trained and evaluated on the same diverse dataset under uniform experimental conditions to ensure a fair comparison.

The study highlights that YOLOv12 significantly outperforms in terms of processing speed, making it ideal for real-time deployment in surveillance systems. In contrast, RF-DETR delivers superior accuracy in high-density environments, attributed to its transformer-based attention mechanisms. This research underscores the critical trade-offs between computational efficiency and detection performance, offering meaningful insights for practitioners selecting appropriate models for real-world crowd monitoring applications using computer vision. A real-time demo using PyCharm with YOLOv12 was also developed to test inference in practical scenarios.

## Table of Contents

1.  [Motivation]
2.  [Related Work]
3.  [Methodology]
      * [Dataset Collection and Preparation]
      * [Model Architectures].
      * [Image Preprocessing for Models]
      * [Training Configuration]
4.  [Experimental Results and Analysis]
      * [YOLOv12 Performance]
      * [RF-DETR Performance]
      * [Summary of Metrics]
5.  [Real-World Implementation: PyCharm Video Demo]
6.  [Discussion]
7.  [Conclusion and Future Work]
8.  [Setup and Usage]
9.  [Team and Supervisor]
10. [References]


## Motivation

Crowd counting is a fundamental task in computer vision with significant applications in surveillance, crowd safety, and smart city management. Traditional techniques often fall short in accuracy and scalability under real-world conditions. While deep learning object detection models like YOLOv12 (known for speed and modularity with Area Attention and R-ELAN modules and RF-DETR (leveraging transformer-based deformable attention for high accuracy in dense scenes have shown promise, a systematic comparison in people-counting tasks is needed to guide practical deployment choices. This study aims to fill this gap by providing insights into the trade-offs between detection precision and computational speed.

## Related Work

Crowd counting has evolved significantly from early regression-based techniques and handcrafted features with SVMs, which struggled with generalization in complex environments. The advent of deep learning, particularly CNNs like MCNN and CSRNet, improved performance by using multi-scale features for density estimation rather than explicit object detection.

The YOLO (You Only Look Once) family revolutionized real-time object detection with unified single-stage pipelines. Successive versions have enhanced both speed and accuracy, with YOLOv12 incorporating features like Area Attention and R-ELAN backbones for superior performance with lower latency.

In parallel, transformer-based models like DETR introduced an end-to-end approach using self-attention. While initial versions had limitations like slow convergence, variants like Deformable DETR and RT-DETR led to RF-DETR, which balances accuracy and speed using deformable attention and optimized training. While YOLO models generally excel in speed, transformer-based detectors often offer better detection quality in challenging conditions. This work directly compares YOLOv12 and RF-DETR for people counting to evaluate their specific trade-offs.

## Methodology

### Dataset Collection and Preparation

  **Source & Composition:** The dataset comprises 5,030 annotated images, primarily from the ShanghaiTech crowd counting dataset (known for diverse densities and urban settings). It was enriched with images from night walk videos in Shibuya, Tokyo, to include densely populated areas under challenging lighting[cite: 50, 51]. [cite\_start]The dataset reflects varied camera viewpoints (top-down, side-view, oblique) and lighting conditions (daytime, nighttime).
 **Annotation:** All images were annotated with a single class: "person".
  * **Data Splitting & Preprocessing:** Using Roboflow, the dataset was split into:
      Training: 4,388 images (87%) 
      Validation: 428 images (9%) 
      Test: 214 images (4%) 
     Preprocessing steps included automatic orientation correction, resizing to $640 \\times 640$ pixels, and adaptive histogram-based contrast enhancement. Full-color information was preserved (no grayscale conversion).
  **Augmentation:** Each training image was augmented with three variants, incorporating random rotations from $-15^{\\circ}$ to $+15^{\\circ}$ to simulate camera orientation variations.
  * **Sample Images:**
      * Figure 1: Sample annotated images for validation.
      * Figure 2: Sample annotated images for training.
      * Figure 3: Sample annotated images for test.

### Model Architectures

  * **YOLOv12 (Figure 4):**
      * Follows a modular single-stage detection design.
      * Backbone: Series of convolutional layers with R-ELAN (Residual Efficient Layer Aggregation Network) blocks.
      * Neck: Performs feature aggregation via upsampling and concatenation.
      * Head: Uses Flash Attention A2 modules for final prediction at three different scales.
 **RF-DETR (Figure 5):**
      * Transformer-based architecture inspired by DETR.
      * Backbone: DINOv2 ViT-L/14 pre-trained backbone.
      * Encoder-Decoder: Deformable transformer pipeline with multi-scale deformable self-attention in both encoder and decoder stages.
      * Object queries are passed through cross-attention layers for bounding box predictions.

### Image Preprocessing for Models

All images were resized to fit model input requirements (typically 640 x 640 pixels). Basic augmentations like horizontal flipping, brightness adjustment, and scaling were applied. Annotation files were converted to YOLO API key format for YOLOv12 and COCO-style JSON API key for RF-DETR.

### Training Configuration

Both models were trained under standardized conditions. (Table 1)


* **Hardware:**

      * YOLOv12: Google Colab Pro with NVIDIA L4 GPU.
      * RF-DETR: Google Colab Pro with NVIDIA A100 GPU.

  * **Parameters:**

    | Parameter        | YOLOv12 (L4 GPU)                | RF-DETR (A100 GPU)          |
    | ---------------- | ------------------------------- | --------------------------- |
    | Epochs           | 150                             | 107                         |
    | Optimizer        | SGD                             | AdamW                       |
    | Batch Size       | 16                              | 8                           |
    | Learning Rate    | 0.01                            | 0.0001                      |
    | Input Size       | $640 \\times 640$                 | $800 \\times 800$ (scaled)     |
    | Loss Functions   | CIoU Loss, BCE for Class/Object | Hungarian Matching, Focal Loss |
    | Total Params     | 2.5M (YOLOv12n)                 | \~40M (RF-DETR)              |
   

## Experimental Results and Analysis

Models were evaluated on mean Average Precision (mAP), recall, and inference latency.

### YOLOv12 Performance (NVIDIA L4 GPU)

(Figure 6: Training results of YOLOv12 model)

  * Trained for 150 epochs.
  * mAP@50: 0.731 
  * mAP@50-95: 0.456 
  * Precision: 0.79 
  * Recall: 0.652 
  * Inference Speed: 1.9ms inference time + 4.4ms postprocessing per image.
  * Approximately 2.92 FPS on tested hardware.

### RF-DETR Performance (NVIDIA A100 GPU)

(Figure 7: Training results of RF-DETR model)

  * Trained for 107 epochs.
  * mAP@50: 0.688. 
  * mAP@50-95: 0.443.
  * Outperformed YOLOv12 in large-object detection:
      * AP@large: 0.797.
      * AR@large: 0.870.
  * Training Time: Over 8.5 hours.

### Summary of Metrics

(Table 2)

| Metric          | YOLOv12n        | RF-DETR             |
| --------------- | --------------- | ------------------- |
| mAP@50          | 0.731           | 0.688               |
| mAP@50-95       | 0.456           | 0.443               |
| Precision (Box) | 0.79            |                     |
| Recall          | 0.652           | 0.513 (AR all)      |
| AP@large        | 0.732           | 0.797               |
| AP@small        |                 | 0.206               |
| AR@large        |                 | 0.870               |
| Inference Time  | $1.9ms + 4.4ms$ | Slower (not reported) |
| Training Time   | \~2.0 hours      | \~8.5 hours          |
| GPU Used        | NVIDIA L4       | NVIDIA A100         |


## Real-World Implementation: PyCharm Video Demo

A video-based inference demo was developed using PyCharm to demonstrate YOLOv12's practical deployment.

  * **Input:** A "night walk" video from Shibuya, Tokyo, featuring high people density and challenging nighttime lighting.
  * **Tooling:** Ultralytics' `solutions.ObjectCounter` module was used with the custom-trained `YOLOv12.pt` file (`best.pt`).
  * **Functionality:** A predefined polygonal Region of Interest (ROI) constrained counting to a specific area. The model detects individuals, applies counting logic within the ROI, and overlays bounding boxes and counts on output frames.
  * **Technology:** OpenCV for video I/O and annotation rendering.
  * **Performance:** Achieved near-real-time performance on an NVIDIA L4 GPU, suitable for offline analysis and post-event review.
    **Applications:** Highlights YOLOv12's flexibility for crowd flow analysis, event auditing, and pedestrian traffic assessment in urban low-light conditions.
  * (Figure 8: Output frame from YOLOv12 demo )

## Discussion

The analysis reveals distinct trade-offs between YOLOv12 and RF-DETR for people counting[cite: 102].

  * **YOLOv12n:**
      * Consistently superior inference speed (sub-10ms frame processing) due to its lightweight architecture (R-ELAN, Area Attention, anchor-free heads).
      * Well-suited for edge-based deployment where speed and resource constraints are critical.
      * Showed slightly better resilience in low-light environments (e.g., Shibuya footage), possibly due to simpler spatial priors.
  * **RF-DETR:**
    * Slower to train and infer due to its transformer backbone and deformable attention modules.
      * Delivered higher precision in detecting large and well-separated individuals, capturing long-range dependencies and contextual relationships effectively in dense/occluded scenes.
      * Outperformed YOLOv12 in AP@large and AR@large metrics.
      * Performance degraded more in scenes with high contrast or non-standard viewpoints due to positional encoding sensitivity.
      * Occasionally miscounted overlapping individuals in high-density frames due to box suppression conflicts.
  * **General Observations:**
      * The single-class detection task (person) reduced complexity, potentially amplifying YOLOv12's speed advantage.
  * **Recommendation:**
      * YOLOv12 is better for real-time crowd analytics, especially in low-resource environments[cite: 115].
      * RF-DETR is more appropriate for high-precision offline tasks like post-event analysis or forensic video review.
      * Choice depends on application requirements: latency vs. accuracy, streaming vs. batch, visual domain complexity.

## Conclusion and Future Work

This study provides a comprehensive comparison of YOLOv12 and RF-DETR for crowd counting in complex urban environments. YOLOv12 is more suitable for real-time deployment due to its speed, while RF-DETR offers higher accuracy for large/well-separated individuals but at a higher computational cost.

**Limitations and Future Work:**

  * **Dataset Enhancement:** 
      * Expand dataset quantitatively (more samples, cities, densities).
      * Introduce temporal annotations for video-based tracking and flow analysis.
      * Incorporate more edge cases (overlapping individuals, occlusions, multi-scale objects).
  * **Demo Development:**
      * Develop a comparable demo pipeline for RF-DETR to enable fairer qualitative comparison and support offline forensic applications.
  * **Model Exploration:**
      * Explore hybrid architectures combining YOLOv12's speed with RF-DETR's contextual awareness.
      * Integrate tracking algorithms (e.g., DeepSORT, ByteTrack) for complete people flow tracking systems.

This work lays a foundation for deeper exploration of CNN vs. transformer detectors in people counting, aiming to bridge the gap between academic models and real-world systems.

## Setup and Usage



**Prerequisites:**

  * Python (e.g., 3.8+)
  * PyTorch
  * Ultralytics YOLO
  * OpenCV


**Installation:**


**Data Preparation:**

  * Describe where to place the dataset and its expected format (e.g., YOLO format, COCO format).
  * Mention any scripts for data conversion or preprocessing.

**Training Models:**

## Team and Supervisor

  * **Group Students:**
      Vu Duc Thanh - Leader (ID: 22071150, Class: FDB 2022C) 
      
  * **Supervisor:** Associate Professor, Ph.D. Tran Thi Oanh 

## References

*(A full list of references is available in the original report. Key cited works include Chen et al. (2024) for YOLOv12, Carion et al. (2020) for DETR, and Sapkota et al. (2024) for RF-DETR comparisons in other contexts.)*

  * Carion, N., et al. (2020). End-to-End Object Detection with Transformers. 
  * Chen, Y., et al. (2024). YOLOv12: Anchor-Free Real-Time Object Detection with Area Attention and R-ELAN. 
  * Sapkota, R., et al. (2024). RF-DETR Object Detection vs YOLOv12... 
  * Zhu, X., et al. (2020). Deformable DETR... 
  * Zhang, Y., et al. (2016). Single-Image Crowd Counting via Multi-Column Convolutional Neural Network. 

