# Comparative-Analysis-of-YOLOv12-and-RF-DETR-for-Real-Time-Crowd-Counting



## Project Overview

[cite\_start]This project conducts a comprehensive comparative analysis between two state-of-the-art deep learning models—YOLOv12 and RF-DETR—for the task of real-time crowd counting[cite: 15]. [cite\_start]The primary objective is to assess their effectiveness based on detection accuracy, inference speed, and resilience under varying levels of crowd density and challenging environmental conditions[cite: 16, 18, 19, 20]. [cite\_start]Both models were trained and evaluated on the same diverse dataset under uniform experimental conditions to ensure a fair comparison[cite: 17].

[cite\_start]The study highlights that YOLOv12 significantly outperforms in terms of processing speed, making it ideal for real-time deployment in surveillance systems[cite: 18]. [cite\_start]In contrast, RF-DETR delivers superior accuracy in high-density environments, attributed to its transformer-based attention mechanisms[cite: 19]. [cite\_start]This research underscores the critical trade-offs between computational efficiency and detection performance, offering meaningful insights for practitioners selecting appropriate models for real-world crowd monitoring applications using computer vision[cite: 20]. [cite\_start]A real-time demo using PyCharm with YOLOv12 was also developed to test inference in practical scenarios[cite: 28].

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

[cite\_start]Crowd counting is a fundamental task in computer vision with significant applications in surveillance, crowd safety, and smart city management[cite: 21]. [cite\_start]Traditional techniques often fall short in accuracy and scalability under real-world conditions[cite: 22]. [cite\_start]While deep learning object detection models like YOLOv12 (known for speed and modularity with Area Attention and R-ELAN modules [cite: 24][cite\_start]) and RF-DETR (leveraging transformer-based deformable attention for high accuracy in dense scenes [cite: 25][cite\_start]) have shown promise, a systematic comparison in people-counting tasks is needed to guide practical deployment choices[cite: 26, 29]. [cite\_start]This study aims to fill this gap by providing insights into the trade-offs between detection precision and computational speed[cite: 29].

## Related Work

[cite\_start]Crowd counting has evolved significantly from early regression-based techniques and handcrafted features with SVMs, which struggled with generalization in complex environments[cite: 32, 33]. [cite\_start]The advent of deep learning, particularly CNNs like MCNN and CSRNet, improved performance by using multi-scale features for density estimation rather than explicit object detection[cite: 34, 35, 36].

[cite\_start]The YOLO (You Only Look Once) family revolutionized real-time object detection with unified single-stage pipelines[cite: 37]. [cite\_start]Successive versions have enhanced both speed and accuracy, with YOLOv12 incorporating features like Area Attention and R-ELAN backbones for superior performance with lower latency[cite: 38, 39].

[cite\_start]In parallel, transformer-based models like DETR introduced an end-to-end approach using self-attention[cite: 40, 41]. [cite\_start]While initial versions had limitations like slow convergence, variants like Deformable DETR and RT-DETR led to RF-DETR, which balances accuracy and speed using deformable attention and optimized training[cite: 42, 43]. [cite\_start]While YOLO models generally excel in speed, transformer-based detectors often offer better detection quality in challenging conditions[cite: 44]. [cite\_start]This work directly compares YOLOv12 and RF-DETR for people counting to evaluate their specific trade-offs[cite: 45, 46].

## Methodology

### Dataset Collection and Preparation

  * [cite\_start]**Source & Composition:** The dataset comprises 5,030 annotated images, primarily from the ShanghaiTech crowd counting dataset (known for diverse densities and urban settings)[cite: 48, 49]. [cite\_start]It was enriched with images from night walk videos in Shibuya, Tokyo, to include densely populated areas under challenging lighting[cite: 50, 51]. [cite\_start]The dataset reflects varied camera viewpoints (top-down, side-view, oblique) and lighting conditions (daytime, nighttime)[cite: 51, 52].
  * [cite\_start]**Annotation:** All images were annotated with a single class: "person"[cite: 53].
  * **Data Splitting & Preprocessing:** Using Roboflow, the dataset was split into:
      * [cite\_start]Training: 4,388 images (87%) [cite: 54]
      * [cite\_start]Validation: 428 images (9%) [cite: 54]
      * [cite\_start]Test: 214 images (4%) [cite: 54]
        [cite\_start]Preprocessing steps included automatic orientation correction, resizing to $640 \\times 640$ pixels, and adaptive histogram-based contrast enhancement[cite: 55]. [cite\_start]Full-color information was preserved (no grayscale conversion)[cite: 56].
  * [cite\_start]**Augmentation:** Each training image was augmented with three variants, incorporating random rotations from $-15^{\\circ}$ to $+15^{\\circ}$ to simulate camera orientation variations[cite: 57, 58].
  * **Sample Images:**
      * [cite\_start]Figure 1: Sample annotated images for validation[cite: 59].
      * [cite\_start]Figure 2: Sample annotated images for training[cite: 60].
      * [cite\_start]Figure 3: Sample annotated images for test[cite: 61].

### Model Architectures

  * [cite\_start]**YOLOv12 (Figure 4 [cite: 62]):**
      * [cite\_start]Follows a modular single-stage detection design[cite: 62].
      * [cite\_start]Backbone: Series of convolutional layers with R-ELAN (Residual Efficient Layer Aggregation Network) blocks[cite: 63].
      * [cite\_start]Neck: Performs feature aggregation via upsampling and concatenation[cite: 63].
      * [cite\_start]Head: Uses Flash Attention A2 modules for final prediction at three different scales[cite: 64].
  * [cite\_start]**RF-DETR (Figure 5 [cite: 69]):**
      * [cite\_start]Transformer-based architecture inspired by DETR[cite: 65].
      * [cite\_start]Backbone: DINOv2 ViT-L/14 pre-trained backbone[cite: 66].
      * [cite\_start]Encoder-Decoder: Deformable transformer pipeline with multi-scale deformable self-attention in both encoder and decoder stages[cite: 66, 67].
      * [cite\_start]Object queries are passed through cross-attention layers for bounding box predictions[cite: 68].

### Image Preprocessing for Models

[cite\_start]All images were resized to fit model input requirements (typically $640 \\times 640$ pixels)[cite: 69]. [cite\_start]Basic augmentations like horizontal flipping, brightness adjustment, and scaling were applied[cite: 69]. [cite\_start]Annotation files were converted to YOLO API key format for YOLOv12 and COCO-style JSON API key for RF-DETR[cite: 69, 70].

### Training Configuration

[cite\_start]Both models were trained under standardized conditions[cite: 71, 72]. [cite\_start](Table 1 [cite: 74])

  * **Hardware:**

      * [cite\_start]YOLOv12: Google Colab Pro with NVIDIA L4 GPU[cite: 71].
      * [cite\_start]RF-DETR: Google Colab Pro with NVIDIA A100 GPU[cite: 72].

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
    [cite\_start]*[cite: 74]*

## Experimental Results and Analysis

[cite\_start]Models were evaluated on mean Average Precision (mAP), recall, and inference latency[cite: 75, 76].

### YOLOv12 Performance (NVIDIA L4 GPU)

[cite\_start](Figure 6: Training results of YOLOv12 model [cite: 79])

  * [cite\_start]Trained for 150 epochs[cite: 79].
  * [cite\_start]mAP@50: 0.731 [cite: 79]
  * [cite\_start]mAP@50-95: 0.456 [cite: 79]
  * [cite\_start]Precision: 0.79 [cite: 79]
  * [cite\_start]Recall: 0.652 [cite: 80]
  * [cite\_start]Inference Speed: 1.9ms inference time + 4.4ms postprocessing per image[cite: 80].
  * [cite\_start]Approximately 2.92 FPS on tested hardware[cite: 80].

### RF-DETR Performance (NVIDIA A100 GPU)

[cite\_start](Figure 7: Training results of RF-DETR model [cite: 83])

  * [cite\_start]Trained for 107 epochs[cite: 85].
  * [cite\_start]mAP@50: 0.688 [cite: 85]
  * [cite\_start]mAP@50-95: 0.443 [cite: 85]
  * Outperformed YOLOv12 in large-object detection:
      * [cite\_start]AP@large: 0.797 [cite: 86]
      * [cite\_start]AR@large: 0.870 [cite: 86]
  * [cite\_start]Training Time: Over 8.5 hours[cite: 87].

### Summary of Metrics

[cite\_start](Table 2 [cite: 89])

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
[cite\_start]*[cite: 89]*

## Real-World Implementation: PyCharm Video Demo

[cite\_start]A video-based inference demo was developed using PyCharm to demonstrate YOLOv12's practical deployment[cite: 90].

  * [cite\_start]**Input:** A "night walk" video from Shibuya, Tokyo, featuring high people density and challenging nighttime lighting[cite: 91].
  * [cite\_start]**Tooling:** Ultralytics' `solutions.ObjectCounter` module was used with the custom-trained `YOLOv12.pt` file (`best.pt`)[cite: 92, 93].
  * [cite\_start]**Functionality:** A predefined polygonal Region of Interest (ROI) constrained counting to a specific area[cite: 94]. [cite\_start]The model detects individuals, applies counting logic within the ROI, and overlays bounding boxes and counts on output frames[cite: 95].
  * [cite\_start]**Technology:** OpenCV for video I/O and annotation rendering[cite: 96].
  * [cite\_start]**Performance:** Achieved near-real-time performance on an NVIDIA L4 GPU, suitable for offline analysis and post-event review[cite: 96, 97, 98].
  * [cite\_start]**Applications:** Highlights YOLOv12's flexibility for crowd flow analysis, event auditing, and pedestrian traffic assessment in urban low-light conditions[cite: 99].
  * [cite\_start](Figure 8: Output frame from YOLOv12 demo [cite: 101])

## Discussion

[cite\_start]The analysis reveals distinct trade-offs between YOLOv12 and RF-DETR for people counting[cite: 102].

  * **YOLOv12n:**
      * [cite\_start]Consistently superior inference speed (sub-10ms frame processing) due to its lightweight architecture (R-ELAN, Area Attention, anchor-free heads)[cite: 103, 104].
      * [cite\_start]Well-suited for edge-based deployment where speed and resource constraints are critical[cite: 105].
      * [cite\_start]Showed slightly better resilience in low-light environments (e.g., Shibuya footage), possibly due to simpler spatial priors[cite: 110].
  * **RF-DETR:**
      * [cite\_start]Slower to train and infer due to its transformer backbone and deformable attention modules[cite: 106].
      * [cite\_start]Delivered higher precision in detecting large and well-separated individuals, capturing long-range dependencies and contextual relationships effectively in dense/occluded scenes[cite: 106, 107].
      * [cite\_start]Outperformed YOLOv12 in AP@large and AR@large metrics[cite: 108].
      * [cite\_start]Performance degraded more in scenes with high contrast or non-standard viewpoints due to positional encoding sensitivity[cite: 111].
      * [cite\_start]Occasionally miscounted overlapping individuals in high-density frames due to box suppression conflicts[cite: 114].
  * **General Observations:**
      * [cite\_start]The single-class detection task (person) reduced complexity, potentially amplifying YOLOv12's speed advantage[cite: 112, 113].
  * **Recommendation:**
      * [cite\_start]YOLOv12 is better for real-time crowd analytics, especially in low-resource environments[cite: 115].
      * [cite\_start]RF-DETR is more appropriate for high-precision offline tasks like post-event analysis or forensic video review[cite: 115].
      * [cite\_start]Choice depends on application requirements: latency vs. accuracy, streaming vs. batch, visual domain complexity[cite: 116].

## Conclusion and Future Work

[cite\_start]This study provides a comprehensive comparison of YOLOv12 and RF-DETR for crowd counting in complex urban environments[cite: 117]. [cite\_start]YOLOv12 is more suitable for real-time deployment due to its speed, while RF-DETR offers higher accuracy for large/well-separated individuals but at a higher computational cost[cite: 118, 119, 120, 121].

**Limitations and Future Work:**

  * [cite\_start]**Dataset Enhancement:** [cite: 122, 123]
      * [cite\_start]Expand dataset quantitatively (more samples, cities, densities)[cite: 124].
      * [cite\_start]Introduce temporal annotations for video-based tracking and flow analysis[cite: 125].
      * [cite\_start]Incorporate more edge cases (overlapping individuals, occlusions, multi-scale objects)[cite: 126].
  * **Demo Development:**
      * [cite\_start]Develop a comparable demo pipeline for RF-DETR to enable fairer qualitative comparison and support offline forensic applications[cite: 128, 129].
  * **Model Exploration:**
      * [cite\_start]Explore hybrid architectures combining YOLOv12's speed with RF-DETR's contextual awareness[cite: 130].
      * [cite\_start]Integrate tracking algorithms (e.g., DeepSORT, ByteTrack) for complete people flow tracking systems[cite: 131].

This work lays a foundation for deeper exploration of CNN vs. transformer detectors in people counting, aiming to bridge the gap between academic models and real-world systems[cite: 132, 133].

## Setup and Usage

*(This section should be filled with specific instructions for your codebase.)*

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
      * [cite\_start]Vu Duc Thanh - Leader (ID: 22071150, Class: FDB 2022C) [cite: 2]
      
  * **Supervisor:** Associate Professor, Ph.D. [cite\_start]Tran Thi Oanh [cite: 2]

## References

*(A full list of references is available in the original report. Key cited works include Chen et al. (2024) for YOLOv12, Carion et al. (2020) for DETR, and Sapkota et al. (2024) for RF-DETR comparisons in other contexts.)*

  * Carion, N., et al. (2020). [cite\_start]End-to-End Object Detection with Transformers. [cite: 135]
  * Chen, Y., et al. (2024). [cite\_start]YOLOv12: Anchor-Free Real-Time Object Detection with Area Attention and R-ELAN. [cite: 137]
  * Sapkota, R., et al. (2024). [cite\_start]RF-DETR Object Detection vs YOLOv12... [cite: 139]
  * Zhu, X., et al. (2020). [cite\_start]Deformable DETR... [cite: 141]
  * Zhang, Y., et al. (2016). [cite\_start]Single-Image Crowd Counting via Multi-Column Convolutional Neural Network. [cite: 142]

