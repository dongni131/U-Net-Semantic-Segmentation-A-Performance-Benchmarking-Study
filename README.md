# Semantic Segmentation with U-Net: Training Strategy Comparison


## 📌 Project Overview

This project studies **semantic segmentation for autonomous driving**, focusing on **pixel-level vehicle segmentation** using the **U-Net architecture**.  
We systematically compare **three U-Net training strategies** to understand their impact on **accuracy, robustness, and training efficiency**.

**Training strategies compared:**
- Baseline U-Net (random initialization)
- U-Net with data augmentation
- U-Net with transfer learning (VGG16 encoder)

Experiments are conducted on the **Carvana Image Masking Challenge** dataset.

---

## 🚗 Motivation

Accurate vehicle segmentation is a critical component of autonomous driving systems.  
While U-Net is known for strong segmentation performance, **training strategy choices can significantly affect model stability and convergence**.

This project aims to answer:
- How stable is a randomly initialized U-Net?
- Does data augmentation improve robustness without sacrificing accuracy?
- Can transfer learning achieve both **higher accuracy and faster convergence**?

---

## 🧠 Methodology

### Dataset
- **Carvana Image Masking Challenge**
- 5,086 high-resolution vehicle images
- Split:
  - Train: 3,560
  - Validation: 763
  - Test: 763

### Models

#### 1. Baseline U-Net
- Standard encoder–decoder U-Net
- Random weight initialization
- Optimizer: Adam (lr = 1e-4)
- Early stopping applied

#### 2. Augmented U-Net
- Same architecture as baseline
- Image augmentations:
  - Horizontal / vertical flips
  - Random rotations
  - Brightness & contrast adjustments

#### 3. Transfer Learning U-Net
- U-Net with **VGG16 pretrained encoder**
- Encoder initialized from ImageNet
- Shallow layers frozen, deeper layers fine-tuned
- Decoder trained from scratch

---

## 📊 Evaluation Metrics

- Dice Coefficient (F1 Score)
- Intersection over Union (IoU)
- Pixel Accuracy
- Binary Cross-Entropy Loss
- Training stability (loss curves)
- Training efficiency (time per step)

---

## 🔍 Key Results

| Model | IoU | Dice | Pixel Accuracy | Training Stability |
|------|-----|------|----------------|--------------------|
| Baseline U-Net | 0.9833 | 0.9916 | 0.9964 | ❌ Loss spikes |
| Augmented U-Net | 0.9746 | 0.9871 | 0.9945 | ⚠️ More stable |
| Transfer Learning U-Net | **0.9872** | **0.9936** | **0.9973** | ✅ Smooth |

### Highlights
- Baseline model achieves high accuracy but suffers from **severe validation loss spikes**
- Data augmentation improves robustness but slightly reduces segmentation precision
- **Transfer learning achieves the best overall performance**
  - Highest Dice & IoU
  - **~26% faster training**
  - Most stable optimization behavior

---

## 🏆 Conclusion

**U-Net with VGG16 transfer learning** provides the best balance of:
- Accuracy
- Robustness
- Training efficiency

For high-precision vehicle segmentation tasks, **transfer learning is the most effective strategy** among the three evaluated approaches.

---

## 🔮 Future Work

- Test on more complex datasets (Cityscapes, BDD100K)
- Explore other pretrained backbones (ResNet, EfficientNet)
- Evaluate robustness under varying weather and lighting
- Deploy model for real-time inference

---

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib

---

## Author
Dongni Li
PinTzu Tseng
Ziliang Song
