# Thai Handwritten OCR with Vision Transformer

A deep learning project for recognizing Thai handwritten tone marks using Vision Transformer (ViT) architecture. This model can classify Thai tone marks including เอก (Ek), โท (Tho), ตรี (Tri), and จัตวา (Chattawa).

## 🎯 Project Overview

This project implements an Optical Character Recognition (OCR) system specifically designed for Thai handwritten tone marks using state-of-the-art Transformer architecture. The model leverages Vision Transformer (ViT) from Hugging Face to achieve high accuracy in classifying Thai tone marks.

## 📊 Dataset

The dataset contains handwritten Thai tone mark images organized into 4 classes:
- **เอก01.png** - First tone mark (Mai Ek)
- **โท02.png** - Second tone mark (Mai Tho) 
- **ตรี03.png** - Third tone mark (Mai Tri)
- **จัตวา04.png** - Fourth tone mark (Mai Chattawa)

**Dataset Statistics:**
- Total samples: 1,773 images
- Training set: 80% (1,418 images)
- Test set: 20% (355 images)
- Image size: 224x224 pixels
- Format: RGB

## 🏗️ Model Architecture

### Vision Transformer (ViT)
- **Base Model**: `google/vit-base-patch16-224-in21k`
- **Patch Size**: 16x16
- **Input Resolution**: 224x224
- **Number of Classes**: 4 (Thai tone marks)
- **Fine-tuning**: Classifier layer adapted for Thai tone mark classification

### Model Components
```
ViT Base Model
├── Patch Embedding (16x16 patches)
├── Transformer Encoder (12 layers)
│   ├── Multi-Head Self-Attention
│   ├── MLP Feed Forward
│   └── Layer Normalization
└── Classification Head (4 classes)
```

## 🚀 Getting Started

### Prerequisites
This project runs entirely in Google Colab - no local installation required!

### Google Colab Setup

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/thai-handwritten-ocr/blob/main/ThaiHandWrite_Transformer_Based_OCR_tones.ipynb)

#### Quick Start in Colab
1. **Open the notebook** in Google Colab using the badge above
2. **Mount Google Drive** to access your dataset:
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. **Install required packages** (already included in notebook):
```python
!pip install transformers torch torchvision scikit-learn pandas tqdm
```

4. **Run all cells** to train the model automatically

#### Dataset Setup in Google Drive
```
Google Drive/
└── MyDrive/
    └── characters_dataset/
        └── tones/
            ├── เอก01.png/
            ├── โท02.png/
            ├── ตรี03.png/
            └── จัตวา04.png/
```

#### Training in Colab
The notebook automatically handles:
- Data loading and preprocessing
- Model training with ViT
- Performance evaluation
- Results visualization
- Model checkpointing

## 📈 Performance

### Training Results
- **Training Accuracy**: 90.13% (Epoch 1)
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam (lr=1e-4)
- **Batch Size**: 32

### Model Metrics
| Metric | Score |
|--------|-------|
| Accuracy | 90.13% |
| Precision | TBD |
| Recall | TBD |
| F1-Score | TBD |

## 📁 Project Structure

```
thai-handwritten-ocr/
├── ThaiHandWrite_Transformer_Based_OCR_tones.ipynb  # Main Colab notebook
├── data/                                            # Dataset folder (in Google Drive)
│   └── characters_dataset/
│       └── tones/
│           ├── เอก01.png/                           # 462 images
│           ├── โท02.png/                            # 439 images  
│           ├── ตรี03.png/                           # 463 images
│           └── จัตวา04.png/                         # 319 images
├── models/                                          # Generated in Colab
│   └── vit_model_checkpoint_tones.pth              # Trained model weights
├── results/                                         # Generated outputs
│   ├── classification_report_epoch_5.xlsx          # Training metrics
│   ├── test_classification_report.xlsx             # Test results
│   ├── loss_per_epoch.png                          # Loss visualization
│   └── accuracy_per_epoch.png                      # Accuracy chart
└── README.md                                        # This file
```

## 🔧 Implementation Details

### Data Preprocessing
```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

### Training Configuration
- **Epochs**: 5
- **Learning Rate**: 1e-4
- **Batch Size**: 32
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Device**: CUDA (if available)

## 📊 Results Visualization

The project generates several visualization outputs:
- Training loss per epoch
- Training accuracy per epoch
- Classification reports (Excel format)
- Confusion matrix

## 🛠️ Technical Features

- **Google Colab Ready**: No local setup required - runs entirely in the cloud
- **Vision Transformer Architecture**: State-of-the-art computer vision model
- **Transfer Learning**: Fine-tuned from pre-trained ViT model
- **GPU Acceleration**: Automatic CUDA support in Colab
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Model Checkpointing**: Automatic save and resume training
- **Google Drive Integration**: Seamless dataset and model storage

## 🎓 Educational Value

This project demonstrates:
- Implementation of Vision Transformers for image classification
- Thai language processing challenges
- Transfer learning techniques
- OCR system development
- Deep learning best practices

## 🤝 Contributing

Contributions are welcome! To contribute to this Colab-based project:

1. Fork the repository
2. Open the notebook in Colab
3. Make your improvements to the notebook
4. Test your changes thoroughly
5. Submit a Pull Request with:
   - Clear description of changes
   - Screenshots of results (if applicable)
   - Updated performance metrics

### Development Guidelines
- **Notebook Structure**: Keep cells organized and well-commented
- **Output Management**: Clear outputs before committing to avoid large files
- **Documentation**: Update markdown cells to explain new features
- **Testing**: Ensure all cells run successfully in a fresh Colab environment



