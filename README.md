Thai Handwritten Character Recognition with Vision Transformer
This project aims to build an Optical Character Recognition (OCR) system for Thai handwritten characters using the Vision Transformer (ViT) model.

Project Steps
Data Preparation: Convert Thai handwritten character images into a Tensor format suitable for the Transformer model and split the data into Train, Validation, and Test sets.
Model Creation: Utilize a Vision Transformer (ViT) or Swin Transformer model, potentially leveraging libraries like Hugging Face Transformers.
Model Training: Define a Loss Function and Optimizer to fine-tune the model.
Evaluation: Evaluate the model's performance using metrics such as Accuracy, Precision, Recall, and F1-score.
Explanation
Data Preparation: Images are transformed into a suitable Tensor format for the Transformer model using a ViTFeatureExtractor (though the current code uses torchvision.transforms).
Model Creation: A pretrained Vision Transformer (e.g., google/vit-base-patch16-224) is used and adapted to the number of Thai character classes.
Model Training: The model is trained using an Optimizer (AdamW in the plan, Adam in the code) and a Loss Function (CrossEntropyLoss), with evaluation on the Validation set (not explicitly implemented in the provided code, but included in the plan).
Evaluation: classification_report is used to measure metrics like Precision, Recall, and F1-score.
Getting Started
Prerequisites
Python 3.6+
PyTorch
Transformers library (Hugging Face)
Scikit-learn
Pillow
tqdm
pandas
matplotlib
You can install the required packages using pip:
