# YOLOv8 Plant Disease Detection

## 📌 Project Overview
This project implements a deep learning-based plant disease detection system using the YOLOv8 model. It leverages multiple datasets from Kaggle containing images of diseased and healthy leaves for crops such as mango, cassava, rice, sugarcane, and potato. The trained model is uploaded to Hugging Face for easy accessibility and testing.

## 📂 Dataset
The dataset is sourced from Kaggle and includes images of plant leaves with various diseases. The following datasets were used:
- **Mango Leaf Disease Dataset**
- **Cassava Leaf Disease Classification**
- **Rice Leaf Diseases Detection**
- **Sugarcane Plant Diseases Dataset**
- **Potato Disease Leaf Dataset**

### 📥 Loading Dataset
Each dataset is loaded and processed into a structured format using the `df_maker` function, which extracts file paths and labels for each image. 

## 🛠 Preprocessing & Dataset Preparation
### 🔄 Data Processing
- Merges all individual datasets into a single dataset.
- Renames the label "Healthy" to unique class names to avoid conflicts.
- Splits data into training, validation, and testing sets (80% Train, 10% Validation, 10% Test).
- Resizes images to **448×448** pixels.
- Normalizes images for YOLOv8 training.

## 🔍 YOLOv8 Model Training
### 🔨 Model Training
- Uses **YOLOv8** for classification and disease detection.
- Training hyperparameters:
  - **Epochs**: 15
  - **Image Size**: 448×448
  - **Batch Size**: 32
  - **Optimizer**: AdamW
  - **Learning Rate**: 0.002
  - **Early Stopping**: Enabled (Patience = 3)
  - **GPU Acceleration**: Enabled
- The trained model is saved as `yolov8_plant_disease.pt`.

## 🚀 Model Deployment & Hugging Face Integration
### 📤 Uploading Model to Hugging Face
- Logs into Hugging Face using authentication tokens.
- Uploads the trained model (`yolov8_plant_disease.pt`) to the Hugging Face Hub.
- Repository: [YOLOv8-5_Plant-Diseases](https://huggingface.co/Lucario-K17/YOLOv8-5_Plant-Diseases)

### 📥 Loading & Testing Model from Hugging Face
- Downloads the trained model from Hugging Face.
- Runs inference on a test image to predict plant diseases.
- Displays the detected class along with confidence scores.

## 📊 Disease Severity Estimation
- **Severity is estimated by calculating the infected area ratio.**
- Converts images to grayscale and applies thresholding to segment diseased regions.
- Computes severity percentage and categorizes into:
  - **Mild (<20%)**
  - **Moderate (20%-50%)**
  - **Severe (>50%)**

## 🔬 Disease Information Retrieval
- Uses a dataset (`Plant_disease_prevent.csv`) to fetch textual information on detected plant diseases.
- Searches for the predicted disease in the dataset and returns relevant details.

## 📦 Object Detection using EfficientDet
- Implements **EfficientDet** for further plant disease segmentation.
- Loads TensorFlow Hub's EfficientDet model.
- Visualizes object detection results using bounding boxes.

## 🔧 Installation & Usage
### 📌 Requirements
Install dependencies using:
```bash
pip install ultralytics opencv-python pandas matplotlib tensorflow tensorflow-hub huggingface_hub
```

### 🔄 Training the Model
Run the training script:
```python
from ultralytics import YOLO
model = YOLO("yolov8s.pt")
model.train(data="/kaggle/working/YOLOv8_Dataset", epochs=15, imgsz=448, batch=32)
```

### 📥 Loading Model & Running Inference
```python
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="Lucario-K17/YOLOv8-5_Plant-Diseases", filename="yolov8_plant_disease.pt")
model = YOLO(model_path)
```

### 🖼 Testing with an Image
```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("test_image.jpg")
results = model(img)
plt.imshow(img)
plt.title(f"Predicted: {results[0].probs.top1}")
plt.show()
```

## 📌 Conclusion
This project successfully develops a deep learning model for detecting plant diseases using YOLOv8 and integrates it with Hugging Face for easy deployment. It also includes disease severity estimation and additional segmentation techniques using EfficientDet.

## 📜 License
This project is licensed under the MIT License.

## ✨ Acknowledgments
- Kaggle for the dataset
- Ultralytics for YOLOv8
- TensorFlow Hub for EfficientDet
- Hugging Face for model hosting

---

📌 **Developed by Kishore M** 🚀
