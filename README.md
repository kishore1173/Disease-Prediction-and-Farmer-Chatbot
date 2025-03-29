Crop Disease Detection and Assistance System
Overview
This project provides a comprehensive solution for detecting plant diseases and offering farming assistance. It combines a YOLO-based disease detection model with an LLM-powered chatbot (Aksara V1) to guide farmers with actionable insights on disease management, fertilizers, and traditional remedies.
Features
	• Plant Disease Detection: Uses a YOLO-based model to detect diseases in cassava, sugarcane, mango, wheat, and rice.
	• Disease Severity Estimation: Estimates the severity of detected diseases to suggest appropriate treatments.
	• Farming Assistant Chatbot (Aksara V1): An LLM-based assistant trained on South Indian farming practices, providing:
		○ Fertilizer and pesticide recommendations.
		○ Traditional remedies for plant diseases.
		○ Answers to common farming queries.
Dataset
We use a dataset containing labeled images of plant diseases and their severity levels. The dataset is preprocessed and augmented for better model generalization.
	• Mango Leaf Disease Dataset
	• Cassava Leaf Disease Classification
	• Rice Leaf Diseases Detection
	• Sugarcane Plant Diseases Dataset
	• Potato Disease Leaf Dataset
	
 Preprocessing & Dataset Preparation
 Data Processing
	• Merges all individual datasets into a single dataset.
	• Renames the label "Healthy" to unique class names to avoid conflicts.
	• Splits data into training, validation, and testing sets (80% Train, 10% Validation, 10% Test).
	• Resizes images to 448×448 pixels.
	• Normalizes images for YOLOv8 training.

Model Training
	1. YOLO-based Plant Disease Detection
		○ The YOLO model is trained on the dataset to classify plant diseases.
		○ Outputs disease labels and severity levels.
	2. Aksara V1 Chatbot Integration
		○ The chatbot is fine-tuned to answer crop-related queries.
		○ Uses the detected disease label to provide treatment suggestions.
Disease Severity Estimation
	• Severity is estimated by calculating the infected area ratio.
	• Converts images to grayscale and applies thresholding to segment diseased regions.
	• Computes severity percentage and categorizes into:
		○ Mild (<20%)
		○ Moderate (20%-50%)
		○ Severe (>50%)

Deployment
	• The YOLO model runs on a web-based interface for real-time disease detection.
	• The Aksara V1 chatbot is integrated into the platform to assist farmers with disease management solutions.
How to Use
	1. Upload an Image: Farmers can upload an image of their crop.
	2. Get Disease Prediction: The YOLO model predicts the disease and severity.
	3. Ask Aksara V1: The chatbot provides tailored advice based on the disease detected.
	4. Receive Treatment Suggestions: The chatbot suggests fertilizers, pesticides, and traditional remedies.
Future Enhancements
	• Improve chatbot accuracy using reinforcement learning.
	• Expand the dataset to include more regional crop diseases.
	• Optimize the YOLO model for better precision and recall.
Acknowledgments
	• Dataset: Publicly available plant disease datasets
	• Model: YOLO for detection, Aksara V1 for assistance
License
This project is licensed under the MIT License.

Developed by Kishore M 
![image](https://github.com/user-attachments/assets/1a000eba-d293-4ea1-814a-701a9bd5707c)
