Brain Tumor Classification from MRI Images using Deep Learning Models
Domain: Computer Vision, Machine Learning
Sub-Domain: Deep Learning, Medical Image Analysis
Techniques: Deep Convolutional Neural Networks, Transfer Learning, XceptionNet
Applications: Image Classification, Medical Diagnosis, MRI Image Recognition

Project Highlights
Developed a deep learning-based system to classify brain tumors using MRI scans with a focus on early and accurate detection.

Utilized a pre-trained Xception model, fine-tuned for four-class classification: Glioma, Meningioma, Pituitary, and No Tumor (Healthy).

Performed image preprocessing, data augmentation, and implemented early stopping to prevent overfitting.

A total of 7023 images were used, split into 5712 training and 1311 testing samples, collected from Kaggle and scanning centers.

Built and deployed a web-based interface using Flask, allowing users to upload MRI scans and receive real-time classification.

Dataset Details
Dataset Name: Brain Tumor MRI Dataset (Glioma vs Meningioma vs Pituitary vs Normal)

Number of Classes: 4

Total Images: 7023

Training Images: 5712

Testing Images: 1311

Image Size: 224×224 pixels (Rescaled during preprocessing)


Class	Training Images	Testing Images
Glioma	1321	300
Meningioma	1339	306
Pituitary	1457	300
No Tumor	1595	405
Model Comparison & Results
Out of multiple architectures tested (ResNet-50, VGG16, NASNetMobile, InceptionV3, Xception), the Xception model achieved the best overall performance:


Model	Accuracy	AUC	F1-Score	Recall	Loss
Xception	89.70%	98.14%	89.30%	89.36%	0.29
Inception V3	86.80%	97.60%	86.41%	86.40%	0.33
NASNetMobile	86.50%	97.33%	85.43%	86.26%	0.40
VGG16	85.66%	96.50%	85.23%	85.50%	0.42
ResNet-50	67.96%	87.36%	65.54%	66.54%	0.87
🔍 Note: Though initially considered, ResNet-50 yielded the lowest accuracy due to misclassification issues. Hence, XceptionNet was selected as the final model.

Conclusion
This project presents a robust deep learning framework for brain tumor classification using MRI images. By leveraging Xception, a depthwise separable convolution model, we achieved high accuracy and minimized model loss. The implementation not only outperforms traditional architectures like ResNet-50 and VGG16 but also bridges the gap between clinical diagnosis and AI-assisted tools through a Flask-based web interface.
