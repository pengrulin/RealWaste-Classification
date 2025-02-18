# Automated Waste Sorting System Using Deep Learning: RealWaste Project

## Overview
This project aims to develop an automated waste sorting system using deep learning techniques to enhance waste management and recycling efficiency. By leveraging advanced image classification methods—including a custom-built Convolutional Neural Network (CNN) and a hybrid approach combining a pre-trained ResNet50 model with K-Nearest Neighbors (KNN)—the system classifies waste images into nine distinct categories: Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, Paper, Plastic, Textile Trash, and Vegetation.

The project addresses the inefficiencies of traditional manual waste sorting by automating the classification process. This not only reduces labor and error rates but also promotes sustainability and a greener future by optimizing recycling efforts.

## Project Structure

### Part 1: Problem Statement & Motivation
- **Objective:**  
  Develop an automated waste sorting system using deep learning to accurately classify waste images and improve waste management processes.
- **Motivation:**  
  Manual waste sorting is labor-intensive, time-consuming, and error-prone. By automating this process with deep learning, the system can help reduce environmental impact and enhance recycling efficiency.

### Part 2: Dataset
- **Source:**  
  The RealWaste dataset is obtained from the UCI Machine Learning Repository (Wollongong City Council) and contains 4,752 colored waste images (524x524 pixels) across 9 classes.
- **Dataset Details:**  
  The classes include: Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, Paper, Plastic, Textile Trash, and Vegetation.
- **Handling Complexity:**  
  Due to the dataset's large size and class imbalances, the project utilized Boston University’s SCC for downloading and unzipping files, and a shared Google Folder for backup. Class imbalance was addressed by assigning higher weights to underrepresented classes during training.

### Part 3: Analysis Methodology
- **K-Nearest Neighbors (KNN):**
  - **Approach:**  
    Flattened the image data to one-dimensional arrays and used a KNN model with 3 neighbors as a baseline.
  - **Results:**  
    Achieved an overall accuracy of 22.2%, with significant discrepancies across classes (e.g., high recall but low precision for Class 2).
  
- **Convolutional Neural Network (CNN):**
  - **Architecture:**  
    Built from scratch using Conv2D layers for feature extraction, MaxPool2D for dimensionality reduction, and ReLU activations for non-linearity. Weighted Cross Entropy Loss was used to mitigate class imbalance.
  - **Results:**  
    Achieved a test accuracy of 54.5%, a significant improvement over random guessing, demonstrating the CNN's capability under limited computational resources.
  
- **ResNet50 Hybrid Approach:**
  - **Method:**  
    Utilized the pre-trained ResNet50 model to extract high-level image features, which were then classified using a KNN classifier.
  - **Results:**  
    This hybrid approach achieved an accuracy of 85.6%, indicating its effectiveness in handling the complex features present in the dataset.

### Part 4: Real World Applications
- **Waste Management Facilities:**  
  Integrate the system into sorting mechanisms to automate waste separation, reduce manual labor, and increase recycling rates.
- **Public Use:**  
  Deploy the technology in smart waste bins (e.g., in streets, malls, and public spaces) to sort waste on demand, thereby minimizing mis-sorted waste and reducing landfill accumulation.

### Part 5: Limitations and Challenges
- **Computational Resources:**  
  Limited GPU availability constrained the depth and complexity of the CNN model, necessitating simpler architectures.
- **Dataset Size:**  
  With under 5,000 images, the dataset is relatively small, which may hinder the model's ability to generalize to new, unseen data.
- **Class Imbalance:**  
  Significant disparities in class counts required the use of weighted loss functions to ensure fair model training across all categories.

## Collaborators
- Audrey Seller
- Jaishankar Govindaraj
