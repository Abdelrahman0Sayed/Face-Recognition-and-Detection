# Face-Recognition-and-Detection

## Overview

The project implements a face detection system using Haar Cascades and a face recognition system using the ORL dataset with PCA and SVM, including confidence thresholding and ROC curve analysis for performance evaluation.

## Face Detection

This method uses classical Haar Cascades for fast and effective face detection without deep learning. It converts an input image to grayscale, detects faces with a pre-trained model, draws rectangles around them, and notifies completion. 

![Face Detection](https://github.com/user-attachments/assets/263ac9ff-b051-4218-9317-f388d2fefff5)

## Face Recognition

The system uses the ORL dataset (40 individuals, 10 images each) with images resized to 112x92 pixels. After normalization, PCA reduces dimensionality, and an SVM with an RBF kernel is trained. Evaluation includes accuracy, confusion matrix, classification report, and ROC-AUC scores. 

![Face Recognition 1](https://github.com/user-attachments/assets/15025e14-570e-493d-b104-4c8daa7244b8)
![Face Recognition 2](https://github.com/user-attachments/assets/98c782a8-70d8-4a92-a69c-1e9ff8ae75b6)

## Confidence Threshold

A confidence score determines if a detected face matches a known subject or is labeled "Unknown." If the score falls below the threshold, it’s classified as "Unknown," balancing accuracy and robustness. The threshold is adjustable. ![Confidence Threshold](https://github.com/user-attachments/assets/01f88426-5d7e-4bb9-8283-2d390dcee99c)

## PCA

Principal Component Analysis reduces the dimensionality of face images, focusing on key features while discarding noise. This speeds up training and prediction, improving generalization for the SVM classifier. ![PCA](https://github.com/user-attachments/assets/26cd8fe5-9de7-4663-8104-3dbf1bd16716)

## ROC Curve

The ROC curve analysis uses a One-vs-Rest approach for multi-class classification. It binarizes true labels, computes decision function values, and plots TPR vs. FPR for each class with AUC scores, including a random guessing baseline for comparison. ![ROC Curve](https://github.com/user-attachments/assets/23dec072-d9e5-4645-941a-e5bbf2ccae3b)

## Getting Started

1. Clone the repository: `https://github.com/Abdelrahman0Sayed/Face-Recognition-and-Detection.git`
2. Install dependencies (e.g., OpenCV, scikit-learn, NumPy).
3. Run the application to explore face detection and recognition.

## Contributors

<table>
  <tr>
            <td align="center">
      <a href="https://github.com/Abdelrahman0Sayed">
        <img src="https://avatars.githubusercontent.com/u/113141265?v=4" width="250px;" alt="Abdelrahman Sayed"/>
        <br />
        <sub><b>Abdelrahman Sayed</b></sub>
      </a>
    </td>
        <td align="center">
      <a href="https://github.com/salahmohamed03">
        <img src="https://avatars.githubusercontent.com/u/93553073?v=4" width="250px;" alt="Salah Mohamed"/>
        <br />
        <sub><b>Salah Mohamed</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Ayatullah-ahmed" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/125223938?v=" width="250px;" alt="Ayatullah Ahmed"/>
        <br />
        <sub><b>Ayatullah Ahmed</b></sub>
      </a>
    </td>
        </td>
        <td align="center">
      <a href="https://github.com/AhmeedRaafatt">
        <img src="https://avatars.githubusercontent.com/u/125607744?v=4" width="250px;" alt="Ahmed Raffat"/>
        <br />
        <sub><b>Ahmed Rafaat</b></sub>
      </a>
    </td>
  </tr>
</table>
