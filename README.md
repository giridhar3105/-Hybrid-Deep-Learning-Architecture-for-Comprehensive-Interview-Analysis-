# **Hybrid Deep Learning Architecture for Comprehensive Interview Analysis**  
**Integrating Emotion Recognition, Gaze Estimation, and Human Pose Detection**

---

## **Overview**  
This project introduces a hybrid deep learning system for real-time online interview analysis. By integrating emotion recognition, gaze estimation, and body pose detection, the system captures critical non-verbal cues to provide actionable insights into candidate confidence, stress levels, and attentiveness.  

---

## **Key Features**  
- **Emotion Recognition**:  
  Utilizes VGG19 and CNN Model to classify facial expressions into seven categories, including anger, happiness, and sadness.  

- **Gaze Estimation**:  
  Tracks gaze direction and attentiveness using Haar cascades and MediaPipe.  

- **Pose Detection**:  
  Analyzes body posture through MediaPipe to infer candidate confidence or stress.  

- **Ensemble Learning**:  
  Combines outputs from all three modalities for robust and adaptable predictions.  

- **Real-Time Analysis**:  
  Processes video data dynamically, with a latency of 90ms per frame.  

---

## **Datasets Used**  
The models are trained on the following datasets:  
1. **FER2013**: A comprehensive dataset for facial emotion classification.  
2. **CK+ (Extended Cohn-Kanade)**: Provides posed and spontaneous facial expressions.  
3. **MPII Human Pose Dataset**: Annotated images for human pose estimation.  
4. **Haar Cascade Classifiers**: Pre-trained models for face and eye detection.  

---

## **System Architecture**  
The hybrid system leverages advanced deep learning frameworks:  
- **VGG19 and CNN Model** for emotion recognition.  
- **MediaPipe** for gaze tracking and pose estimation.  
- **Ensemble Learning** to integrate modality outputs and enhance performance.

---

## **Performance Metrics**  
- **Accuracy**: 82.94% across all modalities.  
- **Latency**: 90ms per frame for real-time processing.  
- **Metrics**: Precision, recall, and F1-score validate model robustness.  

---

## **Applications**  
This system has implications beyond recruitment, including:  
- **Education**: Analyze student engagement in virtual classrooms.  
- **Healthcare**: Monitor emotional and physical well-being.  
- **Customer Service**: Enhance user interactions through behavior analysis.  

---

## **Future Enhancements**  
- Incorporating **speech emotion analysis** for richer insights.  
- Expanding datasets for improved generalization across demographics.  
- Ensuring privacy compliance for ethical deployment.  

---

## **Contact**  
**Giridhar Chennuru**   
Email: [giridhar.Chennuru3105@gmail.com](mailto:giridhar.Chennuru3105@gmail.com)

---
