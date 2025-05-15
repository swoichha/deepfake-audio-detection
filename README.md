# DeepFake Voice Detection using Machine Learning
### By: Swoichha Adhikari

This project focuses on detecting deepfake (synthetically generated) audio using machine learning techniques, including both classical ML algorithms and deep learning models (CNN). The dataset used for this project is available on [Kaggle: Deep Voice - DeepFake Voice Recognition](https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition).

---

## 📁 Dataset

* **Source**: [Kaggle DeepFake Voice Dataset](https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition)
* **Classes**: `REAL`, `FAKE`
* **Audio Format**: `.wav` or `.mp3`
* **Samples**:

  * Real: 8
  * Fake: 56

---

## 🧪 Features

* Extracted features:

  * **Mel Spectrograms** for CNN models
  * **MFCC (Mel Frequency Cepstral Coefficients)** for traditional ML models

---

## 🧰 Tools & Libraries

* **Python**
* **TensorFlow / Keras**
* **scikit-learn**
* **Librosa** (for audio processing)
* **Matplotlib & Seaborn** (for visualization)

---

## 🧱 Models Used

### 🎛️ Traditional ML

* **SVM (Support Vector Machine)**
* **Random Forest**

### 🤖 Deep Learning

* **Baseline CNN**
* **Enhanced CNN** with batch normalization, dropout, and multiple convolutional layers

---

## 🛠️ Preprocessing

* Converted audio to Mel Spectrograms and MFCCs
* Normalized and padded sequences
* Balanced dataset by oversampling real samples

---

## 📊 Performance

### CNN Results:

* **Accuracy**: \~87%
* **ROC-AUC**: \~0.93

### Enhanced CNN:

* **Best Accuracy**: \~98.5%
* **F1 Score**: \~0.98
* Trained using callbacks: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`

### SVM & Random Forest:

* **SVM Accuracy**: \~84%
* **Random Forest Accuracy**: \~87%

---

## 📈 Visualizations

* Audio Waveform plots
* Spectrograms and Mel Spectrograms
* Confusion Matrices
* ROC Curves

---

## 🚀 How to Run

1. Clone the repository.
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition) and extract it into the `KAGGLE/AUDIO` directory.
3. Run the notebook or Python scripts:

   ```bash
   python deepfake_voice_detection.py
   ```

---

## 🧠 Future Work

* Integration with real-time streaming audio
* Fine-tuning using transformer-based models (e.g., Wav2Vec, Whisper)
* Model explainability tools like SHAP or LIME

---

