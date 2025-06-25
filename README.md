# 🎭 AI Powered Facial Emotion Recognition Web App

A deep learning-based **Facial Emotion Detection Web App** that can identify human emotions in two ways:

- 🎥 Real-time webcam feed  
- 🖼️ Image upload  

Built using **Python**, **Flask**, **TensorFlow**, **OpenCV**, and **HTML/CSS**.

---

## 🚀 Features

- ✅ Real-time webcam emotion detection
- ✅ Image upload and emotion classification
- ✅ Trained on FER2013 dataset using CNN
- ✅ Smooth UI with dark blue gradient theme and animated effects
- ✅ Responsive and mobile-friendly design

---

## 🏗️ Project Folder Structure

Facial_Emotion_Recognition/
├── app/
│ ├── app.py
│ ├── static/
│ │ ├── css/
│ │ │ └── style.css
│ │ ├── images/
│ │ │ └── landing_banner.png
│ │ └── uploads/
│ └── templates/
│ ├── index.html
│ ├── webcam.html
│ └── upload.html
├── saved_models/
│ └── emotion_model.h5
├── screenshots/
│ ├── landing.png
│ ├── webcam.png
│ └── upload.png
├── requirements.txt
├── README.md
└── .gitignore


---


## ⚙️ Installation & Running Locally

### 1. Clone the repository 👇
```bash
git clone https://github.com/YourUsername/Facial_Emotion_Recognition.git
cd Facial_Emotion_Recognition
```
### 2. Create Virtual Environment (Recommended) 👇
```bash
python -m venv venv
venv\Scripts\activate  # For Windows
# OR
source venv/bin/activate  # For Mac/Linux
```
### 3. Install Dependencies 👇
```bash
pip install -r requirements.txt
```
### 4. Run the Flask App 👇
```bash
Copy code
python app/app.py
```
### 5. Open in Browser 👇
```cpp
Copy code
http://127.0.0.1:5000/
```
### 🧠 AI Model
- Trained using Convolutional Neural Networks (CNN)

- Dataset: FER2013 - Facial Expression Recognition 2013

- Saved model path: saved_models/emotion_model.h5

### 💻 Tech Stack
- Python 3.x

- Flask

- TensorFlow / Keras

- OpenCV

- HTML, CSS (with AOS animations, Google Fonts)

- JavaScript (for AJAX updates)

### 🎨 UI Theme Details
- Theme: Dark Blue → Light Blue gradients

#### CSS Features:
- ✅ Button animations
- ✅ Card hover effects
- ✅ Banner image hover scaling
- ✅ Text shadows for better readability

### 📽️ Future Improvements (Optional for you)
- ✅ Add emotion probability scores

- ✅ Deploy on cloud (Heroku / AWS / GCP)

- ✅ Add face detection bounding box in upload image predictions

### 📝 License
- This project is licensed under the MIT License.