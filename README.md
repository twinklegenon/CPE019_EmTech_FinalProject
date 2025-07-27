# 🧠 CPE019 EmTech Final Project — Rock-Paper-Scissors Classifier

Live App 👉 [Click here to view the Streamlit app](https://twinklegenon-cpe019-emtech-finalproject-app-k0b820.streamlit.app/)

## 👩‍💻 Authors
- **Twinkle S. Genon**  
- **Christian Ivan P. Murao**  
📚 *CPE019 - CPE32S3 | Emerging Technologies 2 in CpE*

---

## 📌 About the Project

This project is a **Rock-Paper-Scissors Classifier** that uses a deep learning model to identify hand gestures from images. It was developed using **TensorFlow** and deployed with **Streamlit**.

Upload an image showing a hand gesture of **rock**, **paper**, or **scissors**, and the model will classify it accordingly.

---

## 🚀 How to Run Locally

### 🧱 Prerequisites
Make sure Python is installed. You also need to install required Python and system packages:

### 1. Clone the repository:
```bash
git clone https://github.com/twinklegenon/cpe019_emtech_finalproject.git
cd cpe019_emtech_finalproject

### 2. Install Python dependencies:
```bash
pip install -r requirements.txt

### 3. Install system dependencies (for image processing):
```bash
bash
Copy
Edit
sudo apt-get install libgl1

### 4. Run the Streamlit app:
```bash
bash
Copy
Edit
streamlit run app.py

twinklegenon-cpe019_emtech_finalproject/
├── README.md              # Project documentation
├── app.py                 # Streamlit app source code
├── packages.txt           # System package list
├── requirements.txt       # Python dependency list
└── rps_classifier.h5      # Pre-trained Keras model (Git LFS)


📦 Requirements
From requirements.txt:

streamlit

tensorflow

opencv-python-headless

From packages.txt:

libgl1 (required for image processing)

⚙️ Model Information
Model file: rps_classifier.h5

Managed using Git Large File Storage (LFS)
Ensure you have Git LFS installed to clone the model.


📝 Usage Notes
📷 Image input: Upload a .jpg or .png image of your hand.

✌️ Scissors pose guideline:
The scissors pose hand gesture is composed of the thumb, index finger (point finger), and middle finger, with the thumb extended away from the hand and the index and middle fingers kept straight and parallel.

✅ Best results are achieved when the gesture is clearly visible with minimal background clutter.


🎓 Acknowledgement
This project was developed as part of the final requirement for the course:

CPE019 - Emerging Technologies 2 in Computer Engineering
