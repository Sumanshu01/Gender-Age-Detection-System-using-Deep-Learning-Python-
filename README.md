# 🧠 Gender & Age Detection System using Deep Learning (Python)
A real-time **Gender and Age Detection System** that analyzes a user’s face through a webcam and predicts their **gender** and **age group** using **deep learning and computer vision techniques**.

This project demonstrates the practical application of **OpenCV’s DNN module**, **pre-trained CNN models**, and **real-time video processing**, making it ideal for **college projects, hackathons, and AI portfolios**.

---

## 📌 Features

✅ Real-time face detection via webcam  
✅ Simultaneous **Gender** and **Age group** prediction  
✅ Deep Learning–based inference (CNN)  
✅ No dataset training required  
✅ Lightweight and fast execution  
✅ Beginner-friendly yet industry-oriented  
## 🧩 Tech Stack
| Component | Technology |
|---------|------------|
| Programming Language | Python 3.x |
| Computer Vision | OpenCV |
| Deep Learning Framework | Caffe (via OpenCV DNN) |
| Numerical Computing | NumPy |
| Input Source | Live Webcam |

---
## 🏗️ System Architecture
<img width="2906" height="3711" alt="image" src="https://github.com/user-attachments/assets/b7fb8882-f849-4882-ae08-dfc043efe6df" />

## 🔢 Age Categories

The system predicts **age ranges**, not exact age values, which is a standard practice in age estimation systems:
(0–2), (4–6), (8–12), (15–20), (25–32), (38–43), (48–53), (60–100)

## 🧠 Technical Details

### Face Detection
- Uses OpenCV's DNN-based face detector
- More accurate than traditional Haar Cascades
- Optimized for real-time performance

### Prediction Pipeline
1. Detected face resized to **227×227 pixels**
2. Mean normalization applied
3. CNN inference produces:
   - **Gender:** Male or Female
   - **Age:** One of 8 predefined ranges

### Factors Affecting Accuracy
- Lighting conditions
- Facial occlusions (glasses, masks)
- Camera quality
- Demographic representation in training data

---


## 👨‍💻 About Me

**Sumanshu Jindal**  
🎓 Computer Science Engineer  

I am deeply passionate about **Artificial Intelligence**, **Machine Learning**, **Computer Vision**, and **Software Development**.  
My focus lies in designing and building **intelligent systems** that solve real-world problems and create meaningful impact through technology.

🚀 Always learning. Always building.
