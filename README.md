# 🌱 Crop Disease Detection AI

An AI-powered web application that detects crop diseases from leaf images using deep learning.

## 🎯 What it does
Upload a photo of a Tomato, Potato, or Pepper leaf → AI instantly identifies the disease and confidence score.

## 📊 Results
- **Validation Accuracy:** 92.6%
- **Dataset:** PlantVillage (16,500+ images, 15 disease classes)
- **Model:** ResNet50 with transfer learning and fine-tuning

## 🔗 Live Demo
👉 [Try it here](https://arshtramboo-crop-disease-ai.hf.space)

## 🛠️ Tech Stack
- TensorFlow / Keras
- ResNet50 (Transfer Learning)
- Gradio
- Python
- Google Colab (Training)
- Hugging Face Spaces (Deployment)

## 📈 Training Journey
| Round | Val Accuracy |
|-------|-------------|
| Initial | 18.9% |
| Round 2 | 77% |
| Final | 92.6% |

## 🌿 Supported Crops
- Tomato (10 classes)
- Potato (3 classes)
- Pepper (2 classes)

## ⚠️ Limitations
This model only works for Tomato, Potato, and Pepper leaves. Results for other plant types or non-leaf images are not valid.

## 👨‍💻 Author
**Arsh Altaf Tramboo**
BTech Artificial Intelligence, SKUAST-Kashmir, 2026

## 📄 License
MIT License
