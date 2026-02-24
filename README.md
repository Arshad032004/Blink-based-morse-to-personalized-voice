# Blink-to-Speech Morse Code with Voice Cloning

An AI-based assistive communication system that allows users to type using eye blinks (Morse code) and convert the text into natural speech using voice cloning.

This project is designed to help individuals with limited mobility communicate effectively using only eye movements.

---

## 🚀 Features

- Real-time eye blink detection using webcam
- Morse code input using blink patterns
- Word prediction using NLTK dictionary
- Sentence formation and editing
- Custom voice cloning using Tortoise TTS
- Audio generation and playback
- GPU (CUDA) support for faster performance
- User-friendly PyQt5 graphical interface

---

## 🧠 System Modules

### 1. Blink-to-Speech Application
- Detects eye blinks using MediaPipe Face Mesh
- Converts blink duration into Morse code
- Decodes Morse into text
- Provides word suggestions
- Allows backspace and text control using eye gestures

### 2. Voice Cloning
- Uses Tortoise TTS for realistic speech generation
- Supports multiple voice samples
- Generates high-quality audio from typed text

### 3. Voice Test Script
- Standalone script for testing voice cloning
- Loads sample audio files
- Generates `output.wav`

---

## 🛠️ Technologies Used

- Python
- OpenCV
- MediaPipe
- PyQt5
- NumPy
- NLTK
- PyTorch
- Torchaudio
- Tortoise TTS

---

## 📦 Installation

### Clone the repository

```bash
git clone https://github.com/your-username/blink-to-speech.git
cd blink-to-speech
