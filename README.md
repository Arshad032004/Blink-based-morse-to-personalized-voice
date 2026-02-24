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

Install dependencies
pip install torch torchaudio
pip install opencv-python mediapipe pyqt5 numpy nltk
pip install tortoise-tts

NLTK word dataset will be downloaded automatically on first run.

📁 Project Structure
project/
│
├── blink_to_speech.py        # Main application
├── blink_to_speech_v2.py     # Version with clear text feature
├── voice_clone_test.py       # Standalone voice cloning script
│
├── samples/
│   ├── 1.wav
│   ├── 2.wav
│   └── 3.wav
│
└── README.md
▶️ Usage

Run the main application:

python blink_to_speech.py
Steps

Click Start Blink Detection

Load 3 voice samples

Blink to type Morse code

Long blink (2 seconds) to generate speech

Play the generated audio

👁️ Eye Control Instructions
Action	Function
Short blink	Dot (.)
Medium blink	Dash (-)
Long blink (2 sec)	Speak text
Left eye long close	Backspace
Left wink	Move up suggestions
Right wink	Move down suggestions
No eye movement (2 sec)	Add word to sentence
🎤 Voice Sample Requirements

WAV format

Mono channel

Clear voice recording

Same speaker for all samples

Recommended duration: 5–10 seconds each

⚡ Performance Notes

GPU recommended for faster voice generation

CPU mode works but is slow

If CUDA out-of-memory error:

Close other applications

Restart Python

Or switch to CPU mode

🎯 Use Cases

Assistive technology for speech-impaired users

Human-computer interaction research

Accessibility projects

Eye-tracking applications

🔮 Future Improvements

Lightweight TTS model for faster performance

Mobile version

Blink sensitivity calibration

Multi-language support

Cloud deployment

📜 License

This project is for educational and research purposes.
Ensure you have permission before cloning any person’s voice.

👤 Author

Arshad Dylan
