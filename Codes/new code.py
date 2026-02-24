import sys
import cv2
import time
import mediapipe as mp
import numpy as np
import nltk
from nltk.corpus import words
import torch
import torchaudio
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio
import threading
import os
import tempfile

import warnings
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                             QLabel, QPushButton, QFileDialog, QListWidget, QTextEdit,
                             QSizePolicy, QTabWidget)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QSize, QUrl
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

# -------------------- SETUP --------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

nltk.download('words')
WORD_LIST = [w.lower() for w in words.words() if w.isalpha()]

MORSE_CODE_DICT = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z'
}

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tts = TextToSpeech(device=device)


# -------------------- HELPER FUNCTIONS --------------------
def eye_aspect_ratio(eye, lm, h, w):
    a = np.linalg.norm(np.array([lm[eye[1]].x * w, lm[eye[1]].y * h]) -
                       np.array([lm[eye[5]].x * w, lm[eye[5]].y * h]))
    b = np.linalg.norm(np.array([lm[eye[2]].x * w, lm[eye[2]].y * h]) -
                       np.array([lm[eye[4]].x * w, lm[eye[4]].y * h]))
    c = np.linalg.norm(np.array([lm[eye[0]].x * w, lm[eye[0]].y * h]) -
                       np.array([lm[eye[3]].x * w, lm[eye[3]].y * h]))
    return (a + b) / (2.0 * c)


def decode_morse(code):
    return MORSE_CODE_DICT.get(code, '')


def suggest_words(prefix):
    return [w for w in WORD_LIST if w.startswith(prefix.lower())][:5]


# -------------------- COMMUNICATION OBJECT --------------------
class Communicate(QObject):
    update_frame = pyqtSignal(QImage)
    update_morse = pyqtSignal(str)
    update_word = pyqtSignal(str)
    update_text = pyqtSignal(str)
    update_suggestions = pyqtSignal(list)
    update_selected = pyqtSignal(int)
    status_message = pyqtSignal(str)
    audio_playback_finished = pyqtSignal()
    enable_play_button = pyqtSignal()


# -------------------- MAIN WINDOW --------------------
class MorseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.voice_samples = []
        self.setup_ui()
        self.setup_camera()
        self.setup_audio_player()
        self.comm = Communicate()
        self.connect_signals()
        self.last_eye_gaze_time = time.time()

    def create_instruction_panel(self):
        instruction_panel = QWidget()
        instruction_layout = QVBoxLayout(instruction_panel)
        instruction_layout.setContentsMargins(10, 10, 10, 10)

        title = QLabel("Eye Gaze Instructions")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet("color: #FF69B4;")
        instruction_layout.addWidget(title)

        instructions = [
            "• <b>Both eyes blink:</b> Create Morse code (short = '.', long = '-')",
            "• <b>Long blink (2+ sec):</b> Speak the current text",
            "• <b>Left eye closed:</b> Move up in suggestions",
            "• <b>Right eye closed:</b> Move down in suggestions",
            "• <b>Left eye long close (1 sec):</b> Backspace/delete",
            "• <b>No eye gaze (2+ sec):</b> Add current word to text",
            "• <b>Both eyes open:</b> Normal state (waiting for input)"
        ]

        for instruction in instructions:
            label = QLabel(instruction)
            label.setFont(QFont("Arial", 12))
            label.setStyleSheet("color: white;")
            label.setWordWrap(True)
            instruction_layout.addWidget(label)

        return instruction_panel

    def setup_ui(self):
        self.setWindowTitle("Blink-to-Speech Morse Code")
        self.setGeometry(100, 100, 1200, 800)

        # Colorful palette
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(30, 30, 60))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Button, QColor(70, 70, 120))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.Highlight, QColor(255, 105, 180))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(palette)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Left panel (camera and controls)
        left_panel = QVBoxLayout()
        left_panel.setContentsMargins(0, 0, 0, 0)
        left_panel.setSpacing(10)

        # Camera view with fixed size
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setMaximumSize(640, 480)
        self.camera_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.camera_label.setStyleSheet("background-color: black; border: 3px solid #FF69B4;")
        left_panel.addWidget(self.camera_label, 0, Qt.AlignCenter)

        # Morse code display
        self.morse_label = QLabel("Morse Code: ")
        self.morse_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.morse_label.setStyleSheet("color: #FF69B4;")
        left_panel.addWidget(self.morse_label)

        # Current word display
        self.word_label = QLabel("Current Word: ")
        self.word_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.word_label.setStyleSheet("color: #7FFFD4;")
        left_panel.addWidget(self.word_label)

        # Status message
        self.status_label = QLabel("Status: Ready")
        self.status_label.setFont(QFont("Arial", 12))
        self.status_label.setStyleSheet("color: #FFD700;")
        left_panel.addWidget(self.status_label)

        # Voice sample buttons
        voice_layout = QHBoxLayout()
        self.load_btn1 = QPushButton("Load Voice Sample 1")
        self.load_btn2 = QPushButton("Load Voice Sample 2")
        self.load_btn3 = QPushButton("Load Voice Sample 3")
        for btn in [self.load_btn1, self.load_btn2, self.load_btn3]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #483D8B;
                    padding: 8px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #6A5ACD;
                }
            """)
            voice_layout.addWidget(btn)

        left_panel.addLayout(voice_layout)

        # Control buttons
        control_layout = QHBoxLayout()

        # Blink detection controls
        self.start_detection_btn = QPushButton("Start Blink Detection")
        self.start_detection_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.stop_detection_btn = QPushButton("Stop Blink Detection")
        self.stop_detection_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        self.stop_detection_btn.setEnabled(False)

        # Audio playback controls
        self.play_audio_btn = QPushButton("Play Audio")
        self.play_audio_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.play_audio_btn.setEnabled(False)

        self.stop_audio_btn = QPushButton("Stop Audio")
        self.stop_audio_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #e68a00;
            }
        """)
        self.stop_audio_btn.setEnabled(False)

        control_layout.addWidget(self.start_detection_btn)
        control_layout.addWidget(self.stop_detection_btn)
        control_layout.addWidget(self.play_audio_btn)
        control_layout.addWidget(self.stop_audio_btn)
        left_panel.addLayout(control_layout)

        # Backspace button
        self.backspace_btn = QPushButton("Backspace (Close Left Eye)")
        self.backspace_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF6347;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #FF4500;
            }
        """)
        left_panel.addWidget(self.backspace_btn)

        # Right panel (text, suggestions and instructions)
        right_panel = QVBoxLayout()
        right_panel.setContentsMargins(10, 0, 0, 10)
        right_panel.setSpacing(10)

        # Create a tab widget for text/suggestions and instructions
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #9370DB;
                border-radius: 5px;
            }
            QTabBar::tab {
                background: #483D8B;
                color: white;
                padding: 8px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background: #6A5ACD;
            }
        """)

        # Text/Suggestions tab
        text_tab = QWidget()
        text_layout = QVBoxLayout(text_tab)
        text_layout.setContentsMargins(5, 5, 5, 5)

        # Final text display
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Arial", 16))
        self.text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E2E;
                color: white;
                border: 2px solid #9370DB;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        text_layout.addWidget(QLabel("Final Text:"))
        text_layout.addWidget(self.text_edit)

        # Suggestions list with fixed height
        self.suggestions_list = QListWidget()
        self.suggestions_list.setFont(QFont("Arial", 14))
        self.suggestions_list.setMaximumHeight(150)
        self.suggestions_list.setStyleSheet("""
            QListWidget {
                background-color: #1E1E2E;
                color: white;
                border: 2px solid #9370DB;
                border-radius: 5px;
            }
            QListWidget::item {
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: #FF69B4;
                color: black;
            }
        """)
        text_layout.addWidget(QLabel("Word Suggestions:"))
        text_layout.addWidget(self.suggestions_list)

        # Text action buttons
        text_buttons_layout = QHBoxLayout()

        # Speak button
        self.speak_btn = QPushButton("Generate & Play Audio")
        self.speak_btn.setStyleSheet("""
            QPushButton {
                background-color: #32CD32;
                padding: 12px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #3CB371;
            }
        """)
        text_buttons_layout.addWidget(self.speak_btn)

        # Clear text button
        self.clear_text_btn = QPushButton("Clear Text")
        self.clear_text_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF6347;
                padding: 12px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #FF4500;
            }
        """)
        text_buttons_layout.addWidget(self.clear_text_btn)

        text_layout.addLayout(text_buttons_layout)

        # Instructions tab
        instructions_tab = self.create_instruction_panel()

        # Add tabs to the tab widget
        tab_widget.addTab(text_tab, "Text")
        tab_widget.addTab(instructions_tab, "Instructions")

        right_panel.addWidget(tab_widget)

        # Add panels to main layout
        main_layout.addLayout(left_panel, 60)
        main_layout.addLayout(right_panel, 40)

        # Connect buttons
        self.load_btn1.clicked.connect(lambda: self.load_voice_sample(1))
        self.load_btn2.clicked.connect(lambda: self.load_voice_sample(2))
        self.load_btn3.clicked.connect(lambda: self.load_voice_sample(3))
        self.speak_btn.clicked.connect(self.speak_text)
        self.clear_text_btn.clicked.connect(self.clear_text)
        self.backspace_btn.clicked.connect(self.backspace_action)
        self.start_detection_btn.clicked.connect(self.start_detection)
        self.stop_detection_btn.clicked.connect(self.stop_detection)
        self.play_audio_btn.clicked.connect(self.play_audio)
        self.stop_audio_btn.clicked.connect(self.stop_audio)

    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

        self.morse_code = ''
        self.current_word = ''
        self.final_text = ''
        self.suggestions = []
        self.selected_idx = 0
        self.blink_start = None
        self.blinking = False
        self.last_morse_time = time.time()
        self.last_letter_time = time.time()
        self.last_suggestion_time = time.time()
        self.suggestion_mode = False
        self.long_blink_threshold = 2.0  # 2 seconds for long blink
        self.backspace_mode = False
        self.audio_playing = False
        self.detection_active = False
        self.current_audio_file = None

        # Start camera timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def setup_audio_player(self):
        self.media_player = QMediaPlayer()
        self.media_player.setVolume(100)
        self.media_player.mediaStatusChanged.connect(self.handle_media_status)

    def handle_media_status(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.audio_playing = False
            self.comm.audio_playback_finished.emit()
            self.comm.status_message.emit("Audio playback finished")
            # Clean up the temporary file after playback is complete
            if self.current_audio_file and os.path.exists(self.current_audio_file):
                try:
                    os.remove(self.current_audio_file)
                except:
                    pass
                self.current_audio_file = None

    def connect_signals(self):
        self.comm.update_frame.connect(self.set_image)
        self.comm.update_morse.connect(lambda text: self.morse_label.setText(f"Morse Code: {text}"))
        self.comm.update_word.connect(lambda text: self.word_label.setText(f"Current Word: {text}"))
        self.comm.update_text.connect(lambda text: self.text_edit.setText(text))
        self.comm.update_suggestions.connect(self.update_suggestions_list)
        self.comm.update_selected.connect(self.highlight_suggestion)
        self.comm.status_message.connect(lambda text: self.status_label.setText(f"Status: {text}"))
        self.comm.audio_playback_finished.connect(self.enable_morse_detection)
        self.comm.enable_play_button.connect(lambda: self.play_audio_btn.setEnabled(True))

    def load_voice_sample(self, num):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, f"Load Voice Sample {num}", "",
            "Audio Files (*.wav *.mp3);;All Files (*)", options=options)

        if file_name:
            try:
                sample = load_audio(file_name, 22050)
                if len(self.voice_samples) < num:
                    self.voice_samples.append(sample)
                else:
                    self.voice_samples[num - 1] = sample

                self.comm.status_message.emit(f"Voice sample {num} loaded successfully!")
            except Exception as e:
                self.comm.status_message.emit(f"Error loading voice sample: {str(e)}")

    def speak_text(self):
        text = self.text_edit.toPlainText()
        if text and self.voice_samples:
            self.disable_morse_detection()
            threading.Thread(target=self.synthesize_and_play_voice, args=(text,)).start()
            self.comm.status_message.emit("Generating and playing audio...")
        else:
            self.comm.status_message.emit("No text or voice samples loaded!")

    def synthesize_and_play_voice(self, text):
        try:
            with torch.inference_mode():
                gen = tts.tts_with_preset(
                    text,
                    voice_samples=self.voice_samples,
                    preset="ultra_fast",
                    k=1,
                    num_autoregressive_samples=1,
                    diffusion_iterations=20,
                    length_penalty=1.0,
                    temperature=0.8
                )

            # Create a temporary file for playback
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                output_path = temp_file.name

            torchaudio.save(output_path, gen.squeeze(0).cpu(), 24000)

            # Store the audio file path for later playback
            self.current_audio_file = output_path

            # Enable the play button in the UI thread
            self.comm.enable_play_button.emit()
            self.comm.status_message.emit("Audio generated and ready to play!")

        except Exception as e:
            self.comm.status_message.emit(f"Error synthesizing voice: {str(e)}")
            self.enable_morse_detection()

    def play_audio(self):
        if self.current_audio_file and os.path.exists(self.current_audio_file):
            self.audio_playing = True
            url = QUrl.fromLocalFile(self.current_audio_file)
            content = QMediaContent(url)
            self.media_player.setMedia(content)
            self.media_player.play()
            self.comm.status_message.emit("Playing audio...")
            self.play_audio_btn.setEnabled(False)
            self.stop_audio_btn.setEnabled(True)

    def stop_audio(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.stop()
            self.audio_playing = False
            self.comm.status_message.emit("Audio stopped")
            self.play_audio_btn.setEnabled(True)
            self.stop_audio_btn.setEnabled(False)

    def start_detection(self):
        self.detection_active = True
        self.start_detection_btn.setEnabled(False)
        self.stop_detection_btn.setEnabled(True)
        self.comm.status_message.emit("Blink detection started")

    def stop_detection(self):
        self.detection_active = False
        self.start_detection_btn.setEnabled(True)
        self.stop_detection_btn.setEnabled(False)
        self.comm.status_message.emit("Blink detection stopped")

    def disable_morse_detection(self):
        self.audio_playing = True
        self.comm.status_message.emit("Morse detection paused during audio playback")

    def enable_morse_detection(self):
        self.audio_playing = False
        self.comm.status_message.emit("Morse detection enabled")

    def clear_text(self):
        """Clear all text content in the application"""
        self.final_text = ''
        self.current_word = ''
        self.morse_code = ''
        self.suggestions = []
        self.suggestion_mode = False
        self.comm.update_text.emit(self.final_text)
        self.comm.update_word.emit(self.current_word)
        self.comm.update_morse.emit(self.morse_code)
        self.comm.update_suggestions.emit(self.suggestions)
        self.comm.status_message.emit("All text cleared")

    def update_suggestions_list(self, suggestions):
        self.suggestions_list.clear()
        self.suggestions = suggestions
        self.suggestions_list.addItems(suggestions)
        if suggestions:
            self.suggestions_list.setCurrentRow(0)

    def highlight_suggestion(self, idx):
        if 0 <= idx < self.suggestions_list.count():
            self.suggestions_list.setCurrentRow(idx)

    def backspace_action(self):
        if self.suggestion_mode:
            self.suggestion_mode = False
            self.suggestions = []
            self.comm.update_suggestions.emit([])
        elif self.current_word:
            self.current_word = self.current_word[:-1]
            self.comm.update_word.emit(self.current_word)
            self.comm.status_message.emit("Last character removed")
        elif self.final_text:
            # Remove last word if current_word is empty
            words = self.final_text.split()
            if words:
                self.final_text = ' '.join(words[:-1]) + ' ' if len(words) > 1 else ''
                self.comm.update_text.emit(self.final_text)
                self.comm.status_message.emit("Last word removed")

    def update_frame(self):
        if not self.detection_active or self.audio_playing:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        current_time = time.time()

        # Check for no eye gaze (no blink detection) for more than 2 seconds
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            left_ear = eye_aspect_ratio(LEFT_EYE, lm, h, w)
            right_ear = eye_aspect_ratio(RIGHT_EYE, lm, h, w)
            ear = (left_ear + right_ear) / 2

            # Reset the timer if eyes are open (normal state)
            if ear > 0.25:
                self.last_eye_gaze_time = current_time
        else:
            # No face detected counts as no eye gaze
            self.last_eye_gaze_time = current_time

        # If no eye gaze for more than 2 seconds and we have a current word
        if (current_time - self.last_eye_gaze_time > 2.0 and
                self.current_word and
                not self.suggestion_mode):
            self.final_text += self.current_word + ' '
            self.current_word = ''
            self.comm.update_word.emit(self.current_word)
            self.comm.update_text.emit(self.final_text)
            self.last_eye_gaze_time = current_time  # Reset timer

        # Rest of the existing blink detection code...
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            left_ear = eye_aspect_ratio(LEFT_EYE, lm, h, w)
            right_ear = eye_aspect_ratio(RIGHT_EYE, lm, h, w)
            ear = (left_ear + right_ear) / 2

            # Check for backspace (only left eye closed)
            if left_ear < 0.21 and right_ear > 0.25:
                if not self.backspace_mode:
                    self.backspace_mode = True
                    self.backspace_start = time.time()
                elif time.time() - self.backspace_start > 1.0:  # 1 second hold for backspace
                    self.backspace_action()
                    self.backspace_start = time.time()  # Reset timer
                    time.sleep(0.5)  # Small delay to prevent multiple triggers
            else:
                self.backspace_mode = False

            # Normal blink detection (both eyes)
            if ear < 0.21:
                if not self.blinking:
                    self.blink_start = time.time()
                    self.blinking = True
            elif self.blinking:
                blink_dur = time.time() - self.blink_start
                self.blinking = False
                self.last_morse_time = time.time()

                # Check for long blink (more than 2 seconds)
                if blink_dur > self.long_blink_threshold:
                    if self.final_text.strip():  # Only speak if there's text
                        self.speak_text()
                else:
                    # Normal morse code detection
                    if blink_dur < 0.25:
                        self.morse_code += '.'
                    elif blink_dur < 0.8:
                        self.morse_code += '-'

            if self.morse_code and (time.time() - self.last_morse_time > 1):
                letter = decode_morse(self.morse_code)
                if letter:
                    self.current_word += letter
                    self.suggestions = suggest_words(self.current_word)
                    self.suggestion_mode = bool(self.suggestions)
                    self.selected_idx = 0
                    self.last_suggestion_time = time.time()
                    self.comm.update_suggestions.emit(self.suggestions)
                self.morse_code = ''

            if self.suggestion_mode:
                if left_ear < 0.2 < right_ear:  # Left wink - move up
                    self.selected_idx = max(0, self.selected_idx - 1)
                    self.last_suggestion_time = time.time()
                    time.sleep(0.25)
                    self.comm.update_selected.emit(self.selected_idx)
                elif right_ear < 0.2 < left_ear:  # Right wink - move down
                    self.selected_idx = min(len(self.suggestions) - 1, self.selected_idx + 1)
                    self.last_suggestion_time = time.time()
                    time.sleep(0.25)
                    self.comm.update_selected.emit(self.selected_idx)

            if self.suggestion_mode and time.time() - self.last_suggestion_time > 2:
                if self.suggestions:
                    selected_word = self.suggestions[self.selected_idx]
                    self.final_text += selected_word + ' '
                    self.current_word = ''
                    self.suggestions = []
                    self.suggestion_mode = False
                    self.comm.update_suggestions.emit([])

        # Draw landmarks on frame
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if idx in LEFT_EYE or idx in RIGHT_EYE:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

        # Update UI elements
        self.comm.update_morse.emit(self.morse_code)
        self.comm.update_word.emit(self.current_word)
        self.comm.update_text.emit(self.final_text)

        # Convert frame to QImage with fixed size
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, (640, 480))
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.comm.update_frame.emit(qt_image)

    def set_image(self, image):
        self.camera_label.setPixmap(QPixmap.fromImage(image))

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        if hasattr(self, 'media_player'):
            self.media_player.stop()
        # Clean up any temporary audio files
        if hasattr(self, 'current_audio_file') and self.current_audio_file and os.path.exists(self.current_audio_file):
            try:
                os.remove(self.current_audio_file)
            except:
                pass
        event.accept()


# -------------------- MAIN --------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    window = MorseApp()
    window.show()
    sys.exit(app.exec_())