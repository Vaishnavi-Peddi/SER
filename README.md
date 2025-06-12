# SER
# 🎙️ Speech Emotion Recognition (SER) Project

This project is a **Speech Emotion Recognition (SER)** system that identifies human emotions (e.g., happy, sad, angry) from audio recordings. It supports multilingual audio input (Telugu, Hindi, English) and uses a combination of **CNN-LSTM deep learning** for raw audio emotion classification and **NLP-based text analysis** using **GoEmotions** for emotion detection from transcribed speech.

## 🚀 Features

- 🎧 Recognizes emotions from speech audio files
- 🌐 Multilingual: Telugu, Hindi, English
- 🔊 Uses CNN + LSTM model trained on the RAVDESS dataset
- 🧠 Integrates GoEmotions for text-based emotion inference
- 🔄 Speech-to-text with automatic translation (if needed)
- 🌍 Web interface for uploading and analyzing emotions
- 📊 Displays emotion with emoji and probability

## 🛠️ Tech Stack

- **Frontend**: HTML, CSS, Bootstrap
- **Backend**: Python (Flask)
- **Machine Learning**:
  - CNN-LSTM model for audio classification
  - Transformers (Hugging Face) for text emotion detection
- **Speech Recognition**: `speech_recognition`, `ffmpeg`
- **Translation**: `deep_translator` (Google Translator API)
- **Visualization**: Emojis and probability graphs (optional)

## 📁 Project Structure

```
SER/
│
├── app.py                       # Flask server
├── prediction_model.py         # Audio emotion model (CNN+LSTM)
├── templates/
│   └── index.html              # Main web UI
├── static/
│   └── style.css               # Styling (if needed)
├── uploads/                    # Stores uploaded audio
├── processed/                  # Preprocessed/converted audio
├── requirements.txt            # Required Python packages
└── README.md                   # Project documentation
```

## 🔧 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Vaishnavi-Peddi/SER.git
   cd SER
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the Flask app**
   ```bash
   python app.py
   ```

5. **Open in browser**
   ```
   http://127.0.0.1:5000/
   ```

## 🎓 Datasets Used

- **RAVDESS**: For audio-based training and testing
- **GoEmotions**: For emotion labeling via text

## 🧪 Example Emotions Detected

| Language | Input Text/Speech         | Emotion Output |
|----------|----------------------------|----------------|
| English  | "I’m so happy today!"      | 😀 Happy       |
| Hindi    | "मुझे गुस्सा आ रहा है"     | 😠 Angry       |
| Telugu   | "నాకు చాలా బాధగా ఉంది"     | 😢 Sad         |

## 📌 To-Do / Enhancements

- [ ] Add multi-user login and emotion history
- [ ] Enable live microphone input
- [ ] Deploy on cloud (Render/Heroku/AWS)
- [ ] Add emotion graph or timeline

## ✅ Requirements

- Python 3.8+
- `ffmpeg` installed and added to system PATH

## 📦 Dependencies

```
Flask
speechrecognition
deep-translator
transformers
torch
torchaudio
librosa
ffmpeg-python
```

You can install them all using:
```bash
pip install -r requirements.txt
```

## 🧑‍💻 Contributors

- Vaishu (Project Lead)
- [YourName] (Model Development)
- [Optional Contributor]

## 📄 License

This project is open-source under the [MIT License](LICENSE).

---

Made with ❤️ for emotion-aware computing!
