import os
import base64
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import ffmpeg
import speech_recognition as sr
from deep_translator import GoogleTranslator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from prediction_model import prediction  # Your custom RAVDESS model

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

emoji_map = {
    "neutral": "😐", "calm": "😌", "happy": "😄",
    "sad": "😢", "angry": "😠", "fear": "😨",
    "disgust": "🤢", "surprise": "😲"
}

tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/bert-base-go-emotion")
model = AutoModelForSequenceClassification.from_pretrained("bhadresh-savani/bert-base-go-emotion")

EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

mapping = {
    'happy': ['joy', 'love', 'optimism', 'excitement', 'amusement', 'gratitude', 'relief'],
    'sad': ['sadness', 'disappointment', 'grief'],
    'angry': ['anger', 'annoyance', 'disapproval', 'remorse'],
    'fear': ['fear', 'nervousness'],
    'surprise': ['surprise', 'realization'],
    'disgust': ['disgust', 'embarrassment'],
    'neutral': ['neutral', 'approval', 'pride', 'curiosity', 'confusion', 'admiration', 'caring', 'desire']
}

emojis = {
    'happy': '😄', 'sad': '😢', 'angry': '😠', 'fear': '😨',
    'surprise': '😲', 'disgust': '🤢', 'neutral': '😐'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['wav', 'webm', 'mp3', 'ogg']

def convert_to_wav(input_path, output_path):
    ffmpeg.input(input_path).output(output_path, format='wav', acodec='pcm_s16le').overwrite_output().run(quiet=True)

def speech_to_text(audio_path, lang='te_IN'):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio, language=lang)
    except:
        return None

def translate_to_english(text):
    if not text:
        return None
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return None

def detect_emotion(text, threshold=0.2):
    if not text:
        return None, None, {}

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits)[0].detach().numpy()
    raw_emotions = {label: float(prob) for label, prob in zip(EMOTION_LABELS, probs) if prob > threshold}

    core_emotions = {core: sum(raw_emotions.get(lbl, 0) for lbl in labels) for core, labels in mapping.items()}
    top_emotion = max(core_emotions, key=core_emotions.get, default=None)

    return top_emotion, emojis.get(top_emotion, "❓"), core_emotions

@app.route('/', methods=['GET', 'POST'])
def index():
    audio_emotion = audio_emoji = text_emotion = text_emoji = None
    transcribed_text = translated_text = None
    emotion_scores = {}

    if request.method == 'POST':
        language_code = request.form.get('language', 'te_IN')

        if 'analyze_audio' in request.form:
            file = request.files.get('file')
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)

                if not filename.endswith('.wav'):
                    wav_path = os.path.join(PROCESSED_FOLDER, f"{os.path.splitext(filename)[0]}.wav")
                    convert_to_wav(file_path, wav_path)
                    os.remove(file_path)
                else:
                    wav_path = file_path

                audio_emotion = prediction(wav_path)
                audio_emoji = emoji_map.get(audio_emotion.lower(), "❓")
                os.remove(wav_path)

        elif 'analyze_text' in request.form:
            audio_data_b64 = request.form.get('audio_data')
            if audio_data_b64 and audio_data_b64.startswith('data:audio'):
                header, encoded = audio_data_b64.split(',', 1)
                audio_bytes = base64.b64decode(encoded)
                raw_path = os.path.join(UPLOAD_FOLDER, 'recorded_audio.webm')
                with open(raw_path, 'wb') as f:
                    f.write(audio_bytes)

                wav_path = os.path.join(PROCESSED_FOLDER, 'recorded_audio.wav')
                convert_to_wav(raw_path, wav_path)
                os.remove(raw_path)

                transcribed_text = speech_to_text(wav_path, language_code)
                os.remove(wav_path)

                if transcribed_text:
                    translated_text = translate_to_english(transcribed_text)
                    if translated_text:
                        text_emotion, text_emoji, emotion_scores = detect_emotion(translated_text)
                    else:
                        text_emotion, text_emoji = "Translation failed", "❓"
                else:
                    text_emotion, text_emoji = "Transcription failed", "❓"

    return render_template(
        'index.html',
        audio_emotion=audio_emotion,
        audio_emoji=audio_emoji,
        text_emotion=text_emotion,
        text_emoji=text_emoji,
        transcribed_text=transcribed_text,
        translated_text=translated_text,
        emotion_scores=emotion_scores
    )

if __name__ == '__main__':
    app.run(debug=True)