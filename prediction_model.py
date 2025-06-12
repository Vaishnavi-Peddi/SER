import os
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import model_from_json

# === Load model and supporting files ===

# Set paths
BASE_DIR = os.path.dirname(__file__)
MODEL_JSON = os.path.join(BASE_DIR, "CNN_model.json")
MODEL_WEIGHTS = os.path.join(BASE_DIR, "best_model1.weights.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler2.pickle")
ENCODER_PATH = os.path.join(BASE_DIR,"encoder2.pickle")

# Load model architecture
with open(MODEL_JSON, "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

# Load weights
loaded_model.load_weights(MODEL_WEIGHTS)

# Load scaler and encoder
with open(SCALER_PATH, 'rb') as f:
    scaler2 = pickle.load(f)
with open(ENCODER_PATH, 'rb') as f:
    encoder2 = pickle.load(f)

# === Feature Extraction Functions ===

def zcr(data, frame_length=2048, hop_length=512):
    return np.squeeze(librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length))

def rmse(data, frame_length=2048, hop_length=512):
    return np.squeeze(librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length))

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13, hop_length=hop_length, n_fft=frame_length)
    return mfccs.flatten() if flatten else mfccs

def pad_or_truncate(feature, target_len):
    if len(feature) < target_len:
        return np.pad(feature, (0, target_len - len(feature)))
    else:
        return feature[:target_len]

def extract_features(data, sr=22050, frame_length=2048, hop_length=512, fixed_frames=108):
    zcr_feat = pad_or_truncate(zcr(data, frame_length, hop_length), fixed_frames)
    rmse_feat = pad_or_truncate(rmse(data, frame_length, hop_length), fixed_frames)
    mfcc_feat = pad_or_truncate(mfcc(data, sr, frame_length, hop_length), fixed_frames * 13)
    
    zcr_feat = pad_or_truncate(zcr_feat, fixed_frames)
    rmse_feat = pad_or_truncate(rmse_feat, fixed_frames)
    mfcc_feat = pad_or_truncate(mfcc_feat, fixed_frames * 13)

    # Concatenate all features into a single feature vector
    result = np.hstack((zcr_feat, rmse_feat, mfcc_feat))
    return result
    return np.hstack((zcr_feat, rmse_feat, mfcc_feat))

# === Main Prediction Functions ===

def get_predict_feat(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    features = extract_features(data, sr=sample_rate)
    result = np.array(features).reshape(1, -1)
    scaled = scaler2.transform(result)
    final_input = np.expand_dims(scaled, axis=2)
    return final_input
emotions1={1:'Neutral', 2:'Calm', 3:'Happy', 4:'Sad', 5:'Angry', 6:'Fear', 7:'Disgust',8:'Surprise'}
def prediction(audio_path):
    try:
        # Your actual prediction logic
        print(f"Processing file: {audio_path}")
        processed_input = get_predict_feat(audio_path)
        prediction_probs = loaded_model.predict(processed_input)
        predicted_label = encoder2.inverse_transform(prediction_probs)
        return predicted_label[0][0]
    except Exception as e:
        print("Prediction error:", str(e))
        traceback.print_exc()
        raise
import traceback




result=prediction(r"C:\Users\peddi\OneDrive\Desktop\Mini\ravdess\Actor_01\03-01-07-02-02-01-01.wav")
print(f"Predicted Emotion: {result}")

