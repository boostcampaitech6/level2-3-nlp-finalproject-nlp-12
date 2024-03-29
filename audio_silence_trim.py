import librosa
from pysndfx import AudioEffectsChain
import argparse
import soundfile as sf
import noisereduce as nr

aec = AudioEffectsChain()

def reduce_background(y, sr):
    reduced_array = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.7)
    return reduced_array

def trim_silence(y):
    y_trimmed, _ = librosa.effects.trim(y=y, top_db=5, frame_length=2, hop_length=500)
    trimmed_length = librosa.get_duration(y=y) - librosa.get_duration(y=y_trimmed)

    return y_trimmed, trimmed_length

def preprocess(file_path):
    y, sr = librosa.load(file_path)
    y_trimmed, _ = trim_silence(y)
    y_reduced = reduce_background(y=y_trimmed, sr=sr)

    new_path = file_path[:-4] + "_preprocessed.wav"
    sf.write(new_path, y_reduced, sr)

    return new_path

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default=None)
    parser.add_argument("--db_threshold", type=float, default=5.0)
    args = parser.parse_args()

    preprocess(args.file_path)