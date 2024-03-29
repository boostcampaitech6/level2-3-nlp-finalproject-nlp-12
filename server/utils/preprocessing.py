import librosa
import soundfile as sf
import noisereduce as nr
import os

def reduce_background(y, sr):
    reduced_array = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.7)
    return reduced_array

def trim_silence(y):
    y_trimmed, _ = librosa.effects.trim(y=y, top_db=15, frame_length=2, hop_length=500)
    trimmed_length = librosa.get_duration(y=y) - librosa.get_duration(y=y_trimmed)

    return y_trimmed, trimmed_length

def preprocessing_record(file_dir, file_name):
    file_path = os.path.join(file_dir, file_name)
    y, sr = librosa.audioread
    y_trimmed, _ = trim_silence(y)
    y_reduced = reduce_background(y=y_trimmed, sr=sr)

    new_path = os.path.join(file_dir, file_name + "_preprocessed.wav")
    sf.write(new_path, y_reduced, sr)

    return new_path