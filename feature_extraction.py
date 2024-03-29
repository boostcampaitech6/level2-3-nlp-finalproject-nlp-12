import opensmile
import os
import numpy as np
import pandas as pd
from typing import Optional
from tqdm import tqdm

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)

file_names = []

for name in os.listdir('/Users/hwang_insoo/opensmile-tutorial/daicwoz/data'):
    file_names.append(name[:-2])

file_paths = [f'/Users/hwang_insoo/opensmile-tutorial/daicwoz/data/{name}_P/{name}_AUDIO.wav' for name in file_names]

audio_data = pd.DataFrame()
for idx, file_path in tqdm(enumerate(file_paths)):
    features = smile.process_file(file_path)
    audio_data = pd.concat([audio_data, features])

audio_data = audio_data.reset_index().drop(['start', 'end'])

normalize_columns = audio_data.columns[1:]

audio_data[normalize_columns] = (audio_data[normalize_columns] - audio_data[normalize_columns].mean()) / audio_data[normalize_columns].std()

audio_data = audio_data.to_csv('feature_normalized.csv', index=False)