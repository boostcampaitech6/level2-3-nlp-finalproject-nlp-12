import opensmile
import pandas as pd
from loguru import logger
import time

def load_opensmile():
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals
    )
    logger.info("load_opensmile complete")
    return smile

def extract_feature(smile, file_path):
    start_time = time.time()

    features = smile.process_file(file_path)
    audio_data = pd.DataFrame(features)
    
    logger.info(f"Feature extraction time taken: {time.time() - start_time}")
    return audio_data