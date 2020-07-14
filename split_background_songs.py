import random
import numpy as np 
import pandas as pd

import librosa 
import soundfile as sf 
from pathlib import Path		
from joblib import delayed, Parallel
 
ROOT = Path.cwd().parent
INPUT_ROOT = ROOT / "data"
RAW_DATA = INPUT_ROOT / "input/"
TRAIN_AUDIO_DIR = RAW_DATA / "train_audio"
TRAIN_SINGING_DIR = Path("processed_data/train_audio_singing")
TRAIN_BACKGROUND_DIR = Path("processed_data/train_audio_background")
TRAIN_SINGING_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_BACKGROUND_DIR.mkdir(parents=True, exist_ok=True)

 def split_sound(row_number):
    """Returns the sound array, sample rate and
    x_split = intervals where sound is louder than top db
    """
    species, audio_path = get_audio_path(row_number)
    x , sr = librosa.load(audio_path)
    db = librosa.core.amplitude_to_db(x)
    mean_db = np.abs(db).mean()
    std_db = db.std()
    x_split = librosa.effects.split(y=x, top_db = mean_db - std_db)
    return x, sr, x_split
		
	
def split_singing_background(clip, sample='background'):
    """Removes silence from clip
    """
    intervals = split_sound(clip)
    singing = []
    background = clip[0:intervals[0][0]]
    for i in range(len(intervals)-1):
        background = np.append(background, clip[intervals[i][1]:intervals[i+1][0]])
    background = np.append(background, clip[intervals[-1][1]:])
   
    for inter in intervals:
        singing.extend(clip[inter[0]:inter[1]])
    singing = np.array(singing)
    if sample = 'background':
	background = take_random_sample(background)
    elif sample = 'all':
	silence = take_random_sample(silence)
	singing = take_random_sample(singing)
    else: 
	print('no sampling') 
    return singing , silence
		
	def remove_silence_from_file(ebird_code: str, filename: str, source_dir: str, target_sr: int = 32000, sample: str = 'background'):
    ebird_dir = TRAIN_SINGING_DIR / ebird_code
    background_dir = TRAIN_BACKGROUND_DIR / ebird_code
    filename = filename.replace('.mp3', '.wav')
    try:
        y, _ = librosa.load(
            source_dir / ebird_code / filename,
            sr=target_sr, mono=True, res_type="kaiser_fast")
        sound, background = split_singing_background(y)
        sf.write(str(ebird_dir / filename), sound, target_sr)
        sf.write(str(background_dir / filename), background, target_sr)
    except Exception as e:
        print(e)
        with open("skipped.txt", "a") as f:
            file_path = str(source_dir / ebird_code / filename)
            f.write(file_path + ' ' + str(e) + "\n")
		


if __name__ == 'main': 
	for ebird_code in train.ebird_code.unique():
	    ebird_dir = TRAIN_SINGING_DIR / ebird_code
	    background_dir = TRAIN_BACKGROUND_DIR / ebird_code
	    ebird_dir.mkdir(exist_ok=True)
	    background_dir.mkdir(exist_ok=True)
	train = pd.read_csv("data/birdsong-recognition/train.csv", parse_dates=['date'])
	train_audio_infos = train[["ebird_code", "filename"]].values.tolist()
	source_dir = TRAIN_AUDIO_DIR
	Parallel(n_jobs=-1, verbose=5)(delayed(remove_silence_from_file)(ebird_code, file_name, source_dir) for ebird_code, file_name in train_audio_infos)
