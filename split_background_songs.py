import random
import numpy as np
import pandas as pd
import logging
import librosa
import soundfile as sf
from pathlib import Path
from joblib import delayed, Parallel

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")

ROOT = Path.cwd().parent
DATA = ROOT / "data"
INPUT_DATA = DATA / "input"
TRAIN_AUDIO_DIRS = [
    INPUT_DATA / folder for folder in ["1_a-b", "2_c-f", "3_g-m", "4_n-r", "5_s-y"]
]
OUTPUT_DATA = DATA / "output"
TRAIN_SINGING_DIRS = [
    OUTPUT_DATA / "singing" / folder
    for folder in ["1_a-b", "2_c-f", "3_g-m", "4_n-r", "5_s-y"]
]
TRAIN_BACKGROUND_DIRS = [
    OUTPUT_DATA / "background" / folder
    for folder in ["1_a-b", "2_c-f", "3_g-m", "4_n-r", "5_s-y"]
]


def split_sound(clip):
    """Returns the sound array, sample rate and
    x_split = intervals where sound is louder than top db
    """
    db = librosa.core.amplitude_to_db(clip)
    mean_db = np.abs(db).mean()
    std_db = db.std()
    x_split = librosa.effects.split(y=clip, top_db=mean_db - std_db)
    return x_split


def take_random_sample(clip, sample_len=5, sample_rate=32000):
    if len(clip) > sample_len * sample_len:
        idx = random.randint(0, len(clip) - sample_len * sample_len)
        sample = clip[idx : idx + sample_rate * sample_len]
        return sample
    else:
        return clip


def split_singing_background(clip, sample="background"):
    """Removes silence from clip
    """
    intervals = split_sound(clip)
    singing = []
    background = clip[0 : intervals[0][0]]
    for i in range(len(intervals) - 1):
        background = np.append(background, clip[intervals[i][1] : intervals[i + 1][0]])
    background = np.append(background, clip[intervals[-1][1] :])

    for inter in intervals:
        singing.extend(clip[inter[0] : inter[1]])
    singing = np.array(singing)
    if sample == "background":
        background = take_random_sample(background)
    elif sample == "all":
        silence = take_random_sample(background)
        singing = take_random_sample(singing)
    else:
        print("no sampling")
    return singing, silence


def remove_silence_from_file(
    ebird_code,
    filename,
    source_dir,
    singing_dir,
    background_dir,
    target_sr=32000,
    sample="background",
):
    bird_singing_dir = singing_dir / ebird_code
    bird_background_dir = background_dir / ebird_code
    filename = filename.replace(".mp3", ".wav")
    try:
        y, _ = librosa.load(
            source_dir / ebird_code / filename,
            sr=target_sr,
            mono=True,
            res_type="kaiser_fast",
        )
        sound, background = split_singing_background(y)
        sf.write(str(bird_singing_dir / filename), sound, target_sr)
        sf.write(str(bird_background_dir / filename), background, target_sr)
    except Exception as e:
        print(e)
        with open("skipped.txt", "a") as f:
            file_path = str(source_dir / ebird_code / filename)
            f.write(file_path + " " + str(e) + "\n")


def get_train_list():
    train = pd.read_csv("data/train.csv", parse_dates=["date"])
    logging.debug(f"train shape = {train.shape}")
    train_list = [
        train[train["ebird_code"].str.startswith(("a", "b"))],
        train[train["ebird_code"].str.startswith(("c", "d", "e", "f"))],
        train[train["ebird_code"].str.startswith(("g", "h", "i", "j", "k", "l", "m"))],
        train[train["ebird_code"].str.startswith(("n", "o", "p", "q", "r"))],
        train[
            train["ebird_code"].str.startswith(("s", "t", "u", "v", "w", "x", "y", "z"))
        ],
    ]
    return train_list


if __name__ == "main":
    train_list = get_train_list()

    for i in range(5):
        logging.debug("Treating {TRAIN_AUDIO_DIRS[i]}")
        for ebird_code in train_list[i].ebird_code.unique():
            ebird_dir = TRAIN_SINGING_DIRS[i] / ebird_code
            background_dir = TRAIN_BACKGROUND_DIRS[i] / ebird_code
            ebird_dir.mkdir(exist_ok=True)
            background_dir.mkdir(exist_ok=True)
        train_audio_infos = train_list[i][["ebird_code", "filename"]].values.tolist()
        source_dir = TRAIN_AUDIO_DIRS[i]
        singing_output_dir = TRAIN_SINGING_DIRS[i]
        background_output_dir = TRAIN_BACKGROUND_DIRS[i]
        Parallel(n_jobs=-1, verbose=5)(
            delayed(remove_silence_from_file)(
                ebird_code,
                file_name,
                source_dir,
                singing_output_dir,
                background_output_dir,
            )
            for ebird_code, file_name in train_audio_infos
        )
