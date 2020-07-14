import pandas as pd
from joblib import delayed, Parallel

if __name__ == 'main': 
  train = pd.read_csv("data/birdsong-recognition/train.csv", parse_dates=['date'])
  train_list = [train[train['ebird_code'].str.startswith(('a', 'b))],  
                train[train['ebird_code'].str.startswith(('c', 'd', 'e', 'f'))],
                train[train['ebird_code'].str.startswith(('g' , 'h', 'i', 'j', 'k', 'l', 'm'))],
                train[train['ebird_code'].str.startswith(('n', 'o', 'p', 'q', 'r'))],
                train[train['ebird_code'].str.startswith(('s', 't', 'u', 'v', 'w', 'x', 'y', 'z'))]
               ]
  for i in range(5): 
      train_audio_infos = train_list[i][["ebird_code", "filename"]].values.tolist()
      source_dir = TRAIN_RESAMPLED_AUDIO_DIRS[i]
      Parallel(n_jobs=-1, verbose=5)(
           delayed(remove_silence_from_file)(ebird_code, file_name, source_dir) for ebird_code, file_name in train_audio_infos)
