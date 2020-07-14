kaggle datasets download ttahara/birdsong-resampled-train-audio-00 -p data & 
kaggle datasets download ttahara/birdsong-resampled-train-audio-01 -p data &
kaggle datasets download ttahara/birdsong-resampled-train-audio-02 -p data &
kaggle datasets download ttahara/birdsong-resampled-train-audio-03 -p data &
kaggle competitions download -f train.csv birdsong-recognition -p data &
unzip ./data/*.zip -p data/input & 

