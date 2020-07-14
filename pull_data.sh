for ds_part in {0..4}
do 
kaggle datasets download ttahara/birdsong-resampled-train-audio-0{$ds_part} -p data &
unzip data/birdsong-resampled-train-audio-0{$ds_part}.zip -d data/input
rm data/birdsong-resampled-train-audio-0{$ds_part}.zip
kaggle competitions download -f train.csv birdsong-recognition -p data &
unzip data/train.csv.zip -d data/input
