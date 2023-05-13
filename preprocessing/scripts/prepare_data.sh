ROOT_PATH=data
DATASET_PATH=$ROOT_PATH/UCF101
ORIGIN_PATH=$DATASET_PATH/UCF101
CLASSIFY_PATH=$DATASET_PATH/videos_classified
VIDEO_PATH=$DATASET_PATH/videos_jpeg
ANNO_PATH=$DATASET_PATH/annotations

python3 utils/classify_video.py $ORIGIN_PATH $CLASSIFY_PATH 
python3 utils/ucf_jpeg.py $CLASSIFY_PATH $VIDEO_PATH
python3 utils/n_frames.py $VIDEO_PATH

python3 utils/ucf101_json.py $ANNO_PATH

rm -rf $ORIGIN_PATH $CLASSIFY_PATH