ROOT_PATH=data
DATASET_PATH=$ROOT_PATH/UCF101
ANNO_PATH=$DATASET_PATH/annotations

# using only one class
echo "using only one class: PlayingViolin"
mv $ANNO_PATH/classInd.txt $ANNO_PATH/classInd.txt.bak
echo "1 PlayingViolin" >> $ANNO_PATH/classInd.txt

python3 utils/ucf101_json.py $ANNO_PATH
