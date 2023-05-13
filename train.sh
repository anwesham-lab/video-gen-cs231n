var="python3.8 main.py --adv_loss hinge --parallel True --gpus 0 --num_workers 8 \
--use_tensorboard True --ds_chn 64 --dt_chn 64 --g_chn 64 --n_frames 8 --k_sample 4 --batch_size 8 \
--n_class 101 \
--root_path preprocessing/data/UCF101 \
--annotation_path annotations/ucf101_01.json \
--log_path logdir \
--model_save_path outputs/models \
--sample_path outputs/samples \
"
echo $var
exec $var