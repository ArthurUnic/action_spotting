export PYTHONPATH=$PWD/:$PYTHONPATH:/youtu_pedestrian_detection/zhuhe/soccernet
cd /youtu_pedestrian_detection/zhuhe/soccernet


jobid=0;

python scripts/run_action_classification_temporal_inf.py test.lst /youtu_pedestrian_detection/junweil/action_datasets/soccernet/ \
/youtu_pedestrian_detection/zhuhe/soccernet/exps/train_size_224_batch_64_lr_1e-4_epoch_50_real/checkpoints/checkpoint_epoch_00045.pyth \
test/size_224_batch_1024_lr_1e-4_epoch_50 \
--model_dataset aicity --frame_length 16 --frame_stride 4 --proposal_length 64 \
--proposal_stride 16 --video_fps 25.0  --frame_size 224 \
--pyslowfast_cfg configs/Aicity/MVITV2_FULL_B_16x4_CONV.yaml \
--jobid $jobid \
--batch_size 100 --num_cpu_workers 4 \
# 1>/youtu_pedestrian_detection/zhuhe/soccernet/info/log.txt 2>/youtu_pedestrian_detection/zhuhe/soccernet/info/err.txt 