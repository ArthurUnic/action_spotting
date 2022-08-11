
export PYTHONPATH=$PWD/:$PYTHONPATH:/youtu_pedestrian_detection/zhuhe/soccernet
cd /youtu_pedestrian_detection/zhuhe/soccernet


jobid=1;

python scripts/run_action_extract.py all.lst /youtu_pedestrian_detection/junweil/action_datasets/soccernet/ \
/youtu-reid/junweil/checkpoints/k700_train_mvitV2_full_16x4_fromscratch_e200.pyth  \
 /youtu_pedestrian_detection/zhuhe/soccernet/test/extract_smallfeas \
--model_dataset aicity --frame_length 16 --frame_stride 4 --proposal_length 64 \
--proposal_stride 16 --video_fps 25.0  --frame_size 224 \
--pyslowfast_cfg configs/Aicity/MVITV2_FULL_B_16x4_CONV.yaml \
--jobid $jobid \
--batch_size 1 --num_cpu_workers 8 \

