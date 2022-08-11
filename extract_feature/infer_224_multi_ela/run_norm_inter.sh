# for((i=0;i<=220;i++));
for i in 19 24 
do

cat infer.json | jq '.task_flag="chengzhiinfer_with_bg_part_'$i'_vi1n1"' >split.json

echo '
export PYTHONPATH=$PWD/:$PYTHONPATH:/youtu_pedestrian_detection/zhuhe/soccernet
cd /youtu_pedestrian_detection/zhuhe/soccernet


jobid='$i';

python scripts/run_action_extract.py all.lst /youtu_pedestrian_detection/junweil/action_datasets/soccernet/ \
/youtu_pedestrian_detection/zhuhe/soccernet/exps/train_size_224_lr_1e-4_epoch_100_with_bg_all_ela/checkpoints/checkpoint_epoch_00095.pyth \
/youtu_pedestrian_detection/zhuhe/soccernet/test/extract_feas_norm \
--model_dataset aicity --frame_length 16 --frame_stride 4 --proposal_length 64 \
--proposal_stride 16 --video_fps 25.0  --frame_size 224 \
--pyslowfast_cfg configs/Aicity/MVITV2_FULL_B_16x4_CONV.yaml \
--jobid $jobid \
--batch_size 64 --num_cpu_workers 8 --jobinter 1 \
'>start.sh


jizhi_client start -scfg split.json
done
