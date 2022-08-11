for((i=0;i<220;i++));
do


echo '
Location= your path
export PYTHONPATH=$PWD/:$PYTHONPATH:$Location
cd $Location


jobid='$i';

python scripts/run_action_extract.py all.lst {path of your dataset} \
$Location/exps/train_size_224_lr_1e-4_epoch_50_with_bg_ela/checkpoints/checkpoint_epoch_00050.pyth \
$Location/test/extract_feas \
--model_dataset aicity --frame_length 16 --frame_stride 4 --proposal_length 64 \
--proposal_stride 16 --video_fps 25.0  --frame_size 224 \
--pyslowfast_cfg configs/Aicity/MVITV2_FULL_B_16x4_CONV.yaml \
--jobid $jobid \
--batch_size 64 --num_cpu_workers 8 \
'>start.sh


jizhi_client start -scfg split.json
done
