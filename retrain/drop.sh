for ws in 2
do
for drop in 0.2 0.1 0.5
do
    CUDA_VISIBLE_DEVICES=1 python src/main.py --SoccerNet_path=/youtu_pedestrian_detection/junweil/action_datasets/soccernet \
    --model_name=MVIT_DIFFDROP/${drop} \
    --features mvit.npy --framerate 1.923  --window_size $ws  \
    --drop $drop 
done 
done 
