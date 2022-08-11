for ws in 1 
do
for bs in 384
do
    python src/main.py --SoccerNet_path=/youtu_pedestrian_detection/junweil/action_datasets/soccernet \
    --model_name=MVIT_DIFFPARMAS/${ws}_${bs} \
    --features mvit.npy --framerate 1.923  --window_size $ws  \
    --batch_size $bs 
done 
done 
