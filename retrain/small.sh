for ws in 2
do
    CUDA_VISIBLE_DEVICES=2 python src/main.py --SoccerNet_path=path of your dataset \
    --model_name=MVIT_SMALL \
    --features mvitsmall.npy --framerate 1.923  --window_size $ws  
done 
