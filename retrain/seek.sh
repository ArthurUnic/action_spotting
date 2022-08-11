for ws in 2
do
for bs in 384
do
    root=models/MVIT_FULL
    echo $root 
    python seek.py --source_test_path ${root}/outputs_test --output_test_path ${root}/result_test --source_challenge_path ${root}/outputs_challenge --output_challenge_path ${root}/result_challenge \
    --window_start 2 --window_end 5
done 
done 