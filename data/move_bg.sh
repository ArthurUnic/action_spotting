for((i=0;i<=34;i++));
do
# cd /youtu_pedestrian_detection/zhuhe/soccernet/data/bg_clips_3;
# mv $i /youtu_pedestrian_detection/zhuhe/soccernet/data/all_clips_3/;

cd /youtu_pedestrian_detection/zhuhe/soccernet/data/all_clips_3/$i;
mv * ..;
done