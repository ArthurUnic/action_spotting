# get clips of 3 seconds
# coding=utf-8



import argparse
import os
import decord
from collections import defaultdict
import numpy as np
from SoccerNet.utils import getListGames
import json
from tqdm import tqdm


parser = argparse.ArgumentParser()
# parser.add_argument("--anno_file",default='')
parser.add_argument("--video_path",default='')  # the path of video
parser.add_argument("--resolution", default="800:450", help="ffmpeg -vf scale=")
parser.add_argument('--part',default='all',choices=['train_val','all'],help='train_val or all')
parser.add_argument("--out_anno_file",default='/youtu_pedestrian_detection/zhuhe/soccernet/data/annotations/all_3_with_bg.csv')  #path to save the annotations
parser.add_argument("--clip_length", default="3")  
parser.add_argument("--clip_cmds",default='/youtu_pedestrian_detection/zhuhe/soccernet/split_cmd/all_cut_3_with_bg/cut_bg.sh')  
parser.add_argument("--target_path",default='/youtu_pedestrian_detection/zhuhe/soccernet/data/bg_clips_3')
parser.add_argument("--visibility", default='all', choices=['visible', 'all'])


args=parser.parse_args()

actions={
    0:'Ball out of play',
    1:'Throw-in',
    2:'Foul',
    3:'Indirect free-kick',
    4:'Clearance',
    5:'Shots on target',
    6:'Shots off target',
    7:'Corner',
    8:'Substitution',
    9:'Kick-off',
    10:'Direct free-kick',
    11:'Offside',
    12:'Yellow card',
    13:'Goal',
    14:'Penalty',
    15:'Red card',
    16:'Yellow->red card',
    17:'Background'
}
actions_rever={actions[key]:key for key in actions}


def time2int(time_str):
    # 00:18 to integer seconds
    minutes, seconds = time_str.split(":")
    minutes = int(minutes)
    seconds = int(seconds)
    return minutes*60 + seconds

def int2time(secs):
    # seconds to 00:00
    m, s = divmod(secs, 60)
    if s >= 10.0:
        return "%02d:%.3f" % (m, s)
    else:
        return "%02d:0%.3f" % (m, s)


data = defaultdict(list)  # video_file to segments
users = {}
action_lengths = []

action_id_to_count_shown = defaultdict(int)
action_id_to_count_unshown = defaultdict(int)

vid_to_seg = defaultdict(dict)  # video_file to segment, make sure no overlap
anno_files=[]
# compute some stats
# 1.get all the anno
if args.part=='train_val':
    source_cata=getListGames(split=["train", "valid"])
else:
    source_cata=getListGames(split=["train", "valid", "test"])
for cata in source_cata:
    anno_files.append(os.path.join(args.video_path,cata,'Labels-v2.json'))

    
    
count=0
data_empty = defaultdict(list)
# min_gap=[10000,'a',0,0]
for anno_file in tqdm(anno_files):
    with open(anno_file) as f:
        anno=json.load(f)
    video_root=os.path.join(args.video_path,anno['UrlLocal'].rstrip('/'))
    annotations=anno['annotations']
    last_time=0
    for annotation in annotations:
        if args.visibility=='visible' and annotation['visibility']!='visible':
            continue
        
        half_game,_=annotation['gameTime'].split(' - ')
        video_file=os.path.join(video_root,half_game+'_720p.mkv')
        
        tmp=video_file.split('/')
        video_name=tmp[-2]+'_'+tmp[-1].split('_')[0]
        
        time=int(annotation['position'])/1000
        action_id=actions_rever[annotation['label']]
        
#         #seek the min gap between actions
#         if last_time==0 or time-last_time<=0:
#             last_time=time
#         else:
#             if min_gap[0]>time-last_time:
#                 min_gap[0]=time-last_time
#                 min_gap[1]=video_root
#                 min_gap[2]=time

#             last_time=time
            
        #seek the min gap between actions
        if time==0 or time-last_time<=0 or video_file!=last_video_file: 
            pass
        else:
            if time-last_time > 2* int(args.clip_length):
                count+=2
                mid_time=(time+last_time)/2.0
                data[video_file].append((video_name,(mid_time+last_time)/2.0, 17))
                data[video_file].append((video_name,(mid_time+time)/2.0, 17))
                
                data_empty[video_file].append((video_name,(mid_time+last_time)/2.0, 17))
                data_empty[video_file].append((video_name,(mid_time+time)/2.0, 17))
                
        
        last_time=time
        last_video_file=video_file
        
        # action_id = action_id.strip()
        if annotation['visibility']=='visible':
            action_id_to_count_shown[action_id] += 1
        else:
            action_id_to_count_unshown[action_id] += 1
        

        
        #all of the annotations
        data[video_file].append((video_name,time, action_id))

# write the annotation file
video_clips = []  # video_name _ start  _  end.mp4
with open(args.out_anno_file, "w") as f:

    for video_file in data:

        anno_segs = data[video_file]
        # empty_segs = data_empty[video_file]
        for video_name,time, action_id in anno_segs:
            clip_time=float(args.clip_length)
            
            start=max(0,time-clip_time/2.0)
            end=min(2700,time+clip_time/2.0)
            
            #保证视频片段起码有2.56秒
            if start==0 and end<clip_time: end=clip_time
            if end==2700 and start>2700-clip_time: start=2700-clip_time
            
            
            
            video_id = "%s_%.3f_%.3f.mkv" % (
                video_name, start,end)
            # if action_id == "NA":
            #     action_id = -1
            # elif action_id == "empty":
            #     action_id = -2

            video_clips.append((video_file, int2time(start), int2time(end), video_id))

            f.writelines("%s %d\n" % (video_id.replace(' ','\ '), action_id))


#最终版   -ss放前面，加快速度，直接定位到那里；  -sn放中间（不能放前面），去除字幕

n=len(video_clips)
duration=int2time(int(args.clip_length))
for i in range(n//10000+1):
    with open(args.clip_cmds.rstrip('.sh')+'_'+str(i)+'.sh', "w") as f:
        for ori_video, start, end, target_clip in video_clips[10000*i:min(n,10000*(i+1))]:
            os.makedirs(os.path.join(args.target_path,str(i)),exist_ok=True)
            f.writelines("ffmpeg -nostdin -ss %s -y -i %s -vf scale=%s -c:v libx264 -sn -threads 10 -preset ultrafast -t %s %s\n" % (
                start,
                ori_video.replace(' ','\ '),
                args.resolution,
                duration,
                os.path.join(args.target_path, str(i),target_clip.replace(' ','\ '))))


# Only split by
# write the annotation file
video_clips = []  # video_name _ start  _  end.mp4
# with open(args.out_anno_file, "w") as f:

for video_file in data_empty:

    anno_segs = data_empty[video_file]
    # empty_segs = data_empty[video_file]
    for video_name,time, action_id in anno_segs:
        clip_time=float(args.clip_length)

        start=max(0,time-clip_time/2.0)
        end=min(2700,time+clip_time/2.0)

        #保证视频片段起码有2.56秒
        if start==0 and end<clip_time: end=clip_time
        if end==2700 and start>2700-clip_time: start=2700-clip_time



        video_id = "%s_%.3f_%.3f.mkv" % (
            video_name, start,end)
        # if action_id == "NA":
        #     action_id = -1
        # elif action_id == "empty":
        #     action_id = -2

        video_clips.append((video_file, int2time(start), int2time(end), video_id))

            # f.writelines("%s %d\n" % (video_id.replace(' ','\ '), action_id))


#只分割bg
n=len(video_clips)
duration=int2time(int(args.clip_length))
process_num=5000
for i in tqdm(range(n//process_num+1)):
    with open(args.clip_cmds.rstrip('.sh')+'_'+str(i)+'.sh', "w") as f:
        for ori_video, start, end, target_clip in video_clips[process_num*i:min(n,process_num*(i+1))]:
            os.makedirs(os.path.join(args.target_path,str(i)),exist_ok=True)
            f.writelines("ffmpeg -nostdin -ss %s -y -i %s -vf scale=%s -c:v libx264 -sn -threads 10 -preset ultrafast -t %s %s\n" % (
                start,
                ori_video.replace(' ','\ '),
                args.resolution,
                duration,
                os.path.join(args.target_path, str(i),target_clip.replace(' ','\ '))))