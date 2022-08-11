import pickle
import os
import json
from tqdm import tqdm
import numpy as np
from SoccerNet.utils import getListGames
import argparse
import copy
import matplotlib.pyplot as plt
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--anno_csv",default='data/annotations/train_val_test_3/test.csv')
parser.add_argument("--pred_pickle_path",default='test/size_224_lr_1e-4_epoch_50_bce')
parser.add_argument('--target',default='submit/size_224_lr_1e-4_epoch_50_bce_max')
parser.add_argument('--dataset',default='test')
parser.add_argument("--num_class", default=17, type=int)
parser.add_argument("--agg_method", default="avg", choices=["avg", "max"])
parser.add_argument("--graph_path", default='graph/size_224_lr_1e-4_epoch_50_bce',
                    help="generate aggregated prediction graph along with gt")


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
    16:'Yellow->red card'
}

video_fps = 25.0
if args.agg_method == "avg":
    aggregate_func = np.mean
elif args.agg_method == "max":
    aggregate_func = np.max
    
classes = range(17)
os.makedirs(args.graph_path, exist_ok=True)

    anno_data = defaultdict(list)  # vid -> list of classes, vid is user_id + 0/1
    all_pickle = {}
    os.makedirs(args.graph_path, exist_ok=True) 
    for line in open(args.anno_csv, "r").readlines():
        video_file, action_class = line.strip().rsplit(' ',1)
        file_id, half, t0, t1 = video_file.rstrip('.mkv').rsplit("_",3)
        #assert file_id[-1] in ["0", "1"], file_id  # could be 3/4
        vid = "%s.%s" % (file_id, file_id[-1])
        # {'24491_1': [('Rightside_user_id_24491_1', '24491', '0', '17', '0'),
        #('Rightside_user_id_24491_1', '24491', '18', '45', '3'),
        #('Rightside_user_id_24491_1', '24491', '45', '54', '14'),
        #('Rightside_user_id_24491_1', '24491', '74', '105', '2')
        anno_data[vid].append((file_id, half, float(t0), float(t1), int(action_class)))

        all_pickle[file_id] = 1

def frame2time(frame,fps):
    second=float(frame/fps)
    m,s=divmod(round(second),60)
    
    time='%d:%d'%(m,s)
        
    return time,int(1000*second)

def aggregate_predictions(pred_list, aggregate_func, num_class):

    # frame_idx are 0-indexed
    frame_idxs = [t[0] for t in pred_list]
    frame_idxs += [t[1] for t in pred_list]
    min_frame_idx = min(frame_idxs)
    max_frame_idx = max(frame_idxs)
    frame_num = max_frame_idx - min_frame_idx

    # construct a list per frame_idx for all scores
    # assume scores are between 0.0 to 1.
    # len == num_frame, each is a list of predictions of all classes
    score_list_per_frame = [
        [np.zeros((num_class), dtype="float32")]
        for i in range(frame_num)]

    # t1- t0 == 64
    for t0, t1, cls_data in pred_list:
        for t in range(t0, t1):
            save_idx = t - min_frame_idx
            score = cls_data  # num_class
            assert len(score) == num_class
            score_list_per_frame[save_idx].append(score)

    # aggregate the scores at each frame idx
    # get the chunks in (t0, t1) with scores >= thres
    agg_score_per_frame = []
    for i in range(len(score_list_per_frame)):
        # stack all the scores first
        if len(score_list_per_frame[i]) > 1:
            score_list_per_frame[i].pop(0)  # remove the zero padding
        # [K, num_class]
        stacked_scores = np.vstack(score_list_per_frame[i])
        # [num_class]
        this_frame_scores = aggregate_func(stacked_scores, axis=0)
        this_frame_idx = min_frame_idx + i
        agg_score_per_frame.append(this_frame_scores)

    return np.vstack(agg_score_per_frame)  # [num_frame, num_class]

pickle_data = {} 
for video_path in tqdm(getListGames(split=args.dataset)):
    json_data = {} 
    pre={}
    json_data["UrlLocal"]=video_path
    json_data["predictions"]=[]
    
    
    
    target_path=os.path.join(args.target,video_path)
    os.makedirs(target_path,exist_ok=True)
    video_name=video_path.rsplit('/',1)[-1]
    
    pred_file=os.path.join(args.pred_pickle_path,video_name+'.1.pkl')
    with open(pred_file, "rb") as f:
        pred = pickle.load(f)
    
    pickle_data[video_name+'.1'] =  aggregate_predictions(pred, aggregate_func, args.num_class)
    # pkl_data = aggregate_predictions(pred, aggregate_func, args.num_class)
    
#     for frame in range(6,len(pkl_data),13):
#         for action_id in range(17):
#             if pkl_data[frame][action_id]<0.5: continue
            
#             time,millsec=frame2time(frame,25.0)
#             pre={}
#             pre['gameTime']="1 - "+time
#             pre['label']=actions[action_id]
#             pre['position']=str(millsec)
#             pre['half']="1"
#             pre['confidence']=str(pkl_data[frame][action_id])
#             json_data["predictions"].append(copy.copy(pre))
            
           
    
    
    
    
    pred_file=os.path.join(args.pred_pickle_path,video_name+'.2.pkl')
    with open(pred_file, "rb") as f:
        pred = pickle.load(f)
    pickle_data[video_name+'.2'] =  aggregate_predictions(pred, aggregate_func, args.num_class)
    # pickle_data[video_path].append(aggregate_predictions(pred, aggregate_func, args.num_class))
    
#     for frame in range(6,len(pkl_data),13):
#         for action_id in range(17):
#             if pkl_data[frame][action_id]<0.5: continue
            
#             time,millsec=frame2time(frame,25.0)
#             pre={}
#             pre['gameTime']="2 - "+time
#             pre['label']=actions[action_id]
#             pre['position']=str(millsec)
#             pre['half']="2"
#             pre['confidence']=str(pkl_data[frame][action_id])
#             json_data["predictions"].append(copy.copy(pre))
    

#     with open(os.path.join(target_path,'results_spotting.json'),'w') as f:
#         json.dump(json_data,f,indent=2)
    # break

for video_path in tqdm(getListGames(split=args.dataset)):
    json_data = {} 
    pre={}
    json_data["UrlLocal"]=video_path
    json_data["predictions"]=[]
    
    
    
    target_path=os.path.join(args.target,video_path)
    os.makedirs(target_path,exist_ok=True)
    video_name=video_path.rsplit('/',1)[-1]
    
    pkl_data=pickle_data[video_name+'.1']
    for frame in range(6,len(pkl_data),13):
        for action_id in range(17):
            if pkl_data[frame][action_id]<0.5: continue

            time,millsec=frame2time(frame,25.0)
            pre={}
            pre['gameTime']="1 - "+time
            pre['label']=actions[action_id]
            pre['position']=str(millsec)
            pre['half']="1"
            pre['confidence']=str(pkl_data[frame][action_id])
            json_data["predictions"].append(copy.copy(pre))
     
    
    pkl_data=pickle_data[video_name+'.2']
    for frame in range(6,len(pkl_data),13):
        for action_id in range(17):
            if pkl_data[frame][action_id]<0.5: continue

            time,millsec=frame2time(frame,25.0)
            pre={}
            pre['gameTime']="1 - "+time
            pre['label']=actions[action_id]
            pre['position']=str(millsec)
            pre['half']="1"
            pre['confidence']=str(pkl_data[frame][action_id])
            json_data["predictions"].append(copy.copy(pre))