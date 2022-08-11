import argparse
from SoccerNet.utils import getListGames
import os
import json
from tqdm import tqdm
import numpy as np
from SoccerNet.Evaluation.ActionSpotting import evaluate


parser = argparse.ArgumentParser()
# parser.add_argument("--anno_file",default='')
parser.add_argument("--source_test_path",default='/youtu_pedestrian_detection/chengzhilin/program/soccernet/models/MVIT_WIN5/outputs_test')#输入test文件夹 
parser.add_argument("--output_test_path",default='/youtu_pedestrian_detection/chengzhilin/program/soccernet/process/test')  #筛选后test文件夹

parser.add_argument("--source_challenge_path",default='/youtu_pedestrian_detection/chengzhilin/program/soccernet/models/MVIT_WIN5/outputs_challenge')#输入challenge文件夹 
parser.add_argument("--output_challenge_path", default="/youtu_pedestrian_detection/chengzhilin/program/soccernet/process/challenge")  #输出challenge文件夹
parser.add_argument('--window_start', type=int,   default=4,     help='Size of the chunk (in seconds)' )
parser.add_argument('--window_end', type=int,   default=6,     help='Size of the chunk (in seconds)' )
parser.add_argument('--window_gap',  type=int,   default=1,     help='Size of the chunk (in seconds)' )

args=parser.parse_args()


os.makedirs(args.output_challenge_path,exist_ok='True')

def process_window(window):
    action_dic={}
    res=[]
    for pre in window:
        if pre['label'] not in action_dic:
            action_dic[pre['label']]=pre
        else:
            if float(pre['confidence']) > float(action_dic[pre['label']]['confidence']):
                action_dic[pre['label']]=pre
    for label in action_dic.keys():
        res.append(action_dic[label])
    return res


total_results=[]
window_size=args.window_start
while window_size<=args.window_end:
    print(window_size)

    for cata in getListGames(split="test"):

        file=os.path.join(args.source_test_path,cata,'results_spotting.json')
        with open(file,'r') as f:
            data=json.load(f)
        pres=data['predictions']
        pres.sort(key=lambda x:(int(x['half']),int(x['position'])))

        json_data = {} 
        total_pre={}
        json_data["UrlLocal"]=data['UrlLocal']
        json_data["predictions"]=[]

        last_half=1
        start,end=0,window_size
        window=[]
        for cur_pre,pre in enumerate(pres):


            if start<= int(pre['position'])/1000.0 <end:
                window.append(pre)
            else:
                json_data["predictions"]+=process_window(window)
                window=[pre]
                start+=window_size
                end+=window_size

                if last_half==1 and int(pre['half'])==2:
                    last_half=2
                    start,end=0,window_size

        json_data["predictions"]+=process_window(window)

        os.makedirs(os.path.join(args.output_test_path,cata),exist_ok='True')
        with open(os.path.join(args.output_test_path,cata,'results_spotting.json'),'w') as f:
            json.dump(json_data,f,indent=2)
    
    PATH_DATASET='/youtu_pedestrian_detection/junweil/action_datasets/soccernet'
    PATH_PREDICTIONS=args.output_test_path


    from SoccerNet.Evaluation.ActionSpotting import evaluate
    results = evaluate(SoccerNet_path=PATH_DATASET, Predictions_path=PATH_PREDICTIONS,
                       split="test", version=2, prediction_file="results_spotting.json", metric="tight")

    a_map,a_mAP_visible,a_mAP_unshown=results["a_mAP"],results["a_mAP_visible"],results["a_mAP_unshown"]
    print(a_map)
    total_results.append([window_size,a_map,a_mAP_visible,a_mAP_unshown])
    
    
    window_size+=args.window_gap

total_results.sort(key=lambda x:-x[1])
print(total_results)
window_size=total_results[0][0]

for cata in getListGames(split="challenge"):
    
    file=os.path.join(args.source_challenge_path,cata,'results_spotting.json')
    with open(file,'r') as f:
        data=json.load(f)
    pres=data['predictions']
    pres.sort(key=lambda x:(int(x['half']),int(x['position'])))
    
    json_data = {} 
    total_pre={}
    json_data["UrlLocal"]=data['UrlLocal']
    json_data["predictions"]=[]
    
    last_half=1
    start,end=0,window_size
    window=[]
    for cur_pre,pre in enumerate(pres):

        
        if start<= int(pre['position'])/1000.0 <end:
            window.append(pre)
        else:
            json_data["predictions"]+=process_window(window)
            window=[pre]
            start+=window_size
            end+=window_size
            
            if last_half==1 and int(pre['half'])==2:
                last_half=2
                start,end=0,window_size
         
    json_data["predictions"]+=process_window(window)
    
    os.makedirs(os.path.join(args.output_challenge_path,cata),exist_ok='True')
    with open(os.path.join(args.output_challenge_path,cata,'results_spotting.json'),'w') as f:
        json.dump(json_data,f,indent=2)
    
    
