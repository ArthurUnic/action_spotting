import argparse
from SoccerNet.utils import getListGames
import os
import json
from tqdm import tqdm
import numpy as np
from SoccerNet.Evaluation.ActionSpotting import evaluate
from collections import defaultdict

parser = argparse.ArgumentParser()
# parser.add_argument("--anno_file",default='')
parser.add_argument("--source_test_path",default='') 
parser.add_argument("--output_test_path",default='')  

parser.add_argument("--source_challenge_path",default='') 
parser.add_argument("--output_challenge_path", default="")  
# parser.add_argument('--window_start', type=int,   default=10,     help='Size of the chunk (in seconds)' )
# parser.add_argument('--window_end', type=int,   default=50,     help='Size of the chunk (in seconds)' )
# parser.add_argument('--window_gap',  type=int,   default=0.5,     help='Size of the chunk (in seconds)' )

parser.add_argument('--window_size',  type=int,   default=26.5,     help='Size of the chunk (in seconds)' )

parser.add_argument('--thred_start', type=int,   default=0,     help='Size of the chunk (in seconds)' )
parser.add_argument('--thred_end', type=int,   default=0.9,     help='Size of the chunk (in seconds)' )
parser.add_argument('--thred_gap',  type=int,   default=0.1,     help='Size of the chunk (in seconds)' )

args=parser.parse_args()


os.makedirs(args.output_challenge_path,exist_ok='True')

thred={
    'Ball out of play':0,
    'Throw-in':0,
    'Foul':0,
    'Indirect free-kick':0,
    'Clearance':0,
    'Shots on target':0,
    'Shots off target':0,
    'Corner':0,
    'Substitution':0,
    'Kick-off':0,
    'Direct free-kick':0,
    'Offside':0,
    'Yellow card':0,
    'Goal':0,
    'Penalty':0,
    'Red card':0,
    'Yellow->red card':0,
    'Background':0
}

def process_window(window):
    action_dic={}
    res=[]
    for pre in window:
        if float(pre['confidence']) <  thred[pre['label']]: continue
        
        if pre['label'] not in action_dic:
            action_dic[pre['label']]=pre
        else:
            if float(pre['confidence']) > float(action_dic[pre['label']]['confidence']):
                action_dic[pre['label']]=pre
    for label in action_dic.keys():
        res.append(action_dic[label])
    return res

total_results=defaultdict(list)
window_size=args.window_size

for action in tqdm(thred.keys()):
    thred[action]=args.thred_start
    for tmp in [0.1, 1e-2, 1e-4]:
        thred[action] = tmp        
        print(thred[action])
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

            for pre in pres:
                preact = pre['label']
                precon = float(pre['confidence'])
                if preact != action or precon >= thred[action]:
                    json_data['predictions'].append(pre)                   

            os.makedirs(os.path.join(args.output_test_path,cata),exist_ok='True')
            with open(os.path.join(args.output_test_path,cata,'results_spotting.json'),'w') as f:
                json.dump(json_data,f,indent=2)
        print( len(json_data["predictions"]), len(pres))
        PATH_DATASET='path of your dataset'
        PATH_PREDICTIONS=args.output_test_path


        from SoccerNet.Evaluation.ActionSpotting import evaluate
        results = evaluate(SoccerNet_path=PATH_DATASET, Predictions_path=PATH_PREDICTIONS,
                           split="test", version=2, prediction_file="results_spotting.json", metric="tight")

        a_map,a_mAP_visible,a_mAP_unshown=results["a_mAP"],results["a_mAP_visible"],results["a_mAP_unshown"]
        print(a_map)
        # total_results.append([window_size,a_map,a_mAP_visible,a_mAP_unshown])
        total_results[action].append([thred[action],a_map,a_mAP_visible,a_mAP_unshown])

        thred[action]+=args.thred_gap
        
    thred[action]=0

print(total_results[action])
for action in (thred.keys()):               
    total_results[action].sort(key=lambda x:-x[1])
    thred[action]=total_results[action][0][0]
print(thred)             
for cata in tqdm(getListGames(split="challenge")):
    
    file=os.path.join(args.source_challenge_path,cata,'results_spotting.json')
    with open(file,'r') as f:
        data=json.load(f)
    pres=data['predictions']
    pres.sort(key=lambda x:(int(x['half']),int(x['position'])))
    
    json_data = {} 
    total_pre={}
    json_data["UrlLocal"]=data['UrlLocal']
    json_data["predictions"]=[]
    
    print(thred[action])
    for pre in pres:
        preact = pre['label']
        precon = float(pre['confidence'])
        if preact != action or precon >= thred[action]:
            json_data['predictions'].append(pre)    
    
    os.makedirs(os.path.join(args.output_challenge_path,cata),exist_ok='True')
    with open(os.path.join(args.output_challenge_path,cata,'results_spotting.json'),'w') as f:
        json.dump(json_data,f,indent=2)
    
    
