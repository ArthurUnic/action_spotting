import torch 
import os 
import os.path as osp 
import numpy as np 
from tqdm import tqdm 

video_lst='../all.lst'
video_dir='path of your dataset'
with open(video_lst) as f:
    video_files = [os.path.join(video_dir, line.strip())
                    for line in f]

idx = 0 
i=76
# print(video_files[i*5:(i+1)*5])
# assert False 

def get_length(video_fn):
    import cv2

    cap = cv2.VideoCapture(video_fn)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length 


def process_feature(prefix, out):
    idx = 0 
    for video_file in tqdm(video_files):
        video_name_part=video_file.split('/')
        video_name = video_name_part[-2]+'.'+video_name_part[-1].split('_')[0]
        video_fn = osp.join(video_dir, video_file)
        videolen = get_length(video_fn)
        mpath = osp.join('your path/test/extract_feas', video_name + '.pth')
        m2path = osp.join('your path/test/{}'.format(prefix), video_name + '.pth')
        if not osp.exists(m2path):
            print('don\'t exist', idx, idx // 5, video_name)
            idx += 1 
            continue 
        m = torch.load(mpath)
        mvit_feas = m['features']

        m2 = torch.load(m2path)
        m2vit_feas = m2['features']

        pdir = osp.dirname(video_file)
        empath = osp.join(video_dir, pdir, video_name_part[-1].split('_')[0] + '_ResNET_TF2.npy')
        embeds = np.load(empath)
        frate = mvit_feas.shape[0] / (videolen / 25)
        
        framerate =  mvit_feas.shape[0]  / embeds.shape[0] 

        if mvit_feas.shape != m2vit_feas.shape:
            print('shape', idx, idx // 5,  mvit_feas.shape, m2vit_feas.shape)
            idx += 1 
            continue   

        fn = osp.join(video_dir, pdir, video_name_part[-1].split('_')[0] + '_mvit.npy')
        np.save(fn,  mvit_feas.numpy())


        fn2 = osp.join(video_dir, pdir, video_name_part[-1].split('_')[0] + '_mvit{}.npy'.format(out))
        np.save(fn2,  m2vit_feas.numpy())

        idx += 1

# process_feature('extract_feas_norm', out='full')
process_feature('extract_feas_small', out='small')

# m = torch.load('../test/extract_test/2015-08-23 - 15-30 West Brom 2 - 3 Chelsea.2.pth')
# print(m['times'], m['features'].shape)