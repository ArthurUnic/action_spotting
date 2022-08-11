# Action Spotting for SoccerNet Challenge

## Introduction
The [Action Spotting Challenge for SoccerNet Challenge](https://www.soccer-net.org/tasks/action-spotting) is a task to temporally localize the actions in the soccer games. Our model is pre-tained on the Kinetics-700 dataset and fine-tuned on the SoccerNet dataset.

## Requirement
* ffmpeg >= 3.4 for cutting the videos into clips for training.
* python 3.8, tqdm, decord, opencv, pyav, pytorch>=1.9.0, fairscale

## Data Preparation
1. Download all the data from the [official website](https://www.soccer-net.org/download). 
2. run process/1.split_cmd to get the commands to split the videos to obtain the 3 seconds clips. It includes the commands to split semgents with actions and without actions.

```
--video_path: the path of soccernet video
--out_anno_file: the path to save annotation files
--clip_cmds：the path to save the clip_cmds
--target_path: where the clip_cmds will split the videos to 
```

3. Run these commands to obtain the video segments, and put them in a single category directly.
4. The pre-trained model is in https://drive.google.com/file/d/1orjIzC_WOcoKcI08piEmdP5D8K74_VP7/view?usp=sharing. Download it and put it in  model/.

## Training
First we need to add the code file path to PYTHONPATH:
`export PYTHONPATH=$PWD/:$PYTHONPATH;`



Then we begin to train the model on the SoccerNet dataset:
```
mkdir -p exps/soccernet
cd exps/soccernet

BATCH_SIZE=128;
EPOCH=100;
WARMUP_EPOCH=20.0;
CROP_SIZE=224;
MODEL_PATH=../../models/k700_train_mvitV2_full_16x4_fromscratch_e200.pyth;
ANNO_PATH=../../data/annotations/all_with_bg_drop


python ../../tools/run_net.py \
--init_method "tcp://${CHIEF_IP}:${my_port}" --num_shards ${NUM_MACHINE} --shard_id ${INDEX} \
--cfg ../../configs/Aicity/MVITV2_FULL_B_16x4_CONV.yaml \
TRAIN.CHECKPOINT_FILE_PATH ${MODEL_PATH} \
DATA.PATH_PREFIX ../../data/all_clips_3 \
DATA.PATH_TO_DATA_DIR ${ANNO_PATH} \
TRAIN.ENABLE True TRAIN.BATCH_SIZE ${BATCH_SIZE} NUM_GPUS 8 TEST.BATCH_SIZE ${BATCH_SIZE} TEST.ENABLE False \
DATA_LOADER.NUM_WORKERS 8 SOLVER.BASE_LR 1e-4 SOLVER.WARMUP_START_LR 1e-7 \
SOLVER.WARMUP_EPOCHS ${WARMUP_EPOCH} SOLVER.COSINE_END_LR 1e-7 SOLVER.MAX_EPOCH ${EPOCH} LOG_PERIOD 1000 \
TRAIN.CHECKPOINT_PERIOD 1 TRAIN.EVAL_PERIOD 5 USE_TQDM True \
DATA.DECODING_BACKEND decord DATA.TRAIN_CROP_SIZE ${CROP_SIZE} DATA.TEST_CROP_SIZE ${CROP_SIZE} \
TRAIN.AUTO_RESUME True TRAIN.CHECKPOINT_EPOCH_RESET True \
TRAIN.MIXED_PRECISION False MODEL.ACT_CHECKPOINT True \
TENSORBOARD.ENABLE True TENSORBOARD.LOG_DIR tb_log \
MIXUP.ENABLE False MODEL.LOSS_FUNC cross_entropy \
MODEL.DROPOUT_RATE 0.5 MVIT.DROPPATH_RATE 0.4 \
SOLVER.OPTIMIZING_METHOD adamw \
```
Then we get the models in exps/, and we select the epoch_00095.pyth as the model.
Put it in the exps/
## Extract the features
```
cd extract_feature/infer
bash run.sh
bash run_norm.sh
```
The features are seperately called f1 and f2.

## Train NetVLAD++ with thses features
* Change the format of features
  ```
  cd scripts/
  python read_extfeas.py
  ```
* Train the model
```
cd retrain/
bash train_mvit.sh
bash full.sh
```


## Inference
```
python src/main.py --SoccerNet_path=path/to/SoccerNet/ --model_name=MVIT_FULL --test_only
```
