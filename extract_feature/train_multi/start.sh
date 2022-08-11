export NCCL_IB_DISABLE=1;
export my_port=23456;
export PYTHONPATH=$PWD/:$PYTHONPATH:/youtu_pedestrian_detection/zhuhe/soccernet
dir=/youtu_pedestrian_detection/zhuhe/soccernet/exps/train_chengzhi 
cd $dir 

BATCH_SIZE=512;
NUM_MACHINE=64;
EPOCH=50;
WARMUP_EPOCH=10.0;
CROP_SIZE=448;
# CROP_SIZE=224;
MODEL_PATH=../../models/k700_train_mvitV2_full_16x4_fromscratch_e200.pyth;

python ../../tools/run_net.py \
--init_method "tcp://${CHIEF_IP}:${my_port}" --num_shards ${NUM_MACHINE} --shard_id ${INDEX} \
--cfg ../../configs/Aicity/MVITV2_FULL_B_16x4_CONV_448.yaml \
TRAIN.CHECKPOINT_FILE_PATH ${MODEL_PATH} \
DATA.PATH_PREFIX ../../data/all_clips_3 \
DATA.PATH_TO_DATA_DIR ../../data/annotations/train_val_test_3 \
TRAIN.ENABLE True TRAIN.BATCH_SIZE ${BATCH_SIZE} NUM_GPUS 8 TEST.BATCH_SIZE ${BATCH_SIZE} TEST.ENABLE False \
DATA_LOADER.NUM_WORKERS 8 SOLVER.BASE_LR 1e-4 SOLVER.WARMUP_START_LR 1e-7 \
SOLVER.WARMUP_EPOCHS ${WARMUP_EPOCH} SOLVER.COSINE_END_LR 1e-7 SOLVER.MAX_EPOCH ${EPOCH} LOG_PERIOD 1000 \
TRAIN.CHECKPOINT_PERIOD 5 TRAIN.EVAL_PERIOD 5 USE_TQDM True \
DATA.DECODING_BACKEND decord DATA.TRAIN_CROP_SIZE ${CROP_SIZE} DATA.TEST_CROP_SIZE ${CROP_SIZE} \
TRAIN.AUTO_RESUME True TRAIN.CHECKPOINT_EPOCH_RESET True \
TRAIN.MIXED_PRECISION False MODEL.ACT_CHECKPOINT True \
TENSORBOARD.ENABLE False TENSORBOARD.LOG_DIR tb_log \
MIXUP.ENABLE False MODEL.LOSS_FUNC cross_entropy \
MODEL.DROPOUT_RATE 0.5 MVIT.DROPPATH_RATE 0.4 \
SOLVER.OPTIMIZING_METHOD adamw \
1>${dir}/log.txt 2>${dir}/err.txt 
