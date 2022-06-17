DANCE_NAME=$1
# HOSTDIR=/home/sunyangtian/sytDisk/PAMI
HOSTDIR=../../PAMI
DATASETDIR=../../DanceDataset
py=../../miniconda3/envs/PAMI/bin/python

${py} ./test.py \
    --name ${DANCE_NAME}_multiPose  \
    --checkpoints_dir ${HOSTDIR}/checkpoints \
	--pose_path ${DATASETDIR}/${DANCE_NAME}/openpose_json \
	--pose_tgt_path ${DATASETDIR}/${DANCE_NAME}/openpose_json \
    --img_path ${DATASETDIR}/${DANCE_NAME}/${DANCE_NAME} \
    --no_flip \
    --instance_feat \
    --input_nc 3 \
    --loadSize 512 \
    --resize_or_crop resize \
    --results_dir ${HOSTDIR}/Result/val/${DANCE_NAME} \
    --which_epoch 50 \
    --data_ratio 0.9 \
    --test_val