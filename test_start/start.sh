DANCE_NAME=$1
pid=001
cid=12
HOSTDIR=/home/sunyangtian/sytDisk/PAMI/
HOSTDIR=../../PAMI
DATASETDIR=../../DanceDataset
py=../../miniconda3/envs/PAMI/bin/python

${py} ./test.py \
    --name ${DANCE_NAME}_18Feature_ori_3StaticEpoch10_newPreTrain  \
    --checkpoints_dir ${HOSTDIR}/checkpoints \
	--pose_path ${DATASETDIR}/2_openpose \
	--pose_tgt_path ${DATASETDIR}/${DANCE_NAME}/openpose_json \
    --img_path ${DATASETDIR}/${DANCE_NAME}/${DANCE_NAME} \
    --no_flip \
    --instance_feat \
    --input_nc 3 \
    --loadSize 512 \
    --resize_or_crop resize \
    --results_dir ${HOSTDIR}/Result/test/src_${pid}_${cid}/tgt_${DANCE_NAME} \
    --which_epoch 10 \

