DANCE_NAME=$1
DATASETDIR=../DanceDataset
# HOSTDIR=/home/sunyangtian/sytDisk/PAMI
HOSTDIR=../HOSTDIR

py=../miniconda3/envs/JittorEnv/bin/python

${py} ./train.py  \
--name ${DANCE_NAME}_multiPoseJittor_debug \
--batchSize 1  \
--gpu_ids 0  \
--use_laplace  \
--checkpoints_dir ${HOSTDIR}/checkpoints_testJittor  	  \
--pose_path ${DATASETDIR}/${DANCE_NAME}/openpose_json  	 	\
--mask_path ${DATASETDIR}/${DANCE_NAME}/mask 	  	\
--img_path ${DATASETDIR}/${DANCE_NAME}/${DANCE_NAME} 	  	\
--densepose_path ${DATASETDIR}/${DANCE_NAME}/densepose 	  	\
--bg_path ${DATASETDIR}/${DANCE_NAME}/bg.jpg 	  	\
--texture_path ${DATASETDIR}/${DANCE_NAME}/texture.jpg 	  	\
--flow_path ${DATASETDIR}/${DANCE_NAME}/flow    \
--flow_inv_path ${DATASETDIR}/${DANCE_NAME}/flow_inv \
--no_flip  \
--instance_feat  \
--input_nc 3  \
--loadSize 512  \
--resize_or_crop resize  \
--load_pretrain_TransG ${DATASETDIR}/  \
--which_epoch_TransG 48   \
--lambda_L2 500  \
--lambda_UV 1000  \
--lambda_Prob 10  \
--use_densepose_loss  \
--save_epoch_freq 10  \
--data_ratio 0.9 \
--lambda_Temp 500 \
--tf_log \