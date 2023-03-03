#!/bin/bash

#Experiments information
experiment_name=2080ti_002_tune
OUTPUTFOLDER=EXPS

DEBUG=0
GPUS=1

NUM_SANITY_VAL_STEPS=0
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fernie
echo source ~/anaconda3/etc/profile.d/conda.sh
echo conda activate fernie

#Dataloader
BATCH_SIZE=320
NUM_WORKERS=10

#Optimization
WEIGHT_DECAY=1e-4
LR=1e-4

#Model structure
CF=592
H=72
DATM=200
DAMINO=200
DCHRG=200
DDIST=200
POOL_TYPE=max
ACTIVATION=relu
DP_RATE=0
BATCHNORM=1

echo \#\#Some global setting
data_root=/y/Aurora/Fernie/data/structure_based_data
ROOT_OUTPUT_DIR=/y/Aurora/Fernie/${OUTPUTFOLDER}/${EXPERIMENT_NAME}

echo \#GPUS=$GPUS
echo \#DEBUG=${DEBUG}
echo \#NUM_SANITY_VAL_STEPS=${NUM_SANITY_VAL_STEPS}
echo
echo \#DATA_ROOT=${DATA_ROOT}
echo \#ROOT_OUTPUT_DIR=${ROOT_OUTPUT_DIR}
echo \#BATCH_SIZE=${BATCH_SIZE}
echo \#NUM_WORKERS=${NUM_WORKERS}

echo
echo \#\#Optimization
echo \#WEIGHT_DECAY=${WEIGHT_DECAY}
echo \#LR=${LR}

echo
echo \#\#Model structure
echo \#CF=${CF}
echo \#H=${H}
echo \#DATM=${DATM}
echo \#DAMINO=${DAMINO}
echo \#DCHRG=${DCHRG}
echo \#DDIST=${DDIST}
echo \#POOL_TYPE=${POOL_TYPE}
echo \#ACTIVATION=${ACTIVATION}
echo \#DP_RATE=${DP_RATE}
echo \#BATCHNORM=${BATCHNORM}

extra_argument="--batch_size ${BATCH_SIZE}\
    --num_workers ${NUM_WORKERS}\
    --weight_decay ${WEIGHT_DECAY}\
    --lr ${LR}\
    --cf ${CF}\
    --h ${H}\
    --datm ${DATM}\
    --damino ${DAMINO}\
    --dchrg ${DCHRG}\
    --ddist ${DDIST}\
    --pool_type ${POOL_TYPE}\
    --activation ${ACTIVATION}\
    --dp_rate ${DP_RATE}\
    --batchnorm ${BATCHNORM}\
    --num_sanity_val_steps ${NUM_SANITY_VAL_STEPS}\
    --gpus ${GPUS}\
    --debug ${DEBUG}"

echo
echo \#------------------------------------------
echo \#Cross validation
echo \#------------------------------------------

echo
echo \#------------------------------------------
echo \#Clustered 3-fold cross-validation on DUD-E
echo \#------------------------------------------
TRAINING_TYPE=holdout
OUTPUT_DIR=cv

DATASET=DUD-E 
echo "mkdir -p ${ROOT_OUTPUT_DIR}/${DATASET}/performances &&\ " 
echo "mkdir -p ${ROOT_OUTPUT_DIR}/${DATASET}/scores &&\ " 

FOLD=0 
TMP_OUTPUT=${ROOT_OUTPUT_DIR}/${OUTPUT_DIR}/${DATASET}/fold_${FOLD} 
command="./main.py --working_mode train --training_type ${TRAINING_TYPE}\
    --train_file ${DATA_ROOT}/${DATASET}_3_fold/train_${FOLD}.pickle\
    --valid_file ${DATA_ROOT}/${DATASET}_3_fold/test_${FOLD}.pickle\
    --test_dir ${DATA_ROOT}/${DATASET}_3_fold/test_${FOLD}\
    -o ${TMP_OUTPUT}\
    ${extra_argument} &&\ "
echo $command
command="./eval.py -d ${TMP_OUTPUT}/scores -o ${TMP_OUTPUT}/performances -m fernie -n ${DATASET}_fold_${FOLD} &&\ "
echo $command
echo "mv ${TMP_OUTPUT}/scores/* ${ROOT_OUTPUT_DIR}/${DATASET}/scores &&\ "

echo
FOLD=1 
TMP_OUTPUT=${ROOT_OUTPUT_DIR}/${OUTPUT_DIR}/${DATASET}/fold_${FOLD} 
command="./main.py --working_mode train --training_type ${TRAINING_TYPE}\
    --train_file ${DATA_ROOT}/${DATASET}_3_fold/train_${FOLD}.pickle\
    --valid_file ${DATA_ROOT}/${DATASET}_3_fold/test_${FOLD}.pickle\
    --test_dir ${DATA_ROOT}/${DATASET}_3_fold/test_${FOLD}\
    -o ${TMP_OUTPUT}\
    ${extra_argument} &&\ " 
echo $command
command="./eval.py -d ${TMP_OUTPUT}/scores -o ${TMP_OUTPUT}/performances -m fernie -n ${DATASET}_fold_${FOLD} &&\ " 
echo $command
echo "mv ${TMP_OUTPUT}/scores/* ${ROOT_OUTPUT_DIR}/${DATASET}/scores &&\ " 

echo
FOLD=2 
TMP_OUTPUT=${ROOT_OUTPUT_DIR}/${OUTPUT_DIR}/${DATASET}/fold_${FOLD} 
command="./main.py --working_mode train --training_type ${TRAINING_TYPE}\
    --train_file ${DATA_ROOT}/${DATASET}_3_fold/train_${FOLD}.pickle\
    --valid_file ${DATA_ROOT}/${DATASET}_3_fold/test_${FOLD}.pickle\
    --test_dir ${DATA_ROOT}/${DATASET}_3_fold/test_${FOLD}\
    -o ${TMP_OUTPUT}\
    ${extra_argument} &&\ "
echo $command
command="./eval.py -d ${TMP_OUTPUT}/scores -o ${TMP_OUTPUT}/performances -m fernie -n ${DATASET}_fold_${FOLD} &&\ " 
echo $command
echo "mv ${TMP_OUTPUT}/scores/* ${ROOT_OUTPUT_DIR}/${DATASET}/scores &&\ " 

echo
command="./eval.py -d ${ROOT_OUTPUT_DIR}/${DATASET}/scores -o ${ROOT_OUTPUT_DIR}/${DATASET}/performances -m fernie -n ${DATASET} &&\ " 
echo $command

echo
echo \#------------------------------------------
echo \#Clustered 3-fold cross-validation on Kernie
echo \#------------------------------------------
DATASET=Kernie 
echo "mkdir -p ${ROOT_OUTPUT_DIR}/${DATASET}/performances &&\ "
echo "mkdir -p ${ROOT_OUTPUT_DIR}/${DATASET}/scores &&\ "

FOLD=0 
TMP_OUTPUT=${ROOT_OUTPUT_DIR}/${OUTPUT_DIR}/${DATASET}/fold_${FOLD} 
command="./main.py --working_mode train --training_type ${TRAINING_TYPE}\
    --train_file ${DATA_ROOT}/${DATASET}_3_fold/train_${FOLD}.pickle\
    --valid_file ${DATA_ROOT}/${DATASET}_3_fold/test_${FOLD}.pickle\
    --test_dir ${DATA_ROOT}/${DATASET}_3_fold/test_${FOLD}\
    -o ${TMP_OUTPUT}\
    ${extra_argument} &&\ "
echo $command
command="./eval.py -d ${TMP_OUTPUT}/scores -o ${TMP_OUTPUT}/performances -m fernie -n ${DATASET}_fold_${FOLD} &&\ "
echo $command
echo "mv ${TMP_OUTPUT}/scores/* ${ROOT_OUTPUT_DIR}/${DATASET}/scores &&\ " 

echo
FOLD=1 
TMP_OUTPUT=${ROOT_OUTPUT_DIR}/${OUTPUT_DIR}/${DATASET}/fold_${FOLD} 
command="./main.py --working_mode train --training_type ${TRAINING_TYPE}\
    --train_file ${DATA_ROOT}/${DATASET}_3_fold/train_${FOLD}.pickle\
    --valid_file ${DATA_ROOT}/${DATASET}_3_fold/test_${FOLD}.pickle\
    --test_dir ${DATA_ROOT}/${DATASET}_3_fold/test_${FOLD}\
    -o ${TMP_OUTPUT}\
    ${extra_argument} &&\ "
echo $command
command="./eval.py -d ${TMP_OUTPUT}/scores -o ${TMP_OUTPUT}/performances -m fernie -n ${DATASET}_fold_${FOLD} &&\ "
echo $command
echo "mv ${TMP_OUTPUT}/scores/* ${ROOT_OUTPUT_DIR}/${DATASET}/scores &&\ " 

echo
FOLD=2 
TMP_OUTPUT=${ROOT_OUTPUT_DIR}/${OUTPUT_DIR}/${DATASET}/fold_${FOLD} 
command="./main.py --working_mode train --training_type ${TRAINING_TYPE}\
    --train_file ${DATA_ROOT}/${DATASET}_3_fold/train_${FOLD}.pickle\
    --valid_file ${DATA_ROOT}/${DATASET}_3_fold/test_${FOLD}.pickle\
    --test_dir ${DATA_ROOT}/${DATASET}_3_fold/test_${FOLD}\
    -o ${TMP_OUTPUT}\
    ${extra_argument} &&\ "
echo $command
command="./eval.py -d ${TMP_OUTPUT}/scores -o ${TMP_OUTPUT}/performances -m fernie -n ${DATASET}_fold_${FOLD} &&\ "
echo $command
echo "mv ${TMP_OUTPUT}/scores/* ${ROOT_OUTPUT_DIR}/${DATASET}/scores &&\ " 

echo
command="./eval.py -d ${ROOT_OUTPUT_DIR}/${DATASET}/scores -o ${ROOT_OUTPUT_DIR}/${DATASET}/performances -m fernie -n ${DATASET} &&\ "
echo $command

echo
echo \#------------------------------------------
echo \#Test on MUV
echo \#------------------------------------------

echo
echo \#------------------------------------------
echo \# 1 Training on DUD-E
echo \#------------------------------------------
TEST_DIR=${DATA_ROOT}/MUV_feature/single_pose 
DATASET=DUD-E
TESTSET=MUV
OUTPUT_DIR=MUV

TRAIN_FILE=DUD-E
TMP_OUTPUT=${ROOT_OUTPUT_DIR}/${OUTPUT_DIR}/${TRAIN_FILE}
command="./main.py --working_mode train --training_type ${TRAINING_TYPE}\
    --train_file ${DATA_ROOT}/${DATASET}_feature/${TRAIN_FILE}.single_pose.undersample.pickle\
    --test_dir ${TEST_DIR}\
    -o ${TMP_OUTPUT}\
    ${extra_argument} &&\ "
echo $command
command="./eval.py -d ${TMP_OUTPUT}/scores -o ${TMP_OUTPUT}/performances -m fernie -n ${TESTSET} &&\ "
echo $command

echo
echo \#------------------------------------------
echo \# 2 Training on DUD-E.Kinase
echo \#------------------------------------------
TRAIN_FILE=DUD-E.Kinase 
TMP_OUTPUT=${ROOT_OUTPUT_DIR}/${OUTPUT_DIR}/${TRAIN_FILE}
command="./main.py --working_mode train --training_type ${TRAINING_TYPE}\
    --train_file ${DATA_ROOT}/${DATASET}_feature/${TRAIN_FILE}.single_pose.undersample.pickle\
    --test_dir ${TEST_DIR}\
    -o ${TMP_OUTPUT}\
    ${extra_argument} &&\ "
echo $command
command="./eval.py -d ${TMP_OUTPUT}/scores -o ${TMP_OUTPUT}/performances -m fernie -n ${TESTSET} &&\ "
echo $command

echo
echo \#------------------------------------------
echo \# 3 Training on DUD-E.Protease
echo \#------------------------------------------
TRAIN_FILE=DUD-E.Protease 
TMP_OUTPUT=${ROOT_OUTPUT_DIR}/${OUTPUT_DIR}/${TRAIN_FILE}
command="./main.py --working_mode train --training_type ${TRAINING_TYPE}\
    --train_file ${DATA_ROOT}/${DATASET}_feature/${TRAIN_FILE}.single_pose.undersample.pickle\
    --test_dir ${TEST_DIR}\
    -o ${TMP_OUTPUT}\
    ${extra_argument} &&\ "
echo $command
command="./eval.py -d ${TMP_OUTPUT}/scores -o ${TMP_OUTPUT}/performances -m fernie -n ${TESTSET} &&\ "
echo $command

echo
echo \#------------------------------------------
echo \# 4 Training on DUD-E.Cytochrome_P450
echo \#------------------------------------------
TRAIN_FILE=DUD-E.Cytochrome_P450 
TMP_OUTPUT=${ROOT_OUTPUT_DIR}/${OUTPUT_DIR}/${TRAIN_FILE}
command="./main.py --working_mode train --training_type ${TRAINING_TYPE}\
    --train_file ${DATA_ROOT}/${DATASET}_feature/${TRAIN_FILE}.single_pose.undersample.pickle\
    --test_dir ${TEST_DIR}\
    -o ${TMP_OUTPUT}\
    ${extra_argument} &&\ "
echo $command
command="./eval.py -d ${TMP_OUTPUT}/scores -o ${TMP_OUTPUT}/performances -m fernie -n ${TESTSET} &&\ "
echo $command

echo
echo \#------------------------------------------
echo \# 5 Training on DUD-E.GPCR
echo \#------------------------------------------
TRAIN_FILE=DUD-E.GPCR 
TMP_OUTPUT=${ROOT_OUTPUT_DIR}/${OUTPUT_DIR}/${TRAIN_FILE}
command="./main.py --working_mode train --training_type ${TRAINING_TYPE}\
    --train_file ${DATA_ROOT}/${DATASET}_feature/${TRAIN_FILE}.single_pose.undersample.pickle\
    --test_dir ${TEST_DIR}\
    -o ${TMP_OUTPUT}\
    ${extra_argument} &&\ "
echo $command
command="./eval.py -d ${TMP_OUTPUT}/scores -o ${TMP_OUTPUT}/performances -m fernie -n ${TESTSET} &&\ "
echo $command

echo
echo \#------------------------------------------
echo \# 6 Training on DUD-E.Ion_Channel
echo \#------------------------------------------
TRAIN_FILE=DUD-E.Ion_Channel 
TMP_OUTPUT=${ROOT_OUTPUT_DIR}/${OUTPUT_DIR}/${TRAIN_FILE}
command="./main.py --working_mode train --training_type ${TRAINING_TYPE}\
    --train_file ${DATA_ROOT}/${DATASET}_feature/${TRAIN_FILE}.single_pose.undersample.pickle\
    --test_dir ${TEST_DIR}\
    -o ${TMP_OUTPUT}\
    ${extra_argument} &&\ "
echo $command
command="./eval.py -d ${TMP_OUTPUT}/scores -o ${TMP_OUTPUT}/performances -m fernie -n ${TESTSET} &&\ "
echo $command

echo
echo \#------------------------------------------
echo \# 7 Training on DUD-E.Miscellaneous
echo \#------------------------------------------
TRAIN_FILE=DUD-E.Miscellaneous 
TMP_OUTPUT=${ROOT_OUTPUT_DIR}/${OUTPUT_DIR}/${TRAIN_FILE}
command="./main.py --working_mode train --training_type ${TRAINING_TYPE}\
    --train_file ${DATA_ROOT}/${DATASET}_feature/${TRAIN_FILE}.single_pose.undersample.pickle\
    --test_dir ${TEST_DIR}\
    -o ${TMP_OUTPUT}\
    ${extra_argument} &&\ "
echo $command
command="./eval.py -d ${TMP_OUTPUT}/scores -o ${TMP_OUTPUT}/performances -m fernie -n ${TESTSET} &&\ "
echo $command

echo
echo \#------------------------------------------
echo \# 8 Training on DUD-E.Nuclear_Receptor
echo \#------------------------------------------
TRAIN_FILE=DUD-E.Nuclear_Receptor 
TMP_OUTPUT=${ROOT_OUTPUT_DIR}/${OUTPUT_DIR}/${TRAIN_FILE}
command="./main.py --working_mode train --training_type ${TRAINING_TYPE}\
    --train_file ${DATA_ROOT}/${DATASET}_feature/${TRAIN_FILE}.single_pose.undersample.pickle\
    --test_dir ${TEST_DIR}\
    -o ${TMP_OUTPUT}\
    ${extra_argument} &&\ "
echo $command
command="./eval.py -d ${TMP_OUTPUT}/scores -o ${TMP_OUTPUT}/performances -m fernie -n ${TESTSET} &&\ "
echo $command

echo
echo \#------------------------------------------
echo \# 9 Training on DUD-E.Other_Enzymes
echo \#------------------------------------------
TRAIN_FILE=DUD-E.Other_Enzymes 
TMP_OUTPUT=${ROOT_OUTPUT_DIR}/${OUTPUT_DIR}/${TRAIN_FILE}
command="./main.py --working_mode train --training_type ${TRAINING_TYPE}\
    --train_file ${DATA_ROOT}/${DATASET}_feature/${TRAIN_FILE}.single_pose.undersample.pickle\
    --test_dir ${TEST_DIR}\
    -o ${TMP_OUTPUT}\
    ${extra_argument} &&\ "
echo $command
command="./eval.py -d ${TMP_OUTPUT}/scores -o ${TMP_OUTPUT}/performances -m fernie -n ${TESTSET} &&\ "
echo $command

echo
echo \#------------------------------------------
echo \# 10 Training on Kernie
echo \#------------------------------------------
DATASET=Kernie 
TRAIN_FILE=Kernie 
TMP_OUTPUT=${ROOT_OUTPUT_DIR}/${OUTPUT_DIR}/${TRAIN_FILE}
command="./main.py --working_mode train --training_type ${TRAINING_TYPE}\
    --train_file ${DATA_ROOT}/${DATASET}_feature/${TRAIN_FILE}.single_pose.undersample.pickle\
    --test_dir ${TEST_DIR}\
    -o ${TMP_OUTPUT}\
    ${extra_argument} &&\ "
echo $command
command="./eval.py -d ${TMP_OUTPUT}/scores -o ${TMP_OUTPUT}/performances -m fernie -n ${TESTSET} &&\ "
echo $command

echo
echo \#------------------------------------------
echo \# 11 Training on Kernie without MUV targets
echo \#------------------------------------------
DATASET=Kernie 
TRAIN_FILE=Kernie-MUV 
TMP_OUTPUT=${ROOT_OUTPUT_DIR}/${OUTPUT_DIR}/${TRAIN_FILE}
command="./main.py --working_mode train --training_type ${TRAINING_TYPE}\
    --train_file ${DATA_ROOT}/${DATASET}_feature/${TRAIN_FILE}.single_pose.undersample.pickle\
    --test_dir ${TEST_DIR}\
    -o ${TMP_OUTPUT}\
    ${extra_argument} &&\ "
echo $command
command="./eval.py -d ${TMP_OUTPUT}/scores -o ${TMP_OUTPUT}/performances -m fernie -n ${TESTSET} &&\ "
echo $command