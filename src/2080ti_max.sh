source /y/home/zdx/anaconda3/etc/profile.d/conda.sh
conda activate 2080ti
##Some global setting
#DEBUG=0
#NUM_SANITY_VAL_STEPS=0

#DATA_ROOT=/y/Aurora/Fernie/data/structure_based_data
#ROOT_OUTPUT_DIR=/y/Aurora/Fernie/EXPS/002_2080ti
#BATCH_SIZE=320
#NUM_WORKERS=10

##Optimization
#WEIGHT_DECAY=1e-5
#LR=1e-3

##Model structure
#CF=400
#H=50
#DATM=200
#DAMINO=200
#DCHRG=200
#DDIST=200
#POOL_TYPE=max
#ACTIVATION=False
#DP_RATE=0
#BATCHNORM=0

#------------------------------------------
#Cross validation
#------------------------------------------

#------------------------------------------
#Clustered 3-fold cross-validation on DUD-E
#------------------------------------------
mkdir -p /y/Aurora/Fernie/EXPS/002_2080ti/DUD-E/performances
mkdir -p /y/Aurora/Fernie/EXPS/002_2080ti/DUD-E/scores
./main.py --working_mode train --training_type holdout --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/train_0.pickle --valid_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/test_0.pickle --test_dir /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/test_0 -o /y/Aurora/Fernie/EXPS/002_2080ti/cv/DUD-E/fold_0 --batch_size 320 --num_workers 10 --weight_decay 1e-5 --lr 1e-3 --cf 400 --h 50 --datm 200 --damino 200 --dchrg 200 --ddist 200 --pool_type max --activation False --dp_rate 0 --batchnorm 0 --num_sanity_val_steps 0 --debug 0
./eval.py -d /y/Aurora/Fernie/EXPS/002_2080ti/cv/DUD-E/fold_0/scores -o /y/Aurora/Fernie/EXPS/002_2080ti/cv/DUD-E/fold_0/performances -m fernie -n DUD-E_fold_0
mv /y/Aurora/Fernie/EXPS/002_2080ti/cv/DUD-E/fold_0/scores/* /y/Aurora/Fernie/EXPS/002_2080ti/DUD-E/scores

./main.py --working_mode train --training_type holdout --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/train_1.pickle --valid_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/test_1.pickle --test_dir /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/test_1 -o /y/Aurora/Fernie/EXPS/002_2080ti/cv/DUD-E/fold_1 --batch_size 320 --num_workers 10 --weight_decay 1e-5 --lr 1e-3 --cf 400 --h 50 --datm 200 --damino 200 --dchrg 200 --ddist 200 --pool_type max --activation False --dp_rate 0 --batchnorm 0 --num_sanity_val_steps 0 --debug 0
./eval.py -d /y/Aurora/Fernie/EXPS/002_2080ti/cv/DUD-E/fold_1/scores -o /y/Aurora/Fernie/EXPS/002_2080ti/cv/DUD-E/fold_1/performances -m fernie -n DUD-E_fold_1
mv /y/Aurora/Fernie/EXPS/002_2080ti/cv/DUD-E/fold_1/scores/* /y/Aurora/Fernie/EXPS/002_2080ti/DUD-E/scores

./main.py --working_mode train --training_type holdout --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/train_2.pickle --valid_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/test_2.pickle --test_dir /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/test_2 -o /y/Aurora/Fernie/EXPS/002_2080ti/cv/DUD-E/fold_2 --batch_size 320 --num_workers 10 --weight_decay 1e-5 --lr 1e-3 --cf 400 --h 50 --datm 200 --damino 200 --dchrg 200 --ddist 200 --pool_type max --activation False --dp_rate 0 --batchnorm 0 --num_sanity_val_steps 0 --debug 0
./eval.py -d /y/Aurora/Fernie/EXPS/002_2080ti/cv/DUD-E/fold_2/scores -o /y/Aurora/Fernie/EXPS/002_2080ti/cv/DUD-E/fold_2/performances -m fernie -n DUD-E_fold_2
mv /y/Aurora/Fernie/EXPS/002_2080ti/cv/DUD-E/fold_2/scores/* /y/Aurora/Fernie/EXPS/002_2080ti/DUD-E/scores

./eval.py -d /y/Aurora/Fernie/EXPS/002_2080ti/DUD-E/scores -o /y/Aurora/Fernie/EXPS/002_2080ti/DUD-E/performances -m fernie -n DUD-E

#------------------------------------------
#Clustered 3-fold cross-validation on Kernie
#------------------------------------------
mkdir -p /y/Aurora/Fernie/EXPS/002_2080ti/Kernie/performances
mkdir -p /y/Aurora/Fernie/EXPS/002_2080ti/Kernie/scores
./main.py --working_mode train --training_type holdout --train_file /y/Aurora/Fernie/data/structure_based_data/Kernie_3_fold/train_0.pickle --valid_file /y/Aurora/Fernie/data/structure_based_data/Kernie_3_fold/test_0.pickle --test_dir /y/Aurora/Fernie/data/structure_based_data/Kernie_3_fold/test_0 -o /y/Aurora/Fernie/EXPS/002_2080ti/cv/Kernie/fold_0 --batch_size 320 --num_workers 10 --weight_decay 1e-5 --lr 1e-3 --cf 400 --h 50 --datm 200 --damino 200 --dchrg 200 --ddist 200 --pool_type max --activation False --dp_rate 0 --batchnorm 0 --num_sanity_val_steps 0 --debug 0
./eval.py -d /y/Aurora/Fernie/EXPS/002_2080ti/cv/Kernie/fold_0/scores -o /y/Aurora/Fernie/EXPS/002_2080ti/cv/Kernie/fold_0/performances -m fernie -n Kernie_fold_0
mv /y/Aurora/Fernie/EXPS/002_2080ti/cv/Kernie/fold_0/scores/* /y/Aurora/Fernie/EXPS/002_2080ti/Kernie/scores

./main.py --working_mode train --training_type holdout --train_file /y/Aurora/Fernie/data/structure_based_data/Kernie_3_fold/train_1.pickle --valid_file /y/Aurora/Fernie/data/structure_based_data/Kernie_3_fold/test_1.pickle --test_dir /y/Aurora/Fernie/data/structure_based_data/Kernie_3_fold/test_1 -o /y/Aurora/Fernie/EXPS/002_2080ti/cv/Kernie/fold_1 --batch_size 320 --num_workers 10 --weight_decay 1e-5 --lr 1e-3 --cf 400 --h 50 --datm 200 --damino 200 --dchrg 200 --ddist 200 --pool_type max --activation False --dp_rate 0 --batchnorm 0 --num_sanity_val_steps 0 --debug 0
./eval.py -d /y/Aurora/Fernie/EXPS/002_2080ti/cv/Kernie/fold_1/scores -o /y/Aurora/Fernie/EXPS/002_2080ti/cv/Kernie/fold_1/performances -m fernie -n Kernie_fold_1
mv /y/Aurora/Fernie/EXPS/002_2080ti/cv/Kernie/fold_1/scores/* /y/Aurora/Fernie/EXPS/002_2080ti/Kernie/scores

./main.py --working_mode train --training_type holdout --train_file /y/Aurora/Fernie/data/structure_based_data/Kernie_3_fold/train_2.pickle --valid_file /y/Aurora/Fernie/data/structure_based_data/Kernie_3_fold/test_2.pickle --test_dir /y/Aurora/Fernie/data/structure_based_data/Kernie_3_fold/test_2 -o /y/Aurora/Fernie/EXPS/002_2080ti/cv/Kernie/fold_2 --batch_size 320 --num_workers 10 --weight_decay 1e-5 --lr 1e-3 --cf 400 --h 50 --datm 200 --damino 200 --dchrg 200 --ddist 200 --pool_type max --activation False --dp_rate 0 --batchnorm 0 --num_sanity_val_steps 0 --debug 0
./eval.py -d /y/Aurora/Fernie/EXPS/002_2080ti/cv/Kernie/fold_2/scores -o /y/Aurora/Fernie/EXPS/002_2080ti/cv/Kernie/fold_2/performances -m fernie -n Kernie_fold_2
mv /y/Aurora/Fernie/EXPS/002_2080ti/cv/Kernie/fold_2/scores/* /y/Aurora/Fernie/EXPS/002_2080ti/Kernie/scores

./eval.py -d /y/Aurora/Fernie/EXPS/002_2080ti/Kernie/scores -o /y/Aurora/Fernie/EXPS/002_2080ti/Kernie/performances -m fernie -n Kernie

#------------------------------------------
#Test on MUV
#------------------------------------------

#------------------------------------------
#Training on DUD-E
#------------------------------------------
./main.py --working_mode train --training_type holdout --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_feature/DUD-E.single_pose.undersample.pickle --test_dir /y/Aurora/Fernie/data/structure_based_data/MUV_feature/single_pose -o /y/Aurora/Fernie/EXPS/002_2080ti/MUV/DUD-E --batch_size 320 --num_workers 10 --weight_decay 1e-5 --lr 1e-3 --cf 400 --h 50 --datm 200 --damino 200 --dchrg 200 --ddist 200 --pool_type max --activation False --dp_rate 0 --batchnorm 0 --num_sanity_val_steps 0 --debug 0
./eval.py -d /y/Aurora/Fernie/EXPS/002_2080ti/MUV/DUD-E/scores -o /y/Aurora/Fernie/EXPS/002_2080ti/MUV/DUD-E/performances -m fernie -n MUV

#------------------------------------------
#Training on DUD-E.Kinase
#------------------------------------------
./main.py --working_mode train --training_type holdout --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_feature/DUD-E.Kinase.undersample.pickle --test_dir /y/Aurora/Fernie/data/structure_based_data/MUV_feature/single_pose -o /y/Aurora/Fernie/EXPS/002_2080ti/MUV/DUD-E.Kinase --batch_size 320 --num_workers 10 --weight_decay 1e-5 --lr 1e-3 --cf 400 --h 50 --datm 200 --damino 200 --dchrg 200 --ddist 200 --pool_type max --activation False --dp_rate 0 --batchnorm 0 --num_sanity_val_steps 0 --debug 0
./eval.py -d /y/Aurora/Fernie/EXPS/002_2080ti/MUV/DUD-E.Kinase/scores -o /y/Aurora/Fernie/EXPS/002_2080ti/MUV/DUD-E.Kinase/performances -m fernie -n MUV

#------------------------------------------
#Training on DUD-E.Protease
#------------------------------------------
./main.py --working_mode train --training_type holdout --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_feature/DUD-E.Protease.undersample.pickle --test_dir /y/Aurora/Fernie/data/structure_based_data/MUV_feature/single_pose -o /y/Aurora/Fernie/EXPS/002_2080ti/MUV/DUD-E.Protease --batch_size 320 --num_workers 10 --weight_decay 1e-5 --lr 1e-3 --cf 400 --h 50 --datm 200 --damino 200 --dchrg 200 --ddist 200 --pool_type max --activation False --dp_rate 0 --batchnorm 0 --num_sanity_val_steps 0 --debug 0
./eval.py -d /y/Aurora/Fernie/EXPS/002_2080ti/MUV/DUD-E.Protease/scores -o /y/Aurora/Fernie/EXPS/002_2080ti/MUV/DUD-E.Protease/performances -m fernie -n MUV

#------------------------------------------
#Training on DUD-E.Cytochrome_P450
#------------------------------------------
./main.py --working_mode train --training_type holdout --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_feature/DUD-E.Cytochrome_P450.undersample.pickle --test_dir /y/Aurora/Fernie/data/structure_based_data/MUV_feature/single_pose -o /y/Aurora/Fernie/EXPS/002_2080ti/MUV/DUD-E.Cytochrome_P450 --batch_size 320 --num_workers 10 --weight_decay 1e-5 --lr 1e-3 --cf 400 --h 50 --datm 200 --damino 200 --dchrg 200 --ddist 200 --pool_type max --activation False --dp_rate 0 --batchnorm 0 --num_sanity_val_steps 0 --debug 0
./eval.py -d /y/Aurora/Fernie/EXPS/002_2080ti/MUV/DUD-E.Cytochrome_P450/scores -o /y/Aurora/Fernie/EXPS/002_2080ti/MUV/DUD-E.Cytochrome_P450/performances -m fernie -n MUV

#------------------------------------------
#Training on DUD-E.GPCR
#------------------------------------------
./main.py --working_mode train --training_type holdout --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_feature/DUD-E.GPCR.undersample.pickle --test_dir /y/Aurora/Fernie/data/structure_based_data/MUV_feature/single_pose -o /y/Aurora/Fernie/EXPS/002_2080ti/MUV/DUD-E.GPCR --batch_size 320 --num_workers 10 --weight_decay 1e-5 --lr 1e-3 --cf 400 --h 50 --datm 200 --damino 200 --dchrg 200 --ddist 200 --pool_type max --activation False --dp_rate 0 --batchnorm 0 --num_sanity_val_steps 0 --debug 0
./eval.py -d /y/Aurora/Fernie/EXPS/002_2080ti/MUV/DUD-E.GPCR/scores -o /y/Aurora/Fernie/EXPS/002_2080ti/MUV/DUD-E.GPCR/performances -m fernie -n MUV

#------------------------------------------
#Training on DUD-E.Ion_Channel
#------------------------------------------
./main.py --working_mode train --training_type holdout --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_feature/DUD-E.Ion_Channel.undersample.pickle --test_dir /y/Aurora/Fernie/data/structure_based_data/MUV_feature/single_pose -o /y/Aurora/Fernie/EXPS/002_2080ti/MUV/DUD-E.Ion_Channel --batch_size 320 --num_workers 10 --weight_decay 1e-5 --lr 1e-3 --cf 400 --h 50 --datm 200 --damino 200 --dchrg 200 --ddist 200 --pool_type max --activation False --dp_rate 0 --batchnorm 0 --num_sanity_val_steps 0 --debug 0
./eval.py -d /y/Aurora/Fernie/EXPS/002_2080ti/MUV/DUD-E.Ion_Channel/scores -o /y/Aurora/Fernie/EXPS/002_2080ti/MUV/DUD-E.Ion_Channel/performances -m fernie -n MUV

#------------------------------------------
#Training on DUD-E.Miscellaneous
#------------------------------------------
./main.py --working_mode train --training_type holdout --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_feature/DUD-E.Miscellaneous.undersample.pickle --test_dir /y/Aurora/Fernie/data/structure_based_data/MUV_feature/single_pose -o /y/Aurora/Fernie/EXPS/002_2080ti/MUV/DUD-E.Miscellaneous --batch_size 320 --num_workers 10 --weight_decay 1e-5 --lr 1e-3 --cf 400 --h 50 --datm 200 --damino 200 --dchrg 200 --ddist 200 --pool_type max --activation False --dp_rate 0 --batchnorm 0 --num_sanity_val_steps 0 --debug 0
./eval.py -d /y/Aurora/Fernie/EXPS/002_2080ti/MUV/DUD-E.Miscellaneous/scores -o /y/Aurora/Fernie/EXPS/002_2080ti/MUV/DUD-E.Miscellaneous/performances -m fernie -n MUV

#------------------------------------------
#Training on DUD-E.Nuclear_Receptor
#------------------------------------------
./main.py --working_mode train --training_type holdout --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_feature/DUD-E.Nuclear_Receptor.undersample.pickle --test_dir /y/Aurora/Fernie/data/structure_based_data/MUV_feature/single_pose -o /y/Aurora/Fernie/EXPS/002_2080ti/MUV/DUD-E.Nuclear_Receptor --batch_size 320 --num_workers 10 --weight_decay 1e-5 --lr 1e-3 --cf 400 --h 50 --datm 200 --damino 200 --dchrg 200 --ddist 200 --pool_type max --activation False --dp_rate 0 --batchnorm 0 --num_sanity_val_steps 0 --debug 0
./eval.py -d /y/Aurora/Fernie/EXPS/002_2080ti/MUV/DUD-E.Nuclear_Receptor/scores -o /y/Aurora/Fernie/EXPS/002_2080ti/MUV/DUD-E.Nuclear_Receptor/performances -m fernie -n MUV

#------------------------------------------
#Training on DUD-E.Other_Enzymes
#------------------------------------------
./main.py --working_mode train --training_type holdout --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_feature/DUD-E.Other_Enzymes.undersample.pickle --test_dir /y/Aurora/Fernie/data/structure_based_data/MUV_feature/single_pose -o /y/Aurora/Fernie/EXPS/002_2080ti/MUV/DUD-E.Other_Enzymes --batch_size 320 --num_workers 10 --weight_decay 1e-5 --lr 1e-3 --cf 400 --h 50 --datm 200 --damino 200 --dchrg 200 --ddist 200 --pool_type max --activation False --dp_rate 0 --batchnorm 0 --num_sanity_val_steps 0 --debug 0
./eval.py -d /y/Aurora/Fernie/EXPS/002_2080ti/MUV/DUD-E.Other_Enzymes/scores -o /y/Aurora/Fernie/EXPS/002_2080ti/MUV/DUD-E.Other_Enzymes/performances -m fernie -n MUV

#------------------------------------------
#Training on Kernie
#------------------------------------------
./main.py --working_mode train --training_type holdout --train_file /y/Aurora/Fernie/data/structure_based_data/Kernie_feature/Kernie.undersample.pickle --test_dir /y/Aurora/Fernie/data/structure_based_data/MUV_feature/single_pose -o /y/Aurora/Fernie/EXPS/002_2080ti/MUV/Kernie --batch_size 320 --num_workers 10 --weight_decay 1e-5 --lr 1e-3 --cf 400 --h 50 --datm 200 --damino 200 --dchrg 200 --ddist 200 --pool_type max --activation False --dp_rate 0 --batchnorm 0 --num_sanity_val_steps 0 --debug 0
./eval.py -d /y/Aurora/Fernie/EXPS/002_2080ti/MUV/Kernie/scores -o /y/Aurora/Fernie/EXPS/002_2080ti/MUV/Kernie/performances -m fernie -n MUV

#------------------------------------------
#Training on Kernie without MUV targets
#------------------------------------------
./main.py --working_mode train --training_type holdout --train_file /y/Aurora/Fernie/data/structure_based_data/Kernie_feature/Kernie-MUV.undersample.pickle --test_dir /y/Aurora/Fernie/data/structure_based_data/MUV_feature/single_pose -o /y/Aurora/Fernie/EXPS/002_2080ti/MUV/Kernie-MUV --batch_size 320 --num_workers 10 --weight_decay 1e-5 --lr 1e-3 --cf 400 --h 50 --datm 200 --damino 200 --dchrg 200 --ddist 200 --pool_type max --activation False --dp_rate 0 --batchnorm 0 --num_sanity_val_steps 0 --debug 0
./eval.py -d /y/Aurora/Fernie/EXPS/002_2080ti/MUV/Kernie-MUV/scores -o /y/Aurora/Fernie/EXPS/002_2080ti/MUV/Kernie-MUV/performances -m fernie -n MUV
