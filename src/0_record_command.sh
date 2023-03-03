#!/bin/bash
###################################################################################

###################################################################################

./all_experiments.py --undersample 1 --exps 2 3 4 5 6 7 8 9 10 11 12  --experiment_name max_cf_680_h_292_dp0.1_over_validNone --gpu_id 0

time prepare_data.py --data_path actives_final_only_smiles.smi --dataset_name zinc --save_dir active_pre/

real    0m4.087s
user    0m7.360s
sys     0m6.317s


####################################################################################
# Generate decoy for Kernie and  DUD-E
####################################################################################

prepare_data.py --data_path PATH_TO_DATA --dataset_name NAME_OF_DATASET --save_dir OUTPUT_LOCATION

# Kernie n101
# python /y/Aurora/Fernie/tune/ZINC/DeepCoy/DeepCoy.py\
--restore /y/Aurora/Fernie/tune/ZINC/DeepCoy/models/DeepCoy_DUDE_model_e09.pickle\
--dataset zinc\
--config '{"generation": true, 
           "number_of_generation_per_valid": 1000, 
           "batch_size": 1, "train_file": 
           "/y/Aurora/Fernie/tune/ZINC/DeepCoy/data/Kernie/molecules_zinc_dude_train.json", 
           "valid_file": "/y/Aurora/Fernie/tune/ZINC/DeepCoy/data/Kernie/molecules_zinc_dude_valid.json", 
           "output_name": "Kernie_DeepCoy_generated_decoy.txt"}'
# DUD-E n100
# python /y/Aurora/Fernie/tune/ZINC/DeepCoy/DeepCoy.py --restore /y/Aurora/Fernie/tune/ZINC/DeepCoy/models/DeepCoy_DUDE_model_e09.pickle --dataset zinc --config '{"generation": true, "number_of_generation_per_valid": 1000, "batch_size": 1, "train_file": "/y/Aurora/Fernie/tune/ZINC/DeepCoy/data/DUD-E/molecules_zinc_dude_train.json", "valid_file": "/y/Aurora/Fernie/tune/ZINC/DeepCoy/data/DUD-E/molecules_zinc_dude_valid.json", "output_name": "DUD-E_DeepCoy_generated_decoy.txt"}'


# python select_and_evaluate_decoys.py --data_path PATH_TO_INPUT_FILE/DIRECTORY --output_path PATH_TO_OUTPUT --dataset_name dude --num_decoys_per_active 50 >> decoy_selection_log.txt


####################################################################################
# 6th tune best hyperparameters （undersample and oversample） 
# n11-1 down (finished)
# n100 (3 4 5 6 7 8 9 10) n100 (0 1 2 11 12) over
####################################################################################

# undersample=1

# batch_size=320
# num_workers=12

# weight_decay=0.001
# lr=0.0001

# cf=600
# h=100
# activation=False
# dp_rate=0.1

# num_sanity_val_steps=0
# gpu_id=0

# experiment_name=cf${cf}_h${h}_dp${dp_rate}_${activation}_batch${batch_size}

# if [ "$undersample" == "1" ]; then
#     experiment_name=${experiment_name}_down
#     echo -------------------------------------------------------------
#     echo experiment_name=$experiment_name
#     echo -------------------------------------------------------------
# else
#     experiment_name=${experiment_name}_over
#     echo -------------------------------------------------------------
#     echo experiment_name=$experiment_name
#     echo -------------------------------------------------------------
# fi

# extra_argument="\
#     --experiment_name ${experiment_name}\
#     --undersample ${undersample}\
#     --batch_size ${batch_size}\
#     --num_workers ${num_workers}\
#     --weight_decay ${weight_decay}\
#     --lr ${lr}\
#     --cf ${cf}\
#     --h ${h}\
#     --activation ${activation}\
#     --dp_rate ${dp_rate}\
#     --num_sanity_val_steps ${num_sanity_val_steps}\
#     --gpu_id ${gpu_id}"

# echo ./all_experiments.py --exps 9 10  $extra_argument


# echo ./all_experiments.py --exps  0 1 $extra_argument
# echo
# echo ./all_experiments.py --exps 2 11 12 $extra_argument
# echo
# echo ./all_experiments.py --exps 3 4 5 6 7 8 9 10  $extra_argument


####################################################################################
# DeepVS best hyperparameters （undersample and oversample）
# n101 down over
####################################################################################

# undersample=1

# batch_size=320
# num_workers=12

# weight_decay=0.001
# lr=0.0001

# cf=400
# h=50
# activation=False
# dp_rate=0.1

# num_sanity_val_steps=0
# gpu_id=0

# experiment_name=cf${cf}_h${h}_dp${dp_rate}_${activation}_batch${batch_size}

# if [ "$undersample" == "1" ]; then
#     experiment_name=${experiment_name}_down
#     echo -------------------------------------------------------------
#     echo experiment_name=$experiment_name
#     echo -------------------------------------------------------------
# else
#     experiment_name=${experiment_name}_over
#     echo -------------------------------------------------------------
#     echo experiment_name=$experiment_name
#     echo -------------------------------------------------------------
# fi

# extra_argument="\
#     --experiment_name ${experiment_name}\
#     --undersample ${undersample}\
#     --batch_size ${batch_size}\
#     --num_workers ${num_workers}\
#     --weight_decay ${weight_decay}\
#     --lr ${lr}\
#     --cf ${cf}\
#     --h ${h}\
#     --activation ${activation}\
#     --dp_rate ${dp_rate}\
#     --num_sanity_val_steps ${num_sanity_val_steps}\
#     --gpu_id ${gpu_id}"

# echo ./all_experiments.py --exps 9 10  $extra_argument

# n101 down
# echo ./all_experiments.py --exps -1 $extra_argument
# n101 over failed
# echo ./all_experiments.py --exps  0 1 2 $extra_argument
# echo
# echo ./all_experiments.py --exps 11 12 3 4 5 6 7 8 9 10 $extra_argument
