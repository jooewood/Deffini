#!/bin/bash


echo \# ----------------------
echo \# 1
echo \# Best hyperparameters:  
echo \# {
echo \#     'weight_decay': 0.0001, 
echo \#     'lr': 0.0001, 
echo \#     'cf': 256, 
echo \#     'h': 64, 
echo \#     'pool_type': 'max', 
echo \#     'activation': 'False', 
echo \#     'dp_rate': 0.05, 
echo \#     'batchnorm': 1, 
echo \#     'batch_2size': 320
echo \# } 
echo \# 7.145016525718901 hours
echo \# -----------------------
name=001_2080ti
gpus_per_trial=0.5
cpu_per_trial=10
num_samples=60
echo ./main.py --working_mode tune --exp_name hyopt_${name} --gpus_per_trial $gpus_per_trial --cpu_per_trial $cpu_per_trial --validation_split .5\
    --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_feature/DUD-E.single_pose.undersample.pickle\
    --num_samples $num_samples --num_workers $cpu_per_trial -o /y/Aurora/Fernie/tune/
echo tensorboard --logdir=/y/Aurora/Fernie/tune/hyopt_${name} --host=0.0.0.0 --port 7000
    # config = {
    #     "weight_decay":  tune.choice([1e-7, 1e-6, 1e-5, 1e-4, 1e-3]), # tune.loguniform(1e-7, 1e-1), 
    #     "lr": tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1]), # tune.loguniform(1e-6, 1),

    #     # ------------------
    #     # hyperparameters for model architecture
    #     # ------------------
    #     "cf": tune.choice([200, 256, 400, 512]), # tune.choice([100, 128, 200, 256, 400, 512]), 
    #     "h": tune.choice([50, 64, 128]), #tune.choice([50, 64, 100, 128, 200, 256]),
    #     # "datm": tune.choice([128, 200, 256]),
    #     # "damino": tune.choice([128, 200, 256]), 
    #     # "dchrg": tune.choice([128, 200, 256]), 
    #     # "ddist": tune.choice([128, 200, 256]), 

    #     # ------------------
    #     # model layer hyperparameters
    #     # ------------------
    #     "pool_type": tune.choice(['max', 'avg']),
    #     "activation": tune.choice(['False', 'relu', 'elu', 'gelu', 'sigmoid']), # 'sigmoid', 'tahn',  'leakyrelu', 'gelu', 'elu'
    #     "dp_rate": tune.choice([0, 0.05, 0.1, 0.2, 0.3]), # tune.uniform(0, 0.5) 
    #     "batchnorm": tune.choice([1, 0]),
    #     "batch_size": tune.choice([320]) # tune.choice([32, 64, 128, 256, 512]),
    # }
echo
echo
echo \# -----------------------
echo \# 2
echo \# Best hyperparameters:
echo \# {
echo \# 'weight_decay': 0.0001, 
echo \# 'lr': 0.0001, 
echo \# 'cf': 592, 
echo \# 'h': 72, 
echo \# 'pool_type': 'max', 
echo \# 'activation': 'relu',
echo \# 'dp_rate': 0.0, 
echo \# 'batchnorm': 1, 
echo \# 'batch_size': 320
echo \# }
echo \# -----------------------
name=002_A40
gpus_per_trial=0.333
cpu_per_trial=10
num_samples=180
echo ./main.py --working_mode tune --exp_name hyopt_${name} --gpus_per_trial $gpus_per_trial --cpu_per_trial $cpu_per_trial --validation_split .5\
    --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_feature/DUD-E.single_pose.undersample.pickle\
    --num_samples $num_samples --num_workers $cpu_per_trial -o /y/Aurora/Fernie/tune/
echo tensorboard --logdir=/y/Aurora/Fernie/tune/hyopt_${name} --host=0.0.0.0 --port 7000
    # config = {
    #     "weight_decay":  tune.choice([0.0001]), # tune.loguniform(1e-7, 1e-1), 
    #     "lr": tune.choice([0.0001]), # tune.loguniform(1e-6, 1),

    #     # ------------------
    #     # hyperparameters for model architecture
    #     # ------------------
    #     "cf": tune.qrandint(128, 512, 16), 
    #     "h": tune.qrandint(32, 256, 8), 

    #     # ------------------
    #     # model layer hyperparameters
    #     # ------------------
    #     "pool_type": tune.choice(['max']),
    #     "activation": tune.choice(['False', 'relu']), # 'sigmoid', 'tahn',  'leakyrelu', 'gelu', 'elu'
    #     "dp_rate":  tune.quniform(0, 0.2, 0.01),
    #     "batchnorm": tune.choice([1]),
    #     "batch_size": tune.choice([320])
    # }

echo
echo
echo \# -----------------------
echo \# 3
echo \# Best hyperparameters found were:  
echo \# {'weight_decay': 0.0001, 
echo \# 'lr': 0.0001, 
echo \# 'cf': 680, 
echo \# 'h': 292, 
echo \# 'pool_type': 'max', 
echo \# 'activation': 'relu', 
echo \# 'dp_rate': 0, 
echo \# 'batchnorm': 1, 
echo \# 'batch_size': 320}
echo \# 17.56661285916964 hours
echo \# -----------------------
name=003_A40
gpus_per_trial=0.333
cpu_per_trial=8
num_workers=12
num_samples=180
echo ./main.py --working_mode tune --exp_name hyopt_${name} --gpus_per_trial $gpus_per_trial --cpu_per_trial $cpu_per_trial --validation_split .5\
    --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_feature/DUD-E.single_pose.undersample.pickle\
    --num_samples $num_samples --num_workers $num_workers -o /y/Aurora/Fernie/tune/
echo tensorboard --logdir=/y/Aurora/Fernie/tune/hyopt_${name} --host=0.0.0.0 --port 7000
#     config = {
#         "weight_decay":  tune.choice([0.0001]), #tune.qloguniform(1e-5, 1e-3, 1e-6),
#         "lr": tune.choice([0.0001]), #tune.qloguniform(1e-5, 1e-3, 1e-6)

#         # ------------------
#         # hyperparameters for model architecture
#         # ------------------
#         "cf": tune.qrandint(500, 700, 8), 
#         "h": tune.qrandint(200, 300, 4), 

#         # ------------------
#         # model layer hyperparameters
#         # ------------------
#         "pool_type": tune.choice(['max']),
#         "activation": tune.choice(['relu']), 
#         "dp_rate":  tune.choice([0]]), # tune.qloguniform(0, 0.05, 0.001),
#         "batchnorm": tune.choice([1]),
#         "batch_size": tune.choice([320])
#     }

echo
echo
echo \# -----------------------
echo \# 4 Failed
echo \# -----------------------
name=004_A40
gpus_per_trial=0.5
cpu_per_trial=16
num_workers=18
num_samples=180

fold=0
echo "./main.py --working_mode tune --exp_name hyopt_${name}_fold_${fold} --gpus_per_trial $gpus_per_trial --cpu_per_trial $cpu_per_trial\
    --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/train_${fold}.pickle\
    --valid_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/test_${fold}.pickle\
    --num_samples $num_samples --num_workers $num_workers -o /y/Aurora/Fernie/tune/ &&\ "

fold=1
echo "./main.py --working_mode tune --exp_name hyopt_${name}_fold_${fold} --gpus_per_trial $gpus_per_trial --cpu_per_trial $cpu_per_trial\
    --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/train_${fold}.pickle\
    --valid_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/test_${fold}.pickle\
    --num_samples $num_samples --num_workers $num_workers -o /y/Aurora/Fernie/tune/ &&\ "

fold=2
echo "./main.py --working_mode tune --exp_name hyopt_${name}_fold_${fold} --gpus_per_trial $gpus_per_trial --cpu_per_trial $cpu_per_trial\
    --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/train_${fold}.pickle\
    --valid_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/test_${fold}.pickle\
    --num_samples $num_samples --num_workers $num_workers -o /y/Aurora/Fernie/tune/"

echo tensorboard --logdir=/y/Aurora/Fernie/tune/hyopt_${name} --host=0.0.0.0 --port 7000
    # config = {
    #     "weight_decay":  tune.qloguniform(1e-5, 1e-4, 5e-6),
    #     "lr": tune.qloguniform(1e-5, 1e-4, 5e-6),

    #     # ------------------
    #     # hyperparameters for model architecture
    #     # ------------------
    #     "cf": tune.choice([680]), 
    #     "h": tune.choice([292]), 

    #     # ------------------
    #     # model layer hyperparameters
    #     # ------------------
    #     "pool_type": tune.choice(['max']),
    #     "activation": tune.choice(['relu']), 
    #     "dp_rate":  tune.choice([0]), #tune.quniform(0, 0.2, 0.0125),
    #     "batchnorm": tune.choice([1]),
    #     "batch_size": tune.qrandint(256, 1024, 64)
    # }
# ./main.py --working_mode tune --exp_name hyopt_004_A40_fold_0 --gpus_per_trial 0.5 --cpu_per_trial 16    --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/train_0.pickle    --valid_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/test_0.pickle    --num_samples 180 --num_workers 18 -o /y/Aurora/Fernie/tune/ &&\
# ./main.py --working_mode tune --exp_name hyopt_004_A40_fold_1 --gpus_per_trial 0.5 --cpu_per_trial 16    --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/train_1.pickle    --valid_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/test_1.pickle    --num_samples 180 --num_workers 18 -o /y/Aurora/Fernie/tune/ &&\
# ./main.py --working_mode tune --exp_name hyopt_004_A40_fold_2 --gpus_per_trial 0.5 --cpu_per_trial 16    --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/train_2.pickle    --valid_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/test_2.pickle    --num_samples 180 --num_workers 18 -o /y/Aurora/Fernie/tune/

echo
echo
echo \# -----------------------
echo \# 5
echo \# -----------------------
name=005_A40
gpus_per_trial=0.333
cpu_per_trial=10
num_workers=12
num_samples=180

fold=0
echo "./main.py --working_mode tune --exp_name hyopt_${name}_fold_${fold} --gpus_per_trial $gpus_per_trial --cpu_per_trial $cpu_per_trial\
    --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/train_${fold}.pickle\
    --valid_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/test_${fold}.pickle\
    --num_samples $num_samples --num_workers $num_workers -o /y/Aurora/Fernie/tune/ &&\ "

fold=1
echo "./main.py --working_mode tune --exp_name hyopt_${name}_fold_${fold} --gpus_per_trial $gpus_per_trial --cpu_per_trial $cpu_per_trial\
    --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/train_${fold}.pickle\
    --valid_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/test_${fold}.pickle\
    --num_samples $num_samples --num_workers $num_workers -o /y/Aurora/Fernie/tune/ &&\ "

fold=2
echo "./main.py --working_mode tune --exp_name hyopt_${name}_fold_${fold} --gpus_per_trial $gpus_per_trial --cpu_per_trial $cpu_per_trial\
    --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/train_${fold}.pickle\
    --valid_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/test_${fold}.pickle\
    --num_samples $num_samples --num_workers $num_workers -o /y/Aurora/Fernie/tune/"

    # config = {
    #     "weight_decay":  tune.choice([1e-4]), # tune.qloguniform(1e-5, 1e-4, 5e-6)
    #     "lr": tune.choice([1e-4]), # tune.qloguniform(1e-5, 1e-4, 5e-6)

    #     # ------------------
    #     # hyperparameters for model architecture
    #     # ------------------
    #     "cf": tune.choice([680]), 
    #     "h": tune.choice([292]), 

    #     # ------------------
    #     # model layer hyperparameters
    #     # ------------------
    #     "pool_type": tune.choice(['max']),
    #     "activation": tune.choice(['relu']), 
    #     "dp_rate":  tune.quniform(0, 0.3, 0.01),
    #     "batchnorm": tune.choice([1]),
    #     "batch_size": tune.qrandint([320])
    # }


echo
echo
echo \# -----------------------
echo \# 6
echo \# -----------------------
name=006_A40
gpus_per_trial=0.333
cpu_per_trial=10
num_workers=12
num_samples=180

# 0.001	0.0001	600	100	max	False	0.1	1

fold=0
echo "./main.py --working_mode tune --exp_name hyopt_${name}_fold_${fold} --gpus_per_trial $gpus_per_trial --cpu_per_trial $cpu_per_trial\
    --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/train_${fold}.pickle\
    --valid_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/test_${fold}.pickle\
    --num_samples $num_samples --num_workers $num_workers -o /y/Aurora/Fernie/tune/ &&\ "

    # config = {
    #     "weight_decay":  tune.choice([1e-5, 1e-4, 1e-3]), # tune.qloguniform(1e-5, 1e-4, 5e-6)
    #     "lr":  tune.choice([1e-5, 1e-4, 1e-3]), # tune.qloguniform(1e-5, 1e-4, 5e-6)

    #     # ------------------
    #     # hyperparameters for model architecture
    #     # ------------------
    #     "cf": tune.choice([200, 256, 400, 512, 600]), 
    #     "h": tune.choice([50, 64, 100, 128]), 

    #     # ------------------
    #     # model layer hyperparameters
    #     # ------------------
    #     "pool_type": tune.choice(['max']),
    #     "activation": tune.choice(['False', 'relu']), 
    #     "dp_rate":  tune.choice([0, 0.05, 0.1]), 
    #     "batchnorm": tune.choice([1]),
    #     "batch_size": tune.qrandint([320])
    # }


echo
echo
echo \# -----------------------
echo \# 7
echo \# -----------------------
name=007_A40
gpus_per_trial=0.5
cpu_per_trial=24
num_workers=24
num_samples=90

# 0.001	0.0001	600	100	max	False	0.1	1

fold=0
echo "./main.py --working_mode tune --exp_name hyopt_${name}_fold_${fold} --gpus_per_trial $gpus_per_trial --cpu_per_trial $cpu_per_trial\
    --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/train_${fold}.pickle\
    --valid_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/test_${fold}.pickle\
    --num_samples $num_samples --num_workers $num_workers -o /y/Aurora/Fernie/tune/ &&\ "


    # config = {
    #     "weight_decay":  tune.choice([1e-5, 1e-4, 1e-3]), # tune.qloguniform(1e-5, 1e-4, 5e-6)
    #     "lr":  tune.choice([1e-5, 1e-4, 1e-3]), # tune.qloguniform(1e-5, 1e-4, 5e-6)

    #     # ------------------
    #     # hyperparameters for model architecture
    #     # ------------------
    #     "cf": tune.choice([600]), 
    #     "h": tune.choice([100]), 

    #     # ------------------
    #     # model layer hyperparameters
    #     # ------------------
    #     "pool_type": tune.choice(['max']),
    #     "activation": tune.choice(['False', 'relu']), 
    #     "dp_rate":  tune.choice([0.1, 0.15, 0.2]), 
    #     "batchnorm": tune.choice([1]),
    #     "batch_size": tune.choice([320, 512, 1024])
    # }