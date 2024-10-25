- [1. Deffini: A family-specific deep neural network model for structure-based virtual screening)
  - [1.1. Data](#11-data)
  - [1.2. Dependencies](#12-dependencies)
- [2. Get all experiments command line](#2-get-all-experiments-command-line)
- [3. Usage](#3-usage)
  - [3.1. Train](#31-train)
  - [3.2. Cross validation by given fold data.](#32-cross-validation-by-given-fold-data)
  - [3.3. Prediction](#33-prediction)
  - [3.4. Evaluation](#34-evaluation)
    - [3.4.1. Evaluate single score file.](#341-evaluate-single-score-file)
    - [3.4.2. Evaluate a dataset](#342-evaluate-a-dataset)
  - [3.5. Hyperparameters Optimization](#35-hyperparameters-optimization)
    - [3.5.1. Tuning Fernie (Take n11-1 as example)](#351-tuning-fernie-take-n11-1-as-example)
    - [3.5.2. Ray + PyTorch Lightning MNIST Example](#352-ray--pytorch-lightning-mnist-example)
  - [3.6. Data Preprocess](#36-data-preprocess)
    - [3.6.1. Docking](#361-docking)
    - [3.6.2. Extract interaction feature (.pickle)](#362-extract-interaction-feature-pickle)
    - [3.6.3. Package multiple targets feature (.pickle)](#363-package-multiple-targets-feature-pickle)
    - [3.6.4. Under sampling keep train and valid 1:1 (.pickle)](#364-under-sampling-keep-train-and-valid-11-pickle)
    - [3.6.5. Generate cross validation Fernie feature data  (.pickle)](#365-generate-cross-validation-fernie-feature-data--pickle)
    - [3.6.6. Generate cross validation ligand-based data (.csv)](#366-generate-cross-validation-ligand-based-data-csv)
    - [3.6.7. Add InChI lines and Drop duplicate data (.csv)](#367-add-inchi-lines-and-drop-duplicate-data-csv)
    - [3.6.8. Split tensor k-fold cross validation (.pt)](#368-split-tensor-k-fold-cross-validation-pt)
    - [3.6.9. Some other scripts](#369-some-other-scripts)

# 1. Deffini: A family-specific deep neural network model for structure-based virtual screening

## 1.1. Data
Located in **/y/Aurora/Fernie/data/**
|Folder|Description|  
|:-|:-|  
|raw|raw data without any process|
|DUD-E|DUD-E dataset after docking(102 targets)|
|Kernie|358 kinase targets|
|MUV|a real screening data|

## 1.2. Dependencies

[**Anaconda Download link**](https://www.anaconda.com/products/individual#Downloads)
```bash
# Download Anaconda
$ wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
# Install Anaconda
$ sha256sum Anaconda3-2020.11-Linux-x86_64.sh # Check file integrity
$ bash Anaconda3-2020.11-Linux-x86_64.sh
$ source ~/.bashrc
```

**Anaconda common commands**

|Command|Description|
|:------|:----------|
|conda search python|Check python version you can install.|
|conda update python| Update the python version in current environment.|
|conda uninstall python|Uninstall python in current environment.|
|conda create -n new_env python=3.9.1|Create a new environment with specific python version.|
|conda activate new_env|Activate a existed environment.|
|conda deactivate|Deactivate current environment.|
|conda info --envs|Check the environment you created.|
|conda install -n new_env numpy|Install package into a specific environment.|
|conda remove -n new_env --all|Delete a environment.|
|python --version| Check the python version in current environment.|
|conda update conda|Update the conda package|
|conda update anaconda|Update the anaconda version|
|conda config --set auto_activate_base false|Set the base environment that does not activate by default|



**Create a new anaconda environment.**
```bash
(base) ~$ conda create -n fernie python=3.8.10 -y &&  conda activate fernie
```

After activate:

```bash
(fernie) ~$
```

**PyTorch**  
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
```
Install pytorch build from source. [Please reference the page](https://blog.csdn.net/zdx1996/article/details/118108551)


**Python-docx**  
```
pip install python-docx
```
**statsmodels**  
```
conda install -c conda-forge statsmodels -y
```
**ONNX**
```
conda install -c conda-forge numpy protobuf==3.16.0 libprotobuf=3.16.0 -y
conda install -c conda-forge onnx -y
```

**PyTorch Lightning**
```
conda install pytorch-lightning -c conda-forge -y
```
**seaborn**  
```
conda install seaborn -y
```
**Spyder**  
```
conda install spyder=5.0.5 -y
```
**openpyxl**  
```
conda install -c anaconda openpyxl -y
```
**easydict**  
```bash
conda install -c conda-forge easydict -y
```

**tqdm**

```bash
conda install -c conda-forge tqdm -y
```

**matplotlib**

```bash
conda install -c anaconda matplotlib -y
```

**scikit-image**
```bash
conda install -c anaconda scikit-image -y
```

**sklearn**

```bash
conda install -c anaconda scikit-learn -y
```

**ray.tune**  

```bash
pip3 install ray['default']
pip3 install ray['tune']
```
**ProDy**  
```
conda install ProDy
```

**pandas**
```
conda install pandas -y
```
**biotitle**
```
conda install -c conda-forge biotite -y
```
**torchmetrics**
```
conda install torchmetrics -c conda-forge -y
```

**numpy** maybe installed by default.

**Check whether GPU is ready?**

```bash
>>> import torch
>>> torch.cuda.is_available()
True
>>> torch.cuda.device_count()
>>> 1
```
# 2. Get all experiments command line
[**See the zommand.sh file**](https://gitlab.yfish.x/aurora/fernie/fernie/-/blob/master/src/zommand.sh)
```
cd src
./all_experiments.sh > zommand.sh
```

# 3. Usage

## 3.1. Train
Training a dataset by auto **hold out** (default 7:3)  
`--working_mode train`: Must to set.  
`--working_mode`: Choose from [train, pred, test, valid, tune]  
`--training_type`: Choose from [cv, holdout]  
`--train_file`: Training set file (.pickle or .pt).  
`--valid_file`: Validation set file (.pickle or .pt).  
`--test_file`: Test set file (.pickle or .pt).  
`--lr`: Learning rate (default=1e-3).  
`--weight_decay`: Weight decay (default=1e-5).
```bash
./main.py --working_mode train --training_type holdout --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/train_0.pickle --test_dir /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/test_0 -o /y/Aurora/Fernie/debug/test_0
```
Training a dataset by given valid file and test file
```bash
./main.py --working_mode train --training_type holdout --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/train_1.pickle --valid_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/test_1.pickle --test_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/test_1.pickle --test_dir /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/test_1 -o /y/Aurora/Fernie/debug/test_1 
```

Training a dataset by auto **cross validation** (defatult 10-fold).
Traing set: DUD-E, Testing set: MUV
```bash
./main.py --working_mode train --training_type cv --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_feature/DUD-E.undersample.pickle --test_dir /y/Aurora/Fernie/data/structure_based_data/MUV_feature/single_pose -o /y/Aurora/Fernie/test_result/MUV/DUD-E
```
## 3.2. Cross validation by given fold data.
`-d`: Device name you are using.  
`-i`: Cross validation folder path which saved train_*.pickle and test_*.pickle.  
`-o`: Output folder path.  
Take our clustered 3-fold cross validation as a example.  
**DUD-E**
```bash
./diff_train.py -d A40 -i /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/ -o /y/Aurora/Fernie/cv/DUD-E
```
**Kernie**
```bash
./diff_train.py -d A40 -i /y/Aurora/Fernie/data/structure_based_data/Kernie_3_fold/ -o /y/Aurora/Fernie/cv/Kernie
```

## 3.3. Prediction  
`--working_mode pred`: Must to set.  
`--checkpoint_path`: Checkpoint path which used to predict.  
`--test_dir`: The path of folder which includes a lot of target feature file (.pickle).  
`-o`: Output folder path.
```bash
./main.py --working_mode pred --checkpoint_path /y/Aurora/Fernie/debug/test_0/checkpoint/epoch=11-val_loss=0.10215-val_acc=0.96408.ckpt --test_dir y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/test_0 -o /y/Aurora/Fernie/debug/test_0_pred
```
Or  
`-i`: Input folder which have many target feature pickle.  
`-c`: Checkpoint path.    
`-o`: Output folder path.  
```
./predict.py -i /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/test_0 -c /y/Aurora/Fernie/output/DUD-E/fernie_PyTorch_Lightning/DUD-E_cv/fold_2/A40/checkpoint/epoch=12-val_loss=0.05295-val_acc=0.98111.ckpt -o /y/Aurora/Fernie/debug/test_2_pred
```
## 3.4. Evaluation
### 3.4.1. Evaluate single score file.  
`-m`: Model name which will used in naming output file.  
`-i`: Single score file which include ID, score, label  
|ID|score|label|
|:-|:-|:-|
|ZINC27298738|1.0|0|
|ZINC01650159|1.0|0|

```bash
./eval.py -i /y/Aurora/Fernie/kinase_project/final_score/DUD-E/scores/aa2ar.score -m fernie
```
### 3.4.2. Evaluate a dataset  
`-d`: Folder which saves prediction files of a dataset.  
`-n`: Dataset name which will used in naming output file.  
`-m`: Model name which will used in naming output file.  
`-o`: Output folder.  
```bash 
./eval.py -d /y/Aurora/Fernie/kinase_project/final_score/DUD-E/scores -m fernie -o /y/Aurora/Fernie/test_result/ -n DUD-E
```
## 3.5. Hyperparameters Optimization


### 3.5.1. Tuning Fernie (Take n11-1 as example)

```
ray_start.sh fernie 1 /y/Aurora/Fernie/tune/tmp 172.22.99.11 6000
```
`--exp_name`: Must be seted.  
`--working_mode tune`: Must be seted.  
`--train_file`: The file used to tune model's hyperparameters (default split 7:3).  
`--num_samples`: Total number of trials.  
`--num_workers`: The number of cpu will be used of each trial.  
`--search_alg`: Search algorighm default PBT.  
`--gpus_per_trial`:  Tell the ray that how many gpu each trial will use (Not true).  
`--cpu_per_trial`: Tell the ray that how many cpu each trial will use (Not true).  
`--validation_split`: The validation set split proportion. default=.3  
`-o`: Output folder.  
```bash
./main.py --working_mode tune --exp_name PBT_001 --gpus_per_trial 0.5 --cpu_per_trial 5 --validation_split .5 --train_file /y/Aurora/Fernie/data/structure_based_data/DUD-E_feature/DUD-E.undersample.pickle --num_samples 500 --num_workers 10 -o /y/Aurora/Fernie/tune/debug/DUD-E_tune
```
**Ray**  
|Command|Description|
|:-|:-|
|ray up| start ray on current node|
|ray stop|stop ray on current node|
Host node
```
ray_start.sh fernie 1 /y/scratch/zdx/tmp 172.22.99.10 6000
```
Client node
```
pdsh -w n10[1,2] "/y/program/bin/ray_start.sh fernie 2 /y/scratch/zdx/tmp 172.22.99.10 6000 && echo 0"
```
Stop node
```
pdsh -w n10[0,1,2]  "/y/program/bin/ray_stop.sh fernie"
```
### 3.5.2. Ray + PyTorch Lightning MNIST Example
```bash
./ray_mnist.py
```
## 3.6. Data Preprocess
### 3.6.1. Docking
`-i`:  A tsv file each row of which is a command line arguments.  
`-s`:  In which row do you want to start.    
`-n`: In which row do you want to end.  
`--stop`: If set this, program will not docking.   
`--nrows_of_file`: Number of the guideline.tsv.  
```bash
./pandas_docking.py -i /y/Aurora/Fernie/Kernie_docking_guideline.tsv
```
### 3.6.2. Extract interaction feature (.pickle)
Extract a target's interaction feature.  
`-m`: If set this, program will extract multiple pose feature.  
`-p`: PDBQT file of protein (A target with only a file).   
`-l`: PDBQT files of ligands (A targets may with a lot of ligands), if path has decoy/active label will be 0/1.
```
./single_target_feature_extractor.py -p /y/Aurora/Fernie/Kernie/A3FQ16/codi/target/receptor.pdbqt -l /y/Aurora/Fernie/Kernie/A3FQ16/codi/docking_pdbqt/active/BK1.pdbqt /y/Aurora/Fernie/Kernie/A3FQ16/codi/docking_pdbqt/active/BK3.pdbqt /y/Aurora/Fernie/Kernie/A3FQ16/codi/docking_pdbqt/decoy/ZINC000003513030.pdbqt /y/Aurora/Fernie/Kernie/A3FQ16/codi/docking_pdbqt/decoy/ZINC000004252506.pdbqt -o ~/Documents/tmp.pickle
```
Sequential extract a dataset interaction feature.  
`-i`: Folder of a dataset.  
`-d`: Depth of the folder. (DUD-E:1, Kernie:2)  
`-o`: Output folder.  
`-m`: If set this, program will extract multiple pose feature.  
```
./sequential_feature_extract.py -i /y/Aurora/Fernie/data/structure_based_data/DUD-E -o /y/Aurora/Fernie/data/structure_based_data/DUD-E_feature/multi_pose/ -d 1 -m
```
### 3.6.3. Package multiple targets feature (.pickle)  
`-i`: Folder includes a lot of pickle files.   
`-o`: Package all the small pickle files into a big one.    
```
./package_data.py -i /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/test_2 -o /y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/test_2.pickle
```
### 3.6.4. Under sampling keep train and valid 1:1 (.pickle)  
Only suit to single pose feature file.  
`-ir`: Folder saves a lot of raw interaction feature files.  
`-or`: Output folder save undersampled interaction feature files.  
`-if`: Single feature file path.   
`-of`: Single output path.  
```
./under_sampling.py -ir /y/Aurora/Fernie/data/structure_based_data/MUV_feature/single_pose -or /y/Aurora/Fernie/data/structure_based_data/MUV_feature/single_pose_undersampling
```
### 3.6.5. Generate cross validation Fernie feature data  (.pickle)
`-r`: Raw single pose interaction feature files (used in test set).  
`-u`: Undersampled single pose interaction (used in training set).  
`-c`: Cross validation information folder.  
`-o`: Output folder.  
```
./generate_cv_data.py -r /y/Aurora/Fernie/data/structure_based_data/Kernie_feature/single_pose -u /y/Aurora/Fernie/data/structure_based_data/Kernie_feature/single_pose_under_sampling -c /y/Aurora/Fernie/data/Kernie_clur_3_fold -o /y/Aurora/Fernie/data/structure_based_data/Kernie_3_fold
```
### 3.6.6. Generate cross validation ligand-based data (.csv)
`-i`: Folder saves a lot of ligand-based interaction information.  
`-c`: Cross validation information folder.  
`-o`: Output folder.  
```
./merge_ligand_data.py -i /y/Aurora/Fernie/data/ligand_based_data/DUD-E -c /y/Aurora/Fernie/data/DUD-E_clur_3_fold -o y/Aurora/Fernie/data/ligand_based_data/DUD-E_cv
```
### 3.6.7. Add InChI lines and Drop duplicate data (.csv)
```
./rm_duplicate.py -i /y/Aurora/Fernie/data/ligand_based_data/DUD-E_cv -o /y/Aurora/Fernie/data/ligand_based_data/DUD-E_cv
```
### 3.6.8. Split tensor k-fold cross validation (.pt)
The program will read pickle file and process it to tensor and split to k-fold and save out in the same folder of input file which named the same as input file.
```
./cv_split.py -i /y/Aurora/Fernie/data/structure_based_data/DUD-E_feature/DUD-E.undersample.pickle
```
### 3.6.9. Some other scripts
|Script|Description|
|:-|:-|
|split_cluster.py|Split data into k-fold by CD-HIT result.|
|torch_summary.py|Show model details.|
|multi_dataset_cv_split.py|Split training set into k-fold tensor(default: 10-fold)|
|model.py|Defination of Fernie|
