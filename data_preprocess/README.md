
### Data preprocessing

Docking results are in /y/Fernie/kinase_project/data/DUD-E/ada/docking_pdbqt for target **ada**.

0. check_after_docking.py
   **Description:** Check the number of pdbqt files after docking and the number of pdbqt files before docking.

1. feature_extract.py
   **Description**: Extract no grid feature from docking pdbqt file, "True": multi pose, "False": single pose.
   **Output**: ada/tmp/train.no_grid.pickle
   ```
   ./feature_extract.py -i /y/Fernie/kernie_mini -o /y/Fernie/output/kernie_output -m False
   ```

2. pdb2FASTA.py
   **Description:** Process all target.pdb files into FASTA files. Extract the protein sequence information from pdb files.
   **Input**: /y/Fernie/kinase_project/data/DUD-E
   **Output:** ada/target/receptor.fst

3. integrate_FASTA.py
   **Description:** Integrate all FASTA files from the previous step into one file.  
   **Input:** /y/Fernie/kinase_project/data/DUD-E  
   **Output:** /y/Fernie/kinase_project/data/DUD-E.fst  


4. Use CD-HIT to cluster the protein targets based on their sequence similarity.
http://weizhong-lab.ucsd.edu/cdhit-web-server/cgi-bin/index.cgi , get file *.fas.1.clstr.sorted, and rename to  XXX_clstr
/y/Fernie/kinase_project/cluster_3_fold

4. gen_3_clus_fold.py  
   **Description**: According to the output of CD-HIT, divide targets into 3 fold
   **Input**: python gen_3_clus_fold.py /y/Fernie/kinase_project/data/kinformation  
   **Output**: /y/Fernie/kinase_project/cluster_3_fold/kinformation/split_files  

        ```
        split_files/  
        ├── test_0.taskfolders  
        ├── test_1.taskfolders
        ├── test_2.taskfolders
        ├── train_0.taskfolders
        ├── train_1.taskfolders
        └── train_2.taskfolders
        ```

5. undersample.py  
   **Description:** Downsample each target's actives and decoys into a set of compounds with active:decoy = 1:1.
     * Choose the top number of decoys whose total matches the number of actives.
   **Input:** /y/Fernie/kinase_project/data/kinformation
   **Output:** /y/Fernie/kinformation/O00444/cidi/tmp/feature_undersample.pickle
     * cidi or codi is conformation.
   

6. gen_cross_valid_data.py  
   * **Description**: Generate cross-valiation data: 3 train_i.pickle and 3 test_i folder which for 3-fold cross-validation.  
   * **Input**: /y/Fernie/kinase_project/cluster_3_fold/DUD-E/split_files  
   * **Output**: /y/Fernie/kinase_project/cluster_3_fold/DUD-E/data
        ```
        data
        ├── test_0
        ├── test_1
        ├── test_2
        ├── train_0.pickle
        ├── train_1.pickle
        └── train_2.pickle
        ```

7. cross_validation.py
   * **Description**: Do clustered three-fold cross validation on DUD-E and kinformation and save models and scores.
   * **Input**: ??
   * **Output**: /y/Fernie/kinase_project/final_score/DUD-E/models ??  /y/Fernie/kinase_project/final_score
        ```
        final_score
        ├── DUD-E
        │   ├── models
        │   └── scores
        └── kinformation
            ├── models
            └── scores
        ```

- Get the performance of the cross-validation

8. fernie_cv_performance.py  
**Descripation:** Get the Fernie performance of clustered three-fold cross validation on a dataset such as DUD-E, kinformation and etc.   
**Usage:** fernie_cv_performance.py  
**Output:** 

```
final_performance
├── DUD-E
│   ├── ddk.performance
│   └── ddk.summary
└── kinformation
    ├── ddk.performance
    └── ddk.summary
```

9. smina_performance.py  
   * **Descripation:** Get the performance of Smina on a dataset such as DUD-E, kinformation and etc.  
   * **Usage**: smina_performance.py  
   * **Output**: 

```
final_performance
├── DUD-E
│   ├── smina.performance
│   └── smina.summary
└── kinformation
    ├── smina.performance
    └── smina.summary
```

### 1.3.4. Independent test or external validation.

Train whole dataset and test it with other independent dataset.
- The data for train model with whole dataset.

10. integrate_a_dataset.py  
**Description**: Undersample each targets active and decoy, and integrate the whole dataset's pickle files into one picklefile.  
**Usage:** python integrate_a_dataset.py   
**Output:** 
```
final_model
├── DUD-E.pickle
└── kinformation.pickle
```
- Train the model by above data prepared.
11. train_a_dataset.py  
**Descripation:** Train a model with a whole dataset and output a *.h5 file.  
**Usage:**  python train_a_dataset.py  
**Output:** 
final_model/XXX.h5

- Test on dependent dataset(MUV)
12.  external_validation.py  
**Description:** Test the model trained by a whole dataset on a independent.   
**Usage:** python external_validation.py  
**Output:**
final_score/MUV/scores/XXX.score