#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""


"""
Package Kernie
"""
./package_data.py -i /y/Aurora/Fernie/data/structure_based_data/Kernie_feature/single_pose -o /y/Aurora/Fernie/data/structure_based_data/Kernie_feature/single_pose.pickle



"""
Package MUV
"""
./package_data.py -i /y/Aurora/Fernie/data/structure_based_data/MUV_feature/single_pose_undersampling -o /y/Aurora/Fernie/data/structure_based_data/MUV_feature/single_pose_undersampling.pickle


"""
Under sample for DUD-E and Kernie
"""
dataset=DUD-E
./under_sampling.py -ir /y/Aurora/Fernie/data/structure_based_data/${dataset}_feature/single_pose -or /y/Aurora/Fernie/data/structure_based_data/${dataset}_feature/single_pose_under_sampling
dataset=Kernie
./under_sampling.py -ir /y/Aurora/Fernie/data/structure_based_data/${dataset}_feature/single_pose -or /y/Aurora/Fernie/data/structure_based_data/${dataset}_feature/single_pose_under_sampling

"""
Generate cross validation data
"""
dataset=DUD-E
./generate_cv_data.py -r /y/Aurora/Fernie/data/structure_based_data/${dataset}_feature/single_pose -u /y/Aurora/Fernie/data/structure_based_data/${dataset}_feature/single_pose_under_sampling -c /y/Aurora/Fernie/data/${dataset}_clur_3_fold -o /y/Aurora/Fernie/data/structure_based_data/${dataset}_3_fold
dataset=Kernie
./generate_cv_data.py -r /y/Aurora/Fernie/data/structure_based_data/${dataset}_feature/single_pose -u /y/Aurora/Fernie/data/structure_based_data/${dataset}_feature/single_pose_under_sampling -c /y/Aurora/Fernie/data/${dataset}_clur_3_fold -o /y/Aurora/Fernie/data/structure_based_data/${dataset}_3_fold

"""
Perpare data .pickle
"""
./extract_specific_subset.py