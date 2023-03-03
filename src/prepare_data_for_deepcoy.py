#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

import os
from utils import copytree, try_copy, build_new_folder
from tqdm import tqdm
from glob import glob
import pandas as pd
from RDKitFileConverter import file2mol, mol2file
import gzip
import shutil
"""
从之前 DUD-E中 拷贝，每个靶点的target/
"""
# input_dir = '/y/Aurora/Fernie/data/structure_based_data/DUD-E'

# targets = os.listdir(input_dir)

# ouput_dir = '/y/Aurora/Fernie/data/structure_based_data/DUD-E_DeepCoy'

# for target in tqdm(targets):
#     src = os.path.join(input_dir, target, 'target')
#     dst = os.path.join(ouput_dir, target, 'target')
    
#     copytree(src, dst)

"""
从 之前的 DUD-E中 拷贝每个靶点的 cmpd_library/active/
"""
# input_dir = '/y/Aurora/Fernie/data/structure_based_data/DUD-E'

# targets = os.listdir(input_dir)

# ouput_dir = '/y/Aurora/Fernie/data/structure_based_data/DUD-E_DeepCoy'

# for target in tqdm(targets):
#     src = os.path.join(input_dir, target, 'cmpd_library/active')
#     dst = os.path.join(ouput_dir, target, 'cmpd_library/active')
    
#     copytree(src, dst)

"""
从原始 DUD-E 中，拷贝 actives_final.mol2.gz， actives_final.sdf.gz
"""

# input_dir = '/y/Aurora/Fernie/data/structure_based_data/raw/DUD-E/*/act*.gz'

# srcs = glob(input_dir)

# for src in tqdm(srcs):
#     # src = srcs[0]
#     target = src.split('/')[-2]
#     file_name = os.path.basename(src)
#     dst_dir = os.path.join(ouput_dir, target, 'cmpd_library')
#     # build_new_folder(dst_dir)
#     dst = os.path.join(dst_dir, file_name)
#     try_copy(src, dst)
    
"""
DUD-E 拷贝 actives_final.smi 并只保留 SMILES和ChEMBL列。
"""
# input_dir = '/y/Aurora/Fernie/data/structure_based_data/raw/DUD-E/*/actives_final.ism'
# ouput_dir = '/y/Aurora/Fernie/data/structure_based_data/DUD-E_DeepCoy'
# srcs = glob(input_dir)
# failed = []
# for src in tqdm(srcs):
#     # src = failed[0]
#     target = src.split('/')[-2]
#     file_name = os.path.basename(src)
#     dst_dir = os.path.join(ouput_dir, target, 'cmpd_library')
#     dst = os.path.join(dst_dir, file_name)
#     dst = dst.replace('ism', 'smi')
#     df = pd.read_csv(src, sep=' ', header=None)
#     try:
#         df.iloc[:,[0,2]].to_csv(dst, sep=' ', index=False, header=False)
#     except:
#         df.to_csv(dst, sep=' ', index=False, header=False)
#         failed.append(src)
# print(failed)
"""
DUD-E 处理 actives_final.smi，只保留 SMILES，保存为 actives_final_only_smiles.smi
"""
# input_dir =  '/y/Aurora/Fernie/data/structure_based_data/DUD-E_DeepCoy/*/*/*.smi'
# srcs = glob(input_dir)
# for src in tqdm(srcs):
#     # src = srcs[0]
#     df = pd.read_csv(src, sep=' ', header=None)
#     dst = src.replace('.smi', '_only_smiles.smi')
#     df.iloc[:,0].to_csv(dst, sep=' ', header=False, index=False)

"""
整理 DUD-E infomation excel 文件，加上了sequence，number of actives.
"""
# input_dir =  '/y/Aurora/Fernie/data/structure_based_data/DUD-E_DeepCoy/*/*/actives_final.smi'
# srcs = glob(input_dir)
# targets = []
# actives_n = []
# for src in tqdm(srcs):
#     # src = srcs[0]
#     df = pd.read_csv(src, sep=' ', header=None)
#     target = src.split('/')[-3]
#     targets.append(target)
#     actives_n.append(len(df))

# df = pd.DataFrame({
#     'target': targets,
#     'no_actives': actives_n
#     })

# DUDE_into = pd.read_csv('/y/Aurora/Fernie/data/DUD-E_info.csv')
# DUDE_seq = pd.read_csv('/y/Aurora/Fernie/data/DUD-E.target.sequence.csv')


# DUDE_into.rename(columns={'Target Name': 'Target_Name'}, inplace=True)
# DUDE_into['Target_Name'] = [x.lower() for x in DUDE_into['Target_Name']]

# DUDE_m = DUDE_into.merge(DUDE_seq, left_on='Target_Name', right_on='target', copy=False)
# del DUDE_m['target']

# DUDE_mm = DUDE_m.merge(df, left_on='Target_Name', right_on='target')
# del DUDE_mm['target']

# DUDE_mm.to_excel('/y/Aurora/Fernie/data/DUD-E_info.xlsx', index=False)
"""
DUD-E 解压 .gz
"""
# input_dir = '/y/Aurora/Fernie/data/structure_based_data/DUD-E_DeepCoy/*/*/*.gz'
# srcs = glob(input_dir)
# for src in tqdm(srcs):
#     with gzip.open(src, 'rb') as f_in:
#         with open(src.replace('.gz', ''), 'wb') as f_out:
#             shutil.copyfileobj(f_in, f_out)

"""
将 target/ 中的 ref_ligand.pdbqt 转成 ref_ligand.sdf
"""

# input_dir = '/y/Aurora/Fernie/data/structure_based_data/DUD-E_DeepCoy/*/*/ref_ligand.pdbqt'
# srcs = glob(input_dir)

# write_failed = []
# for src in tqdm(srcs):
#     # src = '/y/Aurora/Fernie/data/structure_based_data/DUD-E_DeepCoy/fkb1a/target/ref_ligand.pdbqt'
#     # dst = '/y/Aurora/Fernie/data/structure_based_data/DUD-E_DeepCoy/fkb1a/target/ref_ligand.sdf'
#     try:
#         mols = file2mol(src)
#         dst = src.replace('.pdbqt', '.sdf')
#         mol2file(mols, dst)
#     except:
#         write_failed.append(dst)

"""
从之前 Kernie 中拷贝，每个靶点的target/
"""

# input_dir = '/y/Aurora/Fernie/data/structure_based_data/Kernie/'
# ouput_dir = '/y/Aurora/Fernie/data/structure_based_data/Kernie_DeepCoy'

# srcs = glob('/y/Aurora/Fernie/data/structure_based_data/Kernie/*/*/target')

# for src in tqdm(srcs):
#     # src = srcs[0]
#     target, con = src.split('/')[-3:-1]
#     dst = os.path.join(ouput_dir, target, con, 'target')
#     copytree(src, dst)


# ouput_dir = '/y/Aurora/Fernie/data/structure_based_data/Kernie_DeepCoy'

# srcs = glob('/y/Aurora/Fernie/data/structure_based_data/Kernie/*/*/target/ref_ligand.pdbqt')

# for src in tqdm(srcs):
#     # src = srcs[0]
#     target, con = src.split('/')[-4:-2]
#     dst = os.path.join(ouput_dir, target, con, 'ref_ligand.pdbqt')
#     if os.path.exists(dst):
#         os.remove(dst)
#     dst = os.path.join(ouput_dir, target, con, 'target', 'ref_ligand.pdbqt')
#     if os.path.exists(dst):
#         os.remove(dst)
#     if os.path.exists(dst):
#         os.remove(dst)
#     try_copy(src, dst)

    
"""
从 之前的 Kernie 中 拷贝每个靶点的 cmpd_library/active/
"""
# input_dir = '/y/Aurora/Fernie/data/structure_based_data/Kernie/'
# ouput_dir = '/y/Aurora/Fernie/data/structure_based_data/Kernie_DeepCoy'
# targets = os.listdir(input_dir)

# srcs = glob('/y/Aurora/Fernie/data/structure_based_data/Kernie/*/*/target')

# for src in tqdm(srcs):
#     # src = srcs[0]
#     target, con = src.split('/')[-3:-1]
#     dst = os.path.join(ouput_dir, target, con, 'target')
#     copytree(src, dst)

"""
从之前 Kernie 中拷贝，每个靶点的 cmpd_library/active.sdf，并给 sdf 中每个 active 命名

"""

# input_dir = '/y/Aurora/Fernie/data/structure_based_data/Kernie/*/*/cmpd_library/active.sdf'
# ouput_dir = '/y/Aurora/Fernie/data/structure_based_data/Kernie_DeepCoy'
# srcs = glob(input_dir)

# for src in tqdm(srcs):
#     # src = srcs[0]
#     target, con = src.split('/')[-4:-2]
#     mols = file2mol(src)
#     count = 0
#     with open(src, 'r') as f:
#         content = f.readlines()
#     for i, line in enumerate(content):
#         if '>  <ID>' in line:
#             mols[count].title = content[i+1].strip()
#             count += 1
#     dst = os.path.join(ouput_dir, target, con, 'cmpd_library/actives_final.sdf')
#     mol2file(mols, dst)
#     with open(dst, 'r') as f:
#         content = f.readlines()
#     content_new = []
#     for i, line in enumerate(content):
#         if '>  <ID>' in line:
#             continue
#         if '>  <ID>' in content[i-1]:
#             continue
#         content_new.append(line)
#     with open(dst, 'w') as f:
#         for line in content_new:
#             f.write(line)
#     mols = file2mol(dst)
#     dst = os.path.join(ouput_dir, target, con, 'cmpd_library/actives_final.smi')
#     mol2file(mols, dst)
#     df = pd.read_csv(dst, sep=' ', header=None)
#     dst = os.path.join(ouput_dir, target, con, 'cmpd_library/actives_final_only_smiles.smi')
#     df.iloc[:,0].to_csv(dst, sep=' ', header=False, index=False)

"""
Kernie only smiles
"""
# input_dir =  '/y/Aurora/Fernie/data/structure_based_data/Kernie_DeepCoy/*/*/*/actives_final_only_smiles.smi'
# srcs = glob(input_dir)
# for src in srcs:
#     # src = '/y/Aurora/Fernie/data/structure_based_data/Kernie_DeepCoy/Q99986/cidi/cmpd_library/actives_final_only_smiles.smi'
#     df = pd.read_csv(src, sep='	', header=None)
#     df = df.iloc[:,0]
#     df.to_csv(src, header=None, index=False)

"""
Kernie 将 target/ 中的 ref_ligand.pdbqt 转成 ref_ligand.sdf
"""
# input_dir = '/y/Aurora/Fernie/data/structure_based_data/Kernie_DeepCoy/*/*/target/ref_ligand.pdb'

# srcs = glob(input_dir)

# write_failed = []
# for src in tqdm(srcs):
#     # src = srcs[0]
#     # dst = '/y/Aurora/Fernie/data/structure_based_data/DUD-E_DeepCoy/fkb1a/target/ref_ligand.sdf'
#     try:
#         mols = file2mol(src)
#         dst = src.replace('.pdb', '.sdf')
#         mol2file(mols, dst)
#     except:
#         write_failed.append(dst)

"""
MUV Convert reference ligand PDBQT to sdf file.
"""
# input_dir = '/y/Aurora/Fernie/data/structure_based_data/MUV/*/*/target/ligand_ref.pdbqt'
# srcs = glob(input_dir)
# failed = []
# for src in tqdm(srcs):
#     # src = srcs[0]
#     try:
#         mols = file2mol(src)
#         dst = src.replace('pdbqt', 'sdf')
#         mol2file(mols, dst)
#     except:
#         failed.append(src)
# print(failed)
from rdkit import Chem
['/y/Aurora/Fernie/data/structure_based_data/MUV/Nuclear/692/target/ligand_ref.pdbqt', 
 '/y/Aurora/Fernie/data/structure_based_data/MUV/Nuclear/600/target/ligand_ref.pdbqt', 
 '/y/Aurora/Fernie/data/structure_based_data/MUV/Kinase/548/target/ligand_ref.pdbqt', 
 '/y/Aurora/Fernie/data/structure_based_data/MUV/Kinase/644/target/ligand_ref.pdbqt', 
 '/y/Aurora/Fernie/data/structure_based_data/MUV/Protease/852/target/ligand_ref.pdbqt', 
 '/y/Aurora/Fernie/data/structure_based_data/MUV/Protease/846/target/ligand_ref.pdbqt', 
 '/y/Aurora/Fernie/data/structure_based_data/MUV/Protease/832/target/ligand_ref.pdbqt']

file = '/y/Aurora/Fernie/data/structure_based_data/MUV/Nuclear/692/target/ligand_ref.pdbqt'
mols = file2mol(file)
res = Chem.MolFromMolBlock(file,sanitize=False)
