#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import seaborn as sns
import onnx
from os.path import join
from onnx import numpy_helper
import numpy as np

dataset = 'DUD-E'
exp_name = 'cf600_h100_dp0.1_False_batch320_down'
exps_dir = '/y/Aurora/Fernie/EXPS/'


fold = 0

model = onnx.load(join(exps_dir, exp_name, 'cv', dataset, f'fold_{fold}', 'models/fernie.onnx'))

weights = [] 
names = []
for t in model.graph.initializer:
    weights.append(numpy_helper.to_array(t))
    names.append(t.name)

model_weights = {}

for name, weight in zip(names, weights):
    model_weights[name] = weight
    
conv1 = model_weights['conv1.weight']

filters = []
for i in range(len(conv1)):
    filters.append(conv1[i,0,:])
    

filters_np = np.concatenate(filters)
filters_np = np.absolute(filters_np)

filters_np_mean = filters_np.mean(axis=0)

sns.heatmap(filters_np, cmap="YlGnBu")

import seaborn as sns
x = list(range(len(filters_np_mean)))
y = list(filters_np_mean)
sns.barplot(x=x, y=y)

atoms = []
for i in range(1, 27):
    end = i*200
    start = end - 200
    atoms.append(filters_np[:,start:end].mean(axis=1).reshape(600,1))
atom_np = np.concatenate(atoms, axis=1)
sns.heatmap(atom_np, xticklabels=list(range(1,27)), yticklabels=False, 
            cmap="YlGnBu")
atom_np_mean = atom_np.mean(axis=0)
x = list(range(len(atom_np_mean)))
y = list(atom_np_mean)
sns.barplot(x=x, y=y)


ligand_atoms = []
ligand_atoms.append(atom_np[:,0:7])
ligand_atoms.append(atom_np[:,9:15])
ligand_atoms.append(atom_np[:,17:23])


protein_atoms = []
protein_atoms.append(atom_np[:,7:9])
protein_atoms.append(atom_np[:,15:17])
protein_atoms.append(atom_np[:,23:27])

resettled_atom_np = np.concatenate(ligand_atoms + protein_atoms, axis=1)

sns.heatmap(resettled_atom_np, xticklabels=list(range(1,27)), yticklabels=False, 
            cmap="YlGnBu")


resettled_atom_np_mean = resettled_atom_np.mean(axis=0)
x = list(range(len(resettled_atom_np_mean)))
y = list(resettled_atom_np_mean)
sns.barplot(x=x, y=y)


ligand_atoms_np = np.concatenate(ligand_atoms, axis=1)
protein_atoms_np = np.concatenate(protein_atoms, axis=1)
ligand_atoms_np_mean = ligand_atoms_np.mean()
protein_atoms_np_mean = protein_atoms_np.mean()
x = list(range(2))
y = [ligand_atoms_np_mean, protein_atoms_np_mean]
sns.barplot(x=['ligand', 'protein'], y=y)

atom_type = atom_np[0:8].mean()
distance = atom_np[8:16].mean()
charge = atom_np[16:24].mean()
amino = atom_np[24:26].mean()
y = [atom_type, distance, charge, amino]
sns.barplot(x=['Atom type', 'Distance', 'Charge', 'Amino acid'], y=y)





