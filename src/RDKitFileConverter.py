#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
from copy import deepcopy
from os.path import splitext, basename, dirname, join, exists
from os import makedirs
from oddt.toolkits.rdk import readstring, readfile, Outputfile
from tqdm import tqdm

"""
========================== Convert file to mol ================================
"""

def file2mol(file, lazy=False, format_=None, protein=False):
    """
    >>> mols = file2mol('example.sdf')
    The format of input file is sdf
    Convert example.sdf to 1 molecule(s) format.
    Success!
    """
    if protein:
        target = 'protein(s)'
    else:
        target = 'molecule(s)'
    flag = 0
    if '.gz' in file:
        flag = 1
        file_tmp = deepcopy(file)
        file = file.replace('.gz', '')
    if format_ is None:
        format_ = splitext(basename(file))[1].split('.')[1]
        format_ = format_.lower()
        if format_ == 'ism':
            format_ = 'smi'
    if flag == 1:
        file = file_tmp
        flag = 0
    try:
        print('The format of input file is', format_)
        assert format_ in ['pdb', 'pdbqt', 'sdf', 'mol', 'mol2', 'smi', 'inchi']
    except:
        print('ERROR')
        print(f"Your file format {format} is not allowed. We support "
              ".pdb, .pdbqt, .sdf, .mol, .mol2, .smi, .inchi")
        return
    mols = list(readfile(format_, file, lazy=lazy))
    print(f'Convert {file} to {len(mols)} {target} format.')
    if protein:
        for x in mols:
            x.protein=True 
    print('Success!\n')
    return mols

def string2mol(string=None, format_='smi'):
    """
    >>> mol = string2mol('C1CCC1')
    """
    try:
        assert format_ in ['smi']
    except:
        print(f'Your input format {format_} is not collect. We support smi')
    mol = readstring(format_, string)
    return [mol]

"""
========================== Convert mol to file ================================
"""

def mol2file(mols, file, format_=None, names=None, overwrite=True, size=None, 
             split=False, protein=False):
    """
    >>> mol2file(mols, 'sdf', 'example.sdf')
    The format of output file is sdf
    Saving 1 molecule(s) to example.sdf
    Success!
    """
    if format_ is None:
        format_ = splitext(basename(file))[1].split('.')[1]
        format_ = format_.lower()
    if format_ in ['sdf', 'pdbqt', 'mol', 'mol2', 'pdb']:
        for mol  in mols:
            mol.addh()
            mol.make3D()
    if protein:
        target = 'protein(s)'
    else:
        target = 'molecule(s)'
    try:
        print('The format of output file is', format_)
        print(f'Saving {len(mols)} {target} to {file}')
        assert format_ in ['pdb', 'pdbqt', 'sdf', 'mol', 'mol2', 'smi', 'inchi']
    except:
        print('ERROR')
        print(f"Your output file format {format} is not allowed. We support "
              ".pdb, .pdbqt, .sdf, .mol, .mol2, .smi, .inchi")
        return
    if isinstance(mols, list):
        if names is not None:
            if len(names)!=len(mols):
                print('ERROR')
                print(f"Size of names isn't equal to size of {target}.")
                return
            else:
                for mol, name in tqdm(zip(mols, names)):
                    mol.title = name
        if len(mols) == 0:
            print('ERROR')
            print(f'Threre is no {target} in the input list, please check.')
            return
        elif len(mols) == 1:
            mols[0].write(format_, file, overwrite=overwrite, size=size)
        elif len(mols) > 1:
            if not split:
                writer = Outputfile(format_, file, overwrite=overwrite)
                for mol in tqdm(mols):
                    writer.write(mol)
                writer.close()
            else:
                out_dir = dirname(file)
                file_name = splitext(basename(file))[0]
                out_dir = join(out_dir, file_name)
                if not exists(out_dir):
                    makedirs(out_dir)
                for mol in tqdm(mols):
                    file = join(out_dir, '.'.join([mol.title, format_]))
                    mol.write(format_, file, overwrite=overwrite, size=size)
    else:
        print('ERROR\nInput must be list.')
        return
    print('\nSuccess!\n')
