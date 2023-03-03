#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import numpy as np
import pandas as pd
import os
import multiprocessing
from multiprocessing import Pool
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit.Chem.Scaffolds import MurckoScaffold
from moses.metrics import mol_passes_filters, QED, SA, logP
from functools import partial
from rdkit.Chem import rdinchi
from os.path import basename, splitext, join, dirname
from glob import glob


"""
Remove duplicates molecules
"""
def drop_duplicate_mols(file=None, df=None, out=None, sep=None, header=None, 
                              col_names=None, inplace=False, save_out=True,
                              remain_InChI=True):
    if file is not None:
        if splitext(basename(file))[1] == '.smi':
            if sep is None and header is None:
                sep = ' '
                header = None
            df = pd.read_csv(file, sep = sep, header = header)
            if col_names is not None:
                df.columns = col_names
            elif df.shape[1] == 1:
                df.columns = ['SMILES']
            elif df.shape[1] == 2:
                df.columns = ['SMILES', 'ID']
            else:
                print(f'{file} columns names is not given.')
                return
        elif splitext(basename(file))[1] == '.csv':
            if sep is None and header is None:
                sep = ','
                header = 0
            df = pd.read_csv(file, sep = sep, header = header)
        elif splitext(basename(file))[1] == '.tsv':
            if sep is None and header is None:
                sep = '\t'
                header = 0
            df = pd.read_csv(file, sep = sep, header = header)
        else:
            if sep is not None and header is not None:
                df = pd.read_csv(file, sep = sep, header = header)
            else:
                print('You need to give sep and header flag.')
                return
    elif df is None:
        print('please check your input.')
        return
    if 'SMILES' not in df.columns:
        print('The file must have have SMILES column.')
        return
    print('Before:', df.shape[0], 'moleculs.')
    if 'InChI' not in df.columns:
        df = SMILES2mol2InChI(df)
    df.drop_duplicates('InChI', inplace=True)
    
    print('After drop duplicates:', df.shape[0], 'moleculs.')

    if not remain_InChI:
        del df['InChI']

    if save_out:
        if out is not None:
            df.to_csv(out, index=False, header=header, sep=sep)
        elif file is not None:
            if inplace:
                out = file
            else:
                file_dir = dirname(file)
                file_name, suffix = splitext(basename(file))
                file_name = file_name + '_drop_duplicates'
                out = join(file_dir, ''.join([file_name, suffix]))
        else:
            print('Not given a output path.')
            return
        if splitext(basename(out))[1] == '.smi':
            if 'InChI' in df.columns:
                del df['InChI']
        if splitext(basename(out))[1] == '.smi':
            header=False
        df.to_csv(out, index=False, header=header, sep=sep)
        if inplace:
            print(f'Success to replace raw input file {out}.')
        else:
            print(f'Success to save out to {out}')
    
    return df

def drop_multi_files_duplicate_mols(files=None, input_dir=None, 
                                    output_dir = None,
                                    suffix=None,
                                    sep=None, header=None, col_names=None, 
                                    inplace=True, save_out=True):
    if suffix == 'smi':
        sep = ' '
        header = None
    
    if files is None and input_dir is not None:
        files = glob(join(input_dir, f'*.{suffix}'))
    else:
        print('Please check your input.')
        return
    dfs = []
    for file in files:
        df = drop_duplicate_mols(file=file, sep=sep, header=header,
                              col_names=col_names, inplace=inplace, 
                              save_out=save_out)
        dfs.append(df)
    return dfs


"""
=========================== parallelize apply =================================
"""

def parallelize_dataframe(df, func, **kwargs):
    CPUs = multiprocessing.cpu_count()

    num_partitions = CPUs - 1 # number of partitions to split dataframe
    num_cores = CPUs - 1 # number of cores on your machine

    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    func = partial(func, **kwargs)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def judge_whether_is_dir(path):
    if not os.path.isdir(path):
        return False
    else:
        return True

def remain_path_of_dir(x):
    return list(filter(judge_whether_is_dir, x))

"""
-------------------------------------------------------------------------------
String to mol 
-------------------------------------------------------------------------------
"""
def InChI2MOL(inchi):
    try:
        mol = Chem.inchi.MolFromInchi(inchi)
        if not mol == None:
            return mol
        else:
            return np.nan
    except:
        return np.nan
    
def SMILES2MOL(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol == None:
            return mol
        else:
            return np.nan
    except:
        return np.nan

"""
-------------------------------------------------------------------------------
Parallel string to mol
-------------------------------------------------------------------------------
"""
def parall_SMILES2MOL(df):
    df['ROMol'] = df['SMILES'].apply(SMILES2MOL)
    return df

def parall_InChI2MOL(df):
    df['ROMol'] = df['InChI'].apply(InChI2MOL)
    return df
"""
-------------------------------------------------------------------------------
Mol to string
-------------------------------------------------------------------------------
"""
def MOL2SMILES(mol):
    try:
        sm = Chem.MolToSmiles(mol)
        return sm
    except:
        return np.nan

def MOL2InChI(mol):
    try:
        inchi, retcode, message, logs, aux = rdinchi.MolToInchi(mol)
        return inchi
    except:
        return np.nan

def MOL2ECFP4(mol, nbits=2048, radius=2):
    try:
        res = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
        return np.array(res)
    except:
        return np.nan

"""
-------------------------------------------------------------------------------
Parallel add mol to string
-------------------------------------------------------------------------------
"""
def add_SMILES(df):
    df['SMILES'] = df.ROMol.apply(MOL2SMILES)
    return df

def add_InchI(df):
    df['InChI'] = df.ROMol.apply(MOL2InChI)
    return df

def add_ECFP4(df, nbits, radius):
    mol2ECFP4V2 = partial(MOL2ECFP4, nbits=nbits, radius=radius)
    df['ECFP4'] = df['ROMol'].apply(mol2ECFP4V2)
    return df
"""
--------------------------------------------------------------------------------
Parallel convert from one molecular respresentation to another.
--------------------------------------------------------------------------------
"""
def InChI2mol2SMILES(df):
    df = parallelize_dataframe(df, parall_InChI2MOL)
    df = parallelize_dataframe(df, add_SMILES)
    del df['ROMol']
    return df

def SMILES2mol2InChI(df):
    df = parallelize_dataframe(df, parall_SMILES2MOL)
    df = parallelize_dataframe(df, add_InchI)
    del df['ROMol']
    return df

"""
-------------------------------------------------------------------------------
Property functions
-------------------------------------------------------------------------------
"""
def judge_whether_has_rings_4(mol):
    r = mol.GetRingInfo()
    if len([x for x in r.AtomRings() if len(x)==4]) > 0:
        return False
    else:
        return True
    
def add_whether_have_4_rings(data):
    """4 rings"""
    data['4rings'] = data['ROMol'].apply(judge_whether_has_rings_4)
    return data

def four_rings_filter(df):
    df = parallelize_dataframe(df, add_whether_have_4_rings)
    df = df[df['4rings']==True]
    del df['4rings']
    return df

def MW(mol):
    try:
        res = Chem.Descriptors.ExactMolWt(mol)
        return res
    except:
        return np.nan

def HBA(mol):
    try:
        res = Chem.rdMolDescriptors.CalcNumHBA(mol)
        return res
    except:
        return np.nan

def HBD(mol):
    try:
        res = Chem.rdMolDescriptors.CalcNumHBD(mol)
        return res
    except:
        return np.nan

def TPSA(mol):
    try:
        res = Chem.rdMolDescriptors.CalcTPSA(mol)
        return res
    except:
        return np.nan

def NRB(mol):
    try:
        res =  Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
        return res
    except:
        return np.nan
    
def get_num_rings(mol):
    try:
        r = mol.GetRingInfo()
        res = len(r.AtomRings())
        return res
    except:
        return np.nan

def get_num_rings_6(mol):
    try:
        r = mol.GetRingInfo()
        res = len([x for x in r.AtomRings() if len(x) > 6])
        return res
    except:
        return np.nan

def LOGP(mol):
    try:
        res = logP(mol)
        return res
    except:
        return np.nan
    
def MCF(mol):
    """
    Keep molecules whose MCF=True
    MCF=True means toxicity. but toxicity=True is not bad if the patient is dying.
    """
    try:
        res = mol_passes_filters(mol)
        return res
    except:
        return np.nan

def synthesis_availability(mol):
    """
    0-10. smaller, easier to synthezie.
    not very accurate.
    """
    try:
        res = SA(mol)
        return res
    except:
        return np.nan
    
def estimation_drug_likeness(mol):
    """
    0-1. bigger is better.
    """
    try:
        res = QED(mol)
        return res
    except:
        return np.nan

def add_descriptors(df):
    df['MW'] = df.ROMol.apply(MW)
    df['logP'] = df.ROMol.apply(LOGP)
    df['HBA'] = df.ROMol.apply(HBA)
    df['HBD'] =  df.ROMol.apply(HBD)
    df['TPSA'] = df.ROMol.apply(TPSA)
    df['NRB'] = df.ROMol.apply(NRB)
    df['MCF'] = df.ROMol.apply(MCF)
    df['SA'] = df.ROMol.apply(synthesis_availability)
    df['QED'] = df.ROMol.apply(estimation_drug_likeness)
    df['rings'] = df.ROMol.apply(get_num_rings)
    return df

def add_features(df):
    df = parallelize_dataframe(df, add_descriptors)
    df = df.dropna()
    return df

def validity_filter(df):

    print("Start to remove invalid SMILES...")
    
    if "ROMol" not in df.columns:
        df = parallelize_dataframe(df, parall_SMILES2MOL)
        df = df.dropna()
    print("Finished.")
    return df

def property_filter(df, condition):
    """
    -----------------
    descriptor filter
    -----------------
    """
    print('descriptor filter')
    df = parallelize_dataframe(df, parall_SMILES2MOL)
    df = df.dropna()
    df = four_rings_filter(df)
    df = add_features(df)
    df[['MW', 'logP', 'TPSA', 'SA', 'QED']] = df[['MW', 'logP', 'TPSA', 'SA',\
        'QED']].apply(lambda x: round(x, 3))
    df = df.query(condition)
    df = df.reset_index(drop=True)
    return df



def fingerprint_similarity(line): 
    tanimoto = DataStructs.FingerprintSimilarity(line[0], line[1])
    return tanimoto

def molecule_in_patent(sample_fingerprint, patent_fingerprints, l, ths):
    fp_list = [sample_fingerprint] * int(l)
    matrix = pd.DataFrame({'SMILES':fp_list, 'patent':patent_fingerprints})
    matrix['tanimoto'] = matrix.apply(fingerprint_similarity, axis=1)
    if len(matrix.query('tanimoto%s' % ths)) > 0:
        return True
    else:
        return False
    
def add_patent(df, patent_fingerprints, l, ths):
    molecule_in_patentv2 = partial(molecule_in_patent, 
                                   patent_fingerprints=patent_fingerprints, 
                                   l=l, ths=ths)
    df['patent'] = df['ECFP4'].apply(molecule_in_patentv2)
    return df

def hard_patent_filter(df, patent, ths='==1'):
    """
    -------------------------------
    Remove molcules those in patent
    -------------------------------
    """
    if "ROMol" not in df.columns:
        df = parallelize_dataframe(df, parall_SMILES2MOL)
        df = df.dropna()
    if "ROMol" not in patent.columns:
        patent = parallelize_dataframe(patent, parall_SMILES2MOL)
        patent = patent.dropna()
    df = parallelize_dataframe(df, add_ECFP4)
    patent = parallelize_dataframe(patent, add_ECFP4)
    patent_fingerprints = patent.ECFP4
    l = len(patent_fingerprints)
    df = parallelize_dataframe(df, add_patent, 
                               patent_fingerprints=patent_fingerprints, 
                               l=l, ths=ths)
    df = df[df['patent']==False]
    del df['patent']
    df = df.reset_index(drop=True)
    del df['ECFP4']
    return df, patent

def soft_patent_filter(df, patent, ths='>0.85'):
    """
    -----------------------------------
    Remove molcules Tc > 0.85 in patent
    -----------------------------------
    """
    df, patent = hard_patent_filter(df, patent, ths)
    return df, patent

def get_scaffold_mol(mol):
    try: 
        res = GetScaffoldForMol(mol)
        return res
    except:
        return np.nan

def add_atomic_scaffold(df):
    df['atomic_scaffold_mol'] = df.ROMol.apply(get_scaffold_mol)
    return df

def substruct_match(df):
    # Chem.MolFromSmiles
    return df.ROMol.HasSubstructMatch(df.patent_scaffold_mol)

def add_substructure_match(matrix, outname='remain'):
    matrix[outname] = matrix.apply(substruct_match, axis=1)
    return matrix

def scaffold_in_patent(mol, patent_scaffolds, l):
    mol_list = [mol] * int(l)
    matrix = pd.DataFrame({'ROMol':mol_list, 
                           'patent_scaffold_mol':list(patent_scaffolds)})
    matrix = add_substructure_match(matrix)
    if len(matrix.query('remain==True')) > 0:
        return False
    else:
        return True 
    
def judge_substructure(df, col, patent_scaffolds, l, outname='remain'):
    scaffold_in_patentv2 = partial(scaffold_in_patent, 
                     patent_scaffolds=patent_scaffolds,
                     l=l)
    df[outname] = df[col].apply(scaffold_in_patentv2)
    return df

def atom_scaffold_filter(df, patent, col = 'atomic_scaffold_mol'):
    """
    ----------------------
    atomic scaffold filter
    ----------------------
    """
    if "ROMol" not in df.columns:
        df = parallelize_dataframe(df, parall_SMILES2MOL)
        df = df.dropna()
    if "ROMol" not in patent.columns:
        patent = parallelize_dataframe(patent, parall_SMILES2MOL)
        patent = patent.dropna()
    if "atomic_scaffold_mol" not in df.columns:
        df = parallelize_dataframe(df, add_atomic_scaffold)
        df = df.dropna()
    if "atomic_scaffold_mol" not in patent.columns:
        patent = parallelize_dataframe(patent, add_atomic_scaffold)
        patent = patent.dropna()
    patent_scaffolds = set(patent[col])
    l = len(patent_scaffolds)
    df = parallelize_dataframe(df, judge_substructure, 
                               col = col,
                               patent_scaffolds=patent_scaffolds, l=l)
    df = df[df['remain']==True]
    del df['remain']
    df = df.reset_index(drop=True)
    return df, patent

def get_graph_scaffold_mol(atomic_scaffold_mol):
    try:
        #atomic_scaffold_mol.Compute2DCoords()
        graph_scaffold_mol = MurckoScaffold.MakeScaffoldGeneric( 
            atomic_scaffold_mol)
        return graph_scaffold_mol
    except:
        return np.nan
    
def add_graph_scaffold(df, col='atomic_scaffold_mol', 
                       outname='graph_scaffold_mol'):
    df[outname] = df[col].apply(get_graph_scaffold_mol)
    return df

def grap_scaffold_filter(df, patent, col='graph_scaffold_mol'):
    """
    ----------------------
    graph scaffold filter
    ----------------------
    """
    if "ROMol" not in df.columns:
        df = parallelize_dataframe(df, parall_SMILES2MOL)
        df = df.dropna()
    if "ROMol" not in patent.columns:
        patent = parallelize_dataframe(patent, parall_SMILES2MOL)
        patent = patent.dropna()
    if "atomic_scaffold_mol" not in df.columns:
        df = parallelize_dataframe(df, add_atomic_scaffold)
        df = df.dropna()
    if "atomic_scaffold_mol" not in patent.columns:
        patent = parallelize_dataframe(patent, add_atomic_scaffold)
        patent = patent.dropna()
    if "graph_scaffold_mol" not in df.columns:
        df = parallelize_dataframe(df, add_graph_scaffold)
        df = df.dropna()
    if "graph_scaffold_mol" not in patent.columns:
        patent = parallelize_dataframe(patent, add_graph_scaffold)
        patent = patent.dropna()
    df, patent = atom_scaffold_filter(df, patent, col = col)
    return df, patent

def atomic_scaffold(df):
    if "atomic_scaffold" not in df.columns:
        df["atomic_scaffold"] = df["atomic_scaffold_mol"].apply(MOL2SMILES)
    return df

def graph_scaffold(df):
    if "graph_scaffold" not in df.columns:
        df["graph_scaffold"] = df["graph_scaffold_mol"].apply(MOL2SMILES)
    return df

def save_file(df, path):
    if "ROMol" in df.columns:
        del df["ROMol"]
    if "atomic_scaffold_mol" in df.columns:
        df = parallelize_dataframe(df, atomic_scaffold)
        del df['atomic_scaffold_mol']
    if "graph_scaffold_mol" in df.columns:
        df = parallelize_dataframe(df, graph_scaffold)
        del df["graph_scaffold_mol"]
    df.to_csv(path, index=False)


def filter_molecule(input_file, output_dir, condition_file, patent_file):
    try:
        with open(condition_file, 'r') as f:
            condition = f.readline()
            condition = condition.strip()
    except:
        print("Read condition file failed.")
        return
    try:
        df = pd.read_csv(input_file)
    except:
        print("Read compound file failed.")
        return
    try:
        patent = pd.read_csv(patent_file)
    except:
        print("Read patent file failed.")
        return
    df = property_filter(df, condition)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_file(df, os.path.join(output_dir, "property_filter.csv"))
    df, patent = hard_patent_filter(df, patent)
    save_file(df, os.path.join(output_dir, "hard_pat_filter.csv"))
    df, patent = soft_patent_filter(df, patent)
    save_file(df, os.path.join(output_dir, "soft_pat_filter.csv"))
    df, patent = atom_scaffold_filter(df, patent)
    save_file(df, os.path.join(output_dir, "atom_pat_filter.csv"))
    df, patent = grap_scaffold_filter(df, patent)
    save_file(df, os.path.join(output_dir, "grap_pat_filter.csv"))
