import math
import json
from atomicWeightsDecimal import atomicWeightsDecimal as masses
import HoD_parameters as hods
import numpy as np
from rdkit import Chem

from pathlib import Path
PATH = str(Path(__file__).parent)

def get_mol(mol):
    """
    generate dictionary containing the atoms and coordinate of a molecule
    from RDKit mol object.
    :param mol: RDKit mol object
    :return: dict_A (dictionary with atoms and coordinate of molecule)

    """
    mol_d = {'index': [], 'atom': [], 'coordinates': []}
    for i, atom in enumerate(mol.GetAtoms()):
        positions = mol.GetConformer().GetAtomPosition(i)
        mol_d['index'].append(i)
        mol_d['atom'].append(atom.GetSymbol())
        mol_d['coordinates'].append([positions.x, positions.y, positions.z])
    return mol_d


def count_hb(mol_a, mol_b, atom_a, atom_b):
    """
    Counts the number of hydrogen bonds between two molecules.
    Covalent bonds are identified by A-H distance of less than 1.1 A.
    H-Bond are identified by H:B distance of less than 2.5 and angle less than 120°
    :param mol_a: dict: first monomer, see get_mol()
    :param mol_b: dict: second monomer, see get_mol()
    :param atom_a: str: Atomic symbol of element covalently bonded to H as a string
    :param atom_b: str: Atomic symbol of hydrogen bonded element as a string
    :return: int: count of hydrogen bonds of type AH-B between the two molecule
    """
    hb_count = 0

    # Count H-bonds from A to B
    for i, atom in enumerate(mol_a['atom']):
        if atom == atom_a:
            for j, atomH in enumerate(mol_a['atom']):
                if atomH == 'H':
                    disAH = math.dist(mol_a['coordinates'][i], mol_a['coordinates'][j])
                    if disAH <= 1.1:
                        for k, atomb in enumerate(mol_b['atom']):
                            if atomb == atom_b:
                                disHB = math.dist(mol_a['coordinates'][j], mol_b['coordinates'][k])
                                if disHB <= 2.5:
                                    disAB = math.dist(mol_a['coordinates'][i], mol_b['coordinates'][k])
                                    cosx = (disAH ** 2 + disHB ** 2 - disAB ** 2) / (2 * disAH * disHB)
                                    if cosx < -0.5:
                                        hb_count += 1

    # Count H-bonds from B to A
    for i, atom in enumerate(mol_b['atom']):
        if atom == atom_a:
            for j, atomH in enumerate(mol_b['atom']):
                if atomH == 'H':
                    disAH = math.dist(mol_b['coordinates'][i], mol_b['coordinates'][j])
                    if disAH <= 1.1:
                        for k, atomb in enumerate(mol_a['atom']):
                            if atomb == atom_b:
                                disHB = math.dist(mol_b['coordinates'][j], mol_a['coordinates'][k])
                                if disHB <= 2.5:
                                    disAB = math.dist(mol_b['coordinates'][i], mol_a['coordinates'][k])
                                    cosx = (disAH ** 2 + disHB ** 2 - disAB ** 2) / (2 * disAH * disHB)
                                    if cosx < -0.5:
                                        hb_count += 1
    return hb_count


def hod_atom_pair(mola, molb, pair_hd={'C-C': [0, 4, 5]}):
    """
    Calculate atom pairs histograms of distances.
    :param mola: RDKit mol object for monomer A.
    :param molb: RDKit mol object for monomer B.
    :param pair_hd: dict: atoms pairs and associated HoD parameters ([min, max, n_bins]).
    :return: list: atom pair HoD fingerprint.
    """
    hod = []
    for atom_pair in pair_hd:
        bins = tuple(np.linspace(pair_hd[atom_pair][0], pair_hd[atom_pair][1], pair_hd[atom_pair][2]))
        a1, a2 = atom_pair.split('-')

        # Get all distances for the atom pair
        distances = []
        for i, atomi in enumerate(mola['atom']):
            for j, atomj in enumerate(molb['atom']):
                if ((atomi == a1 and atomj == a2) or (atomi == a2 and atomj == a1)):
                    dis = math.dist(mola['coordinates'][i], molb['coordinates'][j])
                    distances.append(dis)

        bins_d = {}
        for i in enumerate(bins[:-1]):
            bin1 = i[1]
            bin2 = bins[i[0] + 1]
            ind = np.where((distances >= bin1) & (distances <= bin2))
            bindis = bin2 - bin1
            if bin1 not in bins_d:
                bins_d[bin1] = []
            if bin2 not in bins_d:
                bins_d[bin2] = []
            for j in ind[0]:
                c1 = 1 - ((np.array(distances[j]) - bin1) / bindis)
                c2 = ((np.array(distances[j]) - bin1) / bindis)
                bins_d[bin1].append(c1)
                bins_d[bin2].append(c2)
        for i in bins_d:
            hod.append(round(np.sum(bins_d[i]), 3))
    return hod


def substructures_center_of_mass(mol, indices):
    mol = get_mol(mol)
    COMs = []
    for i in indices:
        mass = [float(masses[mol['atom'][x]]['standard']) for x in i]
        positions = [mol['coordinates'][x] for x in i]
        COMs.append(np.average(positions, axis=0, weights=mass))
    return COMs


def hod_substructures_pairs(mola, molb, smarts, parameters={"SubstructureFP_SubFP2-PubChem_582": [0, 4, 5]}):
    """
    Calculate substructure pairs histograms of distances.
    :param mola: RDKit mol object for monomer A.
    :param molb: RDKit mol object for monomer B.
    :param smarts: dict: smarts associated with the names of each substructure pairs.
    :param parameters: dict: parameters([min, max, n_bins]) associated with each substructure pairs.
    :return: list: substructure pair fingerprint.
    """
    fingerprint = []
    for fp in parameters:

        mola_sub1 = substructures_center_of_mass(mola, mola.GetSubstructMatches(Chem.MolFromSmarts(smarts[fp][0])))
        mola_sub2 = substructures_center_of_mass(mola, mola.GetSubstructMatches(Chem.MolFromSmarts(smarts[fp][1])))
        molb_sub1 = substructures_center_of_mass(molb, molb.GetSubstructMatches(Chem.MolFromSmarts(smarts[fp][0])))
        molb_sub2 = substructures_center_of_mass(molb, molb.GetSubstructMatches(Chem.MolFromSmarts(smarts[fp][1])))

        distances = []
        if mola_sub1 and molb_sub2:
            for sub_com1 in mola_sub1:
                for sub_com2 in molb_sub2:
                    distances.append(math.dist(sub_com1, sub_com2))
        if mola_sub2 and molb_sub1:
            for sub_com1 in mola_sub2:
                for sub_com2 in molb_sub1:
                    distances.append(math.dist(sub_com1, sub_com2))

        if parameters[fp][2] == 1:
            distances = np.array(distances)
            fingerprint.append(len(np.where((distances >= parameters[fp][0]) & (distances <= parameters[fp][1]))[0]))
        else:
            bins = np.linspace(parameters[fp][0], parameters[fp][1], parameters[fp][2])
            bins_d = {}
            for i in enumerate(bins[:-1]):
                bin1 = i[1]
                bin2 = bins[i[0] + 1]
                ind = np.where((distances >= bin1) & (distances <= bin2))
                bindis = bin2 - bin1

                if bin1 not in bins_d:
                    bins_d[bin1] = []
                if bin2 not in bins_d:
                    bins_d[bin2] = []
                for index in ind[0]:
                    c1 = 1 - ((np.array(distances[index]) - bin1) / bindis)
                    c2 = ((np.array(distances[index]) - bin1) / bindis)
                    bins_d[bin1].append(c1)
                    bins_d[bin2].append(c2)
            for i in bins_d:
                fingerprint.append(round(np.sum(bins_d[i]), 3))
    return fingerprint

def getfingerprint(mol_a, mol_b, model):
    """
    Calculates the empirical features used in dispML models
    :param mol_a: RDKit mol object for monomer A.
    :param mol_b: RDKit mol object for monomer B.
    :param model: str: 'D3-ML', 'D3-ML-S', 'D3-ML-X', 'MBD-ML-S' or 'MBD-ML-X'.
    :return: lsit: fingerprint for model.
    """
    if 'D3' in model:
        pairhod = hods.pairhd_d3
        subshod = hods.subhd_d3
        with open(PATH+"/smarts_d3fp.json") as file:
            smarts = json.load(file)
    elif 'MBD' in model:
        print('hi')
        pairhod = hods.pairhd_mbd
        subshod = hods.subhd_mbd

        with open(PATH+"/smarts_mbdfp.json") as file:
            smarts = json.load(file)

    mol1dict, mol2dict = get_mol(mol_a), get_mol(mol_b)
    fingerprint = [count_hb(mol1dict, mol2dict, *x) for x in [['N', 'O'], ['N', 'N'], ['O','N'], ['O', 'O']]]
    fingerprint = fingerprint + hod_atom_pair(mol1dict, mol2dict, pairhod)
    fingerprint = fingerprint + hod_substructures_pairs(mol_a, mol_b, smarts, subshod)

    return fingerprint
