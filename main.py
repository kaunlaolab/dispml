#!/usr/bin/env python

"""D3-/MBD-ML

This script allows the rapid prediction of the dispersion component of the 
interaction energy, using a baseline model (D3 or MBD) and added ML correction.
It will generate the features used as input to the XGBoost model, and will calculate
the baseline value if using D3.

This tool accepts XYZ or SDF formatted files containing the molecular geometry.
If an XYZ file is used, it will rely on xyz2mol to convert the system 
to a RDKit mol object. The SDF file format is however recommended 
in case of complexe systems to ensure proper placements of bonds and charge,
or if this code runs for a seemingly long time.

This script requires that numpy, RDKit, xyz2mol, scikit-learn and XGBoost
be installed within the Python environment you are running this script in.

This file can also be imported as a module. In such case, the class dispML()
may be used to calculate the baseline Edisp, the ml correction and the predicted
Edisp value.
"""

import saptd3
import feature_generation as fpgen
import sys
from argparse import ArgumentParser
import xyz2mol
from rdkit import Chem
import xgboost as xgb

from pathlib import Path
PATH = str(Path(__file__).parent)

class dispML:
    """
    Calculates Edisp from baseline model and ML correction.
    """
    def __init__(self, monomer_a, monomer_b, model, quantum_features=None, e6=None, e8=None):
        """
           Calculate dispersion energy between 2 monomers from a baseline method (D3 or MBD) + ML correction
           :param monomer_a: RDKit mol object: RDKit mol object of the first monomer.
           :param monomer_b: RDKit mol object: RDKit mol object of the second monomer.
           :param model: str: 'D3-ML', 'D3-ML-S', 'D3-ML-X', 'MBD-ML-S' or 'MBD-ML-X'.
           :param charge: int: total dimer charge.
           :param quantum_features: list: list of 5 floats, [elst, exch, ind, exchind, |homo_a - homo_b|.
           :param e6: float: if using MBD as baseline, MBD@rsSCS component of MBD dispersion energy.
           :param e8: float: if using MBD as baseline, MBD@esDQ component of MBD dispersion energy.
           """
        if 'MBD' in model:
            if e6 is None or e8 is None:
                raise TypeError('Must provide MBD@rsSCS and MBD@esDQ if using MBD based models.')
            else:
                self.e6, self.e8 = e6, e8
                self.eint = e6 + e8

        elif 'D3' in model:
            mola = fpgen.get_mol(monomer_a)
            molb = fpgen.get_mol(monomer_b)
            mold = {key: mola[key] + molb[key] for key in mola}
            mold['index'] = [x for x in range(0, len(mold['atom']))]
            mola['cn'] = saptd3.coordination_numbers(mola)
            molb['cn'] = saptd3.coordination_numbers(molb)
            mold['cn'] = saptd3.coordination_numbers(mold)

            d3a = saptd3.D3(mola, kcal=True)
            d3b = saptd3.D3(molb, kcal=True)
            d3d = saptd3.D3(mold, kcal=True)
            self.eint = d3d.edisp - d3a.edisp - d3b.edisp
            self.e6 = d3d.e6 - d3a.e6 - d3b.e6
            self.e8 = d3d.e8 - d3a.e8 - d3b.e8


        models = {'D3-ML': PATH + '/ml_models/opt-D3-emp.json',
                  'D3-S-ML': PATH + '/ml_models/opt-D3-sapt.json',
                  'D3-X-ML': PATH + '/ml_models/opt-D3-xsapt.json',
                  'MBD-S-ML': PATH + '/ml_models/mbd-opt-sapt.json',
                  'MBD-X-ML': PATH + '/ml_models/mbd-opt-xsapt.json'}

        mlmodel = xgb.XGBRegressor()
        mlmodel.load_model(models[model])

        self.features = [self.e6, self.e8]
        if quantum_features:
            self.features = self.features + quantum_features

        # self.features.append(charge)
        self.features = self.features + fpgen.getfingerprint(monomer_a, monomer_b, model)
        self.corr = mlmodel.predict([self.features])[0]
        self.disp = self.eint + self.corr

def main():
    parser = ArgumentParser()
    parser.add_argument("--name", dest="name", type=str, default=None,
                        help="Name of dimer to be printed with output, optional.")
    parser.add_argument("--model", dest="model", choices=['D3-ML', 'D3-S-ML', 'D3-X-ML', 'MBD-S-ML', 'MBD-X-ML'],
                        default="D3-ML", help="Choice of models between D3-ML, D3-S-ML, D3-X-ML, MBD-S-ML or MBD-X-ML")
    parser.add_argument("--mbde6", dest="mbde6", type=float, default=None,
                        help="MBD@rsSCS component of MBD dispersion energy (kcal/mol)")
    parser.add_argument("--mbde8", dest="mbde8", type=float, default=None,
                        help="scaled MBD@esDQ component of MBD dispersion energy (kcal/mol)")
    parser.add_argument("--elst", dest="elst", type=float, default=None,
                        help="Electrostatics component of Eint, given by SAPT calculation (kcal/mol)")
    parser.add_argument("--exch", dest="exch", type=float, default=None,
                        help="Exchange component of Eint, given by SAPT calculation (kcal/mol)")
    parser.add_argument("--ind", dest="ind", type=float, default=None,
                        help="Induction component of Eint, given by SAPT calculation (kcal/mol)")
    parser.add_argument("--exchind", dest="exchind", type=float, default=None,
                        help="Exchange-Induction component of Eint, given by SAPT calculation (kcal/mol)")
    parser.add_argument("--homo", dest="homo", type=float, default=None,
                        help="Absolute difference of the HOMO of monomer A and HOMO of monomer B (atomic units)")

    (options, args) = parser.parse_known_args()

    files = []
    for argv in sys.argv:
        if len(argv.split(".")) > 1:
            if argv.split(".")[-1] == 'xyz' or argv.split(".")[-1] == 'sdf':
                files.append(argv)
    if len(files) == 0:
        print("\nNo files found. An xyz or sdf file is required.\n")
        sys.exit()
    elif len(files) == 2:
        pass
    else:
        print('invalid number of input files, must provide 2 xyz or sdf files, one for each monomer')
        sys.exit()

    monomers = []
    for xyz in files:
        if xyz.endswith('.xyz'):
            name = xyz.split('/')[-1].split('.')[0]
            atoms, charge, xyz_coordinates = xyz2mol.read_xyz_file(xyz)
            mol = xyz2mol.xyz2mol(atoms, xyz_coordinates, charge)[0]

        elif xyz.endswith('sdf'):
            name = xyz.split('/')[-1].split('.')[0]
            mol = Chem.MolFromMolFile(xyz, removeHs=False)
        monomers.append(mol)
    if options.name:
        name = options.name
    quantum_features = [options.elst, options.exch, options.ind, options.exchind, options.homo]
    if None in quantum_features:
        quantum_features = None
    if 'S' in options.model or 'X' in options.model:
        if quantum_features is None:
            raise TypeError('Must provide quantum features if using quantum features based models.')

    results = dispML(monomers[0], monomers[1], options.model, quantum_features, options.mbde6, options.mbde8)
    print(name, ", Baseline model Edisp:", results.eint, ", ML correction:", results.corr, ", DispML:", results.disp)

if __name__ == "__main__":
    main()
