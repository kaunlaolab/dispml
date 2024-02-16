import saptd3
import feature_generation as fpgen
import sys
from argparse import ArgumentParser
import xyz2mol
from rdkit import Chem
import xgboost as xgb

from pathlib import Path
PATH = str(Path(__file__).parent)

def calculate_dispml(monomers, model, charge=0, quantum_features=None, e6=None, e8=None):
    """
    Calculate dispersion energy between 2 monomers from a baseline method (D3 or MBD) + ML correction
    :param monomers: list: RDKit mol object of the 2 monomers.
    :param model: str: 'D3-ML', 'D3-ML-S', 'D3-ML-X', 'MBD-ML-S' or 'MBD-ML-X'.
    :param charge: int: total dimer charge.
    :param quantum_features: list: list of 5 floats, [elst, exch, ind, exchind, |homo_a - homo_b|.
    :param e6: float: if using MBD as baseline, MBD@rsSCS component of MBD dispersion energy.
    :param e8: float: if using MBD as baseline, MBD@esDQ component of MBD dispersion energy.
    :return: float, float, float: basseline model Edisp, ML correction, ML corrected Edisp value.
    """

    mola = fpgen.get_mol(monomers[0])
    molb = fpgen.get_mol(monomers[1])
    mold = {key: mola[key] + molb[key] for key in mola}
    mold['index'] = [x for x in range(0, len(mold['atom']))]
    mola['cn'] = saptd3.coordination_numbers(mola)
    molb['cn'] = saptd3.coordination_numbers(molb)
    mold['cn'] = saptd3.coordination_numbers(mold)

    if 'D3' in model:
        d3a = saptd3.D3(mola, kcal=True)
        d3b = saptd3.D3(molb, kcal=True)
        d3d = saptd3.D3(mold, kcal=True)
        eint = d3d.edisp - d3a.edisp - d3b.edisp
        e6 = d3d.e6 - d3a.e6 - d3b.e6
        e8 = d3d.e8 - d3a.e8 - d3b.e8
    else:
        eint = e6+e8
    models = {'D3-ML': PATH+'/ml_models/opt-D3-emp.json',
              'D3-ML-S': PATH+'/ml_models/opt-D3-sapt.json',
              'D3-ML-X': PATH+'/ml_models/opt-D3-xsapt.json',
              'MBD-ML-S': PATH+'/ml_models/mbd-opt-sapt.json',
              'MBD-ML-X': PATH+'/ml_models/mbd-opt-xsapt.json'}
    mlmodel = xgb.XGBRegressor()
    mlmodel.load_model(models[model])

    features = [e6, e8]
    if quantum_features:
        features = features + quantum_features
    features.append(charge)
    features = features + fpgen.getfingerprint(monomers[0], monomers[1], model)
    corr = mlmodel.predict([features])[0]

    return eint, corr, eint + corr

def main():
    parser = ArgumentParser()
    parser.add_argument("--charge", dest="charge", type=int, default=0,
                        help = "Total charge of the dimer. Set to 0 by default.")
    parser.add_argument("--model", dest="model", choices=['D3-ML', 'D3-ML-S', 'D3-ML-X', 'MBD-ML-S', 'MBD-ML-X'],
                        default="D3-ML", help="Choice of models between D3-ML, D3-ML-S, D3-ML-X, MBD-ML-S or MBD-ML-X")
    parser.add_argument("--mbde6", dest="mbde6", type=float, default=None,
                        help="MBD@rsSCS component of MBD dispersion energy")
    parser.add_argument("--mbde8", dest="mbde8", type=float, default=None,
                        help="MBD@esDQ component of MBD dispersion energy")
    parser.add_argument("--elst", dest="elst", type=float, default=None,
                        help="Electrostatics component of Eint, given by SAPT calculation")
    parser.add_argument("--exch", dest="exch", type=float, default=None,
                        help="Exchange component of Eint, given by SAPT calculation")
    parser.add_argument("--ind", dest="ind", type=float, default=None,
                        help="Induction component of Eint, given by SAPT calculation")
    parser.add_argument("--exchind", dest="exchind", type=float, default=None,
                        help="Exchange-Induction component of Eint, given by SAPT calculation")
    parser.add_argument("--homo", dest="homo", type=float, default=None,
                        help="Absolute difference of the HOMO of monomer A and HOMO of monomer B")

    (options, args) = parser.parse_known_args()
    print(options)

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

    if options.model == 'D3-ML':
        baseline, corr, dispml = calculate_dispml(monomers, options.model, options.charge)
        print(name, baseline, corr, dispml)
    else:
        quantum_features = [options.elst, options.exch, options.ind, options.exchind, options.homo]
        if None in quantum_features:
            print('Invalid number of quantum features provided.')
        if 'MBD' in options.model:
            if options.mbde6 and options.mbde8:
                baseline, corr, dispml = calculate_dispml(monomers, options.model, options.charge, quantum_features,\
                                                      options.mbde6, options.mbde8)
                print(name, baseline, corr, dispml)
            else:
                print('please provide MBD@rsSCS and MBD@esDQ terms for MBD based models')
        else:
            baseline, corr, dispml = calculate_dispml(monomers, options.model, options.charge, quantum_features)
            print(name, baseline, corr, dispml)

if __name__ == "__main__":
    main()