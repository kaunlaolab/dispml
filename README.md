# D3-/MBD-ML

The details of this code can be found in the work of DOI: To be published

If provided a system in the XYZ format, the model will rely on xyz2mol (https://github.com/jensengroup/xyz2mol) to obtain an RDKit mol object from its coordinates. In case of a charged system, the second line of the XYZ file must be formatted as to explicitly state it. For instance, the second line of the XYZ file containing a positively charged system would be formatted as:
```
charge=1=
```
For more information on xyz2mol, please see the related article at DOI:10.1002/bkcs.10334, or the github link cited above.

For complexe systems or if the system contains multiple charges, using a SDF file is recommended to ensure this program performs to the best of its ability. The SDF format ensures proper placement of charges or bonds, and can be used directly by RDKit without preprocessing by xyz2mol, which may represents the bottleneck of this code. **For instance, in systems where the monomer consists of a single atom, such as noble gases, using an SDF file is essential to bypass issues caused by xyz2mol.**

## Running the code:

This code requires a number of packages available directly via conda. For simplicity, you may install a conda environment using the .yml file contained within the directory:
```
conda env create -n YOUR_ENV_NAME -f environment.yml
```

Once the environment is installed, activate it and run the code:
 ```
 conda activate YOUR_ENV_NAME
 python3 main.py /path/to/monomer_a.xyz /path/to/monomer_b.xyz
 ```
Which will provide D3-ML values by default. For models that rely on quantum features, the (X)SAPT interaction energy components (Electrostatics, Exchange, Indutction and Exchange-Induction) must be passed to the program directly. If using MBD as the baseline model, the MBD@rsSCS and MBD@esDQ terms must be passed directly.
For more information:
```
python3 main.py --help
```

## Dependencies:
```
- Numpy
- RDKit
- Scikit-learn
- XGBoost
- xyz2mol
```
