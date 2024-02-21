# This script is an implementation of the SAPT-D3 method for the dispersion
# component of the interaction energy. It relies on the D3 dispersion correction,
# with modification to the damping function (using Toennis-Tang form) with
# re-optimized parameters and to the coordination numbers.
try:
    from .pars import *
except:
    from pars import *
import math
import sys
from argparse import ArgumentParser

# some useful conversion factor
bohr2angstrom = 0.5291772083
angstrom2bohr = (1.0/0.5291772083)
hartree2kcal = 627.5095


def coordination_numbers(mol):
    """
    Calculates modified coordination numbers based on the SAPT-D3 scheme, as implemented in Cuby4.
    See DOI: 10.1021/acs.jctc.6b01198
    :param mol: dict: containing atoms and associated coordinates. See  read_xyz().
    :return: list: containing coordination numbers, with index associated with the mol dictionary
    """
    cn = []
    for i, atom1 in enumerate(mol['atom']):
        xn = 0
        for j, atom2 in enumerate(mol['atom']):
            if i != j:
                r = math.dist(mol['coordinates'][i], mol['coordinates'][j])
                rco = (rcov_cuby4[atom1] + rcov_cuby4[atom2]) * 1.1
                if r <= rco:
                    xn = xn+1
        cn.append(xn)
    return cn


def read_xyz(xyz):
    """
    Read xyz input file (angstroms).
    :param xyz: str: path to xyz file containing system of interest.
    :return: dict: containing list of atomic symbol and associated index, coordinates, and CN.
    """
    mol = {'index': [], 'atom': [], 'coordinates': []}
    with open(xyz, 'r') as file:
        ind = 0
        for i, line in enumerate(file):
            line = line.split()
            if len(line) == 4 and i >= 2:
                mol['index'].append(ind)
                mol['atom'].append(line[0])
                mol['coordinates'].append([float(x) for x in line[1:]])
                ind = ind + 1
            elif i > 2:
                raise Exception('Error in the XYZ input file on line', i+1, '.')
    mol['cn'] = coordination_numbers(mol)
    return mol


def getc6(atom1, atom2, cn1, cn2, c6ref):
    """
    Get C6 coefficients for a given atom pair
    :param atom1: str: Atomic symbol of atom1
    :param atom2: str: Atomic symbol of atom2
    :param cn1: float: Coordination number of atom1
    :param cn2: float: Coordination number of atom2
    :param c6ref: dict: reference C6 coefficients in dict, such as {'atom1': {'atom2': {cn1: {cn2: C6}}}}
    :return: Interpolated C6 coefficient.
    """
    k3 = -4.0
    if atomic_properties[atom1]['atomic#'] > atomic_properties[atom2]['atomic#']:
        atom1, atom2 = atom2, atom1
        cn1, cn2 = cn2, cn1

    z, w = 0, 0
    for cna in c6ref[atom1][atom2]:
        for cnb in c6ref[atom1][atom2][cna]:
            c6 = c6ref[atom1][atom2][cna][cnb]
            lij = math.exp(k3 * (math.pow((cn1 - cna), 2) + math.pow((cn2 - cnb), 2)))
            z = z + (c6 * lij)
            w = w + lij
            if atom1 == atom2 and cna != cnb:
                lij = math.exp(k3 * (math.pow((cn1 - cnb), 2) + math.pow((cn2 - cna), 2)))
                z = z + (c6 * lij)
                w = w + lij

    return z / w


def getc8(atom1, atom2, c6):
    """
    Calculate C8 coefficient based on C6. See DOI:10.1063/1.3382344 for more information
    :param atom1: str: atomic symbol of first atom
    :param atom2: str: atomic symbol of second atom
    :param c6: float: C6 coefficient of atom pair
    :return: float: C8 coefficient of atom pair
    """
    qa = 0.5 * math.sqrt(atomic_properties[atom1]['atomic#']) * atomic_properties[atom1]['r2r4']
    qb = 0.5 * math.sqrt(atomic_properties[atom2]['atomic#']) * atomic_properties[atom2]['r2r4']
    return 3 * c6 * math.sqrt(qa*qb)


def TTdamping(dist, r0, a1, a2):
    """
    Calculates Toennis-Tang damping values.
    :param dist: float: distance between the two atoms
    :param r0: float: atom pair specific cut-off distance
    :param a1: float: parameter
    :param a2: float: parameter
    :return: float, float: e6 contribution, e8 contribution.
    """
    r0 = r0 * angstrom2bohr
    a1 = a1 * math.pow(angstrom2bohr, -2)
    a2 = a2 * math.pow(angstrom2bohr, -1)

    beta = a1 * r0 + a2
    summation = 0
    for kparam in range(0, 7):
        summation += math.pow((beta * dist), kparam) / math.factorial(kparam)
    damp6 = 1 - math.exp(-beta * dist) * summation

    summation = 0
    for kparam in range(0, 9):
        summation += math.pow((beta * dist), kparam) / math.factorial(kparam)
    damp8 = 1 - math.exp(-beta * dist) * summation
    return damp6, damp8


class D3:
    """
    Calculates D3 energy and stores associated values.
    """
    def __init__(self, mol, s8=1.173, a1=-0.744, a2=5.297, pairwise=False, uchf=False, kcal=False):
        """
        Calculates D3 upon initialization.
        :param mol: dict: contains system of interest (see read_xyz())
        :param s8: float: s8 parameter (D3 associated)
        :param a1: float: a1 parameter (TT damping associated)
        :param a2: float: a2 parameter (TT damping associated)
        :param pairwise: bool: if true, stores a dict containing pairwise informations (self.pairwise).
        :param uchf: if true, uses uchf c6 coefficients as reference (see DOI:10.1021/acs.jctc.8b00548)
        """
        if uchf:
            s8, a1, a2 = 1.0947, -0.297, 4.240

        self.mol = mol
        self.e6 = 0.0
        self.e8 = 0.0

        # which C6 reference coefficients to use.
        c6ref = c6_uchf if uchf else c6_d3

        if pairwise:
            self.pairwise = {'i': [], 'j': [], 'a': [], 'b': [],
                             'edisp': [], 'e6': [], 'e8': [],
                             'C6': [], 'C8': [], 'r0ab': [],  'damp6': [], 'damp8': []}

        for i, ind1 in enumerate(self.mol['index']):
            for ind2 in self.mol['index'][i+1:]:
                atom1, atom2 = self.mol['atom'][ind1], self.mol['atom'][ind2]
                dist = math.dist(self.mol['coordinates'][ind1], self.mol['coordinates'][ind2]) * angstrom2bohr

                # need to reformat r0ab so it leads instead of lags in the order of the periodic table.
                if atomic_properties[self.mol['atom'][ind1]]['atomic#']\
                        < atomic_properties[self.mol['atom'][ind2]]['atomic#']:
                    r0 = r0ab[self.mol['atom'][ind2]][self.mol['atom'][ind1]] #* angstrom2bohr
                else:
                    r0 = r0ab[self.mol['atom'][ind1]][self.mol['atom'][ind2]] #* angstrom2bohr

                damp6, damp8 = TTdamping(dist, r0, a1, a2)

                c6 = getc6(atom1, atom2, self.mol['cn'][ind1], self.mol['cn'][ind2], c6ref)
                c8 = getc8(self.mol['atom'][ind1], self.mol['atom'][ind2], c6)
                e6 = -1 * c6 * math.pow(dist, -6) * damp6
                e8 = -s8 * c8 * math.pow(dist, -8) * damp8

                if kcal:
                    e6, e8 = e6 * hartree2kcal, e8 * hartree2kcal

                if pairwise:
                    self.pairwise['i'].append(ind1)
                    self.pairwise['j'].append(ind2)
                    self.pairwise['a'].append(self.mol['atom'][ind1])
                    self.pairwise['b'].append(self.mol['atom'][ind2])
                    self.pairwise['edisp'].append(e6+e8)
                    self.pairwise['e6'].append(e6)
                    self.pairwise['e8'].append(e8)
                    self.pairwise['C6'].append(c6)
                    self.pairwise['C8'].append(c8)
                    self.pairwise['damp6'].append(damp6)
                    self.pairwise['damp8'].append(damp8)
                    self.pairwise['r0ab'].append(r0)

                self.e6 = self.e6 + e6
                self.e8 = self.e8 + e8
        self.edisp = self.e6 + self.e8


def main():
    parser = ArgumentParser()
    parser.add_argument("--s8", dest="s8", default=1.173, type=float, help="s8 parameter (D3)")
    parser.add_argument("--a1", dest="a1", default=-0.744, type=float, help="a1 parameter (TT damping)")
    parser.add_argument("--a2", dest="a2", default=5.297, type=float, help="a2 parameter (TT damping)")
    parser.add_argument("--kcal", dest="kcal", action="store_true", default=False, help="Print energies in kcal/mol")
    parser.add_argument("--pw", dest="pairwise", action="store_true", default=False,
                        help="Print a more verbose outputs, with a table of pairwise dispersion values "
                             "and associated C6 and C8 coefficients")
    parser.add_argument("--uchf", dest="uchf", action="store_true", default=False,
                        help="Use UCHF C6 coefficients instead of default D3 ones")
    (options, args) = parser.parse_known_args()

    files = []
    for argv in sys.argv:
        if len(argv.split(".")) > 1:
            if argv.split(".")[-1] == 'xyz':
                files.append(argv)
    if len(files) == 0:
        print("\nNo files found. An xyz file is required.\n")
        sys.exit()
    for file in files:
        name = file[:-4]
        mol = read_xyz(file)
        results = D3(mol, options.s8, options.a1, options.a2, options.pairwise, options.uchf, options.kcal)
        if options.pairwise:
            sys.stdout.write('Python implementation of SAPT-D3 for %s\n' % name)
            sys.stdout.write("%s\t%s\t%s\t%s\t%s\n" % ('a', 'b', 'edisp', 'c6', 'c8'))
            for i, val in enumerate(results.pairwise['i']):
                sys.stdout.write("%1d %1d %s %s %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f\n" %
                                 (val + 1, results.pairwise['j'][i] + 1, results.pairwise['a'][i], results.pairwise['b'][i],
                                  results.pairwise['edisp'][i]
                                  , results.pairwise['C6'][i], results.pairwise['C8'][i], results.pairwise['r0ab'][i], results.pairwise['damp6'][i]
                                  , results.pairwise['damp8'][i]))
            sys.stdout.write('E6 term: %12.5f\nE8 term: %12.5f\nEdisp: %12.5f\n' %
                             (results.e6, results.e8, results.edisp))
        else:
            sys.stdout.write("%s %12.5f\n" % (name, results.edisp))

if __name__ == "__main__":
    main()
