import numpy

def get_from_checkpoint(file_name, anchor, nr_elements, nr_on_one_line, is_float=True):

    s = open(file_name).readlines()

    start = 0
    for i in range(len(s)):
        if anchor in s[i]:
            start = i + 1

    if is_float:
        result = numpy.zeros(nr_elements)
    else:
        result = []
        for i in range(nr_elements):
            result.append(0)

    k = 0
    for i in range(nr_elements/nr_on_one_line):
        for j in range(len(s[start + i].split())):
            if is_float:
                result[k] = float(s[start + i].split()[j])
            else:
                result[k] = int(s[start + i].split()[j])
            k += 1

    rest = nr_elements%nr_on_one_line
    i = nr_elements/nr_on_one_line
    if rest > 0:
        # read last line
        for j in range(len(s[start + i].split())):
            if is_float:
                result[k] = float(s[start + i].split()[j])
            else:
                result[k] = int(s[start + i].split()[j])
            k += 1

    return result

def get_hessian(file_name, nr_atoms):

    H_cart_triangular = get_from_checkpoint(file_name, 'Cartesian Force Constants', nr_atoms*3*(nr_atoms*3 + 1)/2, 5)
    nr_coordinates = nr_atoms*3

    # convert triangular hessian to square
    H = numpy.zeros((nr_atoms*3, nr_atoms*3))
    k = 0
    for ic in range(nr_coordinates):
        for jc in range(nr_coordinates):
            if ic >= jc:
                iatom = ic/3
                jatom = jc/3
                f = H_cart_triangular[k]
                H[ic][jc] = f
                H[jc][ic] = f
                k += 1
    return H

def get_atom_coordinates(file_name, nr_atoms, move_to_com=True):
    C = get_from_checkpoint(file_name, 'Current cartesian coordinates', nr_atoms*3, 5)
    R = numpy.zeros((nr_atoms, 3))
    k = 0
    for iatom in range(nr_atoms):
        for ixyz in range(3):
            R[iatom][ixyz] = C[k]
            k += 1

    if move_to_com:
        atom_mass = get_atom_masses(file_name, nr_atoms)
        # find center of mass R_com
        R_com = numpy.zeros(3)
        for i in range(nr_atoms):
            R_com += atom_mass[i]*(R[i])
        R_com /= sum(atom_mass)
        
        # shift atom_xyz so that R_com is the origin
        for i in range(nr_atoms):
            R[i] -= R_com

    return R


def get_atom_masses(file_name, nr_atoms):
    return get_from_checkpoint(file_name, 'Real atomic weights', nr_atoms, 5)

def get_dipole_derv(file_name, nr_atoms):
    T = get_from_checkpoint(file_name, 'Dipole Derivatives', 3*nr_atoms*3, 5)
    U = numpy.zeros((3, nr_atoms*3))
    k = 0
    for j in range(nr_atoms*3):
        for i in range(3):
            U[i][j] = T[k]
            k += 1
    return U

def get_nr_atoms(file_name):
    for line in open(file_name).readlines():
        if 'Number of atoms' in line:
            return int(line.split()[-1])

def get_bonds(file_name, nr_atoms):
    for line in open(file_name).readlines():
        if 'MxBond' in line:
            max_nr_bonds_per_atom = int(line.split()[-1])
    T = get_from_checkpoint(file_name, 'IBond        ', nr_atoms*max_nr_bonds_per_atom, 6, is_float=False)
    B = []
    k = 0
    for iatom in range(nr_atoms):
        for ibond in range(max_nr_bonds_per_atom):
            if T[k] > 0:
                B.append([iatom, T[k]-1])
            k += 1
    return B

def get_integer_atom_weights(file_name, nr_atoms):
    return get_from_checkpoint(file_name, 'Integer atomic weights', nr_atoms, 6, is_float=False)

def get_dipole_moment(file_name):
    return get_from_checkpoint(file_name, 'Dipole Moment', 3, 6)
