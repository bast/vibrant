import sys
import numpy
import gaussian
import gram_schmidt
import conf


def vibrational_analysis(file_name):

    nr_atoms = gaussian.get_nr_atoms(file_name)
    atom_xyz = gaussian.get_atom_coordinates(file_name, nr_atoms)
    atom_mass = gaussian.get_atom_masses(file_name, nr_atoms)
    
    # construct inertia tensor
    I = numpy.zeros((3, 3))
    for i in range(nr_atoms):
        m = atom_mass[i]
        x = atom_xyz[i][0]
        y = atom_xyz[i][1]
        z = atom_xyz[i][2]
        I[0][0] += m*(y*y + z*z)
        I[1][1] += m*(z*z + x*x)
        I[2][2] += m*(x*x + y*y)
        I[0][1] -= m*(x*y)
        I[1][0] -= m*(x*y)
        I[0][2] -= m*(x*z)
        I[2][0] -= m*(x*z)
        I[1][2] -= m*(y*z)
        I[2][1] -= m*(y*z)
    
    # diagonalize
    eig, X = numpy.linalg.eig(I)
    
    # minus sign to match gaussian
    X *= -1.0
    
    # construct B vectors 1-6
    B = numpy.zeros((6, nr_atoms*3))
    A = numpy.zeros(3)
    k = 0
    for i in range(nr_atoms):
        m = numpy.sqrt(atom_mass[i])
        B[0][k+0] = 1.0
        B[1][k+1] = 1.0
        B[2][k+2] = 1.0
      # B[3][k+1] = -atom_xyz[i][2]
      # B[3][k+2] =  atom_xyz[i][1]
      # B[4][k+0] =  atom_xyz[i][2]
      # B[4][k+2] = -atom_xyz[i][0]
      # B[5][k+0] = -atom_xyz[i][1]
      # B[5][k+1] =  atom_xyz[i][0]
      # k += 3
        for j in range(3):
            A[j] = atom_xyz[i][0]*X[0][j] + atom_xyz[i][1]*X[1][j] + atom_xyz[i][2]*X[2][j]
        for j in range(3):
            B[3][k] = (A[1]*X[j][2] - A[2]*X[j][1])
            B[4][k] = (A[2]*X[j][0] - A[0]*X[j][2])
            B[5][k] = (A[0]*X[j][1] - A[1]*X[j][0])
            k += 1
    
    # construct matrix of reciprocal mass square roots
    M = numpy.zeros((nr_atoms*3, nr_atoms*3))
    k = 0
    for iatom in range(nr_atoms):
        for ixyz in range(3):
            M[k][k] = 1.0/numpy.sqrt(atom_mass[iatom])
            k += 1
    
    # orthonormalize vectors
    B = gram_schmidt.gram_schmidt(B, 6)
    
    # construct projection matrix
    P = numpy.eye(nr_atoms*3) - numpy.array(numpy.mat(numpy.transpose(B))*numpy.mat(B))
    
    # read hessian
    H = gaussian.get_hessian(file_name, nr_atoms)
    
    # project out trans-rot
    H_proj = numpy.array(numpy.mat(numpy.transpose(P))*(numpy.mat(H)*numpy.mat(P)))
    
    H_mwc = numpy.array(numpy.mat(M)*(numpy.mat(H_proj)*numpy.mat(M)))
    freq, L = numpy.linalg.eig(H_mwc)
    
    C = numpy.array(numpy.transpose(L)*numpy.mat(M))
    
    reduced_mass = numpy.zeros(nr_atoms*3 - 6)
    for m in range(nr_atoms*3 - 6):
        reduced_mass[m] = 1.0/numpy.dot(C[m], C[m])
    
    T = gaussian.get_dipole_derv(file_name, nr_atoms)
    
    dip_derv = numpy.array(numpy.mat(C)*numpy.mat(numpy.transpose(T)))
    
    I = numpy.zeros(nr_atoms*3 - 6)
    freq_cm = numpy.zeros(nr_atoms*3 - 6)
    for m in range(nr_atoms*3 - 6):
        I[m] = conf.au_to_kmmol*numpy.dot(dip_derv[m], dip_derv[m])
        freq_cm[m] = conf.hartree_to_cm*numpy.sqrt(abs(freq[m])/conf.amu_to_au)

    C_normalized = numpy.zeros((nr_atoms*3 - 6, nr_atoms, 3))
    for m in range(nr_atoms*3 - 6):
        k = 0
        for iatom in range(nr_atoms):
            for ixyz in range(3):
                C_normalized[m][iatom][ixyz] = C[m][k]*reduced_mass[m]**0.5
                k += 1

    sort_list = []
    for m in range(nr_atoms*3 - 6):
        sort_list.append([freq_cm[m], m])
    sort_list.sort()

    freq_cm_sorted = numpy.zeros(nr_atoms*3 - 6)
    I_sorted = numpy.zeros(nr_atoms*3 - 6)
    reduced_mass_sorted = numpy.zeros(nr_atoms*3 - 6)
    C_normalized_sorted = numpy.zeros((nr_atoms*3 - 6, nr_atoms, 3))
    dip_derv_sorted     = numpy.zeros((nr_atoms*3 - 6, 3))

    k = 0
    for s in sort_list:
        m = s[1]
        freq_cm_sorted[k]      = freq_cm[m]
        I_sorted[k]            = I[m]
        reduced_mass_sorted[k] = reduced_mass[m]
        C_normalized_sorted[k] = C_normalized[m]
        dip_derv_sorted[k]     = dip_derv[m]
        k += 1
    
    return freq_cm_sorted, I_sorted, reduced_mass_sorted, C_normalized_sorted, dip_derv_sorted
