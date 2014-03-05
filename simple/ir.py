
import sys
import numpy
import gram_schmidt
import conf

import warnings
warnings.simplefilter("ignore", numpy.ComplexWarning)

# read atom masses
atom_mass = []
for line in open('atom_masses', 'r').readlines():
    atom_mass.append(float(line))

# read info from file and put it in variable xyz
xyz = open('xyz_eq', 'r').readlines()
# read atom names and xyz coordinates
atom_xyz = []
nr_atoms = int(xyz[0])
for i in range(nr_atoms):
    x = float(xyz[i+2].split()[1])
    y = float(xyz[i+2].split()[2])
    z = float(xyz[i+2].split()[3])
    atom_xyz.append([x, y, z])
atom_xyz = numpy.array(atom_xyz)
atom_xyz /= conf.bohr_to_angstrom

# move origin to c.o.m.
# find center of mass R_com
R_com = numpy.zeros(3)
for i in range(nr_atoms):
    R_com += atom_mass[i]*(atom_xyz[i])
R_com /= sum(atom_mass)

# shift atom_xyz so that R_com is the origin
for i in range(nr_atoms):
    atom_xyz[i] -= R_com

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
H = numpy.zeros((nr_atoms*3, nr_atoms*3))
for line in open('molecular_hessian', 'r').readlines():
    i = int(line.split()[0])
    j = int(line.split()[1])
    f = float(line.split()[2])
    H[i][j] = f

# symmetrize it fully
for i in range(nr_atoms*3):
    for j in range(nr_atoms*3):
        if j > i:
            f = H[i][j] + H[j][i]
            H[i][j] = 0.5*f
            H[j][i] = 0.5*f

# project out trans-rot
H_proj = numpy.array(numpy.mat(numpy.transpose(P))*(numpy.mat(H)*numpy.mat(P)))

H_mwc = numpy.array(numpy.mat(M)*(numpy.mat(H_proj)*numpy.mat(M)))
freq_unsorted, L_unsorted = numpy.linalg.eig(H_mwc)

sort_list = []
for i, f in enumerate(freq_unsorted):
    sort_list.append([f, i])
sort_list.sort(reverse=True)

freq = numpy.zeros(nr_atoms*3)
L = numpy.zeros((nr_atoms*3, nr_atoms*3))
for m, pair in enumerate(sort_list):
    j = pair[1]
    freq[m] = freq_unsorted[j]
    for k in range(nr_atoms*3):
        L[k][m] = L_unsorted[k][j]

C = numpy.array(numpy.transpose(L)*numpy.mat(M))

reduced_mass = numpy.zeros(nr_atoms*3 - 6)
for m in range(nr_atoms*3 - 6):
    reduced_mass[m] = 1.0/numpy.dot(C[m], C[m])

# read dipole gradient from file
T = numpy.zeros((3, nr_atoms*3))
for line in open('dipole_gradient', 'r').readlines():
    i = int(line.split()[0])
    j = int(line.split()[1])
    f = float(line.split()[2])
    T[i][j] = f

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

print('frequencies  intensities')
for i in range(len(freq_cm)):
    print('%9.1f    %9.1f' % (freq_cm[i], I[i]))
