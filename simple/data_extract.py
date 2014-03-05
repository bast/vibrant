#!/usr/bin/env python

import numpy

p = 0.001 # in bohr
bohr_to_angstrom = 0.5291772083
nr_centers = 3

def read_mol_grad(file_name):
    output = open(file_name, 'r').readlines()
    for (i, line) in enumerate(output):
        if 'Total molecular gradient' in line:
           i_save = i
    mol_grad = []
    for i in range(nr_centers):
        x = float(output[i_save+3+i].split()[2])
        y = float(output[i_save+3+i].split()[3])
        z = float(output[i_save+3+i].split()[4])
        mol_grad.append([x, y, z])
    return numpy.array(mol_grad)
    

def read_dipole(file_name):
    output = open(file_name, 'r').readlines()
    for (i, line) in enumerate(output):
        if 'Dipole moment' in line:
           i_save = i
    x = float(output[i_save+9].split()[5])
    y = float(output[i_save+10].split()[5])
    z = float(output[i_save+11].split()[5])
    return numpy.array([x, y, z])


mol_hess = numpy.zeros([nr_centers*3, nr_centers*3])
k = 0
for i in range(nr_centers):
    for j in range(3):
        mol_grad_p = read_mol_grad('HF_dip_grad_molecule_%i_%ip.out' % (i+1, j+1))
        mol_grad_m = read_mol_grad('HF_dip_grad_molecule_%i_%im.out' % (i+1, j+1))
        diff = (mol_grad_p - mol_grad_m)/(2.0*p)
        l = 0
        for m in range(nr_centers):
            for n in range(3):
                mol_hess[k][l] = diff[m][n]
                l += 1
        k += 1

f = open('molecular_hessian', 'w')
for k in range(nr_centers*3):
    for l in range(nr_centers*3):
        f.write('%4i%4i%20.12f\n' % (k, l, mol_hess[k][l]))
f.close()

dip_deriv = numpy.zeros([3, nr_centers*3])
k = 0
for i in range(nr_centers):
    for j in range(3):
        dipole_p = read_dipole('HF_dip_grad_molecule_%i_%ip.out' % (i+1, j+1))
        dipole_m = read_dipole('HF_dip_grad_molecule_%i_%im.out' % (i+1, j+1))
        diff = (dipole_p - dipole_m)/(2.0*p)
        for m in range(3):
            dip_deriv[m][k] = diff[m]
        k += 1

f = open('dipole_gradient', 'w')
for k in range(3):
    for l in range(nr_centers*3):
        f.write('%4i%4i%20.12f\n' % (k, l, dip_deriv[k][l]))
f.close()
