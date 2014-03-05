#!/usr/bin/env python
# vim:ft=python

import sys
import gaussian
import molecule_povray
import povray
import ir
import numpy

from optparse import OptionParser

#-------------------------------------------------------------------------------

usage = '''
  example: ./%prog --chk=freq_vcd.fchk --mode=10 --output=test.pov'''

parser = OptionParser(usage)

parser.add_option('--chk',
                  type='string',
                  action='store',
                  default=None,
                  help='File to parse',
                  metavar='FILE')
parser.add_option('--output',
                  type='string',
                  action='store',
                  default=None,
                  help='POV-Ray output file',
                  metavar='FILE')
parser.add_option('--mode',
                  type='int',
                  action='store',
                  default=1)

(options, args) = parser.parse_args()

if len(sys.argv) == 1:
    # user has given no arguments: print help and exit
    print parser.format_help().strip()
    sys.exit()

#-------------------------------------------------------------------------------

nr_atoms = gaussian.get_nr_atoms(options.chk)
atom_xyz = gaussian.get_atom_coordinates(options.chk, nr_atoms)
bonds = gaussian.get_bonds(options.chk, nr_atoms)

file = povray.File(options.output, 'colors.inc')

#location = (15, 20, 0)
location = (0, 0, -15) #nm 134
location = (0, 5, 10) #mto

molecule_povray.setup_scenery(location, location, file)

molecule_povray.draw_molecule(atom_xyz, bonds, 'Gray', file)

freq, I, reduced_mass, C, dipole_derv = ir.vibrational_analysis(options.chk)

for i in range(len(C[options.mode-1])):
    if numpy.dot(C[options.mode-1][i], C[options.mode-1][i]) > 0.1:
        molecule_povray.draw_vector(atom_xyz[i], C[options.mode-1][i], 'Green', 2.0, file)

# dipole moment
d = gaussian.get_dipole_moment(options.chk)
tip = molecule_povray.draw_vector((0, 0, 0), -d, 'Red', 2.0, file)

# change in dipole moment
molecule_povray.draw_vector(tip, -dipole_derv[options.mode-1], 'Orange', 4.0, file)
