from povray import *


def setup_scenery(location, light, file):

    Background("White").write(file)
    Camera(location=location, look_at=(0,0,0)).write(file)
    LightSource(light, color="White").write(file)


def draw_molecule(atom_xyz, bonds, color, file):

    for i in range(len(atom_xyz)):
        Sphere(tuple(atom_xyz[i]),
               0.1,
               Texture(Pigment(color=color))).write(file)
    for i in range(len(bonds)):
        j = bonds[i][0]
        k = bonds[i][1]
        Cylinder(tuple(atom_xyz[j]), tuple(atom_xyz[k]),
                 0.1,
                 Texture(Pigment(color=color))).write(file)


def draw_vector(r, v, color, s, file):

    r1 = [r[0], r[1], r[2]]
    r2 = [r1[0] + 0.8*s*v[0], r1[1] + 0.8*s*v[1], r1[2] + 0.8*s*v[2]]
    r3 = [r1[0] +     s*v[0], r1[1] +     s*v[1], r1[2] +     s*v[2]]

    f1 = 0.07
    f2 = 2.0*f1

    Cylinder(r1, r2, f1, Texture(Pigment(color=color))).write(file)
    Cone(r2, f2, r3, 0.0, Texture(Pigment(color=color))).write(file)

    return r3
