import numpy

def gram_schmidt(M, k):
    for i in range(k):
        for j in range(i):
            # i is orthogonalized against j
            f = numpy.dot(M[i], M[j])/numpy.dot(M[j], M[j])
            M[i] -= f*M[j]
        # normalize j
        M[i] /= numpy.sqrt(numpy.dot(M[i], M[i]))
    return M
