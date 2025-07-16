import subprocess
import numpy as np

def write_ine_file(A, b, fname='poly.ine', directory='.'):
    """Write H-rep of the form Ax >= b to .ine format"""
    m, n = A.shape
    with open(directory+'/'+fname, 'w') as f:
        f.write('H-representation\nbegin\n')
        f.write(f'{m} {n+1} rational\n')
        for i in range(m):
            row = [-b[i]] + A[i].tolist()
            f.write(' '.join(map(str, row)) + '\n')
        f.write('end\n')

def call_lrs(ine_file='poly.ine', ext_file='poly.ext'):
    subprocess.run(['lrs', ine_file], stdout=open(ext_file, 'w'))

def parse_ext_file(fname='poly.ext', directory='.'):
    vertices = []
    
    parsing = False
    with open(directory+'/'+fname) as f:
        for line in f:
            line = line[:-1]
            
            # skip irrelevant lines
            if line=='begin':
                parsing=True
                continue
            elif line=='end':
                parsing=False
                continue
            elif not parsing:
                continue
            
            # skip the header
            if line.startswith('*'):
                continue
            
            # read the real lines
            if line.startswith(' 1'):
                parts = line.strip().split()
                vertices.append([eval(r) for r in parts[1:]])
            elif line.startswith(' 0'):
                raise ValueError(f'Hyperplanes define a polyhedron (there is a ray={[eval(r) for r in parts[1:]]})')
    return np.array(vertices)