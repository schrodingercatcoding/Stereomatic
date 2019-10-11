import random
import sys
import json
from schrodinger.infra import mm
from schrodinger.structure import StructureReader

from generate_stereomatic_step1 import stereomatic_descriptor


def get_bond_order(st, at1, at2):
    # if at1 in at2.bonded_atoms:
    #     return 1.0
    # else:
    #     return 0

    if at1.atomic_number < at2.atomic_number:
        atoms_pair = '%s_%s'%(at1.element, at2.element)
    else:
        atoms_pair = '%s_%s'%(at2.element, at1.element)
    distance = st.measure(at1, at2)
    
    return stereomatic_descriptor(atoms_pair, distance)

def get_stereomatic_desc(st, origin, visited):

    """
    Need to fix for ring system
    only expand on one branch
    """
    sd = {}
    for at in st.atom:
        if at.index == origin:
            # skip self-atom
            continue

        val = get_bond_order(st, st.atom[origin], at)
        if val > 0.5:
            key = at.element
            if at.index in visited:
                sd.setdefault(key, []).append((val, {}))
            else:
                visited.add(at.index)
                sub_sd = get_stereomatic_desc(st, at.index, visited)
                sd.setdefault(key, []).append((val, sub_sd))

    return sd


def main():

    maefile = sys.argv[1]
    origin = int(sys.argv[2])

    print(f'Processing {maefile}')
    st = next(StructureReader(maefile))
    print(f'Using atom {origin} ({st.atom[origin].element}) as origin.\n')

    # Index of atom origin of stereomatic network
    sd = get_stereomatic_desc(st, origin, set([origin]))
    print(sd)

    with open(maefile.replace('.mae', '.json'), 'w') as fh:
        json.dump(sd, fh, indent=4)

if __name__ == "__main__":
    main()

