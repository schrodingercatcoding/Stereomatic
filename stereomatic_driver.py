from schrodinger import structure
import json
import copy
import collections
import argparse
from scipy import linalg
from scipy.special import expit
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

from stereomatic import get_stereomatic_desc
from get_overlap_new import calculate_all_over_lap

with open("new_bond_data.json", 'r') as fh:
    new_bond_data_dict = json.load(fh)

def parse_args():
    """
    parse commandline arguments
    """

    parser = argparse.ArgumentParser(
        description="calculate stereomatic overlap between two molecules give a specific atom",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'st_1',
        type=str,
        help='name of the molecule1 st file'
    )

    parser.add_argument(
        'st_2',
        type=str,
        help='name of the molecule2 st file'
    )

    parser.add_argument(
        '-atom_of_interest',
        dest='atom_pair',
        metavar='<Xi>',
        default=(1, 1),
        nargs=2,
        type=int,
        help='specify two atom indexes for comparison'
    )

    parser.add_argument(
        '-debug',
        default=False,
        dest='debug',
        action='store_true',
        help='debug option to save the json files containing molecular stereomatic descriptor'  
    )

    return parser.parse_args()

def sigmoid_function(c, s, shift, x):
    """
    sigmoid function

    c: center of the function
    s: shape of the function
    shift: shift up by shift
    x: varible

    return f(x) + shift
    """

    return expit(s * (c -x)) + shift

def box_function(width_left, width_right, shift, sigma, x):
    """
    a function equals one at (shift - width_left, shift + width_right)

    sigma: how fast it goes from 1 to zero
    """

    prefactor = 2.0 * 0.25
    left = erf( (1.0/width_left * x + 1.0/width_left * shift  + 1.0) / ( sigma * math.sqrt(2.0)) )
    right = erf( (1.0/width_right * x + 1.0/width_right * shift  - 1.0) / ( sigma * math.sqrt(2.0)) )

    return prefactor * (left - right)


def prepare_data(to_import_data):

    """
    :type  to_import_data: list
    :param to_import_data: bonding information of a certain type of atom pairs

    return bond_length_average, bond_orders, bond_range_list
    """

    if(len(to_import_data) == 3):
        triple_bond, double_bond, single_bond = to_import_data[:]
        single_bond_min, single_bond_max = round(single_bond['bond_min'], 2), round(single_bond['bond_max'],2)
        double_bond_min, double_bond_max = round(double_bond['bond_min'], 2), round(double_bond['bond_max'],2)
        triple_bond_min, triple_bond_max = round(triple_bond['bond_min'], 2), round(triple_bond['bond_max'],2)
        
        triple_bond_center = (triple_bond_max + double_bond_min)/2
        double_bond_center = (double_bond_max + single_bond_min)/2
        single_bond_center = single_bond_max + 0.1
        bond_length_average = [triple_bond_center, double_bond_center, single_bond_center]
        bond_orders = [3, 2, 1]

        triple_bond_left_width, triple_bond_right_width = triple_bond_center, (double_bond_max + double_bond_min)/2 - triple_bond_center
        double_bond_left_width, double_bond_right_width = double_bond_center - (double_bond_max + double_bond_min)/2, (single_bond_max + single_bond_min)/2 - double_bond_center
        single_bond_left_width, single_bond_right_width = single_bond_center - (single_bond_max + single_bond_min)/2, 0.1
        bond_range_list = [(triple_bond_left_width, triple_bond_right_width),(double_bond_left_width, double_bond_right_width),(single_bond_left_width, single_bond_right_width)]

    elif(len(to_import_data) == 2):
        double_bond, single_bond = to_import_data[:]
        single_bond_min, single_bond_max = round(single_bond['bond_min'], 2), round(single_bond['bond_max'],2)
        double_bond_min, double_bond_max = round(double_bond['bond_min'], 2), round(double_bond['bond_max'],2)

        double_bond_center = (double_bond_max + single_bond_min)/2
        single_bond_center = single_bond_max + 0.1
        bond_length_average = [double_bond_center, single_bond_center]
        bond_orders = [2, 1]
        double_bond_left_width, double_bond_right_width = double_bond_center, (single_bond_max + single_bond_min)/2 - double_bond_center
        single_bond_left_width, single_bond_right_width = single_bond_center - (single_bond_max + single_bond_min)/2, 0.1
        bond_range_list = [(double_bond_left_width, double_bond_right_width),(single_bond_left_width, single_bond_right_width)]

    elif(len(to_import_data) == 1):
        single_bond = to_import_data[0]
        single_bond_min, single_bond_max = round(single_bond['bond_min'], 2), round(single_bond['bond_max'],2)
        single_bond_center = single_bond_max + 0.1
        bond_length_average = [single_bond_center]
        bond_orders = [1]
        single_bond_left_width, single_bond_right_width = single_bond_center, 0.1
        bond_range_list = [(single_bond_left_width, single_bond_right_width)]

    return bond_length_average, bond_orders, bond_range_list

def stereomatic_descriptor(atoms_pair, x, database=new_bond_data_dict):
    """
    A function to return the stereomatic value for a type of atom pairs at distance of x

    :type  atoms_pair: string
    :param atoms_pair: the name of atoms pair eg: C_C, C_O

    :type  x: float
    :param x: the bond distance between two atoms

    :type  database: dict
    :param database: the database that store all information of a given atoms pair

    return the new CN value
    """
    
    try:
        data_to_import = database[atoms_pair]
    except KeyError:
        return 0
    bond_length_average, bond_orders, bond_range_list = prepare_data(data_to_import)
    
    ret_value = 0
    for i in range(len(bond_length_average)):
        ret_value += sigmoid_function(bond_length_average[i], 100, bond_orders[i]-1, x) * box_function(bond_range_list[i][0], bond_range_list[i][1], -bond_length_average[i], 0.0001, x)
    
    return ret_value

def sum_keys(array):

    sum = 0
    for num in array:
        sum += int(num)

    return sum

def sort_base_on_keys(array):
    """
    [[value, layer_number], {...}], [[value, layer_number], {...}], [[value, layer_number], {...}]
    """
    myArr = array.copy()
    myArr.sort(key=lambda x: (len(x[1].keys()), sum_keys(x[1].keys())), reverse=True)

    return myArr

def sort_keys(stereomatic_dict):
    """
    sort the second layer base on their environment complexity
    """
    new_stereomatic_dict = {}
    for key1, value1 in stereomatic_dict.items():
        new_value1 = []
        if len(value1) == 1:
            new_value1 = copy.copy(value1)
        else:
            new_value1 = sort_base_on_keys(value1.copy())

        new_stereomatic_dict[key1] = new_value1
            
    return new_stereomatic_dict

def main():
    
    args = parse_args()
    print("args: ", args)

    st1 = structure.Structure.read(args.st_1)
    st2 = structure.Structure.read(args.st_2)
    pka_atom1, pka_atom2 = args.atom_pair
    data1 = get_stereomatic_desc(st1, pka_atom1, set([pka_atom1]))
    data2 = get_stereomatic_desc(st2, pka_atom2, set([pka_atom2]))
    overlap1, overlap2 = calculate_all_over_lap(data1, data2)

    print('overlap1: ', overlap1)
    print('overlap2: ', overlap2)
    if args.debug:
        with open('%s_atom%d_stereomatic_descriptor.json'%(st1.title, pka_atom1), 'w') as fh:
            json.dump(data1, fh, indent=4)
        with open('%s_atom%d_stereomatic_descriptor.json'%(st2.title, pka_atom2), 'w') as fh:
            json.dump(data2, fh, indent=4)

if __name__ == "__main__":
    main()