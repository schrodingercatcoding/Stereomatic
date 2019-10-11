"""
calculate stereomatic overlap between two molecules, to do: include change descriptor later
"""
import json
import sys
import copy
import collections
import numpy as np

def prepare_data_layer1(data1, data2):
    """
    A function to make data1 and data2 same dimension in order to do overlap calculation

    :type  data1: dict
    :type  data2: dict

    return data1_pact, data2_pack
    """
    data1_pack = copy.deepcopy(data1)
    data2_pack = copy.deepcopy(data2)
    keys_1 = data1.keys()
    keys_2 = data2.keys()

    for key1 in keys_1:
        if key1 not in keys_2:
            data2_pack[key1] = [ [0, {}] for i in range(len(data1[key1]))]
        else:
            if len(data1[key1]) < len(data2[key1]):
                # print('data1 needs more at key %s'%key1)
                for i in range(len(data2[key1]) - len(data1[key1])):
                    data1_pack[key1].append([0, {}])
            else:
                continue

    for key2 in keys_2:
        if key2 not in keys_1:
            data1_pack[key2] = [ [0, {}] for i in range(len(data2[key2]))]
        else:
            if len(data2[key2]) < len(data1[key2]):
                # print('data2 needs more at key %s'%key2)
                for i in range(len(data1[key2]) - len(data2[key2])):
                    data2_pack[key2].append([0, {}])
            else:
                continue
    
    return collections.OrderedDict(sorted(data1_pack.items())), collections.OrderedDict(sorted(data2_pack.items()))

def prepare_data_layer2(data1, data2):

    data1_pack = copy.deepcopy(data1)
    data2_pack = copy.deepcopy(data2)
    keys_1 = data1.keys()
    keys_2 = data2.keys()

    for key1, key2 in zip(keys_1, keys_2):
        value1_final = []
        value2_final = []
        for value1, value2 in zip(data1[key1], data2[key2]):
            obj1 = value1[1]
            obj2 = value2[1]
            obj1_p, obj2_p = prepare_data_layer1(obj1, obj2)
            new_value1 = [value1[0], obj1_p]
            new_value2 = [value2[0], obj2_p]
            value1_final.append(new_value1)
            value2_final.append(new_value2)
        data1_pack[key1] = value1_final
        data2_pack[key2] = value2_final

    # return None, None
    return data1_pack, data2_pack

def calculate_overlap_layer1(data1, data2):
    
    total = 0
    for key1, key2 in zip(data1.keys(), data2.keys()):
        arr1 = [ data1[key1][i][0] for i in range(len(data1[key1]))] 
        arr2 = [ data2[key2][i][0] for i in range(len(data2[key2]))] 
        sub_sum = dot_product(arr1, arr2)
        total += sub_sum
    return total

def calculate_overlap_layer2(data1, data2):

    arr1, arr2 = [], []
    for key1, key2 in zip(data1.keys(), data2.keys()):
        for i in range(len(data1[key1])):
            obj1 = data1[key1][i][1]
            for key, value in obj1.items():
                for val in value:
                    arr1.append(val[0])
        for i in range(len(data2[key2])):
            obj2 = data2[key2][i][1]
            for key, value in obj2.items():
                for val in value:
                    arr2.append(val[0])

    return dot_product(arr1, arr2)

def dot_product(arr1, arr2):

    total = 0
    for ele1, ele2 in zip(arr1, arr2):
        total += (ele1 - ele2) * (ele1 - ele2)

    return total

def calculate_all_over_lap(data1, data2):
    """
    A function to calculate overlap1 + overlap2 between two dicts
    """
    # pack data to the same first layer and calculate dot_product
    data1_pack_layer1, data2_pack_layer1 = prepare_data_layer1(data1, data2)
    difference_1 = calculate_overlap_layer1(data1_pack_layer1, data2_pack_layer1)
    overlap1 = np.exp(-difference_1)

    # pack data to the same second layer and calculate dot_product
    data1_pack_pack_layer2, data2_pack_pack_layer2 = prepare_data_layer2(data1_pack_layer1, data2_pack_layer1)
    difference_2 = calculate_overlap_layer2(data1_pack_pack_layer2, data2_pack_pack_layer2)
    overlap2 = np.exp(-difference_2)

    return overlap1, overlap2

if __name__ == "__main__":
    with open(sys.argv[1], 'r') as fh1:
        data1 = json.load(fh1)
    with open(sys.argv[2], 'r') as fh2:
        data2 = json.load(fh2)

    data1_pack_layer1, data2_pack_layer1 = prepare_data_layer1(data1, data2)

    with open(sys.argv[1].replace('.json', '_pack_layer1.json'), 'w') as fh:
        json.dump(data1_pack_layer1, fh, indent=4)
    with open(sys.argv[2].replace('.json', '_pack_layer1.json'), 'w') as fh:
        json.dump(data2_pack_layer1, fh, indent=4)
    
    difference_1 = calculate_overlap_layer1(data1_pack_layer1, data2_pack_layer1)
    overlap1 = np.exp(-difference_1)

    print('overlap1: ', overlap1)

    data1_pack_layer2, data2_pack_layer2 = prepare_data_layer2(data1_pack_layer1, data2_pack_layer1)

    with open(sys.argv[1].replace('.json', '_pack_layer2.json'), 'w') as fh:
        json.dump(data1_pack_layer2, fh, indent=4)
    with open(sys.argv[2].replace('.json', '_pack_layer2.json'), 'w') as fh:
        json.dump(data2_pack_layer2, fh, indent=4)


    difference_2 = calculate_overlap_layer2(data1_pack_layer2, data2_pack_layer2)
    overlap2 = np.exp(-difference_2)

    print('overlap2: ', overlap2)