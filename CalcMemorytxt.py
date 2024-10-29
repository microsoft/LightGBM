import re
import sys

def read_line(line, name, floattype, previous):
    thresholds_count = 0
    if name in line:
        # Extract floats after tiny_tree_thresholds=
        values = re.findall(fr'{name}=\s*(.*)', line)
        if values:
            if floattype:
                num_values = re.findall(r'[-+]?\d*\.\d+|\d+', values[0])
            else:
                num_values = re.findall(r'\b\d+\b', values[0])
            thresholds_count = len(num_values)
            return thresholds_count
        else:
            return previous
    else:
        return previous

def count_values_in_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    tinycounts = {
        "tiny_tree_thresholds": {"value": 0, "float": True, "add": False},
        "tiny_tree_features": {"value": 0, "float": False, "add": False},
        "tiny_tree_ids_features": {"value": 0, "float": False, "add": True},
        "tiny_tree_ids_thresholds": {"value": 0, "float": False, "add": True}
    }
    nativecounts = {
        "split_feature": {"value": 0, "float": False},
        "threshold": {"value": 0, "float": True},
        "left_child": {"value": 0, "float": False},
        "right_child": {"value": 0, "float": False},
        "leaf_value": {"value": 0, "float": True}
    }
    memorycounttiny = 0
    memorycountnative = 0

    for line in lines:
        for key, element in tinycounts.items() :
            if element['add']:
                element['value'] += read_line(line, key, element['float'], 0)
            else:
                element['value'] = read_line(line, key, element['float'], element['value'])

        for key, element in nativecounts.items() :
                element['value'] += read_line(line, key, element['float'], 0)


    for key, data in nativecounts.items():
        multiplier = 4 if data["float"] else 2
        memorycountnative += data['value'] * multiplier

    for key, data in tinycounts.items():
        multiplier = 4 if data["float"] else 2
        memorycounttiny += data['value'] * multiplier

    print(tinycounts)
    print(nativecounts)
    print(f"Assumed memory consumption tiny = {memorycounttiny} Assumed memory consumption native = {memorycountnative}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_values.py <file_path>")
    else:
        file_path = sys.argv[1]
        count_values_in_file(file_path)
