from __future__ import print_function
from decimal import Decimal

import sys
import os
import traceback


def unique_id():
    global unique_node_id
    nid = unique_node_id
    unique_node_id += 1
    return nid


def get_value_string(line):
    return line[line.index('=') + 1:]


def get_array_strings(line):
    return line[line.index('=') + 1:].split()


def get_array_ints(line):
    return map(lambda x: int(x), line[line.index('=') + 1:].split())


def get_array_floats(line):
    return map(lambda x: Decimal(x), line[line.index('=') + 1:].split())


def get_field_name(node_id, prev_node_idx, is_child):
    idx = leaf_parent[node_id - 1] if is_child else prev_node_idx
    return feature_names[split_feature[idx]]


def get_threshold(node_id, prev_node_idx, is_child):
    idx = leaf_parent[node_id - 1] if is_child else prev_node_idx
    return threshold[idx]


def print_simple_predicate(
        tab_length,
        node_id,
        is_left_child,
        prev_node_idx,
        is_leaf,
        pmml_out):
    if is_left_child:
        op = 'equal' if decision_type[prev_node_idx] == 1 else 'lessOrEqual'
    else:
        op = 'notEqual' if decision_type[prev_node_idx] == 1 else 'greaterThan'
    print('\t' * (tab_length + 1) + ("<SimplePredicate field=\"{0}\" " + " operator=\"{1}\" value=\"{2}\" />") .format(
        get_field_name(node_id, prev_node_idx, is_leaf), op, get_threshold(node_id, prev_node_idx, is_leaf)), file=pmml_out)


def print_nodes_pmml(**kwargs):
    node_id = kwargs['node_id']
    pmml_out = kwargs['out_file']
    tab_len = kwargs['tab_length']
    if node_id < 0:
        node_id = -1 * node_id
        score = leaf_value[node_id - 1]
        recordCount = leaf_count[node_id - 1]
        is_leaf = True
    else:
        score = internal_value[node_id]
        recordCount = internal_count[node_id]
        is_leaf = False
    print(
        '\t' *
        tab_len +
        (
            "<Node id=\"{0}\" score=\"{1}\" " +
            " recordCount=\"{2}\">").format(
            unique_id(),
            score,
            recordCount),
        file=pmml_out)
    print_simple_predicate(
        tab_len,
        node_id,
        kwargs['is_left_child'],
        kwargs['prev_node_idx'],
        is_leaf,
        pmml_out)
    if not is_leaf:
        print_nodes_pmml(
            node_id=left_child[node_id],
            tab_length=tab_len + 1,
            is_left_child=True,
            prev_node_idx=node_id,
            out_file=pmml_out)
        print_nodes_pmml(
            node_id=right_child[node_id],
            tab_length=tab_len + 1,
            is_left_child=False,
            prev_node_idx=node_id,
            out_file=pmml_out)
    print('\t' * tab_len + "</Node>", file=pmml_out)


# print out the pmml for a decision tree
def print_pmml(pmml_out):
    # specify the objective as function name and binarySplit for
    # splitCharacteristic because each node has 2 children
    print(
        "\t\t\t\t<TreeModel functionName=\"regression\" splitCharacteristic=\"binarySplit\">",
        file=pmml_out)
    print("\t\t\t\t\t<MiningSchema>", file=pmml_out)
    # list each feature name as a mining field, and treat all outliers as is,
    # unless specified
    for feature in feature_names:
        print(
            "\t\t\t\t\t\t<MiningField name=\"%s\"/>" %
            (feature), file=pmml_out)
    print("\t\t\t\t\t</MiningSchema>", file=pmml_out)
    # begin printing out the decision tree
    print("\t\t\t\t\t<Node id=\"%d\" score=\"%s\" recordCount=\"%d\">" %
          (unique_id(), internal_value[0], internal_count[0]), file=pmml_out)
    print("\t\t\t\t\t\t<True/>", file=pmml_out)
    print_nodes_pmml(
        node_id=left_child[0],
        tab_length=6,
        is_left_child=True,
        prev_node_idx=0,
        out_file=pmml_out)
    print_nodes_pmml(
        node_id=right_child[0],
        tab_length=6,
        is_left_child=False,
        prev_node_idx=0,
        out_file=pmml_out)
    print("\t\t\t\t\t</Node>", file=pmml_out)
    print("\t\t\t\t</TreeModel>", file=pmml_out)

if len(sys.argv) != 2:
    print('usage: pmml.py <input model file>')
    sys.exit(0)

# open the model file and then process it
try:
    with open(sys.argv[1]) as model_in:
        model_content = filter(
            lambda line: line != '',
            model_in.read().strip().split('\n'))
        objective = get_value_string(model_content[4])
        sigmoid = Decimal(get_value_string(model_content[5]))
        feature_names = get_array_strings(model_content[6])
        model_content = model_content[7:]
        line_no = 0
        segment_id = 1

        with open('LightGBM_pmml.xml', 'w') as pmml_out:
            print(
                "<PMML version=\"4.3\" \n" +
                "\t\txmlns=\"http://www.dmg.org/PMML-4_3\"\n" +
                "\t\txmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n" +
                "\t\txsi:schemaLocation=\"http://www.dmg.org/PMML-4_3 http://dmg.org/pmml/v4-3/pmml-4-3.xsd\"" +
                ">",
                file=pmml_out)
            print("\t<Header copyright=\"Microsoft\">", file=pmml_out)
            print("\t\t<Application name=\"LightGBM\"/>", file=pmml_out)
            print("\t</Header>", file=pmml_out)
            # print out data dictionary entries for each column
            print(
                "\t<DataDictionary numberOfFields=\"%d\">" %
                len(feature_names), file=pmml_out)
            # not adding any interval definition, all values are currently
            # valid
            for feature in feature_names:
                print(
                    "\t\t<DataField name=\"" +
                    feature +
                    "\" optype=\"continuous\" dataType=\"double\"/>",
                    file=pmml_out)
            print("\t</DataDictionary>", file=pmml_out)
            print("\t<MiningModel functionName=\"regression\">", file=pmml_out)
            print("\t\t<MiningSchema>", file=pmml_out)
            # list each feature name as a mining field, and treat all outliers
            # as is, unless specified
            for feature in feature_names:
                print(
                    "\t\t\t<MiningField name=\"%s\"/>" %
                    (feature), file=pmml_out)
            print("\t\t</MiningSchema>", file=pmml_out)
            print(
                "\t\t<Segmentation multipleModelMethod=\"sum\">",
                file=pmml_out)
            # read each array that contains pertinent information for the pmml
            # these arrays will be used to recreate the traverse the decision
            # tree
            while model_content[line_no][:4] == 'Tree':
                print("\t\t\t<Segment id=\"%d\">" % segment_id, file=pmml_out)
                print("\t\t\t\t<True/>", file=pmml_out)
                tree_no = model_content[line_no][5:]
                num_leaves = int(get_value_string(model_content[line_no + 1]))
                split_feature = get_array_ints(model_content[line_no + 2])
                threshold = get_array_floats(model_content[line_no + 4])
                decision_type = get_array_ints(model_content[line_no + 5])
                left_child = get_array_ints(model_content[line_no + 6])
                right_child = get_array_ints(model_content[line_no + 7])
                leaf_parent = get_array_ints(model_content[line_no + 8])
                leaf_value = get_array_floats(model_content[line_no + 9])
                leaf_count = get_array_ints(model_content[line_no + 10])
                internal_value = get_array_floats(model_content[line_no + 11])
                internal_count = get_array_ints(model_content[line_no + 12])
                unique_node_id = 0
                print_pmml(pmml_out)
                print("\t\t\t</Segment>", file=pmml_out)
                line_no += 13
                segment_id += 1

            print("\t\t</Segmentation>", file=pmml_out)
            print("\t</MiningModel>", file=pmml_out)
            print("</PMML>", file=pmml_out)
except Exception as ioex:
    print(ioex)
