# coding: utf-8
# pylint: disable = C0111, C0103
"""convert LightGBM model to pmml"""
from __future__ import absolute_import

from sys import argv
from itertools import count


def get_value_string(line):
    return line[line.find('=') + 1:]


def get_array_strings(line):
    return get_value_string(line).split()


def get_array_ints(line):
    return [int(token) for token in get_array_strings(line)]


def get_field_name(node_id, prev_node_idx, is_child):
    idx = leaf_parent[node_id] if is_child else prev_node_idx
    return feature_names[split_feature[idx]]


def get_threshold(node_id, prev_node_idx, is_child):
    idx = leaf_parent[node_id] if is_child else prev_node_idx
    return threshold[idx]


def print_simple_predicate(tab_len, node_id, is_left_child, prev_node_idx, is_leaf):
    if is_left_child:
        op = 'equal' if decision_type[prev_node_idx] == 1 else 'lessOrEqual'
    else:
        op = 'notEqual' if decision_type[prev_node_idx] == 1 else 'greaterThan'
    out_('\t' * (tab_len + 1) + ("<SimplePredicate field=\"{0}\" " + " operator=\"{1}\" value=\"{2}\" />").format(
        get_field_name(node_id, prev_node_idx, is_leaf), op, get_threshold(node_id, prev_node_idx, is_leaf)))


def print_nodes_pmml(node_id, tab_len, is_left_child, prev_node_idx):
    if node_id < 0:
        node_id = ~node_id
        score = leaf_value[node_id]
        recordCount = leaf_count[node_id]
        is_leaf = True
    else:
        score = internal_value[node_id]
        recordCount = internal_count[node_id]
        is_leaf = False
    out_('\t' * tab_len + ("<Node id=\"{0}\" score=\"{1}\" " + " recordCount=\"{2}\">").format(
        next(unique_id), score, recordCount))
    print_simple_predicate(tab_len, node_id, is_left_child, prev_node_idx, is_leaf)
    if not is_leaf:
        print_nodes_pmml(left_child[node_id], tab_len + 1, True, node_id)
        print_nodes_pmml(right_child[node_id], tab_len + 1, False, node_id)
    out_('\t' * tab_len + "</Node>")


# print out the pmml for a decision tree
def print_pmml():
    # specify the objective as function name and binarySplit for
    # splitCharacteristic because each node has 2 children
    out_("\t\t\t\t<TreeModel functionName=\"regression\" splitCharacteristic=\"binarySplit\">")
    out_("\t\t\t\t\t<MiningSchema>")
    # list each feature name as a mining field, and treat all outliers as is,
    # unless specified
    for feature in feature_names:
        out_("\t\t\t\t\t\t<MiningField name=\"%s\"/>" % (feature))
    out_("\t\t\t\t\t</MiningSchema>")
    # begin printing out the decision tree
    out_("\t\t\t\t\t<Node id=\"{0}\" score=\"{1}\" recordCount=\"{2}\">".format(
        next(unique_id), internal_value[0], internal_count[0]))
    out_("\t\t\t\t\t\t<True/>")
    print_nodes_pmml(left_child[0], 6, True, 0)
    print_nodes_pmml(right_child[0], 6, False, 0)
    out_("\t\t\t\t\t</Node>")
    out_("\t\t\t\t</TreeModel>")


if len(argv) != 2:
    raise ValueError('usage: pmml.py <input model file>')

# open the model file and then process it
with open(argv[1], 'r') as model_in:
    # ignore first 6 and empty lines
    model_content = iter([line for line in model_in.read().splitlines() if line][6:])

feature_names = get_array_strings(next(model_content))
segment_id = count(1)

with open('LightGBM_pmml.xml', 'w') as pmml_out:
    def out_(string):
        pmml_out.write(string + '\n')
    out_(
        "<PMML version=\"4.3\" \n" +
        "\t\txmlns=\"http://www.dmg.org/PMML-4_3\"\n" +
        "\t\txmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n" +
        "\t\txsi:schemaLocation=\"http://www.dmg.org/PMML-4_3 http://dmg.org/pmml/v4-3/pmml-4-3.xsd\">")
    out_("\t<Header copyright=\"Microsoft\">")
    out_("\t\t<Application name=\"LightGBM\"/>")
    out_("\t</Header>")
    # print out data dictionary entries for each column
    out_("\t<DataDictionary numberOfFields=\"%d\">" % len(feature_names))
    # not adding any interval definition, all values are currently
    # valid
    for feature in feature_names:
        out_("\t\t<DataField name=\"" + feature + "\" optype=\"continuous\" dataType=\"double\"/>")
    out_("\t</DataDictionary>")
    out_("\t<MiningModel functionName=\"regression\">")
    out_("\t\t<MiningSchema>")
    # list each feature name as a mining field, and treat all outliers
    # as is, unless specified
    for feature in feature_names:
        out_("\t\t\t<MiningField name=\"%s\"/>" % (feature))
    out_("\t\t</MiningSchema>")
    out_("\t\t<Segmentation multipleModelMethod=\"sum\">")
    # read each array that contains pertinent information for the pmml
    # these arrays will be used to recreate the traverse the decision tree
    while True:
        tree_start = next(model_content, '')
        if not tree_start.startswith('Tree'):
            break
        out_("\t\t\t<Segment id=\"%d\">" % next(segment_id))
        out_("\t\t\t\t<True/>")
        tree_no = tree_start[5:]
        num_leaves = int(get_value_string(next(model_content)))
        split_feature = get_array_ints(next(model_content))
        split_gain = next(model_content)  # unused
        threshold = get_array_strings(next(model_content))
        decision_type = get_array_ints(next(model_content))
        left_child = get_array_ints(next(model_content))
        right_child = get_array_ints(next(model_content))
        leaf_parent = get_array_ints(next(model_content))
        leaf_value = get_array_strings(next(model_content))
        leaf_count = get_array_strings(next(model_content))
        internal_value = get_array_strings(next(model_content))
        internal_count = get_array_strings(next(model_content))
        unique_id = count(1)
        print_pmml()
        out_("\t\t\t</Segment>")

    out_("\t\t</Segmentation>")
    out_("\t</MiningModel>")
    out_("</PMML>")
