#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <string>


struct Node {
    int featureid;
    int thresholdid;
};
struct Tree {
    std::vector<double> leaf_values;
    std::vector<int> feature_ids;
    std::vector<int> thresholds_ids;
    std::vector<Node> nodes;
};
struct Sizes {
    int max_depth;
    int featuresize;
    int thresholdssize;
    int maxthresholdsize;
    int maxfeaturesize;
};
Sizes size = {0,0,0,0};
std::vector<Tree> parse_trees_from_file(const std::string& filename, std::vector<int>& features, std::vector<float>& thresholds, int &max_depth) {
    std::ifstream file(filename);
    std::string line;
    std::vector<Tree> trees;
    Tree currentTree;
    while (getline(file, line)) {
        if (line.empty()) { // Skip empty lines
            continue;
        }

        std::istringstream iss(line);
        std::string header;
        iss >> header; // Read the first token in the line as header
        std::string::size_type pos = header.find('=');
        if (pos != std::string::npos) {
            header = header.substr(0, pos+1);
        }
        if (header == "leaf_value=") {
            double value;
            while (iss >> value) { // Read remaining values on the line
                currentTree.leaf_values.push_back(value);
            }
        } else if (header == "tiny_tree_ids_features=") {
            int id;
            while (iss >> id) {
                currentTree.feature_ids.push_back(id);
            }
        } else if (header == "feature_names=") {
            std::string column;
            while (iss >> column) {
                size.featuresize++;
            }
        } else if (header == "tiny_tree_thresholds=") {
            double threshold;
            thresholds.clear();
            while (iss >> threshold) {
                thresholds.push_back(threshold);
            }
        } else if (header == "tiny_tree_ids_thresholds=") {
            double threshold_id;
            while (iss >> threshold_id) {
                currentTree.thresholds_ids.push_back(threshold_id);
            }
        } else if (header == "tiny_tree_features=") {
            int feature;
            features.clear();
            while (iss >> feature) {
                features.push_back(feature);
            }
        } else if (header.find("shrinkage=")!= std::string::npos) {
            trees.push_back(currentTree);
            currentTree = Tree(); // Reset tree for next use
        } else if (header.find("[max_depth:") != std::string::npos) {
            int maxdepth;
            while (iss >> maxdepth) {
              size.max_depth = maxdepth;
            }
        }
    }
    file.close();
    size.thresholdssize = thresholds.size();
    return trees;
}

void printNode(std::ostream& out, Tree tree, int i, std::vector<int> features, std::vector<float> thresholds, int indentLevel =4) {
    std::string indent(indentLevel, '\t');
    if (tree.feature_ids[i] == -1) {
        out  << indent << "result += " << thresholds[tree.thresholds_ids[i]] << ";\n";
    } else {
        out << indent << "if (values[" << features[tree.feature_ids[i]] << "] <= " << thresholds[tree.thresholds_ids[i]] << ") {\n";
        printNode(out, tree, 2 * i + 1,  features, thresholds, indentLevel +1);
        out << indent << "} else {\n";
        printNode(out, tree, 2 * i + 2, features, thresholds, indentLevel + 1);
        out << indent << "}\n";
    }
}

void generate_cpp_file(const std::vector<Tree>& trees, std::vector<int> features, std::vector<float> thresholds, std::string output) {
    features.erase(std::remove(features.begin(), features.end(), -1), features.end());
    std::ofstream out(output + ".h");
    out << "#pragma once" << "\n";
    out << "namespace LightGBM { " << "\n";
    out << "\t\tclass CovTypeClassifier {" << "\n";
    out << "\t\tpublic:" << "\n";
    out << "\t\t\tfloat Predict(const float values[" << size.featuresize << "]) {\n";
    out << "\t\t\t\tfloat result = 0;\n";
    int counter = 0;
    for (int i = 0; i < trees.size(); i++) {
        Tree tree = trees[i];
        out << "\t\t\t\t// tree " << counter << " ...\n";
        printNode(out, tree, 0, features, thresholds);
        counter++;
    }
    out << "\t\t\treturn 1.0f / (1.0f + exp(-1.0 * result));\n";
    out << "\t\t}\n";
    out << "\t};" << "\n";
    out << "}" << "\n";
    out.close();
}

template <typename T>
std::string genArray(std::string name, std::vector<T> array) {
    std::string tmp = "";
    name == "" ? tmp += "{": tmp += "\t\t\t" + name + " = {";
    for (auto element : array) {
        if (element == -1) {
            tmp += "-1, ";
        } else {
            tmp += std::to_string(element) + ", ";
        }
    }
    if (!tmp.empty()) {
        tmp.pop_back();
        tmp.pop_back();
    }
    tmp += "};\n";
    return tmp;
}


template <typename T, typename T2>
std::string genArray(std::string name, std::vector<T> array, std::vector<T2> array_values) {
    std::string tmp = "";
    name == "" ? tmp += "{": tmp += "\t\t\t" + name + " = {";
    for (auto element : array) {
        if (element == -1) {
            tmp += "-1, ";
        } else {
            tmp += std::to_string(array_values[element]) + ", ";
        }
    }
    if (!tmp.empty()) {
        tmp.pop_back();
        tmp.pop_back();
    }
    tmp += "};\n";
    return tmp;
}

template <typename T, typename T2>
void genArraySingle(std::ostream& out, std::string writename, int indentLevel, std::vector<T> array, std::vector<T2> array_values) {
    std::string indent(indentLevel, '\t');
    for (int i = 0; i< array.size(); i++) {
        if (array[i] == -1) {
            out << indent << writename << "[" << i << "] = -1;\n";
        } else {
            out << indent << writename << "[" << i << "] = " << array_values[array[i]] << ";\n";
        }
    }
}
template <typename T2>
void genArraySingle(std::ostream& out, std::string writename, int indentLevel, std::vector<T2> array_values) {
    std::string indent(indentLevel, '\t');
    for (int i = 0; i< array_values.size(); i++) {
        if (array_values[i] == -1) {
            out << indent << writename << "[" << i << "] = -1;\n";
        } else {
            out << indent << writename << "[" << i << "] = " << array_values[i] << ";\n";
        }
    }
}
void genTree(std::ostream& out, Tree tree, int i, int indentLevel, std::vector<int> features) {
    std::string indent(indentLevel, '\t');
    genArraySingle(out, "features", indentLevel, tree.feature_ids, features);
    genArraySingle(out, "thresholdids", indentLevel, tree.thresholds_ids);
}

void generatePredictTree(std::ostream& out, std::string featuretype, std::string thresholdtype) {
    std::string indent(3, '\t');

    out << "\t\tfloat PredictTree(const " << featuretype << "* features, const " << thresholdtype << "* threshold_ids, const float thresholds[" << size.thresholdssize << "], const float values[" << size.featuresize << "]) {\n";
    out << indent << "int node = 0;" << "\n";
    out << indent << "float threshold = 0.0;" << "\n";
    out << indent << "int max_depth =" << size.max_depth << ";\n";
    out << indent << "int featureid = 0;\n";

    out << indent << "for (int i = 0; i < max_depth; i++) {" << "\n";
    out << indent << "\tfeatureid = features[node];" << "\n";
    out << indent << "\tthreshold = thresholds[threshold_ids[node]];" << "\n";
    out << indent << "\tif (featureid == -1) {" << "\n";
    out << indent << "\t\treturn threshold;" << "\n";
    out << indent << "\t}" << "\n";
    out << indent << "\tvalues[featureid] <= threshold ? node = 2 * node + 1 : node = 2 * node + 2;" << "\n";
    out << indent << "}" << "\n";
    out << indent << "return threshold;" << "\n";
    out << "\t\t}" << "\n";
}

void generate_cpp_loop(const std::vector<Tree>& trees, std::vector<int> features, std::vector<float> thresholds, std::string output, int max_depth) {
    features.erase(std::remove(features.begin(), features.end(), -1), features.end());
    std::ofstream out(output + "_loop" + ".h");
    out << "#pragma once" << "\n";
    out << "namespace LightGBM { " << "\n";
    out << "\tclass CovTypeClassifier {" << "\n";
    out << "\tpublic:" << "\n";
    out << "\t\tfloat Predict(const float values[" << size.featuresize << "]) {\n";
    int counter = 0;
    std::string typefeature = "int";
    std::string typethreshold = "int";
    if (size.featuresize < 255) {
        typefeature = "byte";
    }
    if (size.thresholdssize < 255) {
        typethreshold = "byte";
    }

    out << genArray("float thresholds[" + std::to_string(thresholds.size()) + "]", thresholds);
    out << "\t\t\tfloat result = 0;\n";

    out << "\t\t\t" << typefeature << " *features = NULL;\n";
    out << "\t\t\t" << typethreshold << " *thresholdids = NULL;\n";
    out << "\t\t\tfeatures = malloc(" << trees[0].feature_ids.size() << "* sizeof ("<< typefeature << "));\n";
    out << "\t\t\tthresholdids = malloc(" << trees[0].thresholds_ids.size() << "* sizeof ("<< typethreshold << "));\n";

    for (int i = 0; i < trees.size(); i++) {
        out << "\t\t\t// Tree " << i << "\n";
        genTree(out, trees[i], i, 3, features);
        Tree tree = trees[i];
        out << "\t\t\tresult += PredictTree( features, thresholdids, thresholds, values);\n";
        counter++;
        if (tree.feature_ids.size() > size.maxfeaturesize) {
            size.maxfeaturesize = tree.feature_ids.size();
        }
        if (tree.thresholds_ids.size() > size.maxthresholdsize) {
            size.maxthresholdsize = tree.thresholds_ids.size();
        }
    }
    out << "\t\t\treturn 1.0f / (1.0f + exp(-1.0 * result));\n";
    out << "\t\t}\n";
    // TODO: Currently we take the max in any tree. Does it make a difference to have multiple functions in case a tree is smaller?
    generatePredictTree(out, typefeature, typethreshold);

    out << "\t};" << "\n";
    out << "}" << "\n";
    out.close();
}

int main(int argc, char* argv[]) {

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <pathLightGBMModel> <pathOutput>" << std::endl;
        return 1; // Exit with error code
    }
    std::string filename = argv[1];
    std::vector<float> thresholds;
    std::vector<int> features;
    int max_depth = 0;
    std::vector<Tree> trees = parse_trees_from_file(filename, features, thresholds, max_depth);
    generate_cpp_file(trees, features, thresholds, argv[2]);
    generate_cpp_loop(trees, features, thresholds, argv[2], max_depth);
    return 0;
}