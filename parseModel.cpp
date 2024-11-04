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
              max_depth = maxdepth;
            }
        }
    }
    file.close();
    return trees;
}

void printNode(std::ostream& out, Tree tree, int i, int indentLevel, std::vector<int> features, std::vector<float> thresholds) {
    std::string indent(indentLevel, '\t');
    if (tree.feature_ids[i] == -1) {
        out  << indent << "result += " << thresholds[tree.thresholds_ids[i]] << ";\n";
    } else {
        out << indent << "if (values[" << features[tree.feature_ids[i]] << "] <= " << thresholds[tree.thresholds_ids[i]] << ") {\n";
        printNode(out, tree, 2 * i + 1, indentLevel + 1, features, thresholds);
        out << indent << "} else {\n";
        printNode(out, tree, 2 * i + 2, indentLevel + 1, features, thresholds);
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
    out << "\t\t\tfloat Predict(const float values[" << features.size() << "]) {\n";
    out << "\t\t\t\tfloat result = 0;\n";
    // out << "\t\t\t\tfloat thresholds[" << thresholds.size() << "];\n";
    int counter = 0;
    for (int i = 0; i < trees.size(); i++) {
        Tree tree = trees[i];
        out << "\t\t\t\t// tree " << counter << " ...\n";
        printNode(out, tree, 0, 4, features, thresholds);
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
        tmp += std::to_string(element) + ", ";
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
        tmp += std::to_string(array_values[element]) + ", ";
    }
    if (!tmp.empty()) {
        tmp.pop_back();
        tmp.pop_back();
    }
    tmp += "};\n";
    return tmp;
}

void genTree(std::ostream& out, Tree tree, int i, int indentLevel, std::vector<int> features, std::vector<float> thresholds) {
    std::string indent(indentLevel, '\t');
    out << indent << "Forest[" << i << "].feature_ids = new int[" << tree.feature_ids.size() << "] ;\n";
    out << indent << "int initial_feature_ids_" << i << "[" << tree.feature_ids.size() << "] = " << genArray("", tree.feature_ids, features);
    out << indent << "Forest[" << i << "].thresholds_ids = new float[" << tree.thresholds_ids.size() << "];\n";
    out << indent << "float initial_threshold_ids_" << i << "[" << tree.feature_ids.size() << "] = " << genArray("", tree.thresholds_ids, thresholds);
    out << indent << "for (int i = 0; i < " << tree.feature_ids.size() << "; ++i) {" << "\n";
    out << indent << "\tForest[" << i << "].feature_ids[i] = initial_feature_ids_" << i << "[i];" << "\n";
    out << indent << "\tForest[" << i << "].thresholds_ids[i] = initial_threshold_ids_" << i << "[i];" << "\n";
    out << indent << "}" << "\n";
}

void generatePredictTree(std::ostream& out, int indentLevel, int max_depth) {
    std::string indent(indentLevel, '\t');
    out << "\t\tfloat PredictTree(Tree tree, const float values[]) {\n";
    out << indent << "int node = 0;" << "\n";
    out << indent << "float threshold = 0.0;" << "\n";
    out << indent << "int max_depth =" << max_depth << ";\n";
    out << indent << "int i = 0;" << "\n";
    out << indent << "while (i < max_depth) {" << "\n";
    out << indent << "\tint feature_id = tree.feature_ids[node];" << "\n";
    out << indent << "\tthreshold = tree.thresholds_ids[node];" << "\n";
    out << indent << "\tif (feature_id == -1) {" << "\n";
    out << indent << "\t\tbreak;" << "\n";
    out << indent << "\t}" << "\n";
    out << indent << "\tvalues[feature_id] <= threshold ? node = 2 * node + 1 : node = 2 * node + 2;" << "\n";
    out << indent << "\ti++;" << "\n";
    out << indent << "}" << "\n";
    out << indent << "return threshold;" << "\n";
    out << "\t\t}" << "\n";
}

void generate_cpp_loop(const std::vector<Tree>& trees, std::vector<int> features, std::vector<float> thresholds, std::string output, int max_depth) {
    features.erase(std::remove(features.begin(), features.end(), -1), features.end());
    std::ofstream out(output + "_loop" + ".h");
    out << "#pragma once" << "\n";
    out << "namespace LightGBM { " << "\n";
    out << "\tstruct Tree {" << "\n";
    out << "    int* feature_ids;" << "\n";
    out << "    float* thresholds_ids;" << "\n";
    out << "\t};" << "\n";
    out << "\tclass CovTypeClassifier {" << "\n";
    out << "\tpublic:" << "\n";
    out << "\t\tfloat Predict(const float values[" << features.size() << "]) {\n";
    out << "\t\t\tTree Forest["<< trees.size() <<"];\n";
    for (int i = 0; i < trees.size(); i++) {
        genTree(out, trees[i], i, 3, features, thresholds);
    }
    out << "\t\t\tfloat result = 0;\n";
    // out << genArray("float thresholds[" + std::to_string(thresholds.size()) + "]", thresholds);
    // out << genArray("byte features[" +  std::to_string(features.size()) + "]", features);
    int counter = 0;
    for (int i = 0; i < trees.size(); i++) {
        Tree tree = trees[i];
        out << "\t\t\tresult += PredictTree(Forest[" << counter << "], values);\n";
        counter++;
    }
    out << "\t\t\treturn 1.0f / (1.0f + exp(-1.0 * result));\n";
    out << "\t\t}\n";
    generatePredictTree(out, 3, max_depth);

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