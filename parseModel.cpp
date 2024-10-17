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
    std::vector<int> s_feature_ids;
    std::vector<int> thresholds_ids;
    std::vector<int> s_thresholds_ids;
    std::vector<double> thresholds;
    std::vector<Node> nodes;
};

std::vector<Tree> parse_trees_from_file(const std::string& filename) {
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
            header = header.substr(0, pos);
        }
        if (header == "leaf_value") {
            double value;
            while (iss >> value) { // Read remaining values on the line
                currentTree.leaf_values.push_back(value);
            }
        } else if (header == "tiny_tree_ids_features") {
            int id;
            while (iss >> id) {
                currentTree.feature_ids.push_back(id);
            }
        } else if (header == "tiny_tree_thresholds") {
            double threshold;
            while (iss >> threshold) {
                currentTree.thresholds.push_back(threshold);
            }
        } else if (header == "tiny_tree_ids_thresholds") {
            double threshold_id;
            while (iss >> threshold_id) {
                currentTree.thresholds_ids.push_back(threshold_id);
            }
        } else if (header.find("shrinkage")!= std::string::npos) {
            trees.push_back(currentTree);
            currentTree = Tree(); // Reset tree for next use
        }
    }
    file.close();
    return trees;
}

void printNode(std::ostream& out, Tree tree, int i, int& leavecounter, int indentLevel) {
    std::string indent(indentLevel, '\t');

    if (tree.feature_ids[i] == -1) {
        out  << indent << "result += " << tree.leaf_values[leavecounter] << ";\n";
        leavecounter++;
    } else {
        out << indent << "if (values[" << tree.feature_ids[i] << "] <" << "thresholds[" << tree.thresholds_ids[i] <<"]) {" << "\n";
        printNode(out, tree, 2 * i + 1, leavecounter, indentLevel + 1);
        out << indent << "} else {\n";
        printNode(out, tree, 2 * i + 1, leavecounter, indentLevel + 1);
        out << indent << "}\n";
    }
}
void generate_cpp_file(const std::vector<Tree>& trees, std::vector<int> features, std::vector<float> thresholds, std::string output) {
    features.erase(std::remove(features.begin(), features.end(), -1), features.end());
    std::ofstream out(output);
    out << "#pragma once" << "\n";
    out << "namespace LightGBM { " << "\n";
    out << "\t\tclass CovTypeClassifier {" << "\n";
    out << "\t\tpublic:" << "\n";
    out << "\t\t\tfloat predict(const float values[" << features.size() << "]) {\n";
    out << "\t\t\t\tfloat result = 0;\n";
    out << "\t\t\t\tfloat thresholds[" << thresholds.size() << "];\n";
    int counter = 0;
    for (auto threshold : thresholds) {
        out << "\t\t\t\tthresholds[" << counter << "] =" << threshold << ";\n";
        counter++;
    }
    out << "\t\t\t\tint features[" <<  features.size() << "];\n";
    counter = 0;
    for (auto feature : features) {
        out << "\t\t\t\tfeatures[" << counter << "] =" << feature << ";\n";
        counter++;
    }
    counter = 0;
    for (int i = 0; i < trees.size(); i++) {
        Tree tree = trees[i];
        out << "\t\t\t\t// tree " << counter << " ...\n";
        int leavecounter = 0;
        printNode(out, tree, 0, leavecounter, 4);
        counter++;
    }
    out << "\t\t\treturn result;\n";
    out << "\t\t}\n";
    out << "\t};" << "\n";
    out << "}" << "\n";
    out.close();
}

template<typename T>
void getUniqueFromTree(std::vector<Tree>& trees, std::vector<T>& vec, bool feature) {
    for (auto treeit = trees.begin(); treeit != trees.end(); ++treeit){
        Tree tree = *treeit;
        feature ? vec.insert(vec.end(), tree.feature_ids.begin(), tree.feature_ids.end()) : vec.insert(vec.end(), tree.thresholds.begin(), tree.thresholds.end());
    }
    std::sort(vec.begin(), vec.end());
    auto last = std::unique(vec.begin(), vec.end());
    vec.erase(last, vec.end());
}
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <pathLightGBMModel> <pathOutput>" << std::endl;
        return 1; // Exit with error code
    }
    std::string filename = argv[1]; // "examples/binary_classification/LightGBM_model.txt";
    std::vector<Tree> trees = parse_trees_from_file(filename);
    std::vector<float> thresholds;
    std::vector<int> features;
    getUniqueFromTree(trees, thresholds, false);
    getUniqueFromTree(trees, features, true);
    generate_cpp_file(trees, features, thresholds, argv[2]);
    return 0;
}