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

std::vector<Tree> parse_trees_from_file(const std::string& filename, std::vector<int>& features, std::vector<float>& thresholds) {
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
        }
    }
    file.close();
    return trees;
}

void printNode(std::ostream& out, Tree tree, int i, int indentLevel, std::vector<int> features) {
    std::string indent(indentLevel, '\t');
    if (tree.feature_ids[i] == -1) {
        out  << indent << "result += " << "thresholds[" << tree.thresholds_ids[i] << "];\n";
    } else {
        out << indent << "if (values[" << features[tree.feature_ids[i]] << "] <= thresholds[" << tree.thresholds_ids[i] <<"]) {\n";
        printNode(out, tree, 2 * i + 1, indentLevel + 1, features);
        out << indent << "} else {\n";
        printNode(out, tree, 2 * i + 2, indentLevel + 1, features);
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
        printNode(out, tree, 0, 4, features);
        counter++;
    }
    out << "\t\t\treturn 1.0f / (1.0f + exp(-1.0 * result));\n";
    out << "\t\t}\n";
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
    std::vector<Tree> trees = parse_trees_from_file(filename, features, thresholds);
    generate_cpp_file(trees, features, thresholds, argv[2]);
    return 0;
}