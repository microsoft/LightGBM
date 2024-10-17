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
            printf("pushing a tree");
            trees.push_back(currentTree);
            currentTree = Tree(); // Reset tree for next use
        }
    }
    file.close();
    return trees;
}

void printNode(std::ostream& out, Tree tree, int i, int& leavecounter) {
    if (tree.feature_ids[i] == -1) {
        out << "        result += " << tree.leaf_values[leavecounter] << ";\n";
        leavecounter++;
    } else {
        out << "        if (values[" << tree.feature_ids[i] << "] <" << "thresholds[" << tree.thresholds_ids[i] <<"]) {" << "\n";
        printNode(out, tree, 2 * i + 1, leavecounter);
        out << "        } else {\n";
        printNode(out, tree, 2 * i + 1, leavecounter);
        out << "        }\n";
    }
}
void generate_cpp_file(const std::vector<Tree>& trees, std::vector<int> features, std::vector<float> thresholds) {
    features.erase(std::remove(features.begin(), features.end(), -1), features.end());
    std::ofstream out("Predict.hpp");
    out << "#include <vector>\n";
    out << "float predict(const float values[" << features.size() << "]) {\n";
    out << "    float result = 0;\n";
    out << "    float thresholds[" << thresholds.size() << "];\n";
    int counter = 0;
    for (auto threshold : thresholds) {
        out << "    thresholds[" << counter << "] =" << threshold << ";\n";
        counter++;
    }
    out << "    int features[" <<  features.size() << "];\n";
    counter = 0;
    for (auto feature : features) {
        out << "    features[" << counter << "] =" << feature << ";\n";
        counter++;
    }
    counter = 0;
    for (int i = 0; i < trees.size(); i++) {
        Tree tree = trees[i];
        out << "    // " << counter << " tree...\n";
        int leavecounter = 0;
        printNode(out, tree, 0, leavecounter);
    }
    out << "    return result;\n";
    out << "}\n";
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
int main() {
    std::string filename = "examples/min/LightGBM_model.txt";
    std::vector<Tree> trees = parse_trees_from_file(filename);
    std::vector<float> thresholds;
    std::vector<int> features;
    getUniqueFromTree(trees, thresholds, false);
    getUniqueFromTree(trees, features, true);
    generate_cpp_file(trees, features, thresholds);
    return 0;
}