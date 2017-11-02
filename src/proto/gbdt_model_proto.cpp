#include "../boosting/gbdt.h"

#include <LightGBM/tree.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/objective_function.h>
#include <iostream>
#include <fstream>

namespace LightGBM {

void GBDT::SaveModelToProto(int num_iteration, LightGBM::Model& model) const {
    model.set_name(SubModelName());
    model.set_num_class(num_class_);
    model.set_num_tree_per_iteration(num_tree_per_iteration_);
    model.set_label_index(label_idx_);
    model.set_max_feature_idx(max_feature_idx_);
    if (objective_function_ != nullptr) {
        model.set_objective(objective_function_->ToString());
    }
    model.set_average_output(average_output_);
    for(auto feature_name: feature_names_) {
        model.add_feature_names(feature_name);
    }
    for(auto feature_info: feature_infos_) {
        model.add_feature_infos(feature_info);
    }
    
    int num_used_model = static_cast<int>(models_.size());
    if (num_iteration > 0) {
        num_used_model = std::min(num_iteration * num_tree_per_iteration_, num_used_model);
    }
    for (int i = 0; i < num_used_model; ++i) {
        models_[i]->ToProto(*model.add_trees());
    }
}

void GBDT::SaveModelToProto(int num_iteration, const char* filename) const {
    LightGBM::Model model;
    SaveModelToProto(num_iteration, model);

    std::filebuf fb;
    fb.open(filename, std::ios::out | std::ios::binary);
    std::ostream os(&fb);
    if (!model.SerializeToOstream(&os)) {
        Log::Fatal("Cannot serialize model to binary file.");
    }
    fb.close();
}

void GBDT::SaveModelToPythonProto(int num_iteration, const char* filename, const char* pandas_category) const {
    LightGBM::Model model;
    SaveModelToProto(num_iteration, model);

    LightGBM::PythonModel python_model;
    python_model.mutable_model()->CopyFrom(model);
    python_model.set_pandas_category(pandas_category);

    std::filebuf fb;
    fb.open(filename, std::ios::out | std::ios::binary);
    std::ostream os(&fb);
    if (!python_model.SerializeToOstream(&os)) {
        Log::Fatal("Cannot serialize python model to binary file.");
    }
    fb.close();
}

bool GBDT::LoadModelFromProto(const LightGBM::Model& model) {
    models_.clear();
    num_class_ = model.num_class();
    num_tree_per_iteration_ = model.num_tree_per_iteration();
    label_idx_ = model.label_index();
    max_feature_idx_ = model.max_feature_idx();
    average_output_ = model.average_output();
    feature_names_.reserve(model.feature_names_size());
    for (auto feature_name: model.feature_names()) {
        feature_names_.push_back(feature_name);
    }
    feature_infos_.reserve(model.feature_infos_size());
    for (auto feature_info: model.feature_infos()) {
        feature_infos_.push_back(feature_info);
    }
    loaded_objective_.reset(ObjectiveFunction::CreateObjectiveFunction(model.objective()));
    objective_function_ = loaded_objective_.get();
    
    for (auto tree: model.trees()) {
        models_.emplace_back(new Tree(tree));
    }
    Log::Info("Finished loading %d models", models_.size());
    num_iteration_for_pred_ = static_cast<int>(models_.size()) / num_tree_per_iteration_;
    num_init_iteration_ = num_iteration_for_pred_;
    iter_ = 0;
    return true;
}

bool GBDT::LoadModelFromProto(const char* filename) {
    LightGBM::Model model;
    std::filebuf fb;
    if (fb.open(filename, std::ios::in | std::ios::binary))
    {
        std::istream is(&fb);
        if (!model.ParseFromIstream(&is)) {
            Log::Fatal("Cannot parse model from binary file.");
        }
        fb.close();
    } else {
        Log::Fatal("Cannot open file: %s.", filename);
    }

    return LoadModelFromProto(model);
}

bool GBDT::LoadModelFromPythonProto(const char* filename, std::string* pandas_category) {
    LightGBM::PythonModel python_model;
    std::filebuf fb;
    if (fb.open(filename, std::ios::in | std::ios::binary))
    {
        std::istream is(&fb);
        if (!python_model.ParseFromIstream(&is)) {
            Log::Fatal("Cannot parse python model from binary file.");
        }
        fb.close();
    } else {
        Log::Fatal("Cannot open file: %s.", filename);
    }

    *pandas_category = python_model.pandas_category();

    return LoadModelFromProto(python_model.model());
}

void Tree::ToProto(LightGBM::Model_Tree& model_tree) const {

    model_tree.set_num_leaves(num_leaves_);
    model_tree.set_num_cat(num_cat_);
    for (int i = 0; i < num_leaves_ - 1; ++i) {
        model_tree.add_split_feature(split_feature_[i]);
        model_tree.add_split_gain(split_gain_[i]);
        model_tree.add_threshold(threshold_[i]);
        model_tree.add_decision_type(decision_type_[i]);
        model_tree.add_left_child(left_child_[i]);
        model_tree.add_right_child(right_child_[i]);
        model_tree.add_internal_value(internal_value_[i]);
        model_tree.add_internal_count(internal_count_[i]);
    }

    for (int i = 0; i < num_leaves_; ++i) {
        model_tree.add_leaf_value(leaf_value_[i]);
        model_tree.add_leaf_count(leaf_count_[i]);
    }

    if (num_cat_ > 0) {
        for (int i = 0; i < num_cat_ + 1; ++i) {
            model_tree.add_cat_boundaries(cat_boundaries_[i]);
        }
        for (size_t i = 0; i < cat_threshold_.size(); ++i) {
            model_tree.add_cat_threshold(cat_threshold_[i]);
        }
    }
    model_tree.set_shrinkage(shrinkage_);
}

Tree::Tree(const LightGBM::Model_Tree& model_tree) {

    num_leaves_ = model_tree.num_leaves();
    if (num_leaves_ <= 1) { return; }
    num_cat_ = model_tree.num_cat();

    leaf_value_.reserve(model_tree.leaf_value_size());
    for(auto leaf_value: model_tree.leaf_value()) {
        leaf_value_.push_back(leaf_value);
    }

    left_child_.reserve(model_tree.left_child_size());
    for(auto left_child: model_tree.left_child()) {
        left_child_.push_back(left_child);
    }

    right_child_.reserve(model_tree.right_child_size());
    for(auto right_child: model_tree.right_child()) {
        right_child_.push_back(right_child);
    }

    split_feature_.reserve(model_tree.split_feature_size());
    for(auto split_feature: model_tree.split_feature()) {
        split_feature_.push_back(split_feature);
    }
    
    threshold_.reserve(model_tree.threshold_size());
    for(auto threshold: model_tree.threshold()) {
        threshold_.push_back(threshold);
    }

    split_gain_.reserve(model_tree.split_gain_size());
    for(auto split_gain: model_tree.split_gain()) {
        split_gain_.push_back(split_gain);
    }
    
    internal_count_.reserve(model_tree.internal_count_size());
    for(auto internal_count: model_tree.internal_count()) {
        internal_count_.push_back(internal_count);
    }
    
    internal_value_.reserve(model_tree.internal_value_size());
    for(auto internal_value: model_tree.internal_value()) {
        internal_value_.push_back(internal_value);
    }
    
    leaf_count_.reserve(model_tree.leaf_count_size());
    for(auto leaf_count: model_tree.leaf_count()) {
        leaf_count_.push_back(leaf_count);
    }

    decision_type_.reserve(model_tree.decision_type_size());
    for(auto decision_type: model_tree.decision_type()) {
        decision_type_.push_back(decision_type);
    }
    
    if (num_cat_ > 0) {
        cat_boundaries_.reserve(model_tree.cat_boundaries_size());
        for(auto cat_boundaries: model_tree.cat_boundaries()) {
            cat_boundaries_.push_back(cat_boundaries);
        }

        cat_threshold_.reserve(model_tree.cat_threshold_size());
        for(auto cat_threshold: model_tree.cat_threshold()) {
            cat_threshold_.push_back(cat_threshold);
        }
    }

    shrinkage_ = model_tree.shrinkage();
}

}  // namespace LightGBM
