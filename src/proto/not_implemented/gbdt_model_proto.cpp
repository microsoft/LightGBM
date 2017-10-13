#include "../../boosting/gbdt.h"

namespace LightGBM {

void GBDT::SaveModelToProto(int, const char*) const {
    Log::Fatal("Please cmake with -DUSE_PROTO=ON to use protobuf.");
}

bool GBDT::LoadModelFromProto(const char*) {
    Log::Fatal("Please cmake with -DUSE_PROTO=ON to use protobuf.");
    return false;
}

void Tree::ToProto(LightGBM::Model_Tree&) const {
    Log::Fatal("Please cmake with -DUSE_PROTO=ON to use protobuf.");
}

Tree::Tree(const LightGBM::Model_Tree&) {
    Log::Fatal("Please cmake with -DUSE_PROTO=ON to use protobuf.");
}

}  // namespace LightGBM
