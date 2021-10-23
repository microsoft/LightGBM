#pragma once

#include <NeuralInput.h>
#include <basic_types.h>

class InputExtractor;
class MinimalFeatureMap;

InputExtractor* InputExtractorCreateFromInputStr(const std::string& str);
InputExtractor* InputExtractorCreateFromFreeformV2(const char* freeform);
void InputExtractorDispose(InputExtractor* extractor);
MinimalFeatureMap* InputExtractorGetFeatureMap(InputExtractor* extractor);

bool InputExtractorGetInputName(
        InputExtractor* extractor,
        UInt32 inputIndex,
        char* buffer,
        UInt32 sizeOfBuffer,
        UInt32* length);

UInt32 InputExtractorGetInputCount(InputExtractor* extractor);
const DynamicRank::NeuralInput* InputExtractorGetInput(InputExtractor* extractor, UInt32 index);

bool InputExtractorGetSectionContent(
        InputExtractor* extractor, 
        char* sectionName, 
        char* sectionContentBuffer,
        UInt32 sizeOfBuffer,
        UInt32* sectionContentLength);