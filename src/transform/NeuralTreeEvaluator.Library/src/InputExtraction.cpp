#include "InputExtraction.h"
#include "MinimalFeatureMap.h"
#include "LocalFactoryHolder.h"
#include "InputExtractor.h"

#include <NeuralInput.h>
#include <NeuralInputFreeForm2.h>

InputExtractor* InputExtractorCreateFromInputStr(const string& str)
{
    return InputExtractor::CreateFromInputStr(str);
}

InputExtractor* InputExtractorCreateFromFreeformV2(const char* freeform)
{
    return InputExtractor::CreateFromFreeform2(freeform);
}

void InputExtractorDispose(InputExtractor* extractor)
{
    if (extractor != NULL)
    {
        delete extractor;
    }
}

MinimalFeatureMap* InputExtractorGetFeatureMap(InputExtractor* extractor)
{
    if (extractor != NULL)
    {
        return extractor->GetFeatureMap();
    }

    return 0;
}

bool InputExtractorGetInputName(
        InputExtractor* extractor,
        UInt32 inputIndex,
        char* buffer,
        UInt32 sizeOfBuffer,
        UInt32* length)
{
    memset(buffer, 0x00, sizeOfBuffer);
    if(extractor != NULL)
    {
        std::string content = extractor->GetInputName(inputIndex);
        *length = (UInt32)content.length();
        if (content.length() > sizeOfBuffer) return false;
        memcpy(buffer, content.c_str(), content.length());
        return true;
    }
    return false;
}

UInt32 InputExtractorGetInputCount(InputExtractor* extractor)
{
    if(extractor != NULL)
    {
        return extractor->GetInputCount();
    }
    return 0;
}

const DynamicRank::NeuralInput* InputExtractorGetInput(InputExtractor* extractor, UInt32 index)
{
    if(extractor != NULL)
    {
        return extractor->GetInput(index);
    }
    return NULL;
}
