#ifndef FREEFORMLIBTESTSET_H
#define FREEFORMLIBTESTSET_H

#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <string>
#include <cstdio>
#include <cstring>

#include "NeuralInputFreeForm2.h"
#include "FreeForm2CompilerFactory.h"
#include "FreeForm2Compiler.h"
#include "SimpleFeatureMap.h"
#include <LightGBM/utils/log.h>

using namespace LightGBM;

class FreeFormLibTestSet
{
    static void AssertIsNotNull(const void * ptr){
        if(ptr == nullptr)
        {
            Error("AssertIsNotNull Failed!");
        }
        Log::Info("[PASS] AssertIsNotNull\n");
    }

    static void AssertAreEqual(double truth, double value, double error){
        double delta = truth - value;
        delta = delta >= 0? delta: -delta;
        if(delta > error)
        {
            Error("AssertAreEqual(double) Failed!");
        }
        Log::Info("[PASS] AssertAreEqual(double)\n");
    }

    static void AssertAreEqual(int truth, int value){
        if(truth != value)
        {
            Error("AssertAreEqual(int) Failed!");
        }
        Log::Info("[PASS] AssertAreEqual(int)\n");
    }

    static void AssertSzAreEqual(const char * cptr0, const char * cptr1){
        unsigned int offset = 0;
        while(*(cptr0 + offset) && *(cptr1 + offset))
            ++offset;
        if(*(cptr0 + offset) != *(cptr1 + offset))
        {
            Error("AssertSzAreEqual Failed!");
        }
        Log::Info("[PASS] AssertSzAreEqual\n");
    }

    static void Error(const char * s){
        Log::Fatal("%s", s);
        throw s;
    }
    // Test base:
    static void 
    TestInput(
        DynamicRank::Config& p_config, 
        double p_result,
        const char* p_serial,
        const char* p_section)
    {
        SimpleFeatureMap map;
        FreeForm2::CompiledNeuralInputLoader<FreeForm2::NeuralInputFreeForm2> loader("FreeForm2");
        std::auto_ptr<DynamicRank::NeuralInput> input(loader(p_config, p_section, map));
        AssertIsNotNull(input.get());
        std::unique_ptr<FreeForm2::Compiler> compiler(FreeForm2::CompilerFactory::CreateExecutableCompiler(
            FreeForm2::Compiler::c_defaultOptimizationLevel,
            FreeForm2::CompilerFactory::SingleDocumentEvaluation));
        loader.Compile(*compiler);
        AssertAreEqual(p_result, input->Evaluate(NULL), 0.0001);
    
        // Unfortunately, we have to write to disk to test this input,
        // because i'm not sure how to get a FILE* backed by memory.  This
        // would be much easier if inputs wrote to iostreams.
        const char* filename = "TestNeuralInputLoadSave.tmp";
        FILE* f = fopen(filename, "w");
        AssertIsNotNull(f);
        input->Save(f, 0, map);
        fclose(f);
    
        std::ifstream fs(filename);
        std::stringstream buffer;
        buffer << fs.rdbuf();
        std::string content = buffer.str();
        AssertSzAreEqual(p_serial, content.c_str());
        Log::Info("[PASS] TestInput\n");
    }

public:
    // Test functions:
    static void TestParser(const char * freeform) 
    {
        boost::shared_ptr<SimpleFeatureMap> featureMap(new SimpleFeatureMap());
        boost::shared_ptr<FreeForm2::NeuralInputFreeForm2> input = boost::shared_ptr<FreeForm2::NeuralInputFreeForm2>(
            new FreeForm2::NeuralInputFreeForm2(std::string(freeform), "freeform2", *featureMap));
        AssertIsNotNull(input.get());


        std::unique_ptr<FreeForm2::Compiler> comp(FreeForm2::CompilerFactory::CreateExecutableCompiler(2));
        input->Compile(comp.get());
    }

    static void PRINT_MEM(void* ptr, size_t len)
    {
        printf("\nMEM %02X - %02X:\n", ptr, ptr + len - 1);
        for(unsigned int i = 0; i < len; ++i)
        {
            printf("%02X ", *(unsigned char *)(ptr + i));
        }
        printf("\n");
    }

    static void TestNeuralInputLoadSave() 
    {
        std::string path("");
        std::map<std::string, std::map<std::string, std::string>> config_map;
        // PRINT_MEM((void *) &config_map, sizeof(config_map));
        const char* section = "Input:0";
        const char* transform = "FreeForm2";
        config_map[section]["Transform"] = transform;
        config_map[section]["Line1"] = "(+ 1 2";
        config_map[section]["Line2"] = "1 2";
        config_map[section]["Line3"] = ")";
        // PRINT_MEM((void *) &config_map, sizeof(config_map));
        void* config_ptr = malloc(sizeof(DynamicRank::Config));
        memcpy(config_ptr, &path, sizeof(path));
        memcpy(config_ptr + sizeof(path), &config_map, sizeof(config_map));
        DynamicRank::Config config = *(DynamicRank::Config*) config_ptr;
        TestInput(config, 6.0, "\n[Input:0]\nTransform=FreeForm2\nLine1=(+ 1 2\nLine2=1 2\nLine3=)\n", section);
    }
};
#endif
