#include "Executable.h"
#include "FreeForm2Executable.h"

#include "Compiler.h"
#include "FreeForm2Assert.h"
#include "FreeForm2Compiler.h"
#include "FreeForm2CompilerFactory.h"
#include "FreeForm2Result.h"
#include "LlvmCompiler.h"
#include <memory>

FreeForm2::Executable::Executable(std::auto_ptr<ExecutableImpl> p_impl)
    : m_impl(p_impl.release())
{
}


boost::shared_ptr<FreeForm2::Result> 
FreeForm2::Executable::Evaluate(StreamFeatureInput* p_input,
                                const FeatureType p_features[]) const
{
    return m_impl->Evaluate(p_input, p_features);
}


boost::shared_ptr<FreeForm2::Result>
FreeForm2::Executable::Evaluate(const Executable::FeatureType* const* p_features,
                                UInt32 p_currentDocument,
                                UInt32 p_documentCount,
                                Int64* p_cache) const
{
    return m_impl->Evaluate(p_features, p_currentDocument, p_documentCount, p_cache);
}


FreeForm2::Executable::DirectEvalFun 
FreeForm2::Executable::EvaluationFunction() const
{
    return m_impl->EvaluationFunction();
}


FreeForm2::Executable::AggregatedEvalFun
FreeForm2::Executable::AggregatedEvaluationFunction() const
{
    return m_impl->AggregatedEvaluationFunction();
}


const FreeForm2::Type&
FreeForm2::Executable::GetType() const
{
    return m_impl->GetType();
}


size_t
FreeForm2::Executable::GetExternalSize() const
{
    return sizeof(FreeForm2::ExecutableImpl) + m_impl->GetExternalSize();
}


const FreeForm2::ExecutableImpl&
FreeForm2::Executable::GetImplementation() const
{
    return *m_impl;
}


FreeForm2::ExecutableCompilerResults::ExecutableCompilerResults(
    const boost::shared_ptr<Executable>& p_executable)
    : m_executable(p_executable)
{
}


const boost::shared_ptr<FreeForm2::Executable>&
FreeForm2::ExecutableCompilerResults::GetExecutable() const
{
    return m_executable;
}


FreeForm2::ExecutableImpl::~ExecutableImpl()
{
}


std::unique_ptr<FreeForm2::Compiler>
FreeForm2::CompilerFactory::CreateExecutableCompiler(
    unsigned int p_optimizationLevel,
    FreeForm2::CompilerFactory::DestinationFunctionType p_destinationFunctionType)
{
    std::auto_ptr<CompilerImpl> impl;
    impl.reset(new LlvmCompilerImpl(p_optimizationLevel, p_destinationFunctionType));
    return std::unique_ptr<Compiler>(new Compiler(impl));
}

