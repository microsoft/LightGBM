#include "Program.h"
#include "FreeForm2Program.h"

#include "Allocation.h"
#include "AllocationVisitor.h"
#include <boost/tuple/tuple.hpp>

#include "FeatureSpec.h"
#include "FreeForm2Assert.h"
#include <IFeatureMap.h>
#include "ObjectResolutionVisitor.h"
#include "OperandPromotionVisitor.h"
#include "ProcessFeaturesUsed.h"
#include "SExpressionParse.h"
#include <sstream>
#include "TypeCheckingVisitor.h"
#include "TypeImpl.h"

using namespace FreeForm2;

namespace
{
    template <FreeForm2::Program::Syntax p_syntax>
    boost::tuples::tuple<const Expression*, 
                         boost::shared_ptr<ExpressionOwner>,
                         boost::shared_ptr<TypeManager>>
    ParseInternal(SIZED_STRING p_input,
                  DynamicRank::IFeatureMap& p_map,
                  bool p_mustProduceFloat,
                  const ExternalDataManager* p_externalData,
                  std::ostream* p_debugOutput);


    template <>
    boost::tuples::tuple<const Expression*, 
                         boost::shared_ptr<ExpressionOwner>,
                         boost::shared_ptr<TypeManager>>
    ParseInternal<FreeForm2::Program::sexpression>(SIZED_STRING p_input,
                                                   DynamicRank::IFeatureMap& p_map,
                                                   bool p_mustProduceFloat,
                                                   const ExternalDataManager* p_externalData,
                                                   std::ostream* p_debugOutput)
    {
        return SExpressionParse::Parse(p_input, p_map, p_mustProduceFloat, false);
    }


    template <>
    boost::tuples::tuple<const Expression*, 
                         boost::shared_ptr<ExpressionOwner>,
                         boost::shared_ptr<TypeManager>>
    ParseInternal<FreeForm2::Program::aggregatedSExpression>(SIZED_STRING p_input,
                                                             DynamicRank::IFeatureMap& p_map,
                                                             bool p_mustProduceFloat,
                                                             const ExternalDataManager* p_externalData,
                                                             std::ostream* p_debugOutput)
    {
        return SExpressionParse::Parse(p_input, p_map, p_mustProduceFloat, true);
    }

    std::string 
    ConstructParseErrorMessage(
        const char* p_message,
        unsigned int p_line,
        unsigned int p_lineChar)
    {
        std::ostringstream err;
        err << "Parse error: " << p_message
            << " at line " << p_line
            << ", char " << p_lineChar;
        return err.str();
    }
}


FreeForm2::SourceLocation::SourceLocation()
    : m_lineNo(0),
      m_lineOffset(0)
{
}


FreeForm2::SourceLocation::SourceLocation(unsigned int p_lineNo, unsigned int p_lineOffset)
    : m_lineNo(p_lineNo),
      m_lineOffset(p_lineOffset)
{
}


FreeForm2::ParseError::ParseError(
    const std::string& p_message,
    const SourceLocation& p_location)
    : runtime_error(ConstructParseErrorMessage(p_message.c_str(), p_location.m_lineNo, p_location.m_lineOffset)),
      m_message(p_message),
      m_sourceLocation(p_location)
{
}


FreeForm2::ParseError::ParseError(
    const std::exception& p_inner,
    const SourceLocation& p_location)
    : runtime_error(ConstructParseErrorMessage(p_inner.what(), p_location.m_lineNo, p_location.m_lineOffset)),
      m_message(p_inner.what()),
      m_sourceLocation(p_location)
{
}


const std::string&
FreeForm2::ParseError::GetMessage() const
{
    return m_message;
}


const FreeForm2::SourceLocation&
FreeForm2::ParseError::GetSourceLocation() const
{
    return m_sourceLocation;
}


FreeForm2::ProgramImpl::ProgramImpl(const Expression& p_exp, 
                                    boost::shared_ptr<ExpressionOwner> p_owner,
                                    boost::shared_ptr<TypeManager> p_typeManager,
                                    DynamicRank::IFeatureMap& p_map)
    : m_typeImpl(p_exp.GetType()), 
      m_type(m_typeImpl), 
      m_exp(&p_exp), 
      m_owner(p_owner), 
      m_typeManager(p_typeManager),
      m_map(p_map),
      m_allocationVisitor(p_exp)
{
}


const Type&
FreeForm2::ProgramImpl::GetType() const
{
    return m_type;
}


void 
FreeForm2::ProgramImpl::ProcessFeaturesUsed(DynamicRank::INeuralNetFeatures& p_features) const
{
    ProcessFeaturesUsedVisitor visitor(p_features);
    m_exp->Accept(visitor);
}


const Expression& 
FreeForm2::ProgramImpl::GetExpression() const
{
    return *m_exp;
}


DynamicRank::IFeatureMap&
FreeForm2::ProgramImpl::GetFeatureMap() const
{
    return m_map;
}


const std::vector<boost::shared_ptr<Allocation>>&
FreeForm2::ProgramImpl::GetAllocations() const
{
    return m_allocationVisitor.GetAllocations();
}


template <FreeForm2::Program::Syntax p_syntax>
boost::shared_ptr<FreeForm2::Program> 
FreeForm2::Program::Parse(SIZED_STRING p_input, 
                          DynamicRank::IFeatureMap& p_map,
                          bool p_mustProduceFloat,
                          const ExternalDataManager* p_externalData,
                          std::ostream* p_debugOutput)
{
    using boost::tuples::get;
    boost::tuples::tuple<const Expression*, 
                         boost::shared_ptr<ExpressionOwner>,
                         boost::shared_ptr<TypeManager>> ret
        = ::ParseInternal<p_syntax>(p_input, p_map, p_mustProduceFloat, p_externalData, p_debugOutput);

    const Expression* syntaxTree = get<0>(ret);
    FF2_ASSERT(syntaxTree != NULL);
    boost::shared_ptr<ExpressionOwner> owner;
    boost::shared_ptr<TypeManager> typeManager;

    {
        // Resolve unknown object types.
        ObjectResolutionVisitor resolve;
        get<0>(ret)->Accept(resolve);
        syntaxTree = resolve.GetSyntaxTree();

        // Ensure that all type information has been filled out.
        TypeCheckingVisitor typeCheck;
        syntaxTree->Accept(typeCheck);
        
        // Infer all missing type information.
        OperandPromotionVisitor promotion;
        syntaxTree->Accept(promotion);

        syntaxTree = promotion.GetSyntaxTree();
        owner = promotion.GetExpressionOwner();
        typeManager = promotion.GetTypeManager();
    }

    std::auto_ptr<ProgramImpl> ptr(new ProgramImpl(*syntaxTree, owner, typeManager, p_map));
    return boost::shared_ptr<Program>(new Program(ptr));
}


FreeForm2::Program::Program(std::auto_ptr<ProgramImpl> p_impl)
    : m_impl(p_impl.release())
{
}


const FreeForm2::Type&
FreeForm2::Program::GetType() const
{
    return m_impl->GetType();
}


const FreeForm2::Expression&
FreeForm2::Program::GetExpression() const
{
    return m_impl->GetExpression();
}


void 
FreeForm2::Program::ProcessFeaturesUsed(DynamicRank::INeuralNetFeatures& p_features) const
{
    return m_impl->ProcessFeaturesUsed(p_features);
}


const std::vector<boost::shared_ptr<FreeForm2::Allocation>>&
FreeForm2::Program::GetAllocations() const
{
    return m_impl->GetAllocations();
}

FreeForm2::ProgramImpl& 
FreeForm2::Program::GetImplementation()
{
    return *m_impl;
}


const FreeForm2::ProgramImpl& 
FreeForm2::Program::GetImplementation() const
{
    return *m_impl;
}
