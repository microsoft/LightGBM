#include "TypeManager.h"

#include "ArrayType.h"
#include <boost/array.hpp>
#include <boost/foreach.hpp>
#include <boost/make_shared.hpp>
#include <boost/ref.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>
#include "FreeForm2Assert.h"
#include "FreeForm2ExternalData.h"
#include <iostream>
//#include "MetaStreams.h"
#include "ObjectType.h"
//#include "RankerFeatures.h"
#include <sstream>
#include "StateMachineType.h"
#include "StructType.h"
#include "TypeImpl.h"

#include "MigratedApi.h"

using namespace FreeForm2;

namespace
{
    template <typename T>
    void ByteArrayDeleter(T* p_delete)
    {
        // Cast to char*, as this structure was allocated as char[].
        p_delete->~T();
        char* data = reinterpret_cast<char*>(p_delete);
        delete[] data;
    }

    // A class that manages type information. For now it only contains a
    // std::string-to-StructInfo mapping.
    class NamedTypeManager : public TypeManager
    {
    public:
        // Create an empty type manager with the specified parent.
        NamedTypeManager(const TypeManager& p_parent);

        // Constructor to create the global TypeManager.
        NamedTypeManager();

        // Gets the type information for the provided name. Returns NULL if the
        // type is not found.
        virtual const TypeImpl* GetTypeInfo(const std::string& p_name) const override;

        // Create a variable sized array type owned by this TypeManager.
        virtual const ArrayType&
        GetArrayType(const TypeImpl& p_child, 
                     bool p_isConst,
                     unsigned int p_dimensions, 
                     unsigned int p_maxElements) override;

        // Create a fixed-sized array type owned by this TypeManager.
        virtual const ArrayType&
        GetArrayType(const TypeImpl& p_child,
                     bool p_isConst,
                     unsigned int p_dimensions,
                     const unsigned int p_elementCounts[],
                     unsigned int p_maxElements) override;

        // Returns an array type owned by this TypeManager which has the same
        // properties as another ArrayType.
        virtual
        const ArrayType&
        GetArrayType(const ArrayType& p_type) override;

        // Create a struct type owned by this TypeManager. If the specified 
        // name exists, the existing structure is returned iff it is exactly
        // the same as the parameters; otherwise, an the function asserts.
        virtual const StructType&
        GetStructType(const std::string& p_name,
                      const std::string& p_externName,
                      const std::vector<StructType::MemberInfo>& p_members,
                      bool p_isConst) override;

        // Returns a struct type owned by this TypeManager which has the same
        // properties as another StructType.
        virtual
        const StructType&
        GetStructType(const StructType& p_type) override;

        // Get an object type owned by this TypeManager. The TypeManager is not
        // required to allow multiple non-unique names exist in the context of
        // its owned types.
        virtual
        const ObjectType&
        GetObjectType(const std::string& p_name,
                      const std::string& p_externName,
                      const std::vector<ObjectType::ObjectMember>& p_members,
                      bool p_isConst) override;

        // Returns an object type owned by this TypeManager which has the same
        // properties as another StructType.
        virtual
        const ObjectType&
        GetObjectType(const ObjectType& p_type) override;

        // Returns a state machine type owned by this TypeManager with the same
        // semantics as GetStructType.
        virtual
        const StateMachineType&
        GetStateMachineType(const std::string& p_name,
                            const CompoundType::Member* p_members,
                            size_t p_numMembers,
                            boost::weak_ptr<const StateMachineExpression> p_expr) override;

        // Returns a state machine type owned by this TypeManager which has the
        // same properties as another state machine type.
        virtual
        const StateMachineType&
        GetStateMachineType(const StateMachineType& p_type) override;

        // Get a function type owned by this TypeManager. The TypeManager will just store one
        // function type per signature.
        virtual
        const FunctionType&
        GetFunctionType(const TypeImpl& p_returnType,
                        const TypeImpl* const* p_parameters,
                        size_t p_numParameters) override;

        // Returns a function type owned by this TypeManager which has the same
        // properties as another FunctionType.
        virtual
        const FunctionType&
        GetFunctionType(const FunctionType& p_type) override;

    private:
        template <typename T>
        const T& RegisterType(boost::shared_ptr<const T> p_type, const std::string& p_name)
        {
            BOOST_STATIC_ASSERT((boost::is_base_of<TypeImpl, T>::value == true));
            boost::shared_ptr<const TypeImpl> ptr = boost::static_pointer_cast<const TypeImpl>(p_type);
            m_typeMap.insert(std::make_pair(p_name, ptr));
            return *p_type;
        }


        template <typename T>
        const T& RegisterType(boost::shared_ptr<const T> p_type)
        {
            BOOST_STATIC_ASSERT((boost::is_base_of<TypeImpl, T>::value == true));
            return RegisterType(p_type, p_type->GetName());
        }

        // A mapping from names to types.
        std::map<std::string, boost::shared_ptr<const TypeImpl>> m_typeMap;
    };
}


TypeManager::TypeManager(const TypeManager* p_parent) 
    : m_parent(p_parent)
{
}


TypeManager::~TypeManager()
{
}


NamedTypeManager::NamedTypeManager() : TypeManager(NULL)
{
    // Tuples of interest.
    boost::array<StructType::MemberInfo, 3> tuplesOfInterestArray = {{
        StructType::MemberInfo("WordStart", TypeImpl::GetUInt32Instance(true), "iWordStart", 0, 4),
        StructType::MemberInfo("WordEnd", TypeImpl::GetUInt32Instance(true), "iWordEnd", 4, 4),
        StructType::MemberInfo("Weight", TypeImpl::GetUInt32Instance(true), "iWeight", 8, 4)
    }};

    std::vector<StructType::MemberInfo> tuplesOfInterest(tuplesOfInterestArray.begin(),
                                                         tuplesOfInterestArray.end());

    NamedTypeManager::GetStructType("TupleOfInterest","FreeForm2::TupleOfInterest", tuplesOfInterest, true);

    // AllDoublesDecodeIndexes
    boost::array<StructType::MemberInfo, 2> allDoublesDecodeIndexesArray = {{
        StructType::MemberInfo("FirstIndex", TypeImpl::GetUInt32Instance(true), "m_firstIndex", 0, 4),
        StructType::MemberInfo("SecondIndex", TypeImpl::GetUInt32Instance(true), "m_secondIndex", 4, 4)
    }};

    std::vector<StructType::MemberInfo> allDoublesDecodeIndexes(allDoublesDecodeIndexesArray.begin(),
                                                                allDoublesDecodeIndexesArray.end());

    NamedTypeManager::GetStructType("AllDoublesDecodeIndexes","FreeForm2::RuntimeLibrary::AllDoublesDecodeIndexes", allDoublesDecodeIndexes, true);

    // AllTriplesDecodeIndexes
    boost::array<StructType::MemberInfo, 3> allTriplesDecodeIndexesArray = {{
        StructType::MemberInfo("FirstIndex", TypeImpl::GetUInt32Instance(true), "m_firstIndex", 0, 4),
        StructType::MemberInfo("SecondIndex", TypeImpl::GetUInt32Instance(true), "m_secondIndex", 4, 4),
        StructType::MemberInfo("ThirdIndex", TypeImpl::GetUInt32Instance(true), "m_thirdIndex", 8, 4)
    }};

    std::vector<StructType::MemberInfo> allTriplesDecodeIndexes(allTriplesDecodeIndexesArray.begin(),
                                                                allTriplesDecodeIndexesArray.end());

    NamedTypeManager::GetStructType("AllTriplesDecodeIndexes","FreeForm2::RuntimeLibrary::AllTriplesDecodeIndexes", allTriplesDecodeIndexes, true);

    // NumberOfTuples Object.
    std::vector<ObjectType::ObjectMember> numberOfTuplesCommonMembers;

    const FunctionType& numberOfTuplesInitialize
        = NamedTypeManager::GetFunctionType(TypeImpl::GetVoidInstance(), NULL, 0);
    numberOfTuplesCommonMembers.push_back(ObjectType::ObjectMember("Initialize", numberOfTuplesInitialize));

    const FunctionType& numberOfTuplesReset
        = NamedTypeManager::GetFunctionType(TypeImpl::GetVoidInstance(), NULL, 0);
    numberOfTuplesCommonMembers.push_back(ObjectType::ObjectMember("Reset", numberOfTuplesReset));

    const FunctionType& numberOfTuplesInflateMatrix
        = NamedTypeManager::GetFunctionType(TypeImpl::GetVoidInstance(), NULL, 0);
    numberOfTuplesCommonMembers.push_back(ObjectType::ObjectMember("InflateMatrix", numberOfTuplesInflateMatrix));

    const TypeImpl* numberOfTuplesAddWordArray[] = { &TypeImpl::GetWordInstance(true) };
    const FunctionType& numberOfTuplesAddWord
        = NamedTypeManager::GetFunctionType(TypeImpl::GetVoidInstance(), 
                                            numberOfTuplesAddWordArray, 
                                            countof(numberOfTuplesAddWordArray));
    numberOfTuplesCommonMembers.push_back(ObjectType::ObjectMember("AddWord", numberOfTuplesAddWord));

    const TypeImpl* numberOfTuplesValueArray[] = { &TypeImpl::GetUInt32Instance(true),
                                                   &TypeImpl::GetUInt32Instance(true) };
    const FunctionType& numberOfTuplesValue
        = NamedTypeManager::GetFunctionType(TypeImpl::GetInt32Instance(true), 
                                            numberOfTuplesValueArray, 
                                            countof(numberOfTuplesValueArray));
    const FunctionType& offsetOfTuplesValue
        = NamedTypeManager::GetFunctionType(TypeImpl::GetUInt32Instance(true), 
                                            numberOfTuplesValueArray, 
                                            countof(numberOfTuplesValueArray));
    numberOfTuplesCommonMembers.push_back(ObjectType::ObjectMember("numberOfTuples", numberOfTuplesValue));
    numberOfTuplesCommonMembers.push_back(ObjectType::ObjectMember("firstOccurrenceOffsetOfTuples", offsetOfTuplesValue));
    numberOfTuplesCommonMembers.push_back(ObjectType::ObjectMember("lastOccurrenceOffsetOfTuples", offsetOfTuplesValue));
    numberOfTuplesCommonMembers.push_back(ObjectType::ObjectMember("incrementValue", 
                                                                   TypeImpl::GetInt32Instance(false),
                                                                   "m_iIncrementingValue"));

    NamedTypeManager::GetObjectType("NumberOfTuplesCommon",
                                    "CNumberOfTuples",
                                    numberOfTuplesCommonMembers,
                                    false);

    NamedTypeManager::GetObjectType("NumberOfTuplesCommonNoDuplicate",
                                    "CNumberOfTuples",
                                    numberOfTuplesCommonMembers,
                                    false);

    // NumberOfTuplesInTriples Object.
    std::vector<ObjectType::ObjectMember> numberOfTuplesInTriplesCommonMembers;

    const TypeImpl* numberOfTuplesInTriplesInitializeArray[] = { &TypeImpl::GetUInt32Instance(true),
                                                                 &TypeImpl::GetUInt32Instance(true),
                                                                 &TypeImpl::GetUInt32Instance(true) };
    const FunctionType& numberOfTuplesInTriplesInitialize
        = NamedTypeManager::GetFunctionType(TypeImpl::GetVoidInstance(), 
                                            numberOfTuplesInTriplesInitializeArray, 
                                            countof(numberOfTuplesInTriplesInitializeArray));
    numberOfTuplesInTriplesCommonMembers.push_back(ObjectType::ObjectMember("Initialize", numberOfTuplesInTriplesInitialize));

    const TypeImpl* numberOfTuplesInTriplesStartPhraseArray[] = { &TypeImpl::GetUInt32Instance(true) };
    const FunctionType& numberOfTuplesInTriplesStartPhrase
        = NamedTypeManager::GetFunctionType(TypeImpl::GetVoidInstance(), 
                                            numberOfTuplesInTriplesStartPhraseArray, 
                                            countof(numberOfTuplesInTriplesStartPhraseArray));
    numberOfTuplesInTriplesCommonMembers.push_back(ObjectType::ObjectMember("StartPhrase", numberOfTuplesInTriplesStartPhrase));

    const TypeImpl* numberOfTuplesInTriplesAddWordArray[] = { &TypeImpl::GetWordInstance(true) };
    const FunctionType& numberOfTuplesInTriplesAddWord
        = NamedTypeManager::GetFunctionType(TypeImpl::GetVoidInstance(), 
                                            numberOfTuplesInTriplesAddWordArray, 
                                            countof(numberOfTuplesInTriplesAddWordArray));
    numberOfTuplesInTriplesCommonMembers.push_back(ObjectType::ObjectMember("AddWord", numberOfTuplesInTriplesAddWord));

    const FunctionType& numberOfTuplesInTriplesEndPage
        = NamedTypeManager::GetFunctionType(TypeImpl::GetVoidInstance(), NULL, 0);
    numberOfTuplesInTriplesCommonMembers.push_back(ObjectType::ObjectMember("EndPage", numberOfTuplesInTriplesEndPage));

    const TypeImpl* numberOfTuplesInTriplesValueArray[] = { &TypeImpl::GetUInt32Instance(true) };
    const FunctionType& numberOfTuplesInTriplesValue
        = NamedTypeManager::GetFunctionType(TypeImpl::GetUInt32Instance(true), 
                                            numberOfTuplesInTriplesValueArray, 
                                            countof(numberOfTuplesInTriplesValueArray));
    numberOfTuplesInTriplesCommonMembers.push_back(ObjectType::ObjectMember("numberOfTuples", numberOfTuplesInTriplesValue));
    numberOfTuplesInTriplesCommonMembers.push_back(ObjectType::ObjectMember("numberOfTuplesInOrder", numberOfTuplesInTriplesValue));
    
    NamedTypeManager::GetObjectType("NumberOfTuplesInTriplesCommon",
                                    "CNumberOfTuplesInTriples",
                                    numberOfTuplesInTriplesCommonMembers,
                                    false);

     // WeightingCalculator Object
    std::vector<ObjectType::ObjectMember> weightingCalculatorMembers;

    const FunctionType& weightingCalculatorReset
        = NamedTypeManager::GetFunctionType(TypeImpl::GetVoidInstance(), NULL, 0);
    weightingCalculatorMembers.push_back(ObjectType::ObjectMember("Reset", weightingCalculatorReset));

    {
        const TypeImpl* weightingCalculatorAddWordArray[] = { &TypeImpl::GetWordInstance(true) };
        const FunctionType& weightingCalculatorAddWord
            = NamedTypeManager::GetFunctionType(TypeImpl::GetVoidInstance(), 
                                                weightingCalculatorAddWordArray, 
                                                countof(weightingCalculatorAddWordArray));
        weightingCalculatorMembers.push_back(ObjectType::ObjectMember("AddWord", weightingCalculatorAddWord));
    }

    {
        const TypeImpl* weightingCalculatorApplyWeightingArray[] = { &TypeImpl::GetIntInstance(true) };
        const FunctionType& weightingCalculatorApplyWeighting
            = NamedTypeManager::GetFunctionType(TypeImpl::GetIntInstance(true), 
                                                weightingCalculatorApplyWeightingArray, 
                                                countof(weightingCalculatorApplyWeightingArray));
        weightingCalculatorMembers.push_back(ObjectType::ObjectMember("ApplyWeightingRound", 
                                                                      weightingCalculatorApplyWeighting, 
                                                                      "ApplyWeighting"));
    }

    {
        const TypeImpl* weightingCalculatorApplyWeightingArray[] = { &TypeImpl::GetFloatInstance(true) };
        const FunctionType& weightingCalculatorApplyWeighting
            = NamedTypeManager::GetFunctionType(TypeImpl::GetFloatInstance(true), 
                                                weightingCalculatorApplyWeightingArray, 
                                                countof(weightingCalculatorApplyWeightingArray));
        weightingCalculatorMembers.push_back(
            ObjectType::ObjectMember("ApplyWeighting", weightingCalculatorApplyWeighting));
    }

    NamedTypeManager::GetObjectType("AlterationAndTermWeightingCalculator",
                                    "BarramundiWeightingCalculator",
                                    weightingCalculatorMembers,
                                    false);

    NamedTypeManager::GetObjectType("AlterationWeightingCalculator",
                                    "BarramundiWeightingCalculator",
                                    weightingCalculatorMembers,
                                    false);

    // TrueNearDoublesQueue Object
    std::vector<ObjectType::ObjectMember> trueNearDoubleQueueMembers;

    const FunctionType& trueNearDoublesQueueReset = NamedTypeManager::GetFunctionType(TypeImpl::GetVoidInstance(), NULL, 0);
    trueNearDoubleQueueMembers.push_back(ObjectType::ObjectMember("Reset", trueNearDoublesQueueReset));

    {
        const TypeImpl* trueNearDoublesQueueReceiveWordArray[] = { &TypeImpl::GetWordInstance(true), 
                                                                   &TypeImpl::GetUInt32Instance(false), 
                                                                   &TypeImpl::GetUInt32Instance(false) };
        const FunctionType& trueNearDoublesQueueReceiveWord = NamedTypeManager::GetFunctionType(TypeImpl::GetVoidInstance(), 
                                                                                               trueNearDoublesQueueReceiveWordArray,
                                                                                               countof(trueNearDoublesQueueReceiveWordArray));
        trueNearDoubleQueueMembers.push_back(ObjectType::ObjectMember("ReceiveWord", trueNearDoublesQueueReceiveWord));
    }

    NamedTypeManager::GetObjectType("TrueNearDoubleQueue", "TrueNearDoubleQueue", trueNearDoubleQueueMembers, false);

    // BoundedQueue Object
    std::vector<ObjectType::ObjectMember> boundedQueueMembers;

    {
        const FunctionType& boundedQueueClear = NamedTypeManager::GetFunctionType(TypeImpl::GetVoidInstance(), NULL, 0);
        boundedQueueMembers.push_back(ObjectType::ObjectMember("Clear", boundedQueueClear));
    }

    {
        const FunctionType& boundedQueueEmpty = NamedTypeManager::GetFunctionType(TypeImpl::GetBoolInstance(true), NULL, 0);
        boundedQueueMembers.push_back(ObjectType::ObjectMember("Empty", boundedQueueEmpty));
    }

    {
        const FunctionType& boundedQueueFull = NamedTypeManager::GetFunctionType(TypeImpl::GetBoolInstance(true), NULL, 0);
        boundedQueueMembers.push_back(ObjectType::ObjectMember("Full", boundedQueueFull));
    }

    {
        const TypeImpl* boundedQueueGetParamArray[] = { &TypeImpl::GetUInt32Instance(true) };
        const FunctionType& boundedQueueGet = NamedTypeManager::GetFunctionType(TypeImpl::GetIntInstance(true), boundedQueueGetParamArray, 1);
        boundedQueueMembers.push_back(ObjectType::ObjectMember("Get", boundedQueueGet));
    }

    {
        const FunctionType& boundedQueuePop = NamedTypeManager::GetFunctionType(TypeImpl::GetIntInstance(true), NULL, 0);
        boundedQueueMembers.push_back(ObjectType::ObjectMember("Pop", boundedQueuePop));
    }

    {
        const FunctionType& boundedQueuePush = NamedTypeManager::GetFunctionType(TypeImpl::GetIntInstance(true), NULL, 0);
        boundedQueueMembers.push_back(ObjectType::ObjectMember("Push", boundedQueuePush));
    }

    {
        const FunctionType& boundedQueueSize = NamedTypeManager::GetFunctionType(TypeImpl::GetUInt32Instance(true), NULL, 0);
        boundedQueueMembers.push_back(ObjectType::ObjectMember("Size", boundedQueueSize));
    }

    // Visage bounded queues can hold up to 41 indices.
    NamedTypeManager::GetObjectType("BoundedQueue", "FreeForm2::RuntimeLibrary::BoundedQueue<41>", boundedQueueMembers, false);
}


NamedTypeManager::NamedTypeManager(const TypeManager& p_parent) : TypeManager(&p_parent)
{
}


const TypeImpl*
NamedTypeManager::GetTypeInfo(const std::string& p_name) const
{
    std::map<std::string, boost::shared_ptr<const TypeImpl>>::const_iterator info
        = m_typeMap.find(p_name);

    if (info != m_typeMap.end())
    {
        return info->second.get();
    }
    else if (GetParent() != NULL)
    {
        return GetParent()->GetTypeInfo(p_name);
    }
    else
    {
        return NULL;
    }
}


boost::shared_ptr<const ArrayType>
TypeManager::CreateArrayType(const TypeImpl& p_child,
                             bool p_isConst,
                             unsigned int p_dimensions,
                             const unsigned int p_elementCounts[],
                             unsigned int p_maxElements)
{
    const size_t structSize = sizeof(ArrayType) + sizeof(unsigned int) 
        * (p_dimensions > 0 ? p_dimensions - 1 : 0);
    
    char* mem = NULL;
    try 
    {
        mem = new char[structSize];
        return boost::shared_ptr<const ArrayType>(
            new (mem) ArrayType(p_child, p_isConst, p_dimensions, p_elementCounts, p_maxElements, *this), 
            ByteArrayDeleter<ArrayType>);
    }
    catch (...)
    {
        delete[] mem;
        throw;
    }
}


boost::shared_ptr<const ArrayType>
TypeManager::CreateArrayType(const TypeImpl& p_child,
                             bool p_isConst,
                             unsigned int p_dimensions,
                             unsigned int p_maxElements)
{
    return boost::shared_ptr<const ArrayType>(
        new ArrayType(p_child, p_isConst, p_dimensions, p_maxElements, boost::ref(*this)));
}


boost::shared_ptr<const StructType>
    TypeManager::CreateStructType(const std::string& p_name,
    const std::string& p_externName,
    const std::vector<StructType::MemberInfo>& p_members,
    bool p_isConst)
{
    return boost::shared_ptr<const StructType>(
        new StructType(p_name, p_externName, p_members, p_isConst, boost::ref(*this)));
}


boost::shared_ptr<const ObjectType>
TypeManager::CreateObjectType(const std::string& p_name,
                              const std::string& p_externName,
                              const std::vector<ObjectType::ObjectMember>& p_members,
                              bool p_isConst)
{
    return boost::shared_ptr<const ObjectType>(
        new ObjectType(p_name, p_externName, p_members, p_isConst, boost::ref(*this)));
}


boost::shared_ptr<const StateMachineType>
TypeManager::CreateStateMachineType(const std::string& p_name,
                                    const CompoundType::Member* p_members,
                                    size_t p_numMembers,
                                    boost::weak_ptr<const StateMachineExpression> p_expr)
{
    const size_t memSize = sizeof(StateMachineType) 
        + sizeof(CompoundType::Member) * (std::max(p_numMembers, (size_t) 1ULL) - 1);
    char* mem = NULL;

    try
    {
        mem = new char[memSize];
        return boost::shared_ptr<StateMachineType>(new (mem) StateMachineType(*this,
                                                                              p_name, 
                                                                              p_members,
                                                                              p_numMembers,
                                                                              p_expr), 
                                                   ByteArrayDeleter<StateMachineType>);
    }
    catch (...)
    {
        delete[] mem;
        throw;
    }
}


boost::shared_ptr<const FunctionType>
TypeManager::CreateFunctionType(const TypeImpl& p_returnType,
                                const TypeImpl* const* p_params,
                                size_t p_numParams)
{
    const size_t memSize = sizeof(FunctionType) 
        + sizeof(TypeImpl) * (std::max(p_numParams, (size_t) 1ULL) - 1);
    char* mem = NULL;

    try
    {
        mem = new char[memSize];
        return boost::shared_ptr<FunctionType>(new (mem) FunctionType(*this,
                                                                      p_returnType,
                                                                      p_params,
                                                                      p_numParams), 
                                               ByteArrayDeleter<FunctionType>);
    }
    catch (...)
    {
        delete[] mem;
        throw;
    }
}


const TypeImpl&
TypeManager::GetChildType(const TypeImpl& p_type)
{
    const TypeImpl* childType = &p_type;
    switch (childType->Primitive())
    {
        case Type::Struct:
        {
            childType = &GetStructType(static_cast<const StructType&>(*childType));
            break;
        }

        case Type::Array:
        {
            childType = &GetArrayType(static_cast<const ArrayType&>(*childType));
            break;
        }

        case Type::Object:
        {
            childType = &GetObjectType(static_cast<const ObjectType&>(*childType));
            break;
        }

        case Type::StateMachine:
        {
            childType = &GetStateMachineType(static_cast<const StateMachineType&>(*childType));
            break;
        }

        default:
        {
            FF2_ASSERT(childType == &TypeImpl::GetCommonType(childType->Primitive(), childType->IsConst()));
            break;
        }
    }
    return *childType;
}


boost::shared_ptr<const ArrayType>
TypeManager::CopyArrayType(const ArrayType& p_arrayType)
{
    const TypeImpl& childType = GetChildType(p_arrayType.GetChildType());

    if (p_arrayType.IsFixedSize())
    {
        return CreateArrayType(childType, 
                               p_arrayType.IsConst(), 
                               p_arrayType.GetDimensionCount(), 
                               p_arrayType.GetDimensions(), 
                               p_arrayType.GetMaxElements());
    }
    else
    {
        return CreateArrayType(childType, 
                               p_arrayType.IsConst(), 
                               p_arrayType.GetDimensionCount(),
                               p_arrayType.GetMaxElements());
    }
}


boost::shared_ptr<const StructType>
TypeManager::CopyStructType(const StructType& p_structType)
{
    std::vector<StructType::MemberInfo> members(p_structType.GetMembers());

    BOOST_FOREACH(StructType::MemberInfo& info, members)
    {
        FF2_ASSERT(info.m_type != NULL);
        info.m_type = &GetChildType(*info.m_type);
    }

    return CreateStructType(p_structType.GetName(), 
                            p_structType.GetExternName(), 
                            members, 
                            p_structType.IsConst());
}


boost::shared_ptr<const ObjectType>
TypeManager::CopyObjectType(const ObjectType& p_objectType)
{
    std::vector<ObjectType::ObjectMember> members;
    for (std::map<std::string, ObjectType::ObjectMember>::const_iterator memberIterator = p_objectType.m_members.begin();
         memberIterator !=  p_objectType.m_members.end();
         ++memberIterator)
    {
        members.push_back(memberIterator->second);
    }

    return CreateObjectType(p_objectType.GetName(), 
                            p_objectType.GetExternName(), 
                            members, 
                            p_objectType.IsConst());
}


boost::shared_ptr<const StateMachineType>
TypeManager::CopyStateMachineType(const StateMachineType& p_type)
{
    std::vector<CompoundType::Member> members(p_type.BeginMembers(), p_type.EndMembers());

    BOOST_FOREACH(CompoundType::Member& member, members)
    {
        member.m_type = &GetChildType(*member.m_type);
    }

    return CreateStateMachineType(p_type.GetName(), 
                                  &members[0], 
                                  members.size(),
                                  p_type.GetDefinition());
}


boost::shared_ptr<const FunctionType>
TypeManager::CopyFunctionType(const FunctionType& p_type)
{
    std::vector<const TypeImpl*> params;

    for (UInt32 i = 0; i < p_type.GetParameterCount(); i++)
    {
        const TypeImpl& param = *p_type.BeginParameters()[i];
        params.push_back(&GetChildType(param));
    }

    return CreateFunctionType(GetChildType(p_type.GetReturnType()), 
                              params.size() > 0 ? &params[0] : nullptr,
                              params.size());
}


const TypeManager*
TypeManager::GetParent() const
{
    return m_parent;
}

        
const ArrayType&
NamedTypeManager::GetArrayType(const TypeImpl& p_child, 
                               bool p_isConst,
                               unsigned int p_dimensions, 
                               unsigned int p_maxElements)
{
    return GetArrayType(p_child, p_isConst, p_dimensions, NULL, p_maxElements);
}


const ArrayType&
NamedTypeManager::GetArrayType(const TypeImpl& p_child,
                               bool p_isConst,
                               unsigned int p_dimensions,
                               const unsigned int p_elementCounts[],
                               unsigned int p_maxElements)
{
    std::string signature 
        = ArrayType::GetName(p_child, p_isConst, p_dimensions, p_elementCounts, p_maxElements);

    const TypeImpl* type = GetTypeInfo(signature);
    if (type != NULL)
    {
        FF2_ASSERT(type->IsConst() == p_isConst);
        FF2_ASSERT(type->Primitive() == Type::Array);
        const ArrayType& arrayType = static_cast<const ArrayType&>(*type);
        return arrayType;
    }
    else
    {
        if (p_elementCounts != NULL)
        {
            return RegisterType(
                CreateArrayType(p_child, p_isConst, p_dimensions, p_elementCounts, p_maxElements));
        }
        else
        {
            return RegisterType(CreateArrayType(p_child, p_isConst, p_dimensions, p_maxElements));
        }
    }
}


const ArrayType&
NamedTypeManager::GetArrayType(const ArrayType& p_type)
{
    const TypeImpl* type = GetTypeInfo(p_type.GetName());
    if (type != NULL)
    {
        FF2_ASSERT(type->Primitive() == Type::Array);
        return static_cast<const ArrayType&>(*type);
    }
    else
    {
        return RegisterType(CopyArrayType(p_type));
    }
}


const StructType&
NamedTypeManager::GetStructType(const std::string& p_name,
                                const std::string& p_externName,
                                const std::vector<StructType::MemberInfo>& p_members,
                                bool p_isConst)
{
    FF2_ASSERT(p_name.find(' ') == std::string::npos);

    std::string name;
    name.reserve(8 + p_name.size());
    if (!p_isConst)
    {
        name.assign("mutable ");
    }
    name.append(p_name);

    const TypeImpl* type = GetTypeInfo(name);
    if (type != NULL)
    {
        FF2_ASSERT(type->IsConst() == p_isConst);
        FF2_ASSERT(type->Primitive() == Type::Struct);
        const StructType& structType = static_cast<const StructType&>(*type);
        FF2_ASSERT(structType.GetExternName() == p_externName);
        return structType;
    }
    else
    {
        return RegisterType(CreateStructType(p_name, p_externName, p_members, p_isConst), name);
    }
}


const StructType&
NamedTypeManager::GetStructType(const StructType& p_type)
{
    FF2_ASSERT(p_type.GetName().find(' ') == std::string::npos);

    std::string name;
    name.reserve(8 + p_type.GetName().size());
    if (!p_type.IsConst())
    {
        name.assign("mutable ");
    }
    name.append(p_type.GetName());

    const TypeImpl* type = GetTypeInfo(name);
    if (type != NULL)
    {
        FF2_ASSERT(type->Primitive() == Type::Struct);
        return static_cast<const StructType&>(*type);
    }
    else
    {
        return RegisterType(CopyStructType(p_type), name);
    }
}


const ObjectType&
NamedTypeManager::GetObjectType(const std::string& p_name,
                                const std::string& p_externName,
                                const std::vector<ObjectType::ObjectMember>& p_members,
                                bool p_isConst)
{
    FF2_ASSERT(p_name.find(' ') == std::string::npos);

    const TypeImpl* type = GetTypeInfo(p_name);
    if (type != NULL)
    {
        FF2_ASSERT(type->IsConst() == p_isConst);
        FF2_ASSERT(type->Primitive() == Type::Object);
        const ObjectType& structType = static_cast<const ObjectType&>(*type);
        FF2_ASSERT(structType.GetExternName() == p_externName);
        return structType;
    }
    else
    {
        return RegisterType(CreateObjectType(p_name, p_externName, p_members, p_isConst), p_name);
    }
}


const ObjectType&
NamedTypeManager::GetObjectType(const ObjectType& p_type)
{
    FF2_ASSERT(p_type.GetName().find(' ') == std::string::npos);

    std::string name = p_type.GetName();
    const TypeImpl* type = GetTypeInfo(name);
    if (type != NULL)
    {
        FF2_ASSERT(type->Primitive() == Type::Object);
        return static_cast<const ObjectType&>(*type);
    }
    else
    {
        return RegisterType(CopyObjectType(p_type), name);
    }
}


const StateMachineType&
NamedTypeManager::GetStateMachineType(const std::string& p_name,
                                      const CompoundType::Member* p_members,
                                      size_t p_numMembers,
                                      boost::weak_ptr<const StateMachineExpression> p_expr)
{
    const TypeImpl* type = GetTypeInfo(p_name);
    if (type != NULL)
    {
        FF2_ASSERT(type->Primitive() == Type::StateMachine);
        const StateMachineType& machineType = static_cast<const StateMachineType&>(*type);
        FF2_ASSERT(machineType.GetName() == p_name);
        FF2_ASSERT(machineType.GetMemberCount() == p_numMembers);
        const CompoundType::Member* members = machineType.BeginMembers();
        for (size_t i = 0; i < p_numMembers; i++)
        {
            FF2_ASSERT(members[i].m_name == p_members[i].m_name
                && *members[i].m_type == *p_members[i].m_type);
        }
        return machineType;
    }
    else
    {
        return RegisterType(CreateStateMachineType(p_name, p_members, p_numMembers, p_expr));
    }
}


const StateMachineType&
NamedTypeManager::GetStateMachineType(const StateMachineType& p_type)
{
    const TypeImpl* type = GetTypeInfo(p_type.GetName());
    if (type != NULL)
    {
        FF2_ASSERT(*type == p_type);
        const StateMachineType& machineType = static_cast<const StateMachineType&>(*type);
        return machineType;
    }
    else
    {
        return RegisterType(CopyStateMachineType(p_type));
    }
}


const FunctionType&
NamedTypeManager::GetFunctionType(const TypeImpl& p_returnType,
                                  const TypeImpl* const* p_params,
                                  size_t p_numParams)
{
    std::string name
        = FunctionType::GetName(p_returnType, p_params, p_numParams);
    const TypeImpl* type = GetTypeInfo(name);
    if (type != NULL)
    {
        FF2_ASSERT(type->Primitive() == Type::Function);
        const FunctionType& functionType = static_cast<const FunctionType&>(*type);
        FF2_ASSERT(functionType.GetReturnType() == p_returnType);
        FF2_ASSERT(functionType.GetParameterCount() == p_numParams);

        for (size_t i = 0; i < p_numParams; i++)
        {
            FF2_ASSERT(*functionType.BeginParameters()[i] == *p_params[i]);
        }
        return functionType;
    }
    else
    {
        return RegisterType(CreateFunctionType(p_returnType, p_params, p_numParams));
    }
}


const FunctionType&
NamedTypeManager::GetFunctionType(const FunctionType& p_type)
{
    const TypeImpl* type = GetTypeInfo(p_type.GetName());
    if (type != NULL)
    {
        FF2_ASSERT(*type == p_type);
        const FunctionType& functionType = static_cast<const FunctionType&>(*type);
        return functionType;
    }
    else
    {
        return RegisterType(CopyFunctionType(p_type));
    }
}


const FreeForm2::TypeManager&
FreeForm2::TypeManager::GetGlobalTypeManager()
{
    static const NamedTypeManager s_instance;

    return s_instance;
}


std::auto_ptr<FreeForm2::TypeManager> 
FreeForm2::TypeManager::CreateTypeManager()
{
    return std::auto_ptr<TypeManager>(new NamedTypeManager(TypeManager::GetGlobalTypeManager()));
}


std::auto_ptr<FreeForm2::TypeManager> 
FreeForm2::TypeManager::CreateTypeManager(const TypeManager& p_parent)
{
    return std::auto_ptr<TypeManager>(new NamedTypeManager(p_parent));
}


std::auto_ptr<FreeForm2::TypeManager> 
FreeForm2::TypeManager::CreateTypeManager(const ExternalDataManager& p_parent)
{
    return std::auto_ptr<TypeManager>(new NamedTypeManager(p_parent.m_typeFactory->GetTypeManager()));
}


AnonymousTypeManager::AnonymousTypeManager()
    : TypeManager(NULL)
{
}


const TypeImpl*
AnonymousTypeManager::GetTypeInfo(const std::string& p_name) const
{
    return NULL;
}


const ArrayType&
AnonymousTypeManager::GetArrayType(const TypeImpl& p_child,
                                   bool p_isConst,
                                   unsigned int p_dimensions,
                                   const unsigned int p_elementCounts[],
                                   unsigned int p_maxElements)
{
    boost::shared_ptr<const ArrayType> type(
        CreateArrayType(p_child, p_isConst, p_dimensions, p_elementCounts, p_maxElements));
    m_types.push_back(boost::static_pointer_cast<const TypeImpl>(type));
    return *type;
}


const ArrayType&
AnonymousTypeManager::GetArrayType(const TypeImpl& p_child,
                                   bool p_isConst,
                                   unsigned int p_dimensions,
                                   unsigned int p_maxElements)
{
    boost::shared_ptr<const ArrayType> type(
        CreateArrayType(p_child, p_isConst, p_dimensions, p_maxElements));
    m_types.push_back(boost::static_pointer_cast<const TypeImpl>(type));
    return *type;
}


const ArrayType&
AnonymousTypeManager::GetArrayType(const ArrayType& p_arrayType)
{
    boost::shared_ptr<const ArrayType> type(CopyArrayType(p_arrayType));
    m_types.push_back(boost::static_pointer_cast<const TypeImpl>(type));
    return *type;
}


const StructType&
AnonymousTypeManager::GetStructType(const std::string& p_name,
                                    const std::string& p_externName,
                                    const std::vector<StructType::MemberInfo>& p_members,
                                    bool p_isConst)
{
    boost::shared_ptr<const StructType> type(
        CreateStructType(p_name, p_externName, p_members, p_isConst));
    m_types.push_back(boost::static_pointer_cast<const TypeImpl>(type));
    return *type;
}


const StructType&
AnonymousTypeManager::GetStructType(const StructType& p_type)
{
    boost::shared_ptr<const StructType> type(CopyStructType(p_type));
    m_types.push_back(boost::static_pointer_cast<const TypeImpl>(type));
    return *type;
}


const ObjectType&
AnonymousTypeManager::GetObjectType(const std::string& p_name,
                                    const std::string& p_externName,
                                    const std::vector<ObjectType::ObjectMember>& p_members,
                                    bool p_isConst)
{
    boost::shared_ptr<const ObjectType> type(
        CreateObjectType(p_name, p_externName, p_members, p_isConst));
    m_types.push_back(boost::static_pointer_cast<const TypeImpl>(type));
    return *type;
}


const ObjectType&
AnonymousTypeManager::GetObjectType(const ObjectType& p_type)
{
    boost::shared_ptr<const ObjectType> type(CopyObjectType(p_type));
    m_types.push_back(boost::static_pointer_cast<const TypeImpl>(type));
    return *type;
}


const StateMachineType&
AnonymousTypeManager::GetStateMachineType(const std::string& p_name,
                                          const CompoundType::Member* p_members,
                                          size_t p_numMembers,
                                          boost::weak_ptr<const StateMachineExpression> p_expr)
{
    boost::shared_ptr<const StateMachineType> type(
        CreateStateMachineType(p_name, p_members, p_numMembers, p_expr));
    m_types.push_back(boost::static_pointer_cast<const TypeImpl>(type));
    return *type;
}


const StateMachineType&
AnonymousTypeManager::GetStateMachineType(const StateMachineType& p_type)
{
    boost::shared_ptr<const StateMachineType> type(CopyStateMachineType(p_type));
    m_types.push_back(boost::static_pointer_cast<const TypeImpl>(type));
    return *type;
}


const FunctionType&
AnonymousTypeManager::GetFunctionType(const TypeImpl& p_returnType,
                                      const TypeImpl* const* p_parameters,
                                      size_t p_numParameters)
{
    boost::shared_ptr<const FunctionType> type(
        CreateFunctionType(p_returnType, p_parameters, p_numParameters));
    m_types.push_back(boost::static_pointer_cast<const TypeImpl>(type));
    return *type;
}


const FunctionType&
AnonymousTypeManager::GetFunctionType(const FunctionType& p_type)
{
    boost::shared_ptr<const FunctionType> type(CopyFunctionType(p_type));
    m_types.push_back(boost::static_pointer_cast<const TypeImpl>(type));
    return *type;
}
