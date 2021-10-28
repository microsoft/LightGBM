/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include "StateMachine.h"

#include <boost/shared_ptr.hpp>
#include "FreeForm2Assert.h"
#include "Mutation.h"
#include <sstream>
#include "TypeManager.h"
#include "Visitor.h"

namespace
{
    // Custom deleter for StateMachineExpressions. This is required because
    // these objects are allocated using the struct hack.
    void
    DeleteStateMachine(FreeForm2::StateMachineExpression *p_ptr)
    {
        // Explicitly call the destructor.
        p_ptr->~StateMachineExpression();

        // Delete the memory, which is allocated as a char[].
        char *mem = reinterpret_cast<char *>(p_ptr);
        delete[] mem;
    }
}

FreeForm2::StateExpression::StateExpression(const Annotations &p_annotations)
    : Expression(p_annotations)
{
}

void FreeForm2::StateExpression::Accept(Visitor &p_visitor) const
{
    const size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(*this))
    {
        p_visitor.Visit(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

const FreeForm2::TypeImpl &
FreeForm2::StateExpression::GetType() const
{
    return TypeImpl::GetVoidInstance();
}

size_t
FreeForm2::StateExpression::GetNumChildren() const
{
    return 0;
}

boost::shared_ptr<FreeForm2::StateMachineExpression>
FreeForm2::StateMachineExpression::Alloc(const Annotations &p_annotations,
                                         const TypeInitializerExpression &p_initializer,
                                         const StateExpression *const *p_states,
                                         size_t p_numStates,
                                         size_t p_startStateId)
{
    const size_t allocSize = sizeof(StateMachineExpression) + (p_numStates > 0 ? (p_numStates - 1) * sizeof(const StateExpression *) : 0);
    boost::shared_ptr<StateMachineExpression> expr;
    char *mem = NULL;

    try
    {
        mem = new char[allocSize];
        expr.reset(new (mem) StateMachineExpression(p_annotations,
                                                    p_initializer,
                                                    p_states,
                                                    p_numStates,
                                                    p_startStateId),
                   DeleteStateMachine);
    }
    catch (...)
    {
        delete[] mem;
        throw;
    }
    return expr;
}

boost::shared_ptr<FreeForm2::StateMachineExpression>
FreeForm2::StateMachineExpression::Alloc(const Annotations &p_annotations,
                                         const StateMachineType &p_type,
                                         const TypeInitializerExpression &p_initializer,
                                         const StateExpression *const *p_states,
                                         size_t p_numStates,
                                         size_t p_startStateId)
{
    boost::shared_ptr<StateMachineExpression> expr(
        Alloc(p_annotations, p_initializer, p_states, p_numStates, p_startStateId));

    // Link the type and expression.
    expr->m_type = &p_type;
    FF2_ASSERT(p_type.m_expr.expired());
    p_type.m_expr = expr;

    return expr;
}

FreeForm2::StateMachineExpression::StateMachineExpression(const Annotations &p_annotations,
                                                          const TypeInitializerExpression &p_initializer,
                                                          const StateExpression *const *p_states,
                                                          size_t p_numStates,
                                                          size_t p_startStateId)
    : Expression(p_annotations),
      m_startStateId(p_startStateId),
      m_type(NULL),
      m_initializer(p_initializer),
      m_numStates(p_numStates)
{
    memcpy(m_states, p_states, sizeof(const StateExpression *) * p_numStates);
}

FreeForm2::StateMachineExpression::~StateMachineExpression()
{
}

void FreeForm2::StateMachineExpression::Accept(Visitor &p_visitor) const
{
    const size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(*this))
    {
        for (size_t i = 0; i < m_numStates; i++)
        {
            m_states[i]->Accept(p_visitor);
        }

        m_initializer.Accept(p_visitor);

        p_visitor.Visit(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

const FreeForm2::TypeImpl &
FreeForm2::StateMachineExpression::GetType() const
{
    FF2_ASSERT(m_type != NULL);
    return *m_type;
}

size_t
FreeForm2::StateMachineExpression::GetNumChildren() const
{
    return m_numStates + 1;
}

const FreeForm2::TypeInitializerExpression &
FreeForm2::StateMachineExpression::GetInitializer() const
{
    return m_initializer;
}

const FreeForm2::StateExpression *const *
FreeForm2::StateMachineExpression::GetChildren() const
{
    return m_states;
}

size_t
FreeForm2::StateMachineExpression::GetStartStateId() const
{
    return m_startStateId;
}

std::string
FreeForm2::StateMachineExpression::GetAugmentedMemberName(
    const std::string &p_machineName,
    const std::string &p_memberName)
{
    std::ostringstream out;
    out << "__" << p_machineName << "_" << p_memberName;
    return out.str();
}

boost::shared_ptr<FreeForm2::ExecuteMachineExpression>
FreeForm2::ExecuteMachineExpression::Alloc(const Annotations &p_annotations,
                                           const Expression &p_stream,
                                           const Expression &p_machine,
                                           const std::pair<std::string, const Expression *> *p_yieldActions,
                                           const size_t p_numYieldActions)
{
    size_t bytes = sizeof(ExecuteMachineExpression) + (std::max<size_t>(p_numYieldActions, 1) - 1) * sizeof(std::pair<std::string, const Expression *>);

    // Allocate a shared_ptr that deletes an ExecuteMachineExpression
    // allocated in a char[].
    boost::shared_ptr<ExecuteMachineExpression> exp;
    exp.reset(new (new char[bytes]) ExecuteMachineExpression(p_annotations,
                                                             p_stream,
                                                             p_machine,
                                                             p_yieldActions,
                                                             p_numYieldActions),
              DeleteAlloc);
    return exp;
}

void FreeForm2::ExecuteMachineExpression::DeleteAlloc(ExecuteMachineExpression *p_allocated)
{
    for (size_t i = 1; i < p_allocated->m_numYieldActions; ++i)
    {
        p_allocated->m_yieldActions[i].first.~basic_string();
    }

    // Manually call dtor for expression.
    p_allocated->~ExecuteMachineExpression();

    // Dispose of memory, which we allocated in a char[].
    char *mem = reinterpret_cast<char *>(p_allocated);
    delete[] mem;
}

FreeForm2::ExecuteMachineExpression::ExecuteMachineExpression(
    const Annotations &p_annotations,
    const Expression &p_stream,
    const Expression &p_machine,
    const std::pair<std::string, const Expression *> *p_yieldActions,
    size_t p_numYieldActions)
    : Expression(p_annotations),
      m_machine(p_machine),
      m_stream(p_stream),
      m_numYieldActions(p_numYieldActions)
{
    for (size_t i = 0; i < p_numYieldActions; ++i)
    {
        if (i > 0)
        {
            new (&m_yieldActions[i].first) std::string();
        }

        m_yieldActions[i] = p_yieldActions[i];
    }
}

void FreeForm2::ExecuteMachineExpression::Accept(Visitor &p_visitor) const
{
    const size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(*this))
    {
        m_stream.Accept(p_visitor);
        m_machine.Accept(p_visitor);

        for (size_t i = 0; i < m_numYieldActions; ++i)
        {
            m_yieldActions[i].second->Accept(p_visitor);
        }

        p_visitor.Visit(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

const FreeForm2::TypeImpl &
FreeForm2::ExecuteMachineExpression::GetType() const
{
    return TypeImpl::GetVoidInstance();
}

size_t
FreeForm2::ExecuteMachineExpression::GetNumChildren() const
{
    return 2 + m_numYieldActions;
}

const FreeForm2::Expression &
FreeForm2::ExecuteMachineExpression::GetStream() const
{
    return m_stream;
}

const FreeForm2::Expression &
FreeForm2::ExecuteMachineExpression::GetMachine() const
{
    return m_machine;
}

const std::pair<std::string, const FreeForm2::Expression *> *
FreeForm2::ExecuteMachineExpression::GetYieldActions() const
{
    return m_yieldActions;
}

size_t
FreeForm2::ExecuteMachineExpression::GetNumYieldActions() const
{
    return m_numYieldActions;
}

FreeForm2::ExecuteMachineGroupExpression::ExecuteMachineGroupExpression(const Annotations &p_annotations,
                                                                        const FreeForm2::ExecuteMachineGroupExpression::MachineInstance *p_machineInstances,
                                                                        unsigned int p_numMachineInstances,
                                                                        unsigned int p_numBound)
    : Expression(p_annotations),
      m_numMachineInstances(p_numMachineInstances),
      m_numBound(p_numBound)
{
    for (unsigned int i = 0; i < m_numMachineInstances; ++i)
    {
        m_machineInstances[i] = p_machineInstances[i];
    }
}

boost::shared_ptr<FreeForm2::ExecuteMachineGroupExpression>
FreeForm2::ExecuteMachineGroupExpression::Alloc(const Annotations &p_annotations,
                                                const FreeForm2::ExecuteMachineGroupExpression::MachineInstance *p_machineInstances,
                                                unsigned int p_numMachineInstances,
                                                unsigned int p_numBound)
{
    FF2_ASSERT(p_numMachineInstances > 0);

    size_t bytes = sizeof(ExecuteMachineGroupExpression) + (p_numMachineInstances - 1) * sizeof(MachineInstance);

    // Allocate a shared_ptr that deletes an ExecuteMachineExpression
    // allocated in a char[].
    boost::shared_ptr<ExecuteMachineGroupExpression> exp;
    exp.reset(new (new char[bytes]) ExecuteMachineGroupExpression(p_annotations,
                                                                  p_machineInstances,
                                                                  p_numMachineInstances,
                                                                  p_numBound),
              DeleteAlloc);
    return exp;
}

void FreeForm2::ExecuteMachineGroupExpression::DeleteAlloc(ExecuteMachineGroupExpression *p_allocated)
{
    // Manually call dtor for expression.
    p_allocated->~ExecuteMachineGroupExpression();

    // Dispose of memory, which we allocated in a char[].
    char *mem = reinterpret_cast<char *>(p_allocated);
    delete[] mem;
}

void FreeForm2::ExecuteMachineGroupExpression::Accept(Visitor &p_visitor) const
{
    const size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(*this))
    {
        for (size_t i = 0; i < m_numMachineInstances; ++i)
        {
            m_machineInstances[i].m_machineDeclaration->Accept(p_visitor);
            m_machineInstances[i].m_machineExpression->Accept(p_visitor);
        }

        p_visitor.Visit(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

const FreeForm2::TypeImpl &
FreeForm2::ExecuteMachineGroupExpression::GetType() const
{
    return TypeImpl::GetVoidInstance();
}

size_t
FreeForm2::ExecuteMachineGroupExpression::GetNumChildren() const
{
    return 2 * m_numMachineInstances;
}

unsigned int
FreeForm2::ExecuteMachineGroupExpression::GetNumBound() const
{
    return m_numBound;
}

const FreeForm2::ExecuteMachineGroupExpression::MachineInstance *
FreeForm2::ExecuteMachineGroupExpression::GetMachineInstances() const
{
    return m_machineInstances;
}

unsigned int
FreeForm2::ExecuteMachineGroupExpression::GetNumMachineInstances() const
{
    return m_numMachineInstances;
}

FreeForm2::ExecuteStreamRewritingStateMachineGroupExpression::ExecuteStreamRewritingStateMachineGroupExpression(
    const Annotations &p_annotations,
    const FreeForm2::ExecuteStreamRewritingStateMachineGroupExpression::MachineInstance *p_machineInstances,
    unsigned int p_numMachineInstances,
    unsigned int p_numBound,
    VariableID p_machineIndexID,
    unsigned int p_machineArraySize,
    StreamRewritingType p_streamRewritingType,
    const Expression *p_duplicateTermInformation,
    const Expression *p_numQueryPaths,
    const Expression *p_queryPathCandidates,
    const Expression *p_queryLength,
    const Expression *p_tupleOfInterestCount,
    const Expression *p_tuplesOfInterest,
    bool p_isNearChunk,
    unsigned int p_minChunkNumber)
    : Expression(p_annotations),
      m_numMachineInstances(p_numMachineInstances),
      m_numBound(p_numBound),
      m_machineIndexID(p_machineIndexID),
      m_machineArraySize(p_machineArraySize),
      m_streamRewritingType(p_streamRewritingType),
      m_duplicateTermInformation(p_duplicateTermInformation),
      m_numQueryPaths(p_numQueryPaths),
      m_queryPathCandidates(p_queryPathCandidates),
      m_queryLength(p_queryLength),
      m_tupleOfInterestCount(p_tupleOfInterestCount),
      m_tuplesOfInterest(p_tuplesOfInterest),
      m_isNearChunk(p_isNearChunk),
      m_minChunkNumber(p_minChunkNumber)
{
    for (unsigned int i = 0; i < m_numMachineInstances; ++i)
    {
        m_machineInstances[i] = p_machineInstances[i];
    }
}

boost::shared_ptr<FreeForm2::ExecuteStreamRewritingStateMachineGroupExpression>
FreeForm2::ExecuteStreamRewritingStateMachineGroupExpression::Alloc(const Annotations &p_annotations,
                                                                    const FreeForm2::ExecuteStreamRewritingStateMachineGroupExpression::MachineInstance *p_machineInstances,
                                                                    unsigned int p_numMachineInstances,
                                                                    unsigned int p_numBound,
                                                                    VariableID p_machineIndexID,
                                                                    unsigned int p_machineArraySize,
                                                                    StreamRewritingType p_streamRewritingType,
                                                                    const Expression *p_duplicateTermInformation,
                                                                    const Expression *p_numQueryPaths,
                                                                    const Expression *p_queryPathCandidates,
                                                                    const Expression *p_queryLength,
                                                                    const Expression *p_tupleOfInterestCount,
                                                                    const Expression *p_tuplesOfInterest,
                                                                    bool p_isNearChunk,
                                                                    unsigned int p_minChunkNumber)
{
    FF2_ASSERT(p_numMachineInstances > 0 && p_machineArraySize > 0);

    // Check that the machine array size is appropriately matched with the StreamRewritingType.
    // FF2_ASSERT(//(p_streamRewritingType == BodyBlock && p_machineArraySize == MetaWords::BBWM_Max) ||
    //           (p_streamRewritingType == QueryPath && p_machineArraySize == FeatureData::c_maxNumberOfQueryPaths)
    //           || (p_streamRewritingType == Chunk && p_machineArraySize == (FeatureData::c_chunkTypeEndIndex - FeatureData::c_chunkTypeStartIndex + 1)));

    size_t bytes = sizeof(ExecuteStreamRewritingStateMachineGroupExpression) + (p_numMachineInstances - 1) * sizeof(MachineInstance);

    // Allocate a shared_ptr that deletes an ExecuteMachineExpression
    // allocated in a char[].
    boost::shared_ptr<ExecuteStreamRewritingStateMachineGroupExpression> exp;
    exp.reset(new (new char[bytes]) ExecuteStreamRewritingStateMachineGroupExpression(p_annotations,
                                                                                      p_machineInstances,
                                                                                      p_numMachineInstances,
                                                                                      p_numBound,
                                                                                      p_machineIndexID,
                                                                                      p_machineArraySize,
                                                                                      p_streamRewritingType,
                                                                                      p_duplicateTermInformation,
                                                                                      p_numQueryPaths,
                                                                                      p_queryPathCandidates,
                                                                                      p_queryLength,
                                                                                      p_tupleOfInterestCount,
                                                                                      p_tuplesOfInterest,
                                                                                      p_isNearChunk,
                                                                                      p_minChunkNumber),
              DeleteAlloc);
    return exp;
}

void FreeForm2::ExecuteStreamRewritingStateMachineGroupExpression::DeleteAlloc(ExecuteStreamRewritingStateMachineGroupExpression *p_allocated)
{
    // Manually call dtor for expression.
    p_allocated->~ExecuteStreamRewritingStateMachineGroupExpression();

    // Dispose of memory, which we allocated in a char[].
    char *mem = reinterpret_cast<char *>(p_allocated);
    delete[] mem;
}

void FreeForm2::ExecuteStreamRewritingStateMachineGroupExpression::Accept(Visitor &p_visitor) const
{
    const size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(*this))
    {
        for (size_t i = 0; i < m_numMachineInstances; ++i)
        {
            m_machineInstances[i].m_machineDeclaration->Accept(p_visitor);
            m_machineInstances[i].m_machineExpression->Accept(p_visitor);
        }

        p_visitor.Visit(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

const FreeForm2::TypeImpl &
FreeForm2::ExecuteStreamRewritingStateMachineGroupExpression::GetType() const
{
    return TypeImpl::GetVoidInstance();
}

size_t
FreeForm2::ExecuteStreamRewritingStateMachineGroupExpression::GetNumChildren() const
{
    return 2 * m_numMachineInstances;
}

unsigned int
FreeForm2::ExecuteStreamRewritingStateMachineGroupExpression::GetNumBound() const
{
    return m_numBound;
}

const FreeForm2::ExecuteStreamRewritingStateMachineGroupExpression::MachineInstance *
FreeForm2::ExecuteStreamRewritingStateMachineGroupExpression::GetMachineInstances() const
{
    return m_machineInstances;
}

unsigned int
FreeForm2::ExecuteStreamRewritingStateMachineGroupExpression::GetNumMachineInstances() const
{
    return m_numMachineInstances;
}

FreeForm2::VariableID
FreeForm2::ExecuteStreamRewritingStateMachineGroupExpression::GetMachineIndexID() const
{
    return m_machineIndexID;
}

unsigned int
FreeForm2::ExecuteStreamRewritingStateMachineGroupExpression::GetMachineArraySize() const
{
    return m_machineArraySize;
}

FreeForm2::ExecuteStreamRewritingStateMachineGroupExpression::StreamRewritingType
FreeForm2::ExecuteStreamRewritingStateMachineGroupExpression::GetStreamRewritingType() const
{
    return m_streamRewritingType;
}

const FreeForm2::Expression *
FreeForm2::ExecuteStreamRewritingStateMachineGroupExpression::GetDuplicateTermInformation() const
{
    return m_duplicateTermInformation;
}

const FreeForm2::Expression *
FreeForm2::ExecuteStreamRewritingStateMachineGroupExpression::GetNumQueryPaths() const
{
    return m_numQueryPaths;
}

const FreeForm2::Expression *
FreeForm2::ExecuteStreamRewritingStateMachineGroupExpression::GetQueryPathCandidates() const
{
    return m_queryPathCandidates;
}

const FreeForm2::Expression *
FreeForm2::ExecuteStreamRewritingStateMachineGroupExpression::GetQueryLength() const
{
    return m_queryLength;
}

const FreeForm2::Expression *
FreeForm2::ExecuteStreamRewritingStateMachineGroupExpression::GetTupleOfInterestCount() const
{
    return m_tupleOfInterestCount;
}

const FreeForm2::Expression *
FreeForm2::ExecuteStreamRewritingStateMachineGroupExpression::GetTuplesOfInterest() const
{
    return m_tuplesOfInterest;
}

bool FreeForm2::ExecuteStreamRewritingStateMachineGroupExpression::IsNearChunk() const
{
    return m_isNearChunk;
}

unsigned int
FreeForm2::ExecuteStreamRewritingStateMachineGroupExpression::GetMinChunkNumber() const
{
    return m_minChunkNumber;
}

FreeForm2::YieldExpression::YieldExpression(const Annotations &p_annotations,
                                            const std::string &p_machineName,
                                            const std::string &p_name)
    : Expression(p_annotations),
      m_name(p_name),
      m_machineName(p_machineName),
      m_fullName(p_machineName + "::" + p_name)
{
}

void FreeForm2::YieldExpression::Accept(Visitor &p_visitor) const
{
    const size_t stackSize = p_visitor.StackSize();

    p_visitor.Visit(*this);

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

const FreeForm2::TypeImpl &
FreeForm2::YieldExpression::GetType() const
{
    return TypeImpl::GetVoidInstance();
}

size_t
FreeForm2::YieldExpression::GetNumChildren() const
{
    return 0;
}

const std::string &
FreeForm2::YieldExpression::GetName() const
{
    return m_name;
}

const std::string &
FreeForm2::YieldExpression::GetMachineName() const
{
    return m_machineName;
}

const std::string &
FreeForm2::YieldExpression::GetFullName() const
{
    return m_fullName;
}
