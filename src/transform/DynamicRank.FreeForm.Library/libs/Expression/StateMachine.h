#pragma once

#ifndef FREEFORM2_STATEMACHINE_H
#define FREEFORM2_STATEMACHINE_H

#include <boost/shared_ptr.hpp>
#include "StateMachineType.h"
#include "Expression.h"
#include <list>
#include "Declaration.h"

namespace FreeForm2
{
    class SimpleExpressionOwner;
    class TypeInitializerExpression;
    class TypeManager;

    // A StateExpression represents a single state of a finite state machine.
    // States contain actions and transitions, which are described below.
    class StateExpression : public Expression
    {
    public:
        // All actions and matches have a match type, which corresponds to a
        // possible type of word occurrences at the current stream location.
        // An unconstrained match type does not take into account the current
        // word.
        enum MatchType
        {
            Unconstrained,
            MatchWord,
            MatchInstanceHeader,
            MatchBodyBlockHeader,
            EndStream
        };

        // Actions modify the state of a state machine, and are optionally 
        // executed when entering or leaving a state. Leaving actions are
        // contained within the Transition struct; this struct is for entering
        // actions only.
        struct Action
        {
            // Type of the current word for the action.
            MatchType m_matchType;

            // Expression tree for the action.
            const Expression* m_action;
        };

        // Transitions make the edges of the state machine graph. Because the
        // finite state machines are digraphs, each state contains the 
        // transitions originating from that state.
        struct Transition
        {
            // Type of the current word required for this transition.
            MatchType m_matchType;

            // Optional condition for this transition. This should be NULL if
            // not used.
            const Expression* m_condition;

            // Destination state ID in the context of the parent machine.
            size_t m_destinationId;

            // Optional action to execute when leaving the state.
            const Expression* m_leavingAction;
        };

        StateExpression(const Annotations& p_annotations);

        // Methods inherited from Expression.
        virtual void Accept(Visitor& p_visitor) const override;
        virtual const TypeImpl& GetType() const override;
        virtual size_t GetNumChildren() const override;

        // An identifier which is unique within the same parsing context.
        size_t m_id;

        // Actions to perform on this state.
        std::list<Action> m_actions;

        // Transitions leaving this state.
        std::list<Transition> m_transitions;
    };

    // A StateMachineExpression is a machine node, which contains the digraph
    // state machine. Note that even though the machine is a graph, the syntax
    // tree does not contain physical cycles; transitions are referenced by ID
    // as oppose to pointers.
    class StateMachineExpression : public Expression
    {
    public:
        // Allocate an instance of a StateMachineExpression from an existing 
        // StateMachineType. The StateMachineType must not have an associated
        // definition expression.
        static boost::shared_ptr<StateMachineExpression>
        Alloc(const Annotations& p_annotations,
              const StateMachineType& p_type,
              const TypeInitializerExpression& p_initializer,
              const StateExpression* const* p_states,
              size_t p_numStates,
              size_t p_startStateId);

        // Empty destructor for use with custom allocation.
        virtual ~StateMachineExpression();

        // Methods inherited from Expression.
        virtual void Accept(Visitor& p_visitor) const override;
        virtual const TypeImpl& GetType() const override;
        virtual size_t GetNumChildren() const override;

        // Getters for member variables.
        const TypeInitializerExpression& GetInitializer() const;
        const StateExpression* const* GetChildren() const;
        size_t GetStartStateId() const;

        // Generates the augmented name for state machine member names.
        static std::string GetAugmentedMemberName(const std::string& p_machineName,
                                                  const std::string& p_memberName);

    private:
        // Private constructor due to struct hack allocation.
        StateMachineExpression(const Annotations& p_annotations,
                               const TypeInitializerExpression& p_initializer,
                               const StateExpression* const* p_states,
                               size_t p_numStates,
                               size_t p_startStateId);

        // Allocate the StateMachineExpression independent of the TypeImpl.
        static boost::shared_ptr<StateMachineExpression>
        Alloc(const Annotations& p_annotations,
              const TypeInitializerExpression& p_initializer,
              const StateExpression* const* p_states,
              size_t p_numStates,
              size_t p_startStateId);

        // Start state of the machine.
        size_t m_startStateId;

        // The type of the state machine type.
        const StateMachineType* m_type;

        // The type initializer for the state machine.
        const TypeInitializerExpression& m_initializer;

        // Number of children in m_children.
        size_t m_numStates;

        // Children of the machine, allocated using the struct hack. This 
        // should be either StateExpressions or DeclarationExpressions.
        const StateExpression* m_states[1];
    };

    // This expression executes an instantiated state machine on a stream.
    class ExecuteMachineExpression : public Expression
    {
    public:

        static boost::shared_ptr<ExecuteMachineExpression>
        Alloc(const Annotations& p_annotations,
              const Expression& p_stream,
              const Expression& p_machine,
              const std::pair<std::string, const Expression*>* p_yieldActions,
              size_t p_numYieldActions);

        // Methods inherited from Expression.
        virtual void Accept(Visitor& p_visitor) const override;
        virtual const TypeImpl& GetType() const override;
        virtual size_t GetNumChildren() const override;

        // Getters for member variables.
        const Expression& GetStream() const;
        const Expression& GetMachine() const;
        const std::pair<std::string, const Expression*>* GetYieldActions() const;
        size_t GetNumYieldActions() const;

    private:
        ExecuteMachineExpression(const Annotations& p_annotations,
                                 const Expression& p_stream,
                                 const Expression& p_machine,
                                 const std::pair<std::string, const Expression*>* p_yieldActions,
                                 size_t p_numYieldActions);

        static void DeleteAlloc(ExecuteMachineExpression* p_allocated);

        // The stream object to be mached by the state machine
        const Expression& m_stream;

        // The machine variable to execute.
        const Expression& m_machine;

        // The number of available yield actions.
        size_t m_numYieldActions;

        // Action code to be executed on each named yield,
        // allocated using the struct hack.
        std::pair<std::string, const Expression*> m_yieldActions[1];
    };

    // This expression groups several state machine declarations and ExecuteMachineExpressions.
    // This is a wrapper class to help state machine composition.
    class ExecuteMachineGroupExpression : public Expression
    {
    public:
        struct MachineInstance
        {
            // The VariableRefExpression that points to the
            // state machine declaration.
            const Expression* m_machineDeclaration;

            // The ExecuteMachineExpression associated with this
            // group.
            const ExecuteMachineExpression* m_machineExpression;
        };
        
        static boost::shared_ptr<ExecuteMachineGroupExpression>
        Alloc(const Annotations& p_annotations,
              const MachineInstance* p_machines,
              unsigned int p_numMachines,
              unsigned int p_numBound);
        
        // Return the number of symbols bound by immediate children of this 
        // expression, and left open.
        unsigned int GetNumBound() const;

        // Methods inherited from Expression.
        virtual void Accept(Visitor& p_visitor) const override;
        virtual const TypeImpl& GetType() const override;
        virtual size_t GetNumChildren() const override;

        // Getters for member variables.
        const MachineInstance* GetMachineInstances() const;
        unsigned int GetNumMachineInstances() const;

    private:
        ExecuteMachineGroupExpression(const Annotations& p_annotations,
                                      const MachineInstance* p_machines,
                                      unsigned int p_numMachines,
                                      unsigned int p_numBound);

        static void DeleteAlloc(ExecuteMachineGroupExpression* p_allocated);

        // Number of machine instances.
        unsigned int m_numMachineInstances;
 
        // Number of symbols left bound by the machine instances.
        unsigned int m_numBound;

        // Action code to be executed on each named yield,
        // allocated using the struct hack.
        MachineInstance m_machineInstances[1];
    };

    // This expression executes a state machine group where the state machines
    // act on a modified version of the stream.  The original stream 
    // given to the feature group is "rewritten" before the state machines process it.
    class ExecuteStreamRewritingStateMachineGroupExpression : public Expression
    {
    public:
        struct MachineInstance
        {
            // The DeclarationExpression that points to the
            // state machine declaration.
            const DeclarationExpression* m_machineDeclaration;

            // The ExecuteMachineExpression associated with this
            // group.
            const ExecuteMachineExpression* m_machineExpression;
        };

        // The type of stream rewriting mechanism that will be used for this state machine group.
        enum StreamRewritingType
        {
            BodyBlock,
            QueryPath,
            Chunk
        };
        
        static boost::shared_ptr<ExecuteStreamRewritingStateMachineGroupExpression>
        Alloc(const Annotations& p_annotations,
              const MachineInstance* p_machines,
              unsigned int p_numMachines,
              unsigned int p_numBound,
              VariableID p_machineIndexID,
              unsigned int p_machineArraySize,
              StreamRewritingType p_streamRewritingType,
              const Expression* p_duplicateTermInformation,
              const Expression* p_numQueryPaths,
              const Expression* p_queryPathCandidates,
              const Expression* p_queryLength,
              const Expression* p_tupleOfInterestCount,
              const Expression* p_tuplesOfInterest,
              bool p_isNearChunk,
              unsigned int p_minChunkNumber);
        
        // Return the number of symbols bound by immediate children of this 
        // expression, and left open.
        unsigned int GetNumBound() const;

        // Methods inherited from Expression.
        virtual void Accept(Visitor& p_visitor) const override;
        virtual const TypeImpl& GetType() const override;
        virtual size_t GetNumChildren() const override;

        // Getters for member variables.
        const MachineInstance* GetMachineInstances() const;
        unsigned int GetNumMachineInstances() const;
        VariableID GetMachineIndexID() const;
        unsigned int GetMachineArraySize() const;
        StreamRewritingType GetStreamRewritingType() const;
        const Expression* GetDuplicateTermInformation() const;
        const Expression* GetNumQueryPaths() const;
        const Expression* GetQueryPathCandidates() const;
        const Expression* GetQueryLength() const;
        const Expression* GetTupleOfInterestCount() const;
        const Expression* GetTuplesOfInterest() const;
        bool IsNearChunk() const;
        unsigned int GetMinChunkNumber() const;

    private:
        ExecuteStreamRewritingStateMachineGroupExpression(const Annotations& p_annotations,
                                                          const MachineInstance* p_machines,
                                                          unsigned int p_numMachines,
                                                          unsigned int p_numBound,
                                                          VariableID p_machineIndexID,
                                                          unsigned int p_machineArraySize,
                                                          StreamRewritingType p_streamRewritingType,
                                                          const Expression* p_duplicateTermInformation,
                                                          const Expression* p_numQueryPaths,
                                                          const Expression* p_queryPathCandidates,
                                                          const Expression* p_queryLength,
                                                          const Expression* p_tupleOfInterestCount,
                                                          const Expression* p_tuplesOfInterest,
                                                          bool p_isNearChunk,
                                                          unsigned int p_minChunkNumber);

        static void DeleteAlloc(ExecuteStreamRewritingStateMachineGroupExpression* p_allocated);

        // Number of machine instances.
        unsigned int m_numMachineInstances;
 
        // Number of symbols left bound by the machine instances.
        unsigned int m_numBound;

        // Variable ID of state machine index.
        VariableID m_machineIndexID;

        // Size of State Machine array.
        unsigned int m_machineArraySize;

        // Stream rewriting type.
        StreamRewritingType m_streamRewritingType;

        // An (optional) extern expression representing a raw array with information about
        // duplicate terms.
        const Expression* m_duplicateTermInformation;

        // An (optional) extern expression representing the number of query path candidates.
        const Expression* m_numQueryPaths;

        // An (optional) extern expression representing the query path candidates.
        const Expression* m_queryPathCandidates;

        // An (optional) extern expression representing the length of the query.
        const Expression* m_queryLength;

        // An (optional) extern expression representing the number of tuples of interest.
        const Expression* m_tupleOfInterestCount;

        // An (optional) extern expression representing the tuples of interest.
        const Expression* m_tuplesOfInterest;

        // A flag whether or not the chunk match only considers near occurrences or not.
        bool m_isNearChunk;

        // The minimum number of chunks required to process a chunk feature.
        unsigned int m_minChunkNumber; 

        // Action code to be executed on each named yield,
        // allocated using the struct hack.
        MachineInstance m_machineInstances[1];
    };

    // Yield expressions are used to call back to the caller of a state 
    // machine. A yield expression may only appear in a state machine.
    class YieldExpression : public Expression
    {
    public:
        YieldExpression(const Annotations& p_annotations,
                        const std::string& p_machineName,
                        const std::string& p_name);

        // Methods inherited from Expression.
        virtual void Accept(Visitor& p_visitor) const override;
        virtual const TypeImpl& GetType() const override;
        virtual size_t GetNumChildren() const override;

        // Get the name of the yield.
        const std::string& GetName() const;

        // Get the name of machine where this yield expression is originally executed.
        const std::string& GetMachineName() const;

        // Get the name of the yield block that should be invoked.
        const std::string& GetFullName() const;

    private:
        // The name of the yield.
        const std::string m_name;

        // The name of the machine where this yield expression is originally executed.
        const std::string m_machineName;

        // The name of the yield block that should be invoked.
        const std::string m_fullName;
    };
}

#endif
