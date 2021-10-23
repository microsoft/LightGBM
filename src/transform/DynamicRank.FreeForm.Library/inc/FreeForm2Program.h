#pragma once

#ifndef FREEFORM2_INC_PROGRAM_H
#define FREEFORM2_INC_PROGRAM_H

#include <basic_types.h>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/tuple/tuple.hpp>

#include <iosfwd>
#include <memory>
#include <vector>
#include <stdexcept>

namespace DynamicRank
{
    class IFeatureMap;
    class INeuralNetFeatures;
}

namespace FreeForm2
{
    class Allocation;
    class ProgramImpl;
    class Expression;
    class ExternalDataManager;
    class Type;

    // A struct that holds information about the source location in a program.
    struct SourceLocation
    {
        SourceLocation();
        SourceLocation(unsigned int p_lineNo, unsigned int p_lineOffset);

        unsigned int m_lineNo;
        unsigned int m_lineOffset;
    };

    // This class is used to throw errors during parsing.
    class ParseError : public std::runtime_error
    {
    public:
        // Construct an exception with a message and source location.
        ParseError(const std::string& p_message, const SourceLocation& p_sourceLocation);

        // Construct an exception based off an inner exception, with
        // a source location.
        ParseError(const std::exception& p_inner, const SourceLocation& p_sourceLocation);

        // Get the error message, without line information.
        const std::string& GetMessage() const;

        // Get the line number where the parsing error occurred.
        const SourceLocation& GetSourceLocation() const;

    private:
        // The error message, without line information.
        std::string m_message;

        // The source location where the parsing error occurred.
        SourceLocation m_sourceLocation;
    };

    // A program is a representation of a program, that offers functionality in
    // addition to being able to simply execute.
    class Program : boost::noncopyable
    {
    public:
        Program(std::auto_ptr<ProgramImpl> p_impl);

        // Enumeration of available syntaxes.
        enum Syntax
        {
            sexpression,
            /* visage, */
            aggregatedSExpression,
        };

        // Parse a program from a string.  The feature map defines the mapping
        // between names and feature value slots in the p_features array passed
        // to Executable::Evaluate.  p_syntax dictates the syntax used for
        // parsing.  p_mustProduceFloat forces the program to
        // produce a float as final result.  If the result can be sensibly
        // converted to a float (i.e. from an integer) it will be, otherwise an
        // exception will be thrown.  If p_debugOutput is not NULL, debugging
        // information is written to this stream during parsing. The external
        // data manager is an optional argument. If a manager is provided, 
        // extern lookups will be done via this object; otherwise, externs will
        // be disallowed in the program.
        //
        // Note that a reference to the feature map is saved, and that object
        // must persist through the lifetime of the program.
        template <Program::Syntax p_syntax>
        static boost::shared_ptr<Program> Parse(SIZED_STRING p_input, 
                                                DynamicRank::IFeatureMap& p_map,
                                                bool p_mustProduceFloat,
                                                const ExternalDataManager* p_externalData,
                                                std::ostream* p_debugOutput);

        // Get the output type of the program.
        const Type& GetType() const;

        // Get the expression of the program.
        const Expression& GetExpression() const;

        // Provide an interface to extract features used by this program, by
        // calling Process on the provided INeuralNetFeatures object.
        void ProcessFeaturesUsed(DynamicRank::INeuralNetFeatures& p_features) const;

        // Gets the list of all the allocations required by the program.
        const std::vector<boost::shared_ptr<Allocation>>& GetAllocations() const;

        // Implementation accessors.
        ProgramImpl& GetImplementation();
        const ProgramImpl& GetImplementation() const;

    private:
        // Pointer to implementation class (pimpl idiom).
        boost::scoped_ptr<ProgramImpl> m_impl;
    };

    // Explicit instantiation of Program::Parse for all available syntaxes.
    template
    boost::shared_ptr<Program> 
    Program::Parse<Program::sexpression>(SIZED_STRING p_input, 
                                         DynamicRank::IFeatureMap& p_map,
                                         bool p_mustProduceFloat,
                                         const ExternalDataManager* p_externalData,
                                         std::ostream* p_debugOutput);

    template
    boost::shared_ptr<Program> 
    Program::Parse<Program::aggregatedSExpression>(SIZED_STRING p_input, 
                                                   DynamicRank::IFeatureMap& p_map,
                                                   bool p_mustProduceFloat,
                                                   const ExternalDataManager* p_externalData,
                                                   std::ostream* p_debugOutput);
}

#endif
