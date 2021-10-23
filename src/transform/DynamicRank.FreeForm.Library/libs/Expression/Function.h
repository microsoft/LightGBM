#pragma once

#include "Expression.h"
#include "FeatureSpec.h"

namespace FreeForm2
{
    class FunctionType;
    class VariableRefExpression;

    // Function expression represents the declaration of a user-defined function.
    // Formal parameters, together with implicit feature parameters, are assigned
    // during parse time for both function declarations and function calls.
    class FunctionExpression : public Expression
    {
    public:
        // A struct that holds the actual function parameter ref and the feature name
        // metadata, to be able to bind these parameters by name.
        struct Parameter
        {
            const VariableRefExpression* m_parameter;
            bool m_isFeatureParameter;
            FeatureSpecExpression::FeatureName m_featureName;
        };

        // Create an function expression.
        FunctionExpression(const Annotations& p_annotations,
                           const FunctionType& p_type,
                           const std::string& p_name,
                           const std::vector<Parameter>& p_parameters,
                           const Expression& p_body);

        // Accessor methods.
        const FunctionType& GetFunctionType() const;
        const std::string& GetName() const;
        size_t GetNumParameters() const;
        const Expression& GetBody() const;
        const std::vector<Parameter>& GetParameters() const;

        // Methods inherited from Expression.
        virtual void Accept(Visitor&) const override;
        virtual size_t GetNumChildren() const override;
        virtual const TypeImpl& GetType() const override;

    private:

        // The function type.
        const FunctionType& m_type;

        // The function name.
        const std::string m_name;
        
        // The expression that, when evaluated, computes the return value of the
        // function.
        const Expression& m_body;

        // A list of parameters.
        std::vector<Parameter> m_parameters;
    };

    // Function call expressions represent the calling of a function, which can be
    // either external or user-defined.
    class FunctionCallExpression : public Expression
    {
    public:
        // Create an function call with an ExternExpression.
        // p_function must have a Function type.
        static boost::shared_ptr<FunctionCallExpression>
        Alloc(const Annotations& p_annotations,
              const Expression& p_function,
              const std::vector<const Expression*>& p_parameters);

        // Get the function type.
        const FunctionType& GetFunctionType() const;

        // Get the function.
        const Expression& GetFunction() const;

        size_t GetNumParameters() const;
        const Expression* const* GetParameters() const;

        // Methods inherited from Expression.
        virtual void Accept(Visitor&) const override;
        virtual size_t GetNumChildren() const override;
        virtual const TypeImpl& GetType() const override;

    private:
        // Constructors are private, call Alloc instead.
        FunctionCallExpression(const Annotations& p_annotations,
                               const Expression& p_function,
                               const std::vector<const Expression*>& p_parameters);

        // The function type.
        const FunctionType* m_type; 
        
        // The expression that, when evaluated, becomes a callable expression of type
        // Function.
        const Expression& m_function;

        // Number of parameters of this node.
        size_t m_numParameters;

        // Array of parameters of this node, allocated using struct hack.
        static void DeleteAlloc(FunctionCallExpression* p_allocated);
        const Expression* m_parameters[1];
    };

    // The Return expression causes execution to exit out of a function,
    // passing a value back to the caller.
    class ReturnExpression : public Expression
    {
    public:
        // Create a ReturnExpression that returns the value of an expression.
        ReturnExpression(const Annotations& p_annotations,
                         const Expression& p_value);

        // Methods inherited from Expression
        virtual void Accept(Visitor&) const override;
        virtual size_t GetNumChildren() const override;
        virtual const TypeImpl& GetType() const override;

        // Return the value of the expression to be returned by this 
        // expression.
        const Expression& GetValue() const;
    private:

        // The expression to be returned.
        const Expression& m_value;
    };
}
