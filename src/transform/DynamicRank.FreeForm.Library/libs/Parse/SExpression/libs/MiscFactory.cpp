#include "MiscFactory.h"

#include <boost/make_shared.hpp>
#include "ArrayLength.h"
#include "Conditional.h"
#include "ConvertExpression.h"
#include "Expression.h"
#include "ExpressionFactory.h"
#include "FeatureSpec.h"
#include "FreeForm2Assert.h"
#include "SimpleExpressionOwner.h"
#include "SelectNth.h"
#include "TypeUtil.h"
#include "RandExpression.h"
#include <sstream>

using namespace FreeForm2;

namespace FreeForm2
{
    class ConditionalExpressionFactory : public ExpressionFactory
    {
    public:
        ConditionalExpressionFactory()
        {
        }

    private:
        virtual 
        const Expression& 
        CreateExpression(const ProgramParseState::ExpressionParseState& p_state, 
                         SimpleExpressionOwner& p_owner,
                         TypeManager& p_typeManager) const override
        {
            boost::shared_ptr<Expression> expr(
                new ConditionalExpression(Annotations(SourceLocation(1, p_state.m_offset)),
                                          *p_state.m_children[0], 
                                          *p_state.m_children[1], 
                                          *p_state.m_children[2]));
            p_owner.AddExpression(expr);
            return *expr;
        }


        virtual std::pair<unsigned int, unsigned int> Arity() const override
        {
            return std::make_pair(3, 3);
        }
    };

    static const ConditionalExpressionFactory c_cond;

    class ArrayLengthExpressionFactory : public ExpressionFactory
    {
    public:
        ArrayLengthExpressionFactory()
        {
        }

    private:
        // Create array-length expression from results of type-checking and
        // accumulated children.
        virtual 
        const Expression& 
        CreateExpression(const ProgramParseState::ExpressionParseState& p_state, 
                         SimpleExpressionOwner& p_owner,
                         TypeManager& p_typeManager) const override
        {
            boost::shared_ptr<Expression> expr(new ArrayLengthExpression(Annotations(SourceLocation(1, p_state.m_offset)),
                                                                         *p_state.m_children[0]));
            p_owner.AddExpression(expr);
            return *expr;
        }


        // Indicate what the min/max arity (number of arguments) to the
        // array-length expression is (both are one, as array-length takes a
        // single array).
        virtual std::pair<unsigned int, unsigned int> 
        Arity() const override
        {
            return std::make_pair(1, 1);
        }
    };

    // Singleton instance of the array-length expression factory.
    static const ArrayLengthExpressionFactory c_arrayLengthFactory;

    // Base class for numeric conversions, which take a single numeric argument.
    class NumericConversionFactory : public ExpressionFactory
    {
    private:
        virtual std::pair<unsigned int, unsigned int> Arity() const override
        {
            return std::make_pair(1, 1);
        }
    };

    // Factory that accepts a single child expresion and upconverts
    // it to a float: useful as a root for parsing, as the freeform language
    // implicitly returns a float.
    class FloatConversionFactory : public NumericConversionFactory
    {
    private:
        virtual 
        const Expression& 
        CreateExpression(const ProgramParseState::ExpressionParseState& p_state, 
                         SimpleExpressionOwner& p_owner,
                         TypeManager& p_typeManager) const override
        {
            // Convert expression to floating point.
            boost::shared_ptr<Expression> convert(new ConvertToFloatExpression(Annotations(SourceLocation(1, p_state.m_offset)),
                                                                               *p_state.m_children[0]));
            p_owner.AddExpression(convert);
            return *convert;
        }
    };

    // Factory that accepts a single child expresion and implicitly upconverts
    // it to a float: useful as a root for parsing, as the freeform language
    // implicitly returns a float.
    class IntConversionFactory : public NumericConversionFactory
    {
    private:
        virtual 
        const Expression& 
        CreateExpression(const ProgramParseState::ExpressionParseState& p_state, 
                         SimpleExpressionOwner& p_owner,
                         TypeManager& p_typeManager) const override
        {
            // Convert expression to int.
            boost::shared_ptr<Expression> convert(new ConvertToIntExpression(Annotations(SourceLocation(1, p_state.m_offset)),
                                                                             *p_state.m_children[0]));
            p_owner.AddExpression(convert);

            return *convert;
        }
    };

    // Factory that accepts a single child expresion and truncates
    // it to a bool.
    class BoolConversionFactory : public NumericConversionFactory
    {
    private:
        virtual 
        const Expression& 
        CreateExpression(const ProgramParseState::ExpressionParseState& p_state, 
                         SimpleExpressionOwner& p_owner,
                         TypeManager& p_typeManager) const override
        {
            // Convert expression to int.
            boost::shared_ptr<Expression> convert(new ConvertToBoolExpression(Annotations(SourceLocation(1, p_state.m_offset)),
                                                                              *p_state.m_children[0]));
            p_owner.AddExpression(convert);

            return *convert;
        }
    };

    // Factory that accepts a single child expression and then simply 
    // returns it - useful for parsing.
    class IdentityExpressionFactory : public ExpressionFactory
    {
    private:
        virtual 
        const Expression& 
        CreateExpression(const ProgramParseState::ExpressionParseState& p_state, 
                         SimpleExpressionOwner& p_owner,
                         TypeManager& p_typeManager) const override
        {
            return *p_state.m_children[0];
        }

        virtual std::pair<unsigned int, unsigned int> Arity() const override
        {
            return std::make_pair(1, 1);
        }
    };

    static const FloatConversionFactory c_floatFactory;
    static const IntConversionFactory c_intFactory;
    static const BoolConversionFactory c_boolFactory;
    static const IdentityExpressionFactory c_identityFactory;

    class SelectNthExpressionFactory : public ExpressionFactory
    {
    public:
        SelectNthExpressionFactory()
        {
        }

    private:
        virtual 
        const Expression& 
        CreateExpression(const ProgramParseState::ExpressionParseState& p_state, 
                         SimpleExpressionOwner& p_owner,
                         TypeManager& p_typeManager) const override
        {
            boost::shared_ptr<Expression> expr
                = SelectNthExpression::Alloc(Annotations(SourceLocation(1, p_state.m_offset)),
                                             p_state.m_children);
            p_owner.AddExpression(expr);
            return *expr;
        }


        virtual std::pair<unsigned int, unsigned int> Arity() const override
        {
            return std::make_pair(2, UINT_MAX);
        }
    };

    class SelectRangeExpressionFactory : public ExpressionFactory
    {
    public:
        SelectRangeExpressionFactory()
        {
        }

    private:
        virtual 
        const Expression& 
        CreateExpression(const ProgramParseState::ExpressionParseState& p_state, 
                         SimpleExpressionOwner& p_owner,
                         TypeManager& p_typeManager) const override
        {
            boost::shared_ptr<Expression> expr(
                new SelectRangeExpression(Annotations(SourceLocation(1, p_state.m_offset)),
                                          *p_state.m_children[0],
                                          *p_state.m_children[1],
                                          *p_state.m_children[2],
                                          p_typeManager));
            p_owner.AddExpression(expr);
            return *expr;
        }


        virtual std::pair<unsigned int, unsigned int> Arity() const override
        {
            return std::make_pair(3, 3);
        }
    };

    static const SelectNthExpressionFactory c_selectNth;
    static const SelectRangeExpressionFactory c_selectRange;

    // This class creates a derived feature specification expression. It will
    // optionally convert the result value to float before wrapping. The 
    // feature specification expression should be the root of the expression 
    // tree.
    class FeatureSpecExpressionFactory : public ExpressionFactory
    {
    public:
        FeatureSpecExpressionFactory(bool p_mustProduceFloat)
            : m_produceFloat(p_mustProduceFloat)
        {
        }

    private:
        const bool m_produceFloat;

        virtual 
        const Expression& 
        CreateExpression(const ProgramParseState::ExpressionParseState& p_state, 
                         SimpleExpressionOwner& p_owner,
                         TypeManager& p_typeManager) const override
        {
            const Expression* body = p_state.m_children[0];

            if (m_produceFloat)
            {
                boost::shared_ptr<Expression> convert(
                    TypeUtil::Convert(*p_state.m_children[0], Type::Float));
                p_owner.AddExpression(convert);
                body = convert.get();
            }

            // Create the DerivedFeatureSpecExpression. "Feature" is a generic
            // name (FreeForm2 expressions are anonymous). The (NULL, 0) pair 
            // signifies that the expression does not take special parameters.
            boost::shared_ptr<FeatureSpecExpression::PublishFeatureMap> featureMap =
                boost::make_shared<FeatureSpecExpression::PublishFeatureMap>();
            featureMap->emplace(FeatureSpecExpression::FeatureName("Feature"), body->GetType());
            
            boost::shared_ptr<Expression> expr(
                new FeatureSpecExpression(Annotations(SourceLocation(1, p_state.m_offset)),
                                          featureMap, 
                                          *body, 
                                          FeatureSpecExpression::AggregatedDerivedFeature,
                                          true));
            p_owner.AddExpression(expr);
            return *expr;
        }


        virtual std::pair<unsigned int, unsigned int> Arity() const override
        {
            return std::make_pair(1, 1);
        }
    };

    static const FeatureSpecExpressionFactory c_featureSpec(false);
    static const FeatureSpecExpressionFactory c_floatFeatureSpec(true);

    // Factory that returns the instance of the RandFloatExpression singleton
    // for the S-Expression language parser.
    class RandomFloatFactory : public ExpressionFactory
    {
    public:
        RandomFloatFactory()
        {
        }

    private:
        virtual
        const Expression&
        CreateExpression(const ProgramParseState::ExpressionParseState& p_state,
                         SimpleExpressionOwner& p_owner,
                         TypeManager& p_typeManager) const override
        {
            return RandFloatExpression::GetInstance();
        }

        virtual std::pair<unsigned int, unsigned int> Arity() const override
        {
            return std::make_pair(0, 0);
        }
    };


    // Factory that returns an instance of the RandIntExpression.
    class RandomIntFactory : public ExpressionFactory
    {
    public:
        RandomIntFactory()
        {
        }

    private:
        virtual
        const Expression&
        CreateExpression(const ProgramParseState::ExpressionParseState& p_state,
                         SimpleExpressionOwner& p_owner,
                         TypeManager& p_typeManager) const override
        {
            boost::shared_ptr<Expression> randomInteger(new RandIntExpression(Annotations(SourceLocation(1, p_state.m_offset)),
                                                                              *p_state.m_children[0],
                                                                              *p_state.m_children[1]));
            p_owner.AddExpression(randomInteger);
            return *randomInteger;
        }

        virtual std::pair<unsigned int, unsigned int> Arity() const override
        {
            return std::make_pair(2, 2);
        }
    };

    static const RandomFloatFactory c_randFloatFactory;
    static const RandomIntFactory c_randIntFactory;
}
 

const FreeForm2::ExpressionFactory& 
FreeForm2::GetArrayLengthInstance()
{
    return c_arrayLengthFactory;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Conditional::GetIfInstance()
{
    return c_cond;
}


const FreeForm2::ExpressionFactory&
FreeForm2::Convert::GetFloatConvertFactory()
{
    return c_floatFactory;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Convert::GetIntConvertFactory()
{
    return c_intFactory;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Convert::GetBoolConversionFactory()
{
    return c_boolFactory;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Convert::GetIdentityFactory()
{
    return c_identityFactory;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Select::GetSelectNthInstance()
{
    return c_selectNth;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Select::GetSelectRangeInstance()
{
    return c_selectRange;
}


const FreeForm2::ExpressionFactory&
FreeForm2::GetFeatureSpecInstance(bool p_mustConvertToFloat)
{
    if (p_mustConvertToFloat)
    {
        return c_floatFeatureSpec;
    }
    else
    {
        return c_featureSpec;
    }
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Random::GetRandomFloatInstance()
{
    return c_randFloatFactory;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Random::GetRandomIntInstance()
{
    return c_randIntFactory;
}
