#include "Arithmetic.h"

#include "BinaryOperator.h"
#include "OperatorExpressionFactory.h"
#include "UnaryOperator.h"
#include "TypeUtil.h"

using namespace FreeForm2;

namespace
{
    // This class adds ConvertToIntExpressions where appropriate to truncate
    // all operands before passing them to the OperatorExpressionFactory.
    class TruncatingOperatorFactory : public ExpressionFactory
    {
    public:
        TruncatingOperatorFactory(UnaryOperator::Operation p_unaryOp,
                                  BinaryOperator::Operation p_binaryOp)
            : m_factory(p_unaryOp, p_binaryOp, false)
        {
        }

    private:
        // Internal factory to create the operator expression.
        OperatorExpressionFactory m_factory;

        virtual 
        const Expression& 
        CreateExpression(const ProgramParseState::ExpressionParseState& p_state, 
                         SimpleExpressionOwner& p_owner,
                         TypeManager& p_typeManager) const override
        {
            ProgramParseState::ExpressionParseState opState(m_factory, p_state.m_atom, p_state.m_offset);

            for (size_t i = 0; i < p_state.m_children.size(); i++)
            {
                if (p_state.m_children[i]->GetType().Primitive() == Type::Float)
                {
                    boost::shared_ptr<Expression> expr(
                        TypeUtil::Convert(*p_state.m_children[i], Type::Int));
                    p_owner.AddExpression(expr);
                    opState.Add(*expr);
                }
                else
                {
                    opState.Add(*p_state.m_children[i]);
                }
            }
            opState.m_variableIds.insert(opState.m_variableIds.begin(), 
                                         p_state.m_variableIds.begin(),
                                         p_state.m_variableIds.end());
            return opState.Finish(p_owner, p_typeManager);
        };

        
        virtual 
        std::pair<unsigned int, unsigned int> 
        Arity() const override
        {
            return std::make_pair(1, 2);
        }
    };

    typedef OperatorExpressionFactory OperatorFactory;
    static const OperatorFactory c_plusFactory(UnaryOperator::invalid, 
                                               BinaryOperator::plus, 
                                               true);
    static const OperatorFactory c_minusFactory(UnaryOperator::minus, 
                                                BinaryOperator::minus, 
                                                false);
    static const OperatorFactory c_mulFactory(UnaryOperator::invalid, 
                                              BinaryOperator::multiply,
                                              false);
    static const OperatorFactory c_divFactory(UnaryOperator::invalid, 
                                              BinaryOperator::divides, 
                                              false);
    static const OperatorFactory c_modFactory(UnaryOperator::invalid, 
                                              BinaryOperator::mod, 
                                              false);
    static const OperatorFactory c_maxFactory(UnaryOperator::invalid, 
                                              BinaryOperator::max, 
                                              false);
    static const OperatorFactory c_minFactory(UnaryOperator::invalid, 
                                              BinaryOperator::min, 
                                              false);
    static const OperatorFactory c_powFactory(UnaryOperator::invalid, 
                                              BinaryOperator::pow, 
                                              false);
    static const OperatorFactory c_unaryLogFactory(UnaryOperator::log, 
                                                   BinaryOperator::invalid, 
                                                   false);
    static const OperatorFactory c_binaryLogFactory(UnaryOperator::invalid, 
                                                    BinaryOperator::log, 
                                                    false);
    static const OperatorFactory c_log1Factory(UnaryOperator::log1, 
                                               BinaryOperator::invalid, 
                                               false);
    static const OperatorFactory c_absFactory(UnaryOperator::abs, 
                                              BinaryOperator::invalid, 
                                              false);
    static const OperatorFactory c_roundFactory(UnaryOperator::round,
                                                BinaryOperator::invalid,
                                                false);
    static const OperatorFactory c_truncFactory(UnaryOperator::trunc,
                                                BinaryOperator::invalid,
                                                false);
    static const TruncatingOperatorFactory c_intDivFactory(UnaryOperator::invalid,
                                                           BinaryOperator::divides);
    static const TruncatingOperatorFactory c_intModFactory(UnaryOperator::invalid,
                                                           BinaryOperator::mod);
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Arithmetic::GetPlusInstance()
{
    return c_plusFactory;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Arithmetic::GetMinusInstance()
{
    return c_minusFactory;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Arithmetic::GetMultiplyInstance()
{
    return c_mulFactory;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Arithmetic::GetDividesInstance()
{
    return c_divFactory;
}


const FreeForm2::ExpressionFactory&
FreeForm2::Arithmetic::GetIntegerDivInstance()
{
    return c_intDivFactory;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Arithmetic::GetIntegerModInstance()
{
    return c_intModFactory;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Arithmetic::GetModInstance()
{
    return c_modFactory;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Arithmetic::GetMaxInstance()
{
    return c_maxFactory;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Arithmetic::GetMinInstance()
{
    return c_minFactory;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Arithmetic::GetPowInstance()
{
    return c_powFactory;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Arithmetic::GetUnaryLogInstance()
{
    return c_unaryLogFactory;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Arithmetic::GetBinaryLogInstance()
{
    return c_binaryLogFactory;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Arithmetic::GetLog1Instance()
{
    return c_log1Factory;
}


const FreeForm2::ExpressionFactory& 
FreeForm2::Arithmetic::GetAbsInstance()
{
    return c_absFactory;
}


const FreeForm2::ExpressionFactory&
FreeForm2::Arithmetic::GetRoundInstance()
{
    return c_roundFactory;
}


const FreeForm2::ExpressionFactory&
FreeForm2::Arithmetic::GetTruncInstance()
{
    return c_truncFactory;
}
