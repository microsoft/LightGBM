#include "ExpressionFactory.h"

#include <sstream>

const FreeForm2::Expression& 
FreeForm2::ExpressionFactory::Create(const ProgramParseState::ExpressionParseState& p_state, 
                                     SimpleExpressionOwner& p_owner,
                                     TypeManager& p_typeManager) const
{
    std::pair<unsigned int, unsigned int> arity = Arity();
    if (p_state.m_children.size() >= arity.first 
        && p_state.m_children.size() <= arity.second)
    {
        return CreateExpression(p_state, p_owner, p_typeManager);
    }
    else
    {
        // Incorrect arity, throw exception.
        std::ostringstream err;
        err << "Arity of " << std::string(SIZED_STR(p_state.m_atom)) << " was "
            << p_state.m_children.size() << " but was expected to be in range [" 
            << Arity().first << ", "
            << Arity().second << "]";
        throw std::runtime_error(err.str());
    }
}
