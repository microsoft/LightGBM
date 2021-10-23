#include "OperatorExpression.h"

#include "FreeForm2Assert.h"
#include "BinaryOperator.h"
#include "UnaryOperator.h"
#include "Visitor.h"
#include "TypeUtil.h"
#include <sstream>


FreeForm2::UnaryOperatorExpression::UnaryOperatorExpression(const Annotations& p_annotations,
                                                            const Expression& p_child, 
                                                            UnaryOperator::Operation p_op)
    : Expression(p_annotations),
      m_child(p_child), 
      m_op(p_op),
      m_type(InferType())
{
}


void
FreeForm2::UnaryOperatorExpression::Accept(Visitor& p_visitor) const
{
    size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(*this))
    {
        m_child.Accept(p_visitor);
        p_visitor.Visit(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}


const FreeForm2::TypeImpl&
FreeForm2::UnaryOperatorExpression::GetType() const
{
    return m_type;
}


const FreeForm2::TypeImpl&
FreeForm2::UnaryOperatorExpression::InferType() const
{
    const TypeImpl& operandType = UnaryOperator::GetBestOperandType(m_op, m_child.GetType());
    if (!operandType.IsValid())
    {
        std::ostringstream err;
        err << "Invalid operand type " << m_child.GetType()
            << " supplied to operator";
        throw ParseError(err.str(), GetSourceLocation());
    }

    const TypeImpl& type = UnaryOperator::GetReturnType(m_op, operandType);
    FF2_ASSERT(type.IsValid());
    return type.AsConstType();
}


size_t
FreeForm2::UnaryOperatorExpression::GetNumChildren() const
{
    return 1;
}


FreeForm2::BinaryOperatorExpression::BinaryOperatorExpression(const Annotations& p_annotations,
                                                              const std::vector<const Expression*>& p_children,
                                                              const BinaryOperator::Operation p_binaryOp,
                                                              TypeManager& p_typeManager)
  : Expression(p_annotations),
    m_resultType(NULL),
    m_childType(NULL), 
    m_numChildren(p_children.size()),
    m_binaryOp(p_binaryOp)
{
    // Note that we rely on this ctor not throwing exceptions during
    // allocation below.

    // We rely on our allocator to size this object to be big enough to
    // hold all children, and enforce this forcing construction via Alloc.
    FF2_ASSERT(m_numChildren >= 2);
    for (size_t i = 0; i < p_children.size(); i++)
    {
        m_children[i] = p_children[i];
    }

    // Infer the child and return types.
    m_childType = &InferChildType(p_typeManager);
    m_resultType = &BinaryOperator::GetResultType(m_binaryOp, *m_childType);

    m_valueBounds = ValueBounds(*m_resultType);
}


FreeForm2::BinaryOperatorExpression::BinaryOperatorExpression(const Annotations& p_annotations,
                                                              const Expression& p_leftChild, 
                                                              const Expression& p_rightChild,
                                                              const BinaryOperator::Operation p_binaryOp,
                                                              TypeManager& p_typeManager)
  : Expression(p_annotations),
    m_resultType(NULL),
    m_childType(NULL), 
    m_numChildren(2),
    m_binaryOp(p_binaryOp)
{
    // We rely on our allocator to size this object to be big enough to
    // hold all children, and enforce this forcing construction via Alloc.
    m_children[0] = &p_leftChild;
    m_children[1] = &p_rightChild;

    // Infer the child and return types.
    m_childType = &InferChildType(p_typeManager);
    m_resultType = &BinaryOperator::GetResultType(m_binaryOp, *m_childType);
}


void
FreeForm2::BinaryOperatorExpression::Accept(Visitor& p_visitor) const
{
    size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(*this))
    {
        for (size_t i = 0; i < m_numChildren; i++)
        {
            m_children[i]->Accept(p_visitor);
        }

        p_visitor.Visit(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}


const FreeForm2::TypeImpl&
FreeForm2::BinaryOperatorExpression::InferChildType(TypeManager& p_typeManager) const
{
    const TypeImpl* unifiedType(&TypeImpl::GetUnknownType());

    for (size_t i = 0; i < m_numChildren; i++)
    {
        const TypeImpl& type 
            = TypeUtil::Unify(m_children[i]->GetType(), *unifiedType, p_typeManager, false, true);

        if (!type.IsValid())
        {
            std::ostringstream err;
            err << "Arguments to binary operators (except index) "
                << "are expected to be of a unifiable type.  The first "
                << i << " elements are of type '" << *unifiedType
                << "', element " << i + 1 << " is of type '" 
                << m_children[i]->GetType() << "'";
            throw ParseError(err.str(), GetSourceLocation());
        }
        unifiedType = &type;
    }

    if (!unifiedType->IsLeafType() && unifiedType->Primitive() != Type::Unknown)
    {
        std::ostringstream err;
        err << "Expected fixed-size type; got type: "
            << unifiedType;
        throw ParseError(err.str(), GetSourceLocation());
    }

    const TypeImpl& childType = BinaryOperator::GetBestOperandType(m_binaryOp, *unifiedType);

    if (!childType.IsValid())
    {
        std::ostringstream err;
        err << "Invalid operand type " << *unifiedType
            << " supplied to operator";
        throw ParseError(err.str(), GetSourceLocation());
    }

    return childType;
}


const FreeForm2::TypeImpl&
FreeForm2::BinaryOperatorExpression::GetType() const
{
    return *m_resultType;
}


const FreeForm2::TypeImpl&
FreeForm2::BinaryOperatorExpression::GetChildType() const
{
    return *m_childType;
}


size_t
FreeForm2::BinaryOperatorExpression::GetNumChildren() const
{
    return m_numChildren;
}


const FreeForm2::Expression* const*
FreeForm2::BinaryOperatorExpression::GetChildren() const
{
    return m_children;
}


FreeForm2::BinaryOperator::Operation
FreeForm2::BinaryOperatorExpression::GetOperator() const
{
    return m_binaryOp;
}


boost::shared_ptr<FreeForm2::BinaryOperatorExpression> 
FreeForm2::BinaryOperatorExpression::Alloc(const Annotations& p_annotations,
                                           const std::vector<const Expression*>& p_children, 
                                           const BinaryOperator::Operation p_binaryOp,
                                           TypeManager& p_typeManager)
{
    size_t bytes = sizeof(BinaryOperatorExpression) 
        + (p_children.size() - 1) * sizeof(Expression*);

    // Allocate a shared_ptr that deletes an BinaryOperatorExpression 
    // allocated in a char[].
    boost::shared_ptr<BinaryOperatorExpression> exp(new (new char[bytes]) 
        BinaryOperatorExpression(p_annotations, p_children, p_binaryOp, p_typeManager), DeleteAlloc);
    return exp;
}


boost::shared_ptr<FreeForm2::BinaryOperatorExpression> 
FreeForm2::BinaryOperatorExpression::Alloc(const Annotations& p_annotations,
                                           const Expression& p_leftChild,
                                           const Expression& p_rightChild,
                                           const BinaryOperator::Operation p_binaryOp,
                                           TypeManager& p_typeManager)
{
    size_t bytes = sizeof(BinaryOperatorExpression) + sizeof(Expression*);

    // Allocate a shared_ptr that deletes an BinaryOperatorExpression 
    // allocated in a char[].
    boost::shared_ptr<BinaryOperatorExpression> exp(new (new char[bytes]) 
        BinaryOperatorExpression(p_annotations, p_leftChild, p_rightChild, p_binaryOp, p_typeManager), 
            DeleteAlloc);
    return exp;
}


void 
FreeForm2::BinaryOperatorExpression::DeleteAlloc(BinaryOperatorExpression* p_allocated)
{
    // Manually call dtor for operator expression.
    p_allocated->~BinaryOperatorExpression();

    // Dispose of memory, which we allocated in a char[].
    char* mem = reinterpret_cast<char*>(p_allocated);
    delete[] mem;
}


