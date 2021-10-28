/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#pragma once

#ifndef FREEFORM2_MEMBER_ACCESS_EXPRESSION_H
#define FREEFORM2_MEMBER_ACCESS_EXPRESSION_H

#include "Expression.h"
#include "CompoundType.h"

namespace FreeForm2
{
    // An array-dereference expression removes a dimension from an array.
    class MemberAccessExpression : public Expression
    {
    public:
        MemberAccessExpression(const Annotations &p_annotations,
                               const Expression &p_struct,
                               const CompoundType::Member &p_memberInfo,
                               size_t p_version);

        // Methods inherited from Expression.
        virtual size_t GetNumChildren() const override;
        virtual void Accept(Visitor &p_visitor) const override;
        virtual void AcceptReference(Visitor &p_visitor) const override;
        virtual const TypeImpl &GetType() const override;
        virtual bool IsConstant() const override;
        virtual ConstantValue GetConstantValue() const override;

        const Expression &GetStruct() const;
        const CompoundType::Member &GetMemberInfo() const;
        size_t GetVersion() const;

    private:
        // Struct expression whose member will be accessed.
        const Expression &m_struct;

        // Member information.
        const CompoundType::Member &m_memberInfo;

        // A unique version number associated with a particular
        // value for this variable.
        const size_t m_version;
    };

    // An unresolved member to be accessed.
    class UnresolvedAccessExpression : public Expression
    {
    public:
        UnresolvedAccessExpression(const Annotations &p_annotations,
                                   const Expression &p_object,
                                   const std::string &p_memberName,
                                   const TypeImpl &p_expectedType);

        // Methods inherited from Expression.
        virtual size_t GetNumChildren() const override;
        virtual void Accept(Visitor &p_visitor) const override;
        virtual void AcceptReference(Visitor &p_visitor) const override;
        virtual const TypeImpl &GetType() const override;

        const Expression &GetObject() const;
        const std::string &GetMemberName() const;

    private:
        // Struct expression whose member will be accessed.
        const Expression &m_object;

        // Member information.
        std::string m_memberName;

        // The expected type of the member access expression.
        const TypeImpl &m_expectedType;
    };
};

#endif
