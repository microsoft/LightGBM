#pragma once

#include "Expression.h"

namespace FreeForm2
{
    // The Phi node marks places in the code where the value of a variable can
    // change in different branches of the code. This node should be ignored
    // by the backends since it doesn't affect the compiled output.
    //
    // The incoming array refers to the list of variable versions that can reach
    // this point of the code.
    class PhiNodeExpression : public Expression
    {
    public:
        static boost::shared_ptr<PhiNodeExpression>
        Alloc(const Annotations& p_annotations,
              size_t p_version,
              size_t p_incomingVersionsCount,
              const size_t* p_incomingVersions);

        // Methods inherited from Expression
        virtual void Accept(Visitor&) const override;
        virtual size_t GetNumChildren() const override;
        virtual const TypeImpl& GetType() const override;

        // Getter methods.
        size_t GetVersion() const;
        size_t GetIncomingVersionsCount() const;
        const size_t* GetIncomingVersions() const;
    
    private:
        // Create a PhiNode expression.
        PhiNodeExpression(const Annotations& p_annotations,
                          size_t p_version,
                          size_t p_incomingVersionsCount,
                          const size_t* p_incomingVersions);

        static void DeleteAlloc(PhiNodeExpression* p_allocated);

        size_t m_version;

        size_t m_incomingVersionsCount;

        // Allocated using the struct hack.
        size_t m_incomingVersions[1];
    };
}
