#pragma once

#ifndef FREEFORM2_EXTERN_H
#define FREEFORM2_EXTERN_H

#include "Expression.h"
#include <string>

namespace FreeForm2
{
    class TypeManager;
    class ExternalData;

    // Extern expressions represent an external data member declared in the
    // program. External data is associated with a member of the 
    // ExternalData::DataType enum.
    class ExternExpression : public Expression
    {
    public:
        // Opaque structure for resolving external data objects. These objects
        // are not currently supported by the API and must be built into the
        // compiler.
        struct BuiltInObject;
        static const BuiltInObject& c_numberOfTuplesCommonObject;
        static const BuiltInObject& c_numberOfTuplesCommonNoDuplicateObject;
        static const BuiltInObject& c_numberOfTuplesInTriplesCommonObject;
        static const BuiltInObject& c_alterationAndTermWeightObject;
        static const BuiltInObject& c_alterationWeightObject;
        static const BuiltInObject& c_trueNearDoubleQueueObject;
        static const BuiltInObject& c_boundedQueueObject;

        // Find a BuiltInObject by name. This function returns nullptr if not
        // object exists for the name.
        static const BuiltInObject* GetObjectByName(const std::string& p_name);
        static const ExternalData& GetObjectData(const BuiltInObject& p_object);

        // Create an external data reference for the specified piece of data
        // of a declared type.
        ExternExpression(const Annotations& p_annotations,
                         const ExternalData& p_data,
                         const TypeImpl& p_declaredType,
                         VariableID p_id,
                         TypeManager& p_typeManager);

        // Create an external data reference for a basic type external data 
        // member.
        ExternExpression(const Annotations& p_annotations,
                         const ExternalData& p_data,
                         const TypeImpl& p_declaredType);

        // Create an extern expression for an object.
        ExternExpression(const Annotations& p_annotations,
                         const BuiltInObject& p_object,
                         VariableID p_id,
                         TypeManager& p_typeManager);

        // Methods inherited from Expression.
        virtual void Accept(Visitor&) const override;
        virtual size_t GetNumChildren() const override;
        virtual const TypeImpl& GetType() const override;
        virtual bool IsConstant() const override;
        virtual ConstantValue GetConstantValue() const override;

        // Return the data enum entry associated with this extern.
        const ExternalData& GetData() const;

        // Get the array allocation ID associated with this extern. If this
        // extern is not an array type, the return value is not defined.
        VariableID GetId() const;
        
    private:
        // Name of the extern variable.
        const ExternalData& m_data;

        // Allocation ID for the array for this extern, if applicable.
        const VariableID m_id;
    };
}

#endif
