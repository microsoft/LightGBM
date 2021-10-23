#pragma once

#ifndef FREEFORM2_FUNCTION_TYPE_H
#define FREEFORM2_FUNCTION_TYPE_H

#include "FreeForm2Type.h"
#include "TypeImpl.h"
#include <vector>

namespace FreeForm2
{
    class TypeManager;

    // A function type is any type which can be called with certain parameters and
    // returns a value of another type.
    class FunctionType : public TypeImpl
    {
    public:
        virtual ~FunctionType() {}

        // Gets the return type of the function.
        const TypeImpl& GetReturnType() const;

        // Iterate over parameters.
        typedef const TypeImpl** ParameterIterator;
        ParameterIterator BeginParameters() const;
        ParameterIterator EndParameters() const;

        // Get the number of parameters.
        size_t GetParameterCount() const;
        
        // Get a string representation of the type.
        virtual const std::string& GetName() const override;
        static std::string GetName(const TypeImpl& p_returnType,
                                   const TypeImpl* const* p_parameterTypes,
                                   size_t p_numParams);

        // Create derived types based on this type.
        virtual const TypeImpl& AsConstType() const override;
        virtual const TypeImpl& AsMutableType() const override;

    private:
        // Create a compound type of the give type, constness, and type 
        // manager.
        FunctionType(TypeManager& p_typeManager,
                     const TypeImpl& p_returnType,
                     const TypeImpl* const* p_parameterTypes,
                     size_t p_numParams);
      
        // Compare subclass type data.
        virtual bool IsSameSubType(const TypeImpl& p_type, bool p_ignoreConst) const override;

        friend class TypeManager;
        
        // The return type of the function.
        const TypeImpl* m_returnType;

        // The string representation of this type.
        mutable std::string m_name;

        // The number of parameters.
        size_t m_numParameters;
        
        // The types of the parameters. This is allocated using the struct hack.
        const TypeImpl* m_parameterTypes[1];
    };
}

#endif
