#pragma once
#ifndef FREEFORM2_INC_EXTERNAL_DATA_H
#define FREEFORM2_INC_EXTERNAL_DATA_H

#include <boost/scoped_ptr.hpp>
#include "FreeForm2Features.h"
#include "FreeForm2Type.h"
#include "FreeForm2Result.h"
#include <string>
#include <utility>

namespace FreeForm2
{
    class TypeManager;
    
    // This class represents a piece of data, which is defined externally from
    // the perspective of a program compiled by this library.
    class ExternalData
    {
    public:
        // Creates a run-time constant external data member.
        ExternalData(const std::string& p_name, const TypeImpl& p_typeImpl);

        // Creates a compile-time constant external data member.
        ExternalData(const std::string& p_name, const TypeImpl& p_typeImpl, ConstantValue p_value);

        // Force subclassing of this class. This serves two purposes: it 
        // encounrages implementors to add backend-specific data to the
        // ExternalData class to group resource storage, and it disallows the
        // library to copy ExternalData in case the client has subclassed
        // ExternalData.
        virtual ~ExternalData() = 0;

        // Get the name of the external data.
        const std::string& GetName() const;

        // Get the type of the external data.
        const TypeImpl& GetType() const;

        // Return a flag that determines if this external data is a 
        // compile-time constant.
        bool IsCompileTimeConstant() const;

        // Return the compile-time constant value of this external data. If
        // this method is not a compile-time constant, this method throws an
        // exception.
        ConstantValue GetCompileTimeValue() const;

    private:
        // The name of the data.
        std::string m_name;

        // The type of the data.
        const TypeImpl* m_type;

        // This flag indicates whether or not this object is a compile-time
        // constant value.
        bool m_isCompileTimeConst;

        // This value is used only if this object is a compile-time constant 
        // and contains the constant value.
        ConstantValue m_constantValue;
    };

    // This class provides a name-to-data mapping for external data members.
    class ExternalDataManager
    {
    public:
        // The default constructor initializes the type factory.
        ExternalDataManager();
        virtual ~ExternalDataManager();

        // This method returns an ExternalData object based on a name. If no
        // data exists for the name, this method returns nullptr.
        virtual const ExternalData* FindData(const std::string& p_name) const = 0;

    protected:
        // Get the type factory for managing the types associated with this 
        // data manager.
        TypeFactory& GetTypeFactory();

    private:
        // The type manager owns the types created with the above methods.
        std::unique_ptr<TypeFactory> m_typeFactory;

        friend class TypeManager;
    };
}

#endif
