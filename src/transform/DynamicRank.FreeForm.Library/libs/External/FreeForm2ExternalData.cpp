#include "FreeForm2ExternalData.h"

#include "FreeForm2Type.h"
#include <string>
#include "TypeImpl.h"
#include "TypeManager.h"

FreeForm2::ExternalData::ExternalData(const std::string& p_name, const FreeForm2::TypeImpl& p_typeImpl)
    : m_name(p_name), m_type(&p_typeImpl), m_isCompileTimeConst(false)
{
}


FreeForm2::ExternalData::ExternalData(const std::string& p_name, 
                                     const TypeImpl& p_typeImpl,
                                     ConstantValue p_value)
    : m_name(p_name), m_type(&p_typeImpl), m_isCompileTimeConst(true), m_constantValue(p_value)
{
}


FreeForm2::ExternalData::~ExternalData()
{
}


const std::string&
FreeForm2::ExternalData::GetName() const
{
    return m_name;
}


const FreeForm2::TypeImpl&
FreeForm2::ExternalData::GetType() const
{
    return *m_type;
}


bool
FreeForm2::ExternalData::IsCompileTimeConstant() const
{
    return m_isCompileTimeConst;
}


FreeForm2::ConstantValue
FreeForm2::ExternalData::GetCompileTimeValue() const
{
    return m_constantValue;
}


FreeForm2::ExternalDataManager::ExternalDataManager()
    : m_typeFactory(new TypeFactory(TypeManager::CreateTypeManager()))
{
}


FreeForm2::ExternalDataManager::~ExternalDataManager()
{
}


FreeForm2::TypeFactory&
FreeForm2::ExternalDataManager::GetTypeFactory()
{
    return *m_typeFactory;
}
