#pragma once

#ifndef FREEFORM2_FEATURE_SPEC_H
#define FREEFORM2_FEATURE_SPEC_H

#include "Expression.h"
#include "FreeForm2Features.h"
#include <boost/operators.hpp>
#include <boost/shared_ptr.hpp>
#include "SymbolTable.h"
#include <utility>
#include <vector>
#include <map>

namespace FreeForm2
{
    class TypeManager;

    // This class encapsulates a function-like expression (either a feature 
    // specification or an actual function). If features are published within 
    // this spec, the Type should be Void, otherwise, the type is the same as 
    // the returned value.
    class FeatureSpecExpression : public Expression
    {
    public:
        // A class containing information about a feature name, including
        // parameterization.
        class FeatureName : boost::partially_ordered<FeatureName>
        {
        public:
            typedef std::map<std::string, std::string> ParameterMap;
            typedef ParameterMap::value_type Parameter;

            // Construct an empty feature name.
            FeatureName();

            // Construct a feature name without a parameterization.
            explicit FeatureName(const std::string& p_name);

            // Construct a FeatureName was a base name and parameterization.
            FeatureName(const std::string& p_name,
                        const ParameterMap& p_parameters);

            // Parse a feature name from a base name and a parameterization
            // string. The parameterization string may be empty, even if the
            // parameterization flag is true - this yields a parameterized 
            // feature with no parameters, which implies a single feature
            // value exists for the parameter.
            static FeatureName Parse(const std::string& p_name,
                                     bool p_isParameterized,
                                     const std::string& p_parameterization,
                                     const SourceLocation& p_location);

            // Accessor methods.
            const std::string& GetName() const;
            bool IsParameterized() const;
            const ParameterMap& GetParameters() const;
            SymbolTable::Symbol GetSymbol() const;

            // Operators
            bool operator==(const FeatureName& p_other) const;
            bool operator<(const FeatureName& p_other) const;

        private:
            // The name of the feature being imported.
            std::string m_name;

            // The (optional) parameter of the feature being imported.
            ParameterMap m_params;

            // The parameter string for creating a symbol.
            mutable std::string m_paramStr;

            // Whether this feature is parameterized.
            bool m_isParameterized;
        };

        // This structure is a simple functor used to compare feature names,
        // ignoring parameterization. It is useful for publishing features, 
        // when the names themselves must be unique.
        struct IgnoreParameterLess
        {
            bool operator()(const FeatureName& p_left, const FeatureName& p_right) const;
        };
        
        // Mapping of the names of the features being published to their types.
        typedef std::map<FeatureName, const TypeImpl&, IgnoreParameterLess> PublishFeatureMap;

        // The type of the feature specification. These names are against the
        // coding guidelines only to temporarily limit amount of code touched
        // by this CL. To be fixed in TFS ID 472156.
        typedef FeatureInformation::FeatureType FeatureSpecType;
        static const FeatureSpecType MetaStreamFeature = FeatureInformation::MetaStreamFeature;
        static const FeatureSpecType DerivedFeature = FeatureInformation::DerivedFeature;
        static const FeatureSpecType AggregatedDerivedFeature = FeatureInformation::AggregatedDerivedFeature;
        static const FeatureSpecType AbInitioFeature = FeatureInformation::AbInitioFeature;

        // Construct a feature specification.
        FeatureSpecExpression(const Annotations& p_annotations,
                              boost::shared_ptr<PublishFeatureMap> p_publishFeatureMap,
                              const Expression& p_body,
                              FeatureSpecType p_featureSpecType,
                              bool p_returnsValue);

        // Methods inherited from Expression
        virtual void Accept(Visitor&) const override;
        virtual const TypeImpl& GetType() const override;
        virtual size_t GetNumChildren() const override;

        // Accessor methods
        const Expression& GetBody() const;
        bool IsDerived() const;
        FeatureSpecType GetFeatureSpecType() const;
        boost::shared_ptr<PublishFeatureMap> GetPublishFeatureMap() const;

    private:
        // A mapping of feature name to type of the features being published.
        boost::shared_ptr<PublishFeatureMap> m_publishFeatureMap;

        // Body of the feature.
        const Expression& m_body;

        // The type of feature specification.
        FeatureSpecType m_featureSpecType;

        // Whether this FeatureSpec returns a value (versus publishing feature 
        // names). This is a temporary parameter that will be removed when a 
        // Function type is added to this class. This will be addressed with 
        // TFS 321891.
        bool m_returnsValue;
    };

    // This class imports a feature values as a declaration. This class is
    // distinct from FeatureRefExpression in that it imports features which
    // are dependent on the current metastream; it is more efficient to 
    // determine these values at runtime than to have a FeatureRef for each
    // stream and array dimensions.
    class ImportFeatureExpression : public Expression
    {
    public:
        // Import an array of parameterized feature values.
        ImportFeatureExpression(const Annotations& p_annotations,
                                const FeatureSpecExpression::FeatureName& p_featureName,
                                const std::vector<UInt32>& p_dimensions,
                                VariableID p_id,
                                TypeManager& p_typeManager);

        // Import a single per-stream feature value.
        ImportFeatureExpression(const Annotations& p_annotations,
                                const FeatureSpecExpression::FeatureName& p_featureName,
                                VariableID p_id);

        // Methods inherited from Expression.
        virtual void Accept(Visitor&) const override;
        virtual const TypeImpl& GetType() const override;
        virtual size_t GetNumChildren() const override;

        // Accessor methods.
        VariableID GetId() const;
        const FeatureSpecExpression::FeatureName& GetFeatureName() const;

    private:
        // A struct containing feature name information.
        const FeatureSpecExpression::FeatureName m_featureName;

        // The type of the feature. Most importantly, this holds the array
        // dimensions for the feature.
        const TypeImpl& m_type;

        // Allocation ID for the value.
        VariableID m_id;
    };

    // This class wraps several feature specifications into a feature group.
    // This is needed to be able to emit code that is compatible with the current
    // implementation of features in the IFM.
    class FeatureGroupSpecExpression : public Expression
    {
    public:
        FeatureGroupSpecExpression(const Annotations& p_annotations,
                                   const std::string& p_name,
                                   const std::vector<const FeatureSpecExpression*>& p_featureSpecs,
                                   bool p_isExtendedExperimental,
                                   bool p_isSmallExperimental,
                                   bool p_isBlockLevelFeature,
                                   bool p_isBodyBlockFeature,
                                   bool p_isForwardIndexFeature,
                                   const std::string& p_metaStreamName);

        // Methods inherited from Expression.
        virtual void Accept(Visitor&) const override;
        virtual const TypeImpl& GetType() const override;
        virtual size_t GetNumChildren() const override;

        // Gets the type of all the feature specifications.
        FeatureSpecExpression::FeatureSpecType GetFeatureSpecType() const;

        // Accessor methods.
        const std::string& GetName() const;
        const std::vector<const FeatureSpecExpression*>& GetFeatureSpecs() const;
        bool IsExtendedExperimental() const;
        bool IsSmallExperimental() const;
        bool IsBlockLevelFeature() const;
        bool IsBodyBlockFeature() const;
        bool IsForwardIndexFeature() const;
        const std::string& GetMetaStreamName() const;

        // Returns true if the features in this FeatureGroup are per-stream features.
        bool IsPerStream() const;

    private:
        // The name of the feature group.
        const std::string m_name;

        // The child feature specs.
        const std::vector<const FeatureSpecExpression*> m_featureSpecs;

        // Whether this feature group is extended experimental.
        bool m_isExtendedExperimental;

        // Whether this feature group is small experimental.
        bool m_isSmallExperimental;

        // Whether this feature group is block level.
        bool m_isBlockLevelFeature;

        // Whether this feature group is body block.
        bool m_isBodyBlockFeature;

        // Whether this feature group uses the forward index.
        bool m_isForwardIndexFeature;

        // The stream over which this metastream feature is supposed to operate over.
        const std::string m_metaStreamName;

        // The type of all the feature specifications. All the feature spec types must
        // be the same.
        FeatureSpecExpression::FeatureSpecType m_featureSpecType;
    };

    // This expression denotes the beggining of an aggregate block. This 
    // expression type is only allowed inside aggregate features and will be
    // run exactly once per stream on which this feature is being evaluated.
    class AggregateContextExpression : public Expression
    {
    public:
        // Create an aggregate context containing a body expression.
        AggregateContextExpression(const Annotations& p_annotations,
                                   const Expression& p_body);

        // Methods inherited from Expression.
        virtual void Accept(Visitor&) const override;
        virtual const TypeImpl& GetType() const override;
        virtual size_t GetNumChildren() const override;

        // Get the body of the aggregate context.
        const Expression& GetBody() const;

    private:
        // The body of the aggregator loop.
        const Expression& m_body;
    };
}

std::ostream& operator<<(std::ostream& p_out, 
                         const FreeForm2::FeatureSpecExpression::FeatureName& p_name);

#endif
