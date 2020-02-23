/**
 * This SWIG interface extension provides support to
 * allocate, return and manage arrays of strings through
 * the class StringArray.
 *
 * This is then used to generate wrappers that return newly-allocated
 * arrays of strings, so the user can them access them easily as a String[]
 * on the Java side by a single call to StringArray::data(), and even manipulate
 * them.
 *
 * It also implements working wrappers to:
 *  - LGBM_BoosterGetEvalNames    (re-implemented with new API)
 *  - LGBM_BoosterGetFeatureNames (original non-wrapped version didn't work).
 * where the wrappers names end with "SWIG".
 */


#include <memory>

%include "./StringArray.i"

%inline %{

    #define API_OK_OR_VALUE(api_return, return_value) if (api_return == -1) return return_value
    #define API_OK_OR_NULL(api_return) API_OK_OR_VALUE(api_return, nullptr)

    /**
     * @brief Wraps LGBM_BoosterGetEvalNames.
     *
     * In case of success a new StringArray is created and returned,
     * which you're responsible for freeing,
     * @see StringArrayHandle_free().
     * In case of failure such resource is freed and nullptr is returned.
     * Check for that case with null (lightgbmlib) or 0 (lightgbmlibJNI).
     *
     * @param handle Booster handle
     * @return StringArrayHandle with the eval names (or nullptr in case of error)
     */
    StringArrayHandle LGBM_BoosterGetEvalNamesSWIG(BoosterHandle handle)
    {
        std::unique_ptr<StringArray> strings(nullptr);

        // 1) Figure out the necessary allocation size:
        int eval_counts;
        API_OK_OR_NULL(LGBM_BoosterGetEvalCounts(handle, &eval_counts));

        size_t largest_eval_name_size;
        API_OK_OR_NULL(LGBM_BoosterGetLargestEvalNameSize(handle, &largest_eval_name_size));

        // 2) Allocate the strings container:
        try {
            strings.reset(new StringArray(eval_counts, largest_eval_name_size));
        } catch (std::bad_alloc &e) {
            LGBM_SetLastError("Failure to allocate memory.");
            return nullptr;
        }

        // 3) Extract the names:
        API_OK_OR_NULL(LGBM_BoosterGetEvalNames(handle, &eval_counts, strings->data()));
        return strings.release();
    }

    /**
     * @brief Wraps LGBM_BoosterGetFeatureNames.
     *
     * Allocates a new StringArray. You must free it yourself if it succeeds.
     * @see StringArrayHandle_free().
     * In case of failure such resource is freed and nullptr is returned.
     * Check for that case with null (lightgbmlib) or 0 (lightgbmlibJNI).
     *
     * @param handle Booster handle
     * @return StringArrayHandle with the feature names (or nullptr in case of error)
     */
    StringArrayHandle LGBM_BoosterGetFeatureNamesSWIG(BoosterHandle handle)
    {
        std::unique_ptr<StringArray> strings(nullptr);

        // 1) To preallocate memory extract number of features & required size first:
        int num_features;
        API_OK_OR_NULL(LGBM_BoosterGetNumFeature(handle, &num_features));

        size_t max_feature_name_size;
        API_OK_OR_NULL(LGBM_BoosterGetLargestFeatureNameSize(handle, &max_feature_name_size));

        // 2) Allocate an array of strings:
        try {
            strings.reset(new StringArray(num_features, max_feature_name_size));
        } catch (std::bad_alloc &e) {
            LGBM_SetLastError("Failure to allocate memory.");
            return nullptr;
        }

        // 3) Extract feature names:
        API_OK_OR_NULL(LGBM_BoosterGetFeatureNames(handle, &num_features, strings->data()));

        return strings.release();
    }
%}
