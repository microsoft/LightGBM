// Helper function to check if a data point should be blinded
inline bool ShouldBlind(data_size_t row_idx, int feature_idx) const {
  if (!bl_median_bins_ || !bl_masked_rows_) return false;
  
  // Check if this feature has a valid median bin (only numerical features)
  if (feature_idx >= static_cast<int>(bl_median_bins_->size()) || (*bl_median_bins_)[feature_idx] < 0) {
    return false;
  }
  
  // Check if this row is in the masked list for this feature
  const auto& masked_rows = (*bl_masked_rows_)[feature_idx];
  return std::binary_search(masked_rows.begin(), masked_rows.end(), row_idx);
}

// Get the bin value, potentially blinded
inline uint32_t GetBinValue(const VAL_T* data_ptr, int feature_idx, data_size_t row_idx) const {
  uint32_t bin = static_cast<uint32_t>(data_ptr[feature_idx]);
  
  // Apply blinding if needed
  if (ShouldBlind(row_idx, feature_idx)) {
    bin = static_cast<uint32_t>((*bl_median_bins_)[feature_idx]);
  }
  
  return bin;
}
