/*!
 * Copyright (c) 2023 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * Author: Oliver Borchert
 */

#include <gtest/gtest.h>
#include <LightGBM/arrow.h>

#include <cstdlib>
#include <cmath>

using LightGBM::ArrowChunkedArray;
using LightGBM::ArrowTable;

class ArrowChunkedArrayTest : public testing::Test {
 protected:
  void SetUp() override {}

  ArrowArray created_nested_array(const std::vector<ArrowArray*>& arrays) {
    ArrowArray arr;
    arr.buffers = nullptr;
    arr.children = (ArrowArray**)arrays.data();  // NOLINT
    arr.dictionary = nullptr;
    arr.length = arrays[0]->length;
    arr.n_buffers = 0;
    arr.n_children = arrays.size();
    arr.null_count = 0;
    arr.offset = 0;
    arr.private_data = nullptr;
    arr.release = nullptr;
    return arr;
  }

  template <typename T>
  ArrowArray create_primitive_array(const std::vector<T>& values,
                                    int64_t offset = 0,
                                    std::vector<int64_t> null_indices = {}) {
    // NOTE: Arrow arrays have 64-bit alignment but we can safely ignore this in tests
    // 1) Create validity bitmap
    char* validity = nullptr;
    if (!null_indices.empty()) {
      validity = static_cast<char*>(calloc(values.size() + sizeof(char) - 1, sizeof(char)));
      for (size_t i = 0; i < values.size(); ++i) {
        if (std::find(null_indices.begin(), null_indices.end(), i) != null_indices.end()) {
          validity[i / 8] |= (1 << (i % 8));
        }
      }
    }

    // 2) Create buffers
    const void** buffers = (const void**)malloc(sizeof(void*) * 2);
    buffers[0] = validity;
    buffers[1] = values.data() + offset;

    // Create arrow array
    ArrowArray arr;
    arr.buffers = buffers;
    arr.children = nullptr;
    arr.dictionary = nullptr;
    arr.length = values.size() - offset;
    arr.null_count = 0;
    arr.offset = 0;
    arr.private_data = nullptr;
    arr.release = [](ArrowArray* arr) {
      if (arr->buffers[0] != nullptr)
        free((void*)(arr->buffers[0]));  // NOLINT
      free((void*)(arr->buffers));  // NOLINT
    };
    return arr;
  }

  ArrowSchema create_nested_schema(const std::vector<ArrowSchema*>& arrays) {
    ArrowSchema schema;
    schema.format = "+s";
    schema.name = nullptr;
    schema.metadata = nullptr;
    schema.flags = 0;
    schema.n_children = arrays.size();
    schema.children = (ArrowSchema**)arrays.data();  // NOLINT
    schema.dictionary = nullptr;
    schema.private_data = nullptr;
    schema.release = nullptr;
    return schema;
  }

  template <typename T>
  ArrowSchema create_primitive_schema() {
    std::logic_error("not implemented");
  }

  template <>
  ArrowSchema create_primitive_schema<float>() {
    ArrowSchema schema;
    schema.format = "f";
    schema.name = nullptr;
    schema.metadata = nullptr;
    schema.flags = 0;
    schema.n_children = 0;
    schema.children = nullptr;
    schema.dictionary = nullptr;
    schema.private_data = nullptr;
    schema.release = nullptr;
    return schema;
  }
};

TEST_F(ArrowChunkedArrayTest, GetLength) {
  std::vector<float> dat1 = {1, 2};
  auto arr1 = create_primitive_array(dat1);

  ArrowChunkedArray ca1(1, &arr1, nullptr);
  ASSERT_EQ(ca1.get_length(), 2);

  std::vector<float> dat2 = {3, 4, 5, 6};
  auto arr2 = create_primitive_array<float>(dat2);
  ArrowArray arrs[2] = {arr1, arr2};
  ArrowChunkedArray ca2(2, arrs, nullptr);
  ASSERT_EQ(ca2.get_length(), 6);

  arr1.release(&arr1);
  arr2.release(&arr2);
}

TEST_F(ArrowChunkedArrayTest, GetColumns) {
  std::vector<float> dat1 = {1, 2, 3};
  auto arr1 = create_primitive_array(dat1);
  std::vector<float> dat2 = {4, 5, 6};
  auto arr2 = create_primitive_array(dat2);
  std::vector<ArrowArray*> arrs = {&arr1, &arr2};
  auto arr = created_nested_array(arrs);

  auto schema1 = create_primitive_schema<float>();
  auto schema2 = create_primitive_schema<float>();
  std::vector<ArrowSchema*> schemas = {&schema1, &schema2};
  auto schema = create_nested_schema(schemas);

  ArrowTable table(1, &arr, &schema);
  ASSERT_EQ(table.get_num_rows(), 3);
  ASSERT_EQ(table.get_num_columns(), 2);

  auto ca1 = table.get_column(0);
  ASSERT_EQ(ca1.get_length(), 3);
  ASSERT_EQ(*ca1.begin<int32_t>(), 1);

  auto ca2 = table.get_column(1);
  ASSERT_EQ(ca2.get_length(), 3);
  ASSERT_EQ(*ca2.begin<int32_t>(), 4);

  arr1.release(&arr1);
  arr2.release(&arr2);
}

TEST_F(ArrowChunkedArrayTest, IteratorArithmetic) {
  std::vector<float> dat1 = {1, 2};
  auto arr1 = create_primitive_array<float>(dat1);
  std::vector<float> dat2 = {3, 4, 5, 6};
  auto arr2 = create_primitive_array<float>(dat2);
  std::vector<float> dat3 = {7};
  auto arr3 = create_primitive_array<float>(dat3);
  auto schema = create_primitive_schema<float>();

  ArrowArray arrs[3] = {arr1, arr2, arr3};
  ArrowChunkedArray ca(3, arrs, &schema);

  // Arithmetic
  auto it = ca.begin<int32_t>();
  ASSERT_EQ(*it, 1);
  ++it;
  ASSERT_EQ(*it, 2);
  ++it;
  ASSERT_EQ(*it, 3);
  it += 2;
  ASSERT_EQ(*it, 5);
  it += 2;
  ASSERT_EQ(*it, 7);
  --it;
  ASSERT_EQ(*it, 6);

  // Subscripts
  ASSERT_EQ(it[0], 1);
  ASSERT_EQ(it[1], 2);
  ASSERT_EQ(it[2], 3);
  ASSERT_EQ(it[6], 7);

  // End
  auto end = ca.end<int32_t>();
  ASSERT_EQ(end - it, 2);
  ASSERT_EQ(end - ca.begin<int32_t>(), 7);

  arr1.release(&arr1);
  arr2.release(&arr2);
  arr2.release(&arr3);
}

TEST_F(ArrowChunkedArrayTest, OffsetAndValidity) {
  std::vector<float> dat = {0, 1, 2, 3, 4, 5, 6};
  auto arr = create_primitive_array(dat, 2, {0, 1});
  auto schema = create_primitive_schema<float>();
  ArrowChunkedArray ca(1, &arr, &schema);

  auto it = ca.begin<double>();
  ASSERT_TRUE(std::isnan(*it));
  ASSERT_TRUE(std::isnan(*(++it)));
  ASSERT_EQ(it[2], 4);
  ASSERT_EQ(it[4], 6);

  arr.release(&arr);
}
