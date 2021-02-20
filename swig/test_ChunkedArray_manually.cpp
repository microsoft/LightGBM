/**
 * Tests for ChunkedArray.
 *
 * Some tests require visual assessment.
 * We should move this to googletest/Catch2 in the future
 * and get rid of the tests that require visual checks.
 */

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <sstream>

#include "ChunkedArray.hpp"
using namespace std;

using intChunkedArray=ChunkedArray<int>;
using doubleChunkedArray = ChunkedArray<double>;

// Test data
const int out_of_bounds = 4; // test get outside bounds.
const size_t chunk_size = 3;
const std::vector<int> ref = {1, 2, 3, 4, 5, 6, 7};

template<class T>
size_t _get_merged_array_size(ChunkedArray<T> &ca) {
    if (ca.empty()) {
        return 0;
    } else {
        size_t prior_chunks_total_size = (ca.get_chunks_count() - 1) * ca.get_chunk_size();
        return prior_chunks_total_size + ca.get_last_chunk_add_count();
    }
}

template<class T>
void print_container_stats(ChunkedArray<T> &ca) {
    printf("\n\nContainer stats: %ld chunks of size %ld with %ld item(s) on last chunk (#elements=%ld).\n"
           " > Should result in single array of size %ld.\n\n",
        ca.get_chunks_count(),
        ca.get_chunk_size(),
        ca.get_last_chunk_add_count(),
        ca.get_add_count(),
        _get_merged_array_size(ca));
}

template <typename T>
void _print_chunked_data(ChunkedArray<T> &x, T** data, std::ostream &o = std::cout) {
  int chunk = 0;
  int pos = 0;

  for (int i = 0; i < x.get_add_count(); ++i) {
      o << data[chunk][pos] << " ";

      ++pos;
      if (pos == x.get_chunk_size()) {
          pos = 0;
          ++chunk;
          o << "\n";
      }
  }
}

template <typename T>
void print_data(ChunkedArray<T> &x) {
  T **data = x.data();
  cout << "Printing from T** data(): \n";
  _print_chunked_data(x, data);
  cout << "\n^ Print complete ^\n";
}

template <typename T>
void print_void_data(ChunkedArray<T> &x) {
  T **data = reinterpret_cast<T**>(x.data_as_void());
  cout << "Printing from reinterpret_cast<T**>(data_as_void()):\n";
  _print_chunked_data(x, data);
  cout << "\n^ Print complete ^\n";
}

template <typename T>
void print_ChunkedArray_contents(ChunkedArray<T> &ca) {
    int chunk = 0;
    int pos = 0;
    for (int i = 0; i < ca.get_add_count() + out_of_bounds; ++i) {

        bool within_added = i < ca.get_add_count();
        bool within_bounds = ca.within_bounds(chunk, pos);
        cout << "@(" << chunk << "," << pos << ") = " << ca.getitem(chunk, pos, 10)
        << " " << within_added << " " << within_bounds << endl;

        ++pos;

        if (pos == ca.get_chunk_size()) {
            ++chunk;
            pos = 0;
        }
    }
}

/**
 * Ensure coalesce_to works and dumps all the inserted data correctly.
 */
void test_coalesce_to(const intChunkedArray &ca, const std::vector<int> &ref) {
    std::vector<int> coalesced_out(ca.get_add_count());

    ca.coalesce_to(coalesced_out.data());

    assert(ref.size() == coalesced_out.size());
    assert(std::equal(ref.begin(), ref.end(), coalesced_out.begin()));
}

/**
 * By retrieving all the data to a format split by chunks, one can ensure
 * that the data was stored correctly and with the correct memory layout.
 */
template <typename T>
void test_data_layout(ChunkedArray<T> &ca, const std::vector<T> &ref, bool data_as_void) {
    std::stringstream ss, ss_ref;
    T **data = data_as_void? reinterpret_cast<T**>(ca.data_as_void()) : ca.data();
    // Dump each chunk represented by a line with elements split by space:
    for (int i = 0; i < ref.size(); ++i) {
        if ((i > 0) && (i % chunk_size == 0))
            ss_ref << "\n";
        ss_ref << ref[i] << " ";
    }

    _print_chunked_data(ca, data, ss);  // Dump chunked data to this same string format.
    assert(ss_ref.str() == ss.str());
}

/**
 * Test that using, clearing and reusing uses the latest data only.
 */
void test_clear() {
    // Set-up with some data
    const std::vector<int> ref2 = {1, 2, 5, -1};
    ChunkedArray<int> ca = ChunkedArray<int>(chunk_size);
    for (auto v : ref) {
        ca.add(v);
    }
    test_coalesce_to(ca, ref);  // Should have the same contents.

    // Clear & re-use:
    ca.clear();
    for (auto v : ref2) {
        ca.add(v);  // Fill with new contents.
    }

    // Ensure it still works:
    test_coalesce_to(ca, ref2);  // Should match the new reference content.
}

int main() {
    // Initialize test variables. ////////////////////////////////////////////////////
    ChunkedArray<int> ca = ChunkedArray<int>(chunk_size);
    for (auto v : ref) {
        ca.add(v);  // Indirectly test insertions through the retrieval tests.
    }

    // Tests /////////////////////////////////////////////////////////////////////////

    assert(ca.get_add_count() == ref.size());
    test_coalesce_to(ca, ref);

    // Test chunked data layout for retrieval:
    test_data_layout<int>(ca, ref, false);
    test_data_layout<int>(ca, ref, true);

    test_clear();

    // For manual verification - useful outputs //////////////////////////////////////
    print_container_stats(ca);
    print_ChunkedArray_contents(ca);
    print_data<int>(ca);
    print_void_data<int>(ca);
    ca.release(); ca.release(); print_container_stats(ca);  // Check double free behaviour.
    cout << "Done!" << endl;
    return 0;
}
