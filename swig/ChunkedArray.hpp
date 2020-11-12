#ifndef __CHUNKED_ARRAY_H__
#define __CHUNKED_ARRAY_H__

#include <new>
#include <vector>
#include <algorithm>

#include <stdint.h>
#include <assert.h>

/**
 * Container that manages a dynamic array of fixed-length chunks.
 *
 * The class also takes care of allocation & release of the underlying
 * memory. It can be used with either a high or low-level API.
 *
 * The high-level API allocates chunks as needed, manages addresses automatically and keeps
 * track of number of inserted elements, but is not thread-safe (ok as usually input is a streaming iterator).
 * For parallel input sources the low-level API must be used.
 *
 * Note: When using this for `LGBM_DatasetCreateFromMats` use a
 *       chunk_size multiple of #num_cols for your dataset, so each chunk
 *       contains "complete" instances.
 *
 * === High-level insert API intro ===
 *
 * The easiest way to use is:
 *  0. ChunkedArray(chunk_size)  # Choose appropriate size
 *  1. add(value)                # as many times as you want (will generate chunks as needed)
 *  2. data() or void_data()     # retrieves a T** or void** pointer (useful for `LGBM_DatasetCreateFromMats`).
 *
 * Useful query methods (all O(1)):
 *  - get_add_count()   # total count of added elements
 *  - get_chunks_count()  # how many chunks are currently allocated.
 *  - get_current_chunk_added_count()  # for the last add() chunk, how many items there are.
 *  - get_chunk_size()    # Get constant chunk_size from constructor call.
 *
 *  With those you can generate int32_t sizes[]. Last chunk can be smaller than chunk_size, so, for any i:
 *    - sizes[i<last] = get_chunk_size()
 *    - sizes[i==last] = get_add_count()
 *
 *
 * === Low-level insert API intro ===
 *
 * For advanced usage - useful for inserting in parallel - one can also:
 *  1. call new_chunk() at any time for as many chunks as needed.  (thread-UNsafe)
 *  2. call setitem(chunk, idx, value) to insert each value.       (thread-safe)
 *
 */
template <class T>
class ChunkedArray
{
  public:
    ChunkedArray(size_t chunk_size)
      : _chunk_size(chunk_size), _current_chunks_idx(0), _current_chunk_idx(0)
    {
       new_chunk();
    }

    ~ChunkedArray()
    {
        release();
    }

    /**
     * Adds a value to the chunks sequentially.
     * If the last chunk is full it creates a new one and appends to it.
     */
    void add(T value) {
        if (! within_bounds(_current_chunks_idx, _current_chunk_idx)) {
            new_chunk();
            _current_chunks_idx += 1;
            _current_chunk_idx = 0;
        }

        assert (setitem(_current_chunks_idx, _current_chunk_idx, value) == 0);
        _current_chunk_idx += 1;
    }

    size_t get_add_count() const {
        return _current_chunks_idx * _chunk_size + _current_chunk_idx;
    }

    size_t get_chunks_count() const {
        return _chunks.size();
    }

    size_t get_current_chunk_added_count() const {
        return _current_chunk_idx;
    }

    size_t get_chunk_size() const {
        return _chunk_size;
    }

    /**
     * Returns the pointer to the raw chunks data.
     *
     * @return T** pointer to raw data.
     */
    T **data() noexcept
    {
        return _chunks.data();
    }

    /**
     * Returns the pointer to the raw chunks data, but cast to void**.
     * This is so ``LGBM_DatasetCreateFromMats`` accepts it.
     *
     * @return void** pointer to raw data.
     */
    void **data_as_void() noexcept
    {
        return reinterpret_cast<void**>(_chunks.data());
    }

    /**
     * Coalesces (copies chunked data) to an array of the same type.
     * It assumes that ``other`` has enough space to receive that data.
     */
    void coalesce_to(T *other) const {
        if (this->empty()) {
            return;
        }

        const size_t full_chunks = this->get_chunks_count() - 1;

        // Copy full chunks:
        size_t i = 0;
        for(size_t chunk = 0; chunk < full_chunks; ++chunk) {
            T* chunk_ptr = _chunks[chunk];
            for(size_t chunk_pos = 0; chunk_pos < _chunk_size; ++chunk_pos) {
                other[i++] = chunk_ptr[chunk_pos];
            }
        }
        // Copy filled values from last chunk only:
        const size_t last_chunk_elems = this->get_current_chunk_added_count();
        T* chunk_ptr = _chunks[full_chunks];
        for(size_t chunk_pos = 0; chunk_pos < last_chunk_elems; ++chunk_pos) {
            other[i++] = chunk_ptr[chunk_pos];
        }
    }

    /**
     * Return value from array of chunks.
     *
     * @param chunks_index index of the chunk
     * @param index index within chunk
     * @param on_fail_value sentinel value. If out of bounds returns that value.
     *
     * @return pointer or nullptr if index is out of bounds.
     */
    T getitem(size_t chunks_index, size_t index, T on_fail_value) noexcept
    {
        if (within_bounds(chunks_index, index))
            return _chunks[chunks_index][index];
        else
            return on_fail_value;
    }

    /**
     *
     * @param chunks_index index of the chunk
     * @param index index within chunk
     * @param value value to store
     *
     * @return 0 = success, -1 = out of bounds access.
     */
    int setitem(size_t chunks_index, size_t index, T value) noexcept
    {
        if (within_bounds(chunks_index, index))
        {
            _chunks[chunks_index][index] = value;
            return 0;
        } else {
            return -1;
        }
    }

    /**
     * To reset storage call this.
     * Will release existing resources and prepare for reuse.
     */
    void clear() noexcept
    {
        release();
        new_chunk();
    }

    /**
     * Returns true if is empty.
     */
    bool empty() const noexcept
    {
        return get_current_chunk_added_count() == 0;
    }

    /**
     * Deletes all the allocated chunks.
     * Do not use container after this! See ``clear()`` instead.
     */
    void release() noexcept
    {
        std::for_each(_chunks.begin(), _chunks.end(), [](T* c) { delete[] c; });
        _chunks.clear();
        _chunks.shrink_to_fit();
        _current_chunks_idx = 0;
        _current_chunk_idx = 0;
    }

    inline bool within_bounds(size_t chunks_index, size_t index) {
        return (chunks_index < _chunks.size()) && (index < _chunk_size);
    }

    /**
     * Adds a new chunk to the array of chunks. Not thread-safe.
     */
    void new_chunk()
    {
        _chunks.push_back(new (std::nothrow) T[_chunk_size]);

        // Check memory allocation success:
        if (! _chunks[_chunks.size()-1]) {
            release();
            throw std::bad_alloc();
        }
    }

  private:

    const size_t _chunk_size;
    std::vector<T*> _chunks;

    // For add() interface & some of the get_*() queries:
    size_t _current_chunks_idx; //<! Index of chunks
    size_t _current_chunk_idx;  //<! Index within chunk
};


#endif // __CHUNKED_ARRAY_H__
