#pragma once

#ifndef FREEFORM2_ARRAY_RESULT_H
#define FREEFORM2_ARRAY_RESULT_H

#include "ArrayType.h"
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include "Expression.h"
#include "FreeForm2Result.h"
#include "FreeForm2Type.h"
#include "FreeForm2Tokenizer.h"
#include "FreeForm2Assert.h"
#include "ResultIteratorImpl.h"
#include "ValueResult.h"
#include <vector>

namespace FreeForm2
{
    // Class to iterate through elements of an array, returning basic (i.e. 
    // no arrays) values from it.
    template<class T>
    class BaseResultIterator : public ResultIteratorImpl
    {
    public:
        BaseResultIterator(const T* p_pos,
                           unsigned int p_idx,
                           const boost::shared_array<T>& p_space)
            : m_pos(p_pos), 
              m_idx(p_idx),
              m_space(p_space),
              m_result(NULL)
        {
        }


        virtual void 
        increment()
        {
            m_result.reset(NULL);
            m_pos++;
            m_idx++;
        }


        virtual void 
        decrement()
        {
            m_result.reset(NULL);
            m_pos--;
            m_idx--;
        }


        virtual const Result& 
        dereference() const 
        {
            // Note that we need to initialise the result here, because we
            // aren't sure it is valid at any other point.
            if (m_result.get() == NULL)
            {
                m_result.reset(new ValueResult(*m_pos));
            }
            return *m_result;
        }


        virtual void 
        advance(std::ptrdiff_t p_distance) 
        {
            m_result.reset(NULL);
            m_pos += p_distance;
            m_idx += static_cast<int>(p_distance);
        }


        virtual std::auto_ptr<ResultIteratorImpl> 
        Clone() const 
        {
            return std::auto_ptr<ResultIteratorImpl>(
                new BaseResultIterator(m_pos, m_idx, m_space));
        }


        virtual std::pair<const char*, unsigned int>
        Position() const 
        {
            return std::make_pair(reinterpret_cast<const char*>(m_pos),
                                  m_idx);
        }


        virtual unsigned int 
        ElementSize() const
        {
            return sizeof(T);
        }

    private:
        // Current position in array.
        const T* m_pos;

        // Currend index in array.
        unsigned int m_idx;

        // Shared pointer to space allocated to hold the array.
        boost::shared_array<T> m_space;

        // Current result.
        mutable boost::scoped_ptr<ValueResult> m_result;
    };
      

    typedef boost::shared_ptr<std::vector<unsigned int>> SharedDimensions;
    template<class T> class ArrayResultIterator;


    template<class T>
    class ArrayResult : public Result
    {
    public:
        ArrayResult(const TypeImpl& p_type, 
                    unsigned int p_dimensionPos, 
                    const SharedDimensions& p_dimensions, 
                    const T* p_pos, 
                    const boost::shared_array<T>& p_space)
            : m_arrayType(NULL), 
              m_type(p_type), 
              m_dimensionPos(p_dimensionPos),
              m_dimensions(p_dimensions), 
              m_pos(p_pos), 
              m_end(p_pos + ArrayResult<T>::CalculateArrayStep(p_dimensionPos, *p_dimensions)), 
              m_space(p_space)
        {
            FF2_ASSERT(p_type.Primitive() == Type::Array);
            m_arrayType = static_cast<const ArrayType*>(&p_type);
            FF2_ASSERT(m_arrayType->GetDimensionCount() == p_dimensions->size() - p_dimensionPos);
        }


        virtual const Type& 
        GetType() const
        {
            return m_type;
        }


        virtual IntType 
        GetInt() const
        {
            // Can't retrieve an int from an array.
            Unreachable(__FILE__, __LINE__);
        }


        virtual UInt64Type 
        GetUInt64() const
        {
            // Can't retrieve a uint64 from an array.
            Unreachable(__FILE__, __LINE__);
        }


        virtual int
        GetInt32() const
        {
            // Can't retrieve an int32 from an array.
            Unreachable(__FILE__, __LINE__);
        }


        virtual unsigned int
        GetUInt32() const
        {
            // Can't retrieve an uint32 from an array.
            Unreachable(__FILE__, __LINE__);
        }


        virtual FloatType 
        GetFloat() const
        {
            // Can't retrieve a float from an array.
            Unreachable(__FILE__, __LINE__);
        }


        virtual bool 
        GetBool() const
        {
            // Can't retrieve a bool from an array.
            Unreachable(__FILE__, __LINE__);
        }


        virtual ResultIterator 
        BeginArray() const
        {
            if (m_arrayType->GetDimensionCount() > 1)
            {
                // Step type down a dimension.
                return ResultIterator(std::auto_ptr<ResultIteratorImpl>(
                    new ArrayResultIterator<T>(m_arrayType->GetDerefType(), 
                                               m_dimensionPos + 1, 
                                               m_dimensions,
                                               m_pos,
                                               0,
                                               m_space)));
            }
            else
            {
                return ResultIterator(std::auto_ptr<ResultIteratorImpl>(
                    new BaseResultIterator<T>(m_pos, 0, m_space)));
            }
        }


        virtual ResultIterator 
        EndArray() const
        {
            if (m_arrayType->GetDimensionCount() > 1)
            {
                // Step type down a dimension.
                return ResultIterator(std::auto_ptr<ResultIteratorImpl>(
                    new ArrayResultIterator<T>(m_arrayType->GetDerefType(), 
                                               m_dimensionPos + 1, 
                                               m_dimensions, 
                                               m_end,
                                               (*m_dimensions)[m_dimensions->size() - m_dimensionPos - 1],
                                               m_space)));
            }
            else
            {
                return ResultIterator(std::auto_ptr<ResultIteratorImpl>(
                    new BaseResultIterator<T>(m_end, (*m_dimensions)[0], m_space)));
            }
        }

        // Calculate the number of elements at a given array level, providing the
        // step size across the array.
        static unsigned int CalculateArrayStep(unsigned int p_dimensionPos, 
                                               const std::vector<unsigned int>& p_dimensions)
        {
            unsigned int step = 1;
            for (unsigned int i = p_dimensionPos; i < p_dimensions.size(); i++)
            {
                step *= p_dimensions[p_dimensions.size() - 1 - i];
            }
            return step;
        }


    private:
        // Type and TypeImpl (need both as we need to pass the impl to type).
        const ArrayType* m_arrayType;
        Type m_type;

        // Position in the dimension array.  For example, if at 0 we are at the
        // highest dimension, 1 is next down (one level of dereference), etc.
        unsigned int m_dimensionPos;

        // Shared dimension array.
        SharedDimensions m_dimensions;

        // Position in the array space.
        const T* m_pos;

        // End position in the array space.
        const T* m_end;

        // Shared pointer to space allocated to hold array.
        boost::shared_array<T> m_space;
    };


    // Class to iterate through array elements of an array, returning subarrays.
    // no arrays) values from it.
    template<class T>
    class ArrayResultIterator : public ResultIteratorImpl
    {
    public:
        ArrayResultIterator(const TypeImpl& p_type, 
                            unsigned int p_dimensionPos, 
                            const SharedDimensions& p_dimensions, 
                            const T* p_pos,
                            unsigned int p_idx,
                            const boost::shared_array<T>& p_space)
            : m_type(p_type),
              m_dimensionPos(p_dimensionPos),
              m_dimensions(p_dimensions), 
              m_pos(p_pos), 
              m_idx(p_idx),
              m_space(p_space),
              m_step(ArrayResult<T>::CalculateArrayStep(p_dimensionPos, *p_dimensions)),
              m_result(NULL)
        {
            FF2_ASSERT(m_type.Primitive() == Type::Array);
        }


        virtual void 
        increment()
        {
            m_result.reset(NULL);
            m_pos += m_step;
            m_idx++;
        }


        virtual void 
        decrement()
        {
            m_result.reset(NULL);
            m_pos -= m_step;
            m_idx--;
        }


        virtual const Result& 
        dereference() const
        {
            // Note that we need to initialise the result here, because we
            // aren't sure it is valid at any other point.
            if (m_result.get() == NULL)
            {
                m_result.reset(new ArrayResult<T>(m_type, 
                                                  m_dimensionPos, 
                                                  m_dimensions, 
                                                  m_pos,
                                                  m_space));
            }
            return *m_result;
        }


        virtual void 
        advance(std::ptrdiff_t p_distance) 
        {
            m_result.reset(NULL);
            m_pos += p_distance * m_step;
            m_idx += static_cast<int>(p_distance);
        }


        virtual std::auto_ptr<ResultIteratorImpl> 
        Clone() const 
        {
            return std::auto_ptr<ResultIteratorImpl>(
                new ArrayResultIterator<T>(m_type, m_dimensionPos, m_dimensions, m_pos, m_idx, m_space));
        }


        virtual std::pair<const char*, unsigned int>
        Position() const 
        {
            return std::make_pair(reinterpret_cast<const char*>(m_pos),
                             m_idx);
        }


        virtual unsigned int 
        ElementSize() const
        {
            return sizeof(T) * m_step;
        }


    private:
        // Child type of the array
        const TypeImpl& m_type;

        // Position in the dimension array.  For example, if at 0 we are at the
        // highest dimension, 1 is next down (one level of dereference), etc.
        unsigned int m_dimensionPos;

        // Shared dimension array.
        SharedDimensions m_dimensions;

        // Position in the array space.
        const T* m_pos;

        // Index in the array.
        unsigned int m_idx;

        // Shared pointer to space allocated to hold array.
        boost::shared_array<T> m_space;

        // Step size that we're taking for each element.
        unsigned int m_step;

        // Current result.
        mutable boost::scoped_ptr<ArrayResult<T>> m_result;
    };
}

#endif
