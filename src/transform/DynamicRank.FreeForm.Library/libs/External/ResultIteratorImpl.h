#pragma once

#ifndef FREEFORM2_RESULTITERATORIMPL_H
#define FREEFORM2_RESULTITERATORIMPL_H

#include "FreeForm2Result.h"

namespace FreeForm2
{
    class ResultIteratorImpl
    {
    public:
        virtual ~ResultIteratorImpl()
        {
        }

        // Delegated iterator_facade function to increment the iterator.
        virtual void increment() = 0;

        // Delegated iterator_facade function to decrement the iterator.
        virtual void decrement() = 0;

        // Delegated iterator_facade function to get the current element.
        virtual const Result& dereference() const = 0;

        // Delegated iterator_facade function to get the current element.
        virtual void advance(std::ptrdiff_t p_distance) = 0;

        // Virtual copy constructor.
        virtual std::auto_ptr<ResultIteratorImpl> Clone() const = 0;

        // Having an abstract iterator puts us in a tricky position, because
        // some of the iterator methods (like equal) accept another iterator
        // as arg.  Since we may have any number of subclasses, equal must
        // be able to compare iterators that have nothing to do with each
        // other.  As such, we use a couple of (somewhat hacky) methods
        // below to return enough information from each iterator to compare
        // and calculate the difference between them without further
        // knowledge.

        // Returns a pointer indicating current position, plus the element index.
        virtual std::pair<const char*, unsigned int> Position() const = 0;

        // Returns number of bytes per element.
        virtual unsigned int ElementSize() const = 0;
    };
}

#endif

