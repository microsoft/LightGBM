#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _In_z_
#undef _In_z_
#endif
#define _In_z_

typedef uint32_t UInt32;
typedef uint8_t UInt8;
typedef uint64_t UInt64;

typedef int64_t Int64;
typedef int32_t Int32;
typedef unsigned long DWORD;

typedef size_t Size_t;

#define MAX_UINT32 ((UInt32)-1)

struct SIZED_STRING
{
    union
    {
        const UInt8 *pbData;
        const char  *pcData;
    };
    size_t cbData;
};

#define _TRUNCATE ((size_t)-1)

// A utility class for creating temporary SIZED_STRINGs
class CStackSizedString : public SIZED_STRING
{
public:
    // Null-terminated input
    CStackSizedString(_In_z_ const char *szValue)
    {
#if defined(_MSC_VER) && _MSC_VER >= 1910
        // Suppression: error C26490: Don't use reinterpret_cast.
        // reinterpreting the bits of the char* to the UInt* (both 8 bits per item) is necessary here
        [[gsl::suppress(type.1)]]
#endif
        pbData = reinterpret_cast<const UInt8 *>(szValue);
        cbData = strlen(szValue);
    }

    // Name/size pair
    CStackSizedString(
        const UInt8 *pbValue,
        size_t cbValue)
    {
        pbData = pbValue;
        cbData = cbValue;
    }

    // Name/size pair
    CStackSizedString(
        const char *pcValue,
        size_t cbValue)
    {
        pcData = pcValue;
        cbData = cbValue;
    }

private:
    // prevent heap allocation
    void *operator new(size_t);
};

#define UINT_MAX 0xffffffffu


// convenience macros to make SIZED_STRINGs more usable, particularly with the FEX Document class and FexSprintf
// expand a SIZED_STRING:
#define SIZED_STR(sizedstr) (char*)sizedstr.pbData, sizedstr.cbData
// reverse, for printing to FexSprintf:
#define SIZED_STR_REV(sizedstr) sizedstr.cbData, sizedstr.pcData
// References, for loading with document->GetField( SIZED_STR_REF(str) );
#define SIZED_STR_REF(sizedstr) &sizedstr.pcData, &sizedstr.cbData
// Convert a SIZED_STRING to std::basic_string
#define SIZED_STR_STL(sizedstr) (sizedstr.cbData>0 ? std::string(SIZED_STR(sizedstr)) : std::string())

typedef int BOOL;

#ifndef FALSE
#define FALSE 0
#endif

#ifndef TRUE
#define TRUE 1
#endif
