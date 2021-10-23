#pragma once
#include "basic_types.h"
#include "MigratedApi.h"

// This code is copyed from apsSDK/CsHash.h
// lookup3, by Bob Jenkins, public domain
class CsHash32
{
private:
    CsHash32 ();                               // no NEW allowed
    CsHash32 ( const CsHash32& );              // no copy allowed
    CsHash32& operator= ( const CsHash32& );   // no assignment allowed

public:

    //
    // Mix(): cause every bit of a,b,c to affect 32 bits of a,b,c
    // both forwards and in reverse.  Same for pairs of bits in a,b,c.
    // This can be used along with Final() to hash a fixed number of
    // 4-byte integers, for example see the implementation of Guid().
    //
    inline static void Mix( UInt32& a, UInt32& b, UInt32& c)
    {
        a -= c;  a ^= rotl(c, 4);  c += b;
        b -= a;  b ^= rotl(a, 6);  a += c;
        c -= b;  c ^= rotl(b, 8);  b += a;
        a -= c;  a ^= rotl(c,16);  c += b;
        b -= a;  b ^= rotl(a,19);  a += c;
        c -= b;  c ^= rotl(b, 4);  b += a;
    }

    //
    // Final: cause every bit of a,b,c to affect every bit of c, only forward.
    // Same for pairs of bits in a,b,c.  It also causes b to be an OK hash.
    // This is a good way to hash 1 or 2 or 3 integers:
    //   a = k1; b = k2; c = 0;
    //   CsHash32::Final(a,b,c);
    // Use c (and maybe b) as the hash value
    //
    inline static void Final( UInt32& a, UInt32& b, UInt32& c)
    {
        c ^= b; c -= rotl(b,14);
        a ^= c; a -= rotl(c,11);
        b ^= a; b -= rotl(a,25);
        c ^= b; c -= rotl(b,16);
        a ^= c; a -= rotl(c,4);
        b ^= a; b -= rotl(a,14);
        c ^= b; c -= rotl(b,24);
    }

    //
    // Compute2: Compute two hash values for a byte array of known length
    //
    static void Compute2 (
        const void *pData,    // byte array of known length
        Size_t      uSize,    // size of pData
        UInt32      uSeed1,   // first seed
        UInt32      uSeed2,   // second seed
        UInt32     *uHash1,   // OUT: first hash value (may not be null)
        UInt32     *uHash2);  // OUT: second hash value (may not be null)

    //
    // Compute: Compute a hash values for a byte array of known length
    //
    static const UInt32 Compute (
        const void *pData,      // byte array of known length
        Size_t      uSize,      // size of pData
        UInt32      uSeed = 0)  // seed for hash function
    {
        UInt32 uHash2 = 0;
        CsHash32::Compute2(pData, uSize, uSeed, uHash2, &uSeed, &uHash2);
        return uSeed;
    }

    //
    // String: hash of string of unknown length
    //
    static const UInt32 String (
        _In_z_ const char *pString,     // ASCII string to hash case-sensitive
        UInt32      uSeed = 0)   // optional seed for hash
    {
        UInt32      uHash2 = 0;
        Size_t      uSize;

        uSize = strlen(pString);
        CsHash32::Compute2(pString, uSize, uSeed, uHash2, &uSeed, &uHash2);
        return uSeed;
    }

    //
    // StringI2: Produce two case-insensitive 32-bit hashes of an ASCII string
    // The results are identical to Compute2() on an uppercased string
    //
    static void StringI2 (
        const char *pString,   // ASCII string to hash case-insensitive
        Size_t      uSize,     // length of string (required)
        UInt32      uSeed1,    // first seed
        UInt32      uSeed2,    // second seed
        UInt32     *uHash1,    // OUT: first hash
        UInt32     *uHash2);   // OUT: second hash

    //
    // StringI: case insensitive hash of string of unknown length
    //
    static const UInt32 StringI (
        _In_z_ const char *pString,     // ASCII string to hash case-insensitive
        size_t      len = (size_t)-1,
        UInt32      uSeed = 0)   // optional seed for hash
    {
        UInt32      uHash2 = 0;
        Size_t      uSize;

        uSize = (len == (size_t)-1) ? strlen(pString) : len;
        CsHash32::StringI2(pString, uSize, uSeed, uHash2, &uSeed, &uHash2);
        return uSeed;
    }

};

class CsHash64
{
private:
    CsHash64 ();                               // no NEW allowed
    CsHash64 ( const CsHash64& );              // no copy allowed
    CsHash64& operator= ( const CsHash64& );   // no assignment allowed

public:


    //
    // Compute hash for a byte array of known length.
    //
    static const UInt64 Compute (
        const void *pData,         // byte array to hash
        Size_t      uSize,          // length of pData
        UInt64      uSeed = 0 )     // seed to hash function; 0 is an OK value
    {
        UInt32 uHash1, uHash2;
        CsHash32::Compute2( pData, uSize, (UInt32) uSeed, (UInt32) (uSeed >> 32), &uHash1, &uHash2);
        return uHash1 | (((UInt64)uHash2) << 32);
    }

    //
    // case-insensitive hash of null terminated string
    // produce the same hash as Compute on an uppercased string
    //
    static const UInt64 StringI (
        const char *pString,
        Size_t      uSize,
        UInt64      uSeed)
    {
        UInt32 uHash1, uHash2;
        CsHash32::StringI2( pString, uSize, (UInt32) uSeed, (UInt32) (uSeed >> 32), &uHash1, &uHash2);
        return uHash1 | (((UInt64)uHash2) << 32);
    }

};
