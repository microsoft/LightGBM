/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "CsHash.h"

/*
 * Copied and modified from http://burtleburtle.net/bob/c/lookup3.c,
 * where it is Public Domain.
 */

#define UPPER(c) \
  ((UInt32)((((c) >= 'a') && ((c) <= 'z')) ? (c) - ('a' - 'A') : (c)))

// Compute2: Compute two hashes of an array of bytes.
// The first hash is slightly better mixed than the second hash.
void CsHash32::Compute2(
    const void *pData,  // byte array to hash; may be null if uSize==0
    Size_t uSize,       // length of pData
    UInt32 uSeed1,      // first seed
    UInt32 uSeed2,      // second seed
    UInt32 *uHash1,     // OUT: first hash (may not be null)
    UInt32 *uHash2)     // OUT: second hash (may not be null)
{
  UInt32 a, b, c;
  const UInt32 *k = (const UInt32 *)pData;  // read 32-bit chunks
  const UInt8 *k1;

  // Set up the internal state
  a = b = c = 0xdeadbeef + ((UInt32)uSize) + uSeed1;
  c += uSeed2;

  // all but last block: aligned reads and affect 32 bits of (a,b,c)
  while (uSize > 12) {
    a += k[0];
    b += k[1];
    c += k[2];
    Mix(a, b, c);
    uSize -= 12;
    k += 3;
  }

  // handle the last (probably partial) block
  k1 = (const UInt8 *)k;
  switch (uSize) {
    case 12:
      c += k[2];
      b += k[1];
      a += k[0];
      break;
    case 11:
      c += ((UInt32)k1[10]) << 16;  // fall through
    case 10:
      c += ((UInt32)k1[9]) << 8;  // fall through
    case 9:
      c += (UInt32)k1[8];  // fall through
    case 8:
      b += k[1];
      a += k[0];
      break;
    case 7:
      b += ((UInt32)k1[6]) << 16;  // fall through
    case 6:
      b += ((UInt32)k1[5]) << 8;  // fall through
    case 5:
      b += ((UInt32)k1[4]);  // fall through
    case 4:
      a += k[0];
      break;
    case 3:
      a += ((UInt32)k1[2]) << 16;  // fall through
    case 2:
      a += ((UInt32)k1[1]) << 8;  // fall through
    case 1:
      a += k1[0];
      break;
    case 0:
      *uHash1 = c;
      *uHash2 = b;
      return;
  }

  Final(a, b, c);
  *uHash1 = c;
  *uHash2 = b;
  return;
}

// Hash a string of unknown length case insensitive.  I can't just call
// Compute() without allocating a copy of the string, which could have
// complications because there's no max length for strings.
void CsHash32::StringI2(const char *pString, Size_t uSize, UInt32 uSeed1,
                        UInt32 uSeed2, UInt32 *uHash1, UInt32 *uHash2) {
  UInt32 a, b, c;
  const UInt8 *k;

  k = (const UInt8 *)pString;

  // Set up the internal state
  a = b = c = 0xdeadbeef + ((UInt32)uSize) + uSeed1;
  c += uSeed2;

  // all but the last block: affect some 32 bits of (a,b,c)
  while (uSize > 12) {
    a += UPPER(k[0]);
    a += UPPER(k[1]) << 8;
    a += UPPER(k[2]) << 16;
    a += UPPER(k[3]) << 24;
    b += UPPER(k[4]);
    b += UPPER(k[5]) << 8;
    b += UPPER(k[6]) << 16;
    b += UPPER(k[7]) << 24;
    c += UPPER(k[8]);
    c += UPPER(k[9]) << 8;
    c += UPPER(k[10]) << 16;
    c += UPPER(k[11]) << 24;
    Mix(a, b, c);
    uSize -= 12;
    k += 12;
  }

  // last block: affect all 32 bits of (c)
  switch (uSize)  // all the case statements fall through
  {
    case 12:
      c += UPPER(k[11]) << 24;
    case 11:
      c += UPPER(k[10]) << 16;
    case 10:
      c += UPPER(k[9]) << 8;
    case 9:
      c += UPPER(k[8]);
    case 8:
      b += UPPER(k[7]) << 24;
    case 7:
      b += UPPER(k[6]) << 16;
    case 6:
      b += UPPER(k[5]) << 8;
    case 5:
      b += UPPER(k[4]);
    case 4:
      a += UPPER(k[3]) << 24;
    case 3:
      a += UPPER(k[2]) << 16;
    case 2:
      a += UPPER(k[1]) << 8;
    case 1:
      a += UPPER(k[0]);
      break;
    case 0:
      *uHash1 = c;
      *uHash2 = b;
      return;
  }

  Final(a, b, c);
  *uHash1 = c;
  *uHash2 = b;
  return;
}
