#pragma once

// A simple means for extracting the attribute bits and word offset
// fields from a "location" as recorded in the index.

enum WordAttribute
{
    ZeroAttributeState  = 0,

    // UrlWord attribute values

    ServiceUrlWord      = 0,
    SubDomainUrlWord    = 1,
    BaseDomainUrlWord   = 2,
    TopDomainUrlWord    = 3,
    PortUrlWord         = 4,
    PathUrlWord         = 5,
    QueryUrlWord        = 6,
    AnyUrlWord          = 7,

    NumberOfUrlAttributeStates      = 8,

    // BodyText attribute values

    NormalBodyText      = 0,
    NavigationBodyText  = 1,
    Reserved1BodyText   = 2,
    Reserved2BodyText   = 3,
    AnyBodyText         = 4,

    NumberOfTextAttributeStates     = 5,

    MaximumNumberOfAttributeStates  = 8
};
