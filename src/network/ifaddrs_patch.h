/*!
 * Copyright (c) 2009, Jay Loden, Dave Daeschler, Giampaolo Rodola.
 * Licensed under the BSD 3-Clause License.
 * See https://github.com/giampaolo/psutil/blob/master/LICENSE
 */

/*
 * - https://lists.samba.org/archive/samba-technical/2009-February/063079.html
 * - https://github.com/giampaolo/psutil/blob/master/psutil/arch/solaris/v10/ifaddrs.h
 */

#ifndef LIGHTGBM_NETWORK_IFADDRS_PATCH_H_
#define LIGHTGBM_NETWORK_IFADDRS_PATCH_H_

#include <sys/socket.h>
#include <net/if.h>

#undef  ifa_dstaddr
#undef  ifa_broadaddr
#define ifa_broadaddr ifa_dstaddr

struct ifaddrs {
    struct ifaddrs  *ifa_next;
    char            *ifa_name;
    unsigned int     ifa_flags;
    struct sockaddr *ifa_addr;
    struct sockaddr *ifa_netmask;
    struct sockaddr *ifa_dstaddr;
};

extern int getifaddrs(struct ifaddrs **);
extern void freeifaddrs(struct ifaddrs *);

#endif  // LIGHTGBM_NETWORK_IFADDRS_PATCH_H_
