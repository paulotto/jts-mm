// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

#pragma once

/*
 * Minimal test assertion that survives NDEBUG. Bare assert() compiles to
 * nothing in a release build, which silently turns every test into a pass (and
 * drops any side effect written inside it, e.g. assert(ringbuf_push(...))).
 * CHECK always evaluates its argument exactly once and counts failures;
 * CHECK_DONE() makes main() return non-zero if any check failed.
 */

#include <stdio.h>

static int check_failures;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            check_failures++;                                                    \
        }                                                                        \
    } while (0)

#define CHECK_EQ_U(a, b)                                                         \
    do {                                                                         \
        unsigned long long check_a_ = (unsigned long long)(a);                   \
        unsigned long long check_b_ = (unsigned long long)(b);                   \
        if (check_a_ != check_b_) {                                              \
            fprintf(stderr, "%s:%d: CHECK failed: %s == %s (%llu != %llu)\n",    \
                    __FILE__, __LINE__, #a, #b, check_a_, check_b_);             \
            check_failures++;                                                    \
        }                                                                        \
    } while (0)

#define CHECK_DONE(name)                                                         \
    do {                                                                         \
        if (check_failures != 0) {                                               \
            fprintf(stderr, "%s: %d check(s) failed\n", name, check_failures);   \
            return 1;                                                            \
        }                                                                        \
        printf("%s OK\n", name);                                                 \
        return 0;                                                                \
    } while (0)
