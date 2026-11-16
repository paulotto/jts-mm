// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/* Tiny check harness for the fix-B host tests: real checks that survive NDEBUG. */
#pragma once

#include <stdio.h>

static int fixb_failures;
static int fixb_checks;

#define CHECK(cond)                                                                   \
    do {                                                                              \
        fixb_checks++;                                                                \
        if (!(cond)) {                                                                \
            fixb_failures++;                                                          \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                             \
    } while (0)

#define CHECK_EQ(a, b)                                                                  \
    do {                                                                                \
        long long _a = (long long)(a), _b = (long long)(b);                             \
        fixb_checks++;                                                                  \
        if (_a != _b) {                                                                 \
            fixb_failures++;                                                            \
            fprintf(stderr, "%s:%d: CHECK_EQ failed: %s (%lld) != %s (%lld)\n", __FILE__, \
                    __LINE__, #a, _a, #b, _b);                                          \
        }                                                                               \
    } while (0)

static inline int fixb_report(const char *name) {
    printf("%s: %d checks, %d failures\n", name, fixb_checks, fixb_failures);
    return fixb_failures ? 1 : 0;
}
