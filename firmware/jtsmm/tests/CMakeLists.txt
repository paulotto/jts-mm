# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.13)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(jtsmm_host_tests C)

set(CMAKE_C_STANDARD 11)

add_compile_definitions(HOST_TEST)
# Tests use CHECK (tests/check.h), never assert(): building with NDEBUG proves
# no test silently depends on an assert that a release build compiles out.
add_compile_definitions(NDEBUG)
add_compile_definitions(CJSON_NESTING_LIMIT=32) # as in the firmware build

set(FW_SRC ${CMAKE_CURRENT_LIST_DIR}/..)

add_executable(test_host_link_crc
    test_host_link_crc.c
    ${FW_SRC}/src/runtime/host_link.c
    ${FW_SRC}/src/runtime/ringbuf.c
    ${FW_SRC}/tests/stubs/tusb_stub.c
)
target_include_directories(test_host_link_crc PRIVATE
    ${FW_SRC}/include
    ${FW_SRC}/tests/stubs
)

add_executable(test_host_link_tx
    test_host_link_tx.c
    ${FW_SRC}/src/runtime/host_link.c
    ${FW_SRC}/src/runtime/ringbuf.c
    ${FW_SRC}/tests/stubs/tusb_stub.c
)
target_include_directories(test_host_link_tx PRIVATE
    ${FW_SRC}/include
    ${FW_SRC}/tests/stubs
)

add_executable(test_ringbuf
    test_ringbuf.c
    ${FW_SRC}/src/runtime/ringbuf.c
)
target_include_directories(test_ringbuf PRIVATE
    ${FW_SRC}/include
    ${FW_SRC}/tests/stubs
)

add_executable(test_imu_watchdog
    test_imu_watchdog.c
)
target_include_directories(test_imu_watchdog PRIVATE ${FW_SRC}/include ${FW_SRC}/tests)
add_test(NAME imu_watchdog COMMAND test_imu_watchdog)

add_executable(test_config_json
    test_config_json.c
    ${FW_SRC}/src/runtime/config_json.c
    ${FW_SRC}/src/runtime/ringbuf.c  # ringbuf_slot_bytes: ring byte budgets
    ${FW_SRC}/src/domain/config_defaults.c
    ${FW_SRC}/src/domain/bno085_reports.c
    ${FW_SRC}/third_party/cjson/cJSON.c
)
target_include_directories(test_config_json PRIVATE
    ${FW_SRC}/include
    ${FW_SRC}/third_party/cjson
    ${FW_SRC}/tests/stubs
)
target_compile_definitions(test_config_json PRIVATE
    "GOLDEN_MCU_APPLY_JSON=\"${CMAKE_CURRENT_LIST_DIR}/../../../schema/v1/golden/mcu_apply_template.json\""
)

add_executable(test_config_json_keys
    test_config_json_keys.c
    ${FW_SRC}/src/runtime/config_json.c
    ${FW_SRC}/src/runtime/ringbuf.c  # ringbuf_slot_bytes: ring byte budgets
    ${FW_SRC}/src/domain/config_defaults.c
    ${FW_SRC}/src/domain/bno085_reports.c
    ${FW_SRC}/third_party/cjson/cJSON.c
)
target_include_directories(test_config_json_keys PRIVATE
    ${FW_SRC}/include
    ${FW_SRC}/third_party/cjson
    ${FW_SRC}/tests/stubs
)
target_compile_definitions(test_config_json_keys PRIVATE
    "GOLDEN_MCU_APPLY_JSON=\"${CMAKE_CURRENT_LIST_DIR}/../../../schema/v1/golden/mcu_apply_template.json\""
)

add_executable(test_trigger_timing
    test_trigger_timing.c
    ${FW_SRC}/src/drivers/trigger_timing.c
)
target_include_directories(test_trigger_timing PRIVATE
    ${FW_SRC}/include
    ${FW_SRC}/tests/stubs
)

add_executable(test_trigger_config
    test_trigger_config.c
    ${FW_SRC}/src/domain/trigger_config.c
    ${FW_SRC}/src/domain/config_defaults.c
    ${FW_SRC}/src/drivers/trigger_timing.c
    ${FW_SRC}/src/runtime/ringbuf.c
    ${FW_SRC}/tests/stubs/trigger_pio_stub.c
)
target_include_directories(test_trigger_config PRIVATE
    ${FW_SRC}/include
    ${FW_SRC}/tests/stubs
)

enable_testing()
add_test(NAME host_link_crc COMMAND test_host_link_crc)
add_test(NAME host_link_tx COMMAND test_host_link_tx)
add_test(NAME ringbuf COMMAND test_ringbuf)
add_test(NAME config_json COMMAND test_config_json)
add_test(NAME config_json_keys COMMAND test_config_json_keys)
add_test(NAME trigger_timing COMMAND test_trigger_timing)
add_test(NAME trigger_config COMMAND test_trigger_config)

# config_apply state machine against recording stubs of every hardware-facing
# call (public headers only). fix_B_stubs supplies hardware/i2c.h + pico/types.h.
add_executable(test_config_apply
    test_config_apply.c
    ${FW_SRC}/src/runtime/config_apply.c
    ${FW_SRC}/src/runtime/config_json.c
    ${FW_SRC}/src/runtime/ringbuf.c  # ringbuf_slot_bytes: ring byte budgets
    ${FW_SRC}/src/domain/config_defaults.c
    ${FW_SRC}/src/domain/bno085_reports.c
    ${FW_SRC}/src/drivers/trigger_timing.c
    ${FW_SRC}/third_party/cjson/cJSON.c
    ${FW_SRC}/tests/stubs/apply_stubs.c
)
target_include_directories(test_config_apply PRIVATE
    ${FW_SRC}/tests/fix_B_stubs
    ${FW_SRC}/include
    ${FW_SRC}/third_party/cjson
    ${FW_SRC}/tests/stubs
)
target_compile_definitions(test_config_apply PRIVATE
    "GOLDEN_MCU_APPLY_JSON=\"${CMAKE_CURRENT_LIST_DIR}/../../../schema/v1/golden/mcu_apply_template.json\""
)
add_test(NAME config_apply COMMAND test_config_apply)

# Config plumbing: every apply value reaches its driver call. The REAL CDC parser,
# config / domain / capture-session code against recording driver stubs
# (stubs/rec_drivers.c; bench-command back-ends inert in stubs/rec_cmd_stubs.c).
add_executable(test_config_apply_drivers
    test_config_apply_drivers.c
    ${FW_SRC}/src/runtime/config_apply.c
    ${FW_SRC}/src/runtime/config_json.c
    ${FW_SRC}/src/runtime/capture_session.c
    ${FW_SRC}/src/runtime/cmd_parser.c
    ${FW_SRC}/src/runtime/ringbuf.c
    ${FW_SRC}/src/domain/config_defaults.c
    ${FW_SRC}/src/domain/emg_config.c
    ${FW_SRC}/src/domain/imu_config.c
    ${FW_SRC}/src/domain/imu_timing.c
    ${FW_SRC}/src/domain/trigger_config.c
    ${FW_SRC}/src/domain/bno085_reports.c
    ${FW_SRC}/src/drivers/trigger_timing.c
    ${FW_SRC}/third_party/cjson/cJSON.c
    ${FW_SRC}/tests/stubs/rec_drivers.c
    ${FW_SRC}/tests/stubs/rec_cmd_stubs.c
)
target_include_directories(test_config_apply_drivers PRIVATE
    ${FW_SRC}/tests/fix_B_stubs
    ${FW_SRC}/include
    ${FW_SRC}/third_party/cjson
    ${FW_SRC}/tests/stubs
    ${FW_SRC}/tests
)
target_compile_definitions(test_config_apply_drivers PRIVATE
    "GOLDEN_MCU_APPLY_JSON=\"${CMAKE_CURRENT_LIST_DIR}/../../../schema/v1/golden/mcu_apply_template.json\""
)
add_test(NAME config_apply_drivers COMMAND test_config_apply_drivers)

# Register level: payload -> config_json -> emg_config_apply -> the REAL ads1299.c
# -> fake SPI0 with an ADS1299 register model (ads_stubs/fake_ads1299.c).
add_executable(test_ads1299_registers
    test_ads1299_registers.c
    ${FW_SRC}/src/drivers/ads1299.c
    ${FW_SRC}/src/domain/emg_config.c
    ${FW_SRC}/src/runtime/config_json.c
    ${FW_SRC}/src/runtime/ringbuf.c
    ${FW_SRC}/src/domain/config_defaults.c
    ${FW_SRC}/src/domain/bno085_reports.c
    ${FW_SRC}/third_party/cjson/cJSON.c
    ${FW_SRC}/tests/ads_stubs/fake_ads1299.c
)
target_include_directories(test_ads1299_registers PRIVATE
    ${FW_SRC}/tests/ads_stubs
    ${FW_SRC}/tests/fix_B_stubs
    ${FW_SRC}/include
    ${FW_SRC}/third_party/cjson
    ${FW_SRC}/tests/stubs
    ${FW_SRC}/tests
)
target_compile_definitions(test_ads1299_registers PRIVATE
    "GOLDEN_MCU_APPLY_JSON=\"${CMAKE_CURRENT_LIST_DIR}/../../../schema/v1/golden/mcu_apply_template.json\""
)
add_test(NAME ads1299_registers COMMAND test_ads1299_registers)

# Fix package A (ADS1299 / EMG): pure register + stream arithmetic.
add_executable(test_fix_A_ads1299 test_fix_A_ads1299.c)
target_include_directories(test_fix_A_ads1299 PRIVATE ${FW_SRC}/include ${FW_SRC}/tests/stubs)
target_compile_options(test_fix_A_ads1299 PRIVATE -Wall -Wextra -Werror)
add_test(NAME fix_A_ads1299 COMMAND test_fix_A_ads1299)

add_executable(test_fix_A_emg test_fix_A_emg.c)
target_include_directories(test_fix_A_emg PRIVATE ${FW_SRC}/include ${FW_SRC}/tests/stubs)
target_compile_options(test_fix_A_emg PRIVATE -Wall -Wextra -Werror)
add_test(NAME fix_A_emg COMMAND test_fix_A_emg)

# ---- fix package B: IMU / BNO085 / I2C DMA (own stubs in fix_B_stubs, fakes in fix_B_fakes.c) ----
set(FIX_B_INCLUDES ${FW_SRC}/tests/fix_B_stubs ${FW_SRC}/include ${FW_SRC}/tests/stubs ${FW_SRC}/tests)
set(FIX_B_WARN -Wall -Wextra -Wno-unused-parameter -Werror)
foreach(fixb_test test_fix_B_shtp test_fix_B_stream test_fix_B_i2c test_fix_B_recovery test_fix_B_apply
        test_imu_timing)
    add_executable(${fixb_test}
        ${fixb_test}.c
        fix_B_fakes.c
        ${FW_SRC}/src/drivers/bno085.c
        ${FW_SRC}/src/domain/imu_config.c
        ${FW_SRC}/src/domain/imu_timing.c
        ${FW_SRC}/src/domain/bno085_reports.c
        ${FW_SRC}/src/domain/config_defaults.c
        ${FW_SRC}/src/runtime/ringbuf.c
    )
    target_include_directories(${fixb_test} PRIVATE ${FIX_B_INCLUDES})
    target_compile_options(${fixb_test} PRIVATE ${FIX_B_WARN})
    add_test(NAME ${fixb_test} COMMAND ${fixb_test})
endforeach()
# Same sources with both IMU timestamp switches off: the fallback must be the
# previous behaviour (poll time for every report).
add_executable(test_imu_timing_off
    test_imu_timing_off.c
    fix_B_fakes.c
    ${FW_SRC}/src/drivers/bno085.c
    ${FW_SRC}/src/domain/imu_config.c
    ${FW_SRC}/src/domain/imu_timing.c
    ${FW_SRC}/src/domain/bno085_reports.c
    ${FW_SRC}/src/domain/config_defaults.c
    ${FW_SRC}/src/runtime/ringbuf.c
)
target_include_directories(test_imu_timing_off PRIVATE ${FIX_B_INCLUDES})
target_compile_options(test_imu_timing_off PRIVATE ${FIX_B_WARN})
target_compile_definitions(test_imu_timing_off PRIVATE IMU_TS_USE_INT_EDGE=0 IMU_TS_USE_SH2_DELAY=0)
add_test(NAME test_imu_timing_off COMMAND test_imu_timing_off)
# The real i2c_dma.c against register/DMA stubs (i2c_dma_stubs): abort handshake
# vs. TX_ABRT ownership.
add_executable(test_i2c_dma_abort
    test_i2c_dma_abort.c
    ${FW_SRC}/src/platform/i2c_dma.c
)
target_include_directories(test_i2c_dma_abort PRIVATE
    ${FW_SRC}/tests/i2c_dma_stubs ${FW_SRC}/tests/fix_B_stubs ${FW_SRC}/include ${FW_SRC}/tests/stubs)
target_compile_options(test_i2c_dma_abort PRIVATE -Wall -Wextra -Werror)
add_test(NAME i2c_dma_abort COMMAND test_i2c_dma_abort)
# ---- end fix package B ----

# Warnings for package C's tests and the firmware sources they compile (not
# third-party cJSON; the fix_A/fix_B targets set their own flags).
file(GLOB JTSMM_TEST_OWN_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/test_host_link*.c ${CMAKE_CURRENT_LIST_DIR}/test_ringbuf.c
    ${CMAKE_CURRENT_LIST_DIR}/test_config_json*.c ${CMAKE_CURRENT_LIST_DIR}/test_trigger*.c
    ${CMAKE_CURRENT_LIST_DIR}/test_ads1299_registers.c ${CMAKE_CURRENT_LIST_DIR}/ads_stubs/*.c
    ${CMAKE_CURRENT_LIST_DIR}/stubs/*.c)
set(JTSMM_TEST_FW_SOURCES
    ${FW_SRC}/src/runtime/host_link.c ${FW_SRC}/src/runtime/config_json.c
    ${FW_SRC}/src/drivers/trigger_timing.c ${FW_SRC}/src/domain/trigger_config.c
    ${FW_SRC}/src/runtime/config_apply.c)
set_source_files_properties(${JTSMM_TEST_OWN_SOURCES} ${JTSMM_TEST_FW_SOURCES}
    PROPERTIES COMPILE_OPTIONS "-Wall;-Wextra")
# Positional case tables leave trailing flags zero on purpose.
set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/test_config_apply_drivers.c
    PROPERTIES COMPILE_OPTIONS "-Wall;-Wextra;-Wno-missing-field-initializers")
