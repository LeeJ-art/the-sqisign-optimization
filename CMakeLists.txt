# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)
project(SQIsign VERSION 2.0 LANGUAGES C ASM)

set(SQISIGN_SO_VERSION "0")
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CTest)

option(ENABLE_STRICT "Build with strict compile options." ON)
option(ENABLE_TESTS  "Enable compilation of tests." ON)
option(ENABLE_CT_TESTING  "Enable compilation for constant time testing." OFF)
option(ENABLE_SIGN "Build with sign functionality" ON)
set(GMP_LIBRARY "SYSTEM" CACHE STRING "Which version of GMP to use: SYSTEM, BUILD or MINI")
set(GF_RADIX "AUTO" CACHE STRING "Set the radix for the gf module (currently supported values: 32 or 64), or AUTO.")

#default is ref
if (NOT DEFINED SQISIGN_BUILD_TYPE)
  SET(SQISIGN_BUILD_TYPE "ref")
endif()

#add defined title into compiled files
if (${SQISIGN_BUILD_TYPE} MATCHES "ref")
  add_compile_definitions(SQISIGN_BUILD_TYPE_REF SQISIGN_GF_IMPL_REF)
elseif (${SQISIGN_BUILD_TYPE} MATCHES "opt")
  add_compile_definitions(SQISIGN_BUILD_TYPE_OPT SQISIGN_GF_IMPL_REF)
elseif (${SQISIGN_BUILD_TYPE} MATCHES "broadwell")
  add_compile_definitions(SQISIGN_BUILD_TYPE_BROADWELL SQISIGN_GF_IMPL_BROADWELL)
elseif (${SQISIGN_BUILD_TYPE} MATCHES "arm64crypto")
  add_compile_definitions(SQISIGN_BUILD_TYPE_ARM64CRYPTO SQISIGN_GF_IMPL_REF)
endif()

#SVARIANT_S = lvl1
SET(SVARIANT_S "lvl1")

include(.cmake/flags.cmake)
include(.cmake/sanitizers.cmake)
include(.cmake/target.cmake)

#whether using sign functionality, and default is on
if(ENABLE_SIGN)
	include(.cmake/gmpconfig.cmake)
	add_compile_definitions(ENABLE_SIGN)
endif()

#BM_BINS is used to store binary files
set(BM_BINS "" CACHE INTERNAL "List of benchmark binaries")

#${PROJECT_SOURCE_DIR} = absolute path of the-sqisign
set(SELECT_IMPL_TYPE ${PROJECT_SOURCE_DIR}/.cmake/impl_type.cmake)
set(SELECT_SQISIGN_VARIANT ${PROJECT_SOURCE_DIR}/.cmake/sqisign_variant.cmake)
set(INC_PUBLIC ${PROJECT_SOURCE_DIR}/include)

#call src/CMakeLists.txt
add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(test)

include(.cmake/bm.cmake)

#if(ENABLE_TESTS)
#    enable_testing()
#    add_subdirectory(test)
#endif()
