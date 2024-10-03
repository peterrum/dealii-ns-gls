#
# Sanity check:
#

get_filename_component(_source "${CMAKE_SOURCE_DIR}" REALPATH)
get_filename_component(_build  "${CMAKE_BINARY_DIR}" REALPATH)
if("${_source}" STREQUAL "${_build}")
  message(FATAL_ERROR
    "\nRefusing to configure the project in the source directory. This "
    "operation would globber important files. You need to configure in a "
    "separate build directory. It is easiest to simply invoke $ make, which "
    "will configure ryujin in the directory ./build.\n"
    "You will need to clean up ./CMakeCache.txt and ./CMakeFiles by hand, "
    "or by running $ make cleanup_insource which will delete these files "
    "and directories for you"
    )
endif()

#
# Set up project:
#

cmake_minimum_required(VERSION 3.5)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_STANDARD 20)

project(gls)

find_package(deal.II 9.3 QUIET 
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})
if(NOT ${deal.II_FOUND})
  message(
    FATAL_ERROR
      "\n"
      "*** Could not locate deal.II. ***\n\n"
      "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
      "or set an environment variable \"DEAL_II_DIR\" that contains this path.")
endif()
message("-- deal.II found: ${DEAL_II_DIR}")

if(NOT DEAL_II_WITH_P4EST)
  message(
    FATAL_ERROR
      "\nThis library requires a deal.II installation built with support for p4est but it appears to be missing!\n"
  )
endif()

if(NOT DEAL_II_WITH_TRILINOS)
  message(
    FATAL_ERROR
      "\nThis library requires a deal.II installation built with support for Trilinos but it appears to be missing!\n"
  )
endif()
deal_ii_initialize_cached_variables()

deal_ii_query_git_information(NS_GLS)

CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/revision.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/revision.h
)

set(libName "gls")

DEAL_II_INITIALIZE_CACHED_VARIABLES()

include_directories(include ${CMAKE_CURRENT_BINARY_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/include)

file(GLOB_RECURSE sources "${CMAKE_SOURCE_DIR}/include/*.cc")

add_library(${libName} ${sources})

TARGET_COMPILE_DEFINITIONS(${libName} PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

deal_ii_setup_target(${libName})

set(projectName "gls-app")

message("-- building executable: gls-app")
add_executable(${projectName} main.cc)
deal_ii_setup_target(${projectName})

target_link_libraries(${projectName} ${libName})

set(projectName "gls-vmult")

message("-- building executable: gls-vmult")
add_executable(${projectName} performance.cc)
deal_ii_setup_target(${projectName})

target_link_libraries(${projectName} ${libName})

# Define custom targets to easily switch the build type:
add_custom_target(
  debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMENT
    "Switch CMAKE_BUILD_TYPE to Debug\nPlease call 'make' to build the debug project"
)

add_custom_target(
  release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMENT
    "Switch CMAKE_BUILD_TYPE to Release\nPlease call 'make' to build the release project"
)

