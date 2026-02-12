CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
cmake_policy(SET CMP0060 NEW)
FIND_PACKAGE(deal.II 8.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

SET( TARGET_SRC dummy.cc
  include/fe_time.cc
  include/exact_solution.cc
  include/stmg.cc
  include/operators.cc
  include/parameters.cc
  include/stokes.cc
  include/newton.cc
)
INCLUDE_DIRECTORIES(include)

PROJECT(stfem)

ADD_LIBRARY(stfem SHARED ${TARGET_SRC})
target_include_directories(stfem PUBLIC include)

ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug\nPlease call 'make' to build the debug project"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Release\nPlease call 'make' to build the release project"
  )

DEAL_II_INITIALIZE_CACHED_VARIABLES()
DEAL_II_SETUP_TARGET(stfem)

IF(EXISTS ${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt)
  SET(DEAL_II_HAVE_TESTS_DIRECTORY TRUE)
ENDIF()
IF(DEAL_II_HAVE_TESTS_DIRECTORY)
  add_definitions(-DTESTDIRECTORY="${PROJECT_SOURCE_DIR}/tests/")
ENDIF()

enable_testing()
ADD_SUBDIRECTORY(tests)
