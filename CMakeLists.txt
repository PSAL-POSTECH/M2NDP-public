cmake_minimum_required(VERSION 3.15.0)

if(PERFORMANCE_BUILD STREQUAL "1" OR ACCELSIM_BUILD STREQUAL "1")
  set(PROJECT_NAME "NDPSim")
else()
  set(PROJECT_NAME "FuncSim")
endif()

project(${PROJECT_NAME})
set(CONAN_DISABLE_CHECK_COMPILER ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -Wall -O1")

if(DEFINED MEM_ACCESS_SIZE)
  add_definitions(-DMEM_ACCESS_SIZE=${MEM_ACCESS_SIZE})
  message("Setting memory access size to ${MEM_ACCESS_SIZE}")
endif()

file(GLOB_RECURSE SRC_FILES
  "${CMAKE_SOURCE_DIR}/src/*.h"
  "${CMAKE_SOURCE_DIR}/src/*.cc"
)

# conan setup
execute_process(COMMAND "conan" "install" "${CMAKE_SOURCE_DIR}" "--install-folder" "${CMAKE_SOURCE_DIR}/build")
include("${CMAKE_SOURCE_DIR}/build/conanbuildinfo.cmake")
conan_basic_setup()
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/build/bin")
set(LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build/lib")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-register -Wno-deprecated-declarations -ggdb")
if(PERFORMANCE_BUILD STREQUAL "1" OR ACCELSIM_BUILD STREQUAL "1")
  message("Building performance model")
  find_package(FLEX)
  find_package(BISON)
  set(LEXER_OUT "${CMAKE_SOURCE_DIR}/extern/intersim2/lex.yy.c")
  set(PARSER_OUT "${CMAKE_SOURCE_DIR}/extern/intersim2/y.tab.c")
  BISON_TARGET(PARSER "${CMAKE_SOURCE_DIR}/extern/intersim2/config.y" "${PARSER_OUT}" DEFINES_FILE "${CMAKE_SOURCE_DIR}/extern/intersim2/y.tab.h")

  FLEX_TARGET(LEXER "${CMAKE_SOURCE_DIR}/extern/intersim2/config.l" "${LEXER_OUT}")
  ADD_FLEX_BISON_DEPENDENCY(LEXER PARSER)
  file(GLOB EXTERN_FILES
    "${CMAKE_SOURCE_DIR}/extern/ramulator/*.h"
    "${CMAKE_SOURCE_DIR}/extern/ramulator/*.cc"
    "${CMAKE_SOURCE_DIR}/extern/intersim2/*.h"
    "${CMAKE_SOURCE_DIR}/extern/intersim2/*.cc"
    "${CMAKE_SOURCE_DIR}/extern/intersim2/*.hpp"
    "${CMAKE_SOURCE_DIR}/extern/intersim2/*.cpp"
    "${CMAKE_SOURCE_DIR}/extern/intersim2/networks/*.cpp"
    "${CMAKE_SOURCE_DIR}/extern/intersim2/allocators/*.cpp"
    "${CMAKE_SOURCE_DIR}/extern/intersim2/arbiters/*.cpp"
    "${CMAKE_SOURCE_DIR}/extern/intersim2/examples/*.cpp"
    "${CMAKE_SOURCE_DIR}/extern/intersim2/power/*.cpp"
    "${CMAKE_SOURCE_DIR}/extern/intersim2/routers/*.cpp")
  include_directories(${CMAKE_CURRENT_BINARY_DIR})
  include_directories("${CMAKE_SOURCE_DIR}/src")
  include_directories("${CMAKE_SOURCE_DIR}/extern/ramulator")
  include_directories("${CMAKE_SOURCE_DIR}/extern/intersim2")
  include_directories("${CMAKE_SOURCE_DIR}/extern/intersim2/networks")
  include_directories("${CMAKE_SOURCE_DIR}/extern/intersim2/allocators")
  include_directories("${CMAKE_SOURCE_DIR}/extern/intersim2/arbiters")
  include_directories("${CMAKE_SOURCE_DIR}/extern/intersim2/examples")
  include_directories("${CMAKE_SOURCE_DIR}/extern/intersim2/power")
  include_directories("${CMAKE_SOURCE_DIR}/extern/intersim2/routers")
  add_compile_definitions(TIMING_SIMULATION)
  if(ACCELSIM_BUILD STREQUAL "1")
    include_directories("${CMAKE_SOURCE_DIR}/include")
    message("Building accelsim model")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DACCELSIM_BUILD -DTIMING_SIMULATION")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMEM_ACCESS_SIZE=32")
  else()
    set(CMAKE_CXX_STANDARD 17)
    message("Building perf_runner model")
  endif()

  add_library(${PROJECT_NAME}_lib SHARED ${SRC_FILES} ${LEXER_OUT} ${PARSER_OUT} ${EXTERN_FILES}
    "${CMAKE_SOURCE_DIR}/include/m2ndp_module.h")
  target_link_libraries(${PROJECT_NAME}_lib ${FL_LIBRARIES})
  target_link_libraries(${PROJECT_NAME}_lib ${CONAN_LIBS})
  include_directories("${CMAKE_SOURCE_DIR}/perf_runner")
  if(KVRUN STREQUAL "1")
    add_compile_definitions(KVRUN)
    message("Building KV RUN")
    add_executable(${PROJECT_NAME} "${CMAKE_SOURCE_DIR}/perf_runner/kv_runner.cc"
    "${CMAKE_SOURCE_DIR}/perf_runner/kv_runner.h"
    "${CMAKE_SOURCE_DIR}/perf_runner/main.cc")
  elseif(SCALABILITY STREQUAL "1")
    add_compile_definitions(SCALABILITY)
    message("Building Scalability")
    add_executable(${PROJECT_NAME} "${CMAKE_SOURCE_DIR}/perf_runner/scalability_runner.cc"
    "${CMAKE_SOURCE_DIR}/perf_runner/scalability_runner.h"
    "${CMAKE_SOURCE_DIR}/perf_runner/main.cc")
  else()
  add_executable(${PROJECT_NAME} "${CMAKE_SOURCE_DIR}/perf_runner/simulation_runner.h"
    "${CMAKE_SOURCE_DIR}/perf_runner/simulation_runner.cc"
    "${CMAKE_SOURCE_DIR}/perf_runner/main.cc")
  endif()
  
  target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_lib)
  # enable_testing()
  # add_subdirectory(test)
else()
  message("Building functional model")
  set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/build/bin")
  include_directories("${CMAKE_SOURCE_DIR}/src")
  add_executable(${PROJECT_NAME} ${SRC_FILES} "${CMAKE_SOURCE_DIR}/functional_runner/main.cc")
  target_link_libraries(${PROJECT_NAME} ${CONAN_LIBS})
endif()

