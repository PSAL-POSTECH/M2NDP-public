SET(BINARY ${CMAKE_PROJECT_NAME}_test)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin)

include(ExternalProject)
include(GoogleTest)
externalproject_add(
  GoogleTest
  URL https://github.com/google/googletest/archive/release-1.8.1.zip
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/lib
  CMAKE_ARGS -DCMAKE_CXX_FLAGS:STRING="-D_GLIBCXX_USE_CXX11_ABI=0"
  INSTALL_COMMAND ""
)
externalproject_get_property(GoogleTest source_dir)
include_directories(${source_dir}/googletest/include)
include_directories(${source_dir}/googlemock/include)

externalproject_get_property(GoogleTest binary_dir)
set(GTEST_LIBRARY_PATH ${binary_dir}/googlemock/gtest/${CMAKE_FIND_LIBRARY_PREFIXES}gtest.a)
set(GTEST_LIBRARY GTest::GTest)
add_library(${GTEST_LIBRARY} UNKNOWN IMPORTED)
set_target_properties(${GTEST_LIBRARY} PROPERTIES
  IMPORTED_LOCATION ${GTEST_LIBRARY_PATH})
add_dependencies(${GTEST_LIBRARY} GoogleTest)

set(GMOCK_LIBRARY_PATH ${binary_dir}/googlemock/${CMAKE_FIND_LIBRARY_PREFIXES}gmock.a)
set(GMOCK_LIBRARY GTest::GMock)
add_library(${GMOCK_LIBRARY} UNKNOWN IMPORTED)
set_target_properties(${GMOCK_LIBRARY} PROPERTIES
  IMPORTED_LOCATION ${GMOCK_LIBRARY_PATH})
add_dependencies(${GMOCK_LIBRARY} GoogleTest)

file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false *.h *.cc)
add_executable(${BINARY} ${TEST_SOURCES})

include_directories("${CMAKE_SOURCE_DIR}/src")
include_directories("${CMAKE_SOURCE_DIR}/extern/ramulator")
include_directories("${CMAKE_SOURCE_DIR}/extern/intersim2")
include_directories("${CMAKE_SOURCE_DIR}/extern/intersim2/networks")
include_directories("${CMAKE_SOURCE_DIR}/extern/intersim2/allocators")
include_directories("${CMAKE_SOURCE_DIR}/extern/intersim2/arbiters")
include_directories("${CMAKE_SOURCE_DIR}/extern/intersim2/examples")
include_directories("${CMAKE_SOURCE_DIR}/extern/intersim2/power")
include_directories("${CMAKE_SOURCE_DIR}/extern/intersim2/routers")

target_link_libraries(${BINARY} PUBLIC GTest::GTest GTest::GMock)
target_link_libraries(${BINARY} PUBLIC ${PROJECT_NAME}_lib ${CONAN_LIBS})

add_test(NAME ${BINARY} COMMAND ${BINARY})