FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)
FetchContent_MakeAvailable(googletest)

set(TEST_SRC_FILES
    test_performance.cu
    test_utils.cu
)

add_executable(cuda_tests ${TEST_SRC_FILES})
target_link_libraries(cuda_tests gtest gtest_main cuda)
target_include_directories(cuda_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../include)

include(GoogleTest)
gtest_discover_tests(cuda_tests)