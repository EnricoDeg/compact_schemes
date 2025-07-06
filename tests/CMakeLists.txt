set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI library
find_package(MPI REQUIRED)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_NVCC_FLAGS -std=c++17)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/35d0c365609296fa4730d62057c487e3cfa030ff.zip
)
# Do NOT install gtest with the project
option(INSTALL_GTEST "Enable installation of googletest." OFF)
FetchContent_MakeAvailable(googletest)

# Find nvrtc and cuda driver libs
FIND_LIBRARY(NVRTC_LIB nvrtc PATHS "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
FIND_LIBRARY(CUDA_DRIVER_LIB cuda PATHS "${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs")

enable_testing()

# physics
cuda_add_executable( test_physics
                     test_physics.cu
                   )
target_include_directories(test_physics PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(test_physics PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(test_physics ${MPI_CXX_LIBRARIES} GTest::gtest_main yaml-cpp::yaml-cpp)
set_property(TARGET test_physics PROPERTY CUDA_STANDARD 17)

# physics serial
cuda_add_executable( test_physics_serial
                     test_physics_serial.cu
                   )
target_include_directories(test_physics_serial PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(test_physics_serial PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(test_physics_serial ${MPI_CXX_LIBRARIES} GTest::gtest_main yaml-cpp::yaml-cpp)
set_property(TARGET test_physics_serial PROPERTY CUDA_STANDARD 17)

# numerics serial
cuda_add_executable( test_numerics_serial
                     test_numerics_serial.cu
                   )
target_include_directories(test_numerics_serial PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(test_numerics_serial PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(test_numerics_serial GTest::gtest_main ${MPI_CXX_LIBRARIES} ${NVRTC_LIB} ${CUDA_DRIVER_LIB} yaml-cpp::yaml-cpp)
set_property(TARGET test_numerics_serial PROPERTY CUDA_STANDARD 17)

# set(UNIT_TEST test_physics)
# set(MPI_COMMAND mpirun -np 4 ${UNIT_TEST})
# add_custom_command(
#   TARGET ${UNIT_TEST}
#   POST_BUILD
#   COMMAND ${MPI_COMMAND})

# Add all tests to GoogleTest
include(GoogleTest)
set_property(TARGET test_physics
             PROPERTY CROSSCOMPILING_EMULATOR 'mpirun -n 2')
gtest_discover_tests(test_physics)
gtest_discover_tests(test_physics_serial)
gtest_discover_tests(test_numerics_serial)
