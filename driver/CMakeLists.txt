set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI library
find_package(MPI REQUIRED)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_NVCC_FLAGS -std=c++17)

cuda_add_executable( canard_driver
                canard.cu
              )
target_include_directories(canard_driver PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(canard_driver PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(canard_driver ${MPI_CXX_LIBRARIES})
set_property(TARGET canard_driver PROPERTY CUDA_STANDARD 17)

# Install the library
install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/
  DESTINATION ${PROJECT_SOURCE_DIR}/bin
  FILES_MATCHING
  PATTERN "canard_driver"
  )
