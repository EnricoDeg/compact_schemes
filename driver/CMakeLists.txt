set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI library
find_package(MPI REQUIRED)

find_package(CUDA REQUIRED)

include(FetchContent)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG master
)
FetchContent_MakeAvailable(yaml-cpp)


set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_NVCC_FLAGS -std=c++17)

cuda_add_executable( canard_driver
                     canard.cu
                   )
target_include_directories(canard_driver PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(canard_driver PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(canard_driver ${MPI_CXX_LIBRARIES} yaml-cpp::yaml-cpp)
set_property(TARGET canard_driver PROPERTY CUDA_STANDARD 17)

# Find nvrtc and cuda driver libs
FIND_LIBRARY(NVRTC_LIB nvrtc PATHS "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
FIND_LIBRARY(CUDA_DRIVER_LIB cuda PATHS "${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs")

cuda_add_executable( canard_driver_rtc
                     canard_rtc.cu
                   )
target_include_directories(canard_driver_rtc PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(canard_driver_rtc PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(canard_driver_rtc ${MPI_CXX_LIBRARIES} ${NVRTC_LIB} ${CUDA_DRIVER_LIB} yaml-cpp::yaml-cpp)
set_property(TARGET canard_driver_rtc PROPERTY CUDA_STANDARD 17)

# Install the binaries
install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/
  DESTINATION ${PROJECT_SOURCE_DIR}/bin
  FILES_MATCHING
  PATTERN "canard_driver"
  )

install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/
  DESTINATION ${PROJECT_SOURCE_DIR}/bin
  FILES_MATCHING
  PATTERN "canard_driver_rtc"
  )
