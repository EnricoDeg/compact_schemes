cmake_minimum_required(VERSION 3.20)
project(canard CXX)

include(FetchContent)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG master
)
FetchContent_MakeAvailable(yaml-cpp)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(driver)
