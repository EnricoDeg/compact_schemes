# Install the library
install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/
  DESTINATION ${PROJECT_SOURCE_DIR}/include
  FILES_MATCHING
  PATTERN "*.hpp"
  )
