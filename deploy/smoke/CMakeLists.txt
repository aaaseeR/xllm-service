if(NOT BUILD_TESTING)
  return()
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(
  NAME GatewayRetryClientTest
  COMMAND
    ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/retry_client_test.py
)

set_tests_properties(
  GatewayRetryClientTest
  PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}"
)
