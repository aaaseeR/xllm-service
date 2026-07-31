if(NOT BUILD_TESTING)
  return()
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(
  NAME KubernetesDeploymentContractTest
  COMMAND
    ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/validate.py
    --base-dir
    ${CMAKE_CURRENT_SOURCE_DIR}/base
    --http-route
    ${CMAKE_CURRENT_SOURCE_DIR}/httproute.yaml
)

add_test(
  NAME KubernetesDeploymentContractUnitTest
  COMMAND
    ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/validate_test.py
)

set_tests_properties(
  KubernetesDeploymentContractTest
  KubernetesDeploymentContractUnitTest
  PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}"
)
