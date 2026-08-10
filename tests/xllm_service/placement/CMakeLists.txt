include(cc_test)

cc_test(
  NAME
    placement_planner_test
  SRCS
    placement_planner_test.cpp
  DEPS
    :placement_planner
    GTest::gtest_main
)

cc_test(
  NAME
    placement_observation_collector_test
  SRCS
    placement_observation_collector_test.cpp
  DEPS
    :placement_observation_collector
    GTest::gtest_main
)

cc_test(
  NAME
    placement_input_builder_test
  SRCS
    placement_input_builder_test.cpp
  DEPS
    :placement_input_builder
    GTest::gtest_main
)

cc_test(
  NAME
    placement_budget_allocator_test
  SRCS
    placement_budget_allocator_test.cpp
  DEPS
    :placement_budget_allocator
    GTest::gtest_main
)

cc_test(
  NAME
    placement_lifecycle_test
  SRCS
    placement_lifecycle_test.cpp
  DEPS
    :placement_lifecycle
    GTest::gtest_main
)

cc_test(
  NAME
    placement_desired_store_test
  SRCS
    placement_desired_store_test.cpp
  DEPS
    :placement_desired_store
    GTest::gtest_main
)

cc_test(
  NAME
    placement_reconciler_test
  SRCS
    placement_reconciler_test.cpp
  DEPS
    :placement_reconciler
    GTest::gtest_main
)

cc_test(
  NAME
    placement_actuator_test
  SRCS
    placement_actuator_test.cpp
  DEPS
    :placement_actuator
    GTest::gtest_main
)

cc_test(
  NAME
    placement_operation_store_test
  SRCS
    placement_operation_store_test.cpp
  DEPS
    :placement_actuator
    GTest::gtest_main
)

cc_test(
  NAME
    provider_lifecycle_actuator_test
  SRCS
    provider_lifecycle_actuator_test.cpp
  DEPS
    :placement_actuator
    GTest::gtest_main
    brpc-static
    nlohmann_json::nlohmann_json
    proto::proto_http_service
)

cc_test(
  NAME
    placement_deployment_actuator_test
  SRCS
    placement_deployment_actuator_test.cpp
  DEPS
    :placement_deployment_actuator
    GTest::gtest_main
    brpc-static
    nlohmann_json::nlohmann_json
    proto::proto_http_service
)

cc_test(
  NAME
    placement_controller_test
  SRCS
    placement_controller_test.cpp
  DEPS
    :placement_controller
    GTest::gtest_main
)

cc_test(
  NAME
    placement_config_test
  SRCS
    placement_config_test.cpp
  DEPS
    :placement_config
    GTest::gtest_main
)
