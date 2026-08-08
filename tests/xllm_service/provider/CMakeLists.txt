include(cc_test)

cc_test(
  NAME
    provider_contract_test
  SRCS
    provider_contract_test.cpp
  DEPS
    :provider_contract
    :provider_registry
    GTest::gtest_main
    proto_xllm
)

cc_test(
  NAME
    execution_hold_test
  SRCS
    execution_hold_test.cpp
  DEPS
    :execution_hold
    GTest::gtest_main
    proto_xllm
)

cc_test(
  NAME
    provider_route_selector_test
  SRCS
    provider_route_selector_test.cpp
  DEPS
    :provider_route_selector
    GTest::gtest_main
    proto_xllm
)
