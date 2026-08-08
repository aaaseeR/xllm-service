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
