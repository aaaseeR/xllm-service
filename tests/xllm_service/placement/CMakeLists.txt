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
