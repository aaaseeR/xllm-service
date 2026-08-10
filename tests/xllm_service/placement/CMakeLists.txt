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
