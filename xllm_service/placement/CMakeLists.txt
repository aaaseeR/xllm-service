include(cc_library)

cc_library(
  NAME
    placement_types
  HDRS
    placement_types.h
  SRCS
    placement_types.cpp
  DEPS
    proto_xllm
)

cc_library(
  NAME
    placement_planner
  HDRS
    placement_planner.h
  SRCS
    placement_planner.cpp
  DEPS
    :placement_types
)

cc_library(
  NAME
    placement_budget_allocator
  HDRS
    placement_budget_allocator.h
  SRCS
    placement_budget_allocator.cpp
  DEPS
    :placement_types
)
