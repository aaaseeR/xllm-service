include(cc_library)
include(cc_test)

cc_library(
  NAME
    transport
  HDRS
    channel_pool.h
    types.h
  SRCS
    channel_pool.cpp
    types.cpp
  DEPS
    :common
    glog::glog
)
target_link_libraries(transport PRIVATE brpc-static)

cc_test(
  NAME
    channel_pool_test
  SRCS
    channel_pool_test.cpp
  DEPS
    :transport
    GTest::gtest_main
)
