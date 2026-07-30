include(cc_library)
include(cc_test)

cc_library(
  NAME
    dispatcher
  HDRS
    dispatcher.h
  SRCS
    dispatcher.cpp
  DEPS
    :routing
    :transport
    glog::glog
    proto_xllm
)
target_link_libraries(dispatcher PRIVATE brpc-static)

cc_test(
  NAME
    dispatcher_test
  SRCS
    dispatcher_test.cpp
  DEPS
    :dispatcher
    GTest::gtest_main
)
