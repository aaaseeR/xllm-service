include(cc_test)

cc_test(
  NAME
    output_event_sequencer_test
  SRCS
    output_event_sequencer_test.cpp
  DEPS
    :output_event_sequencer
    GTest::gtest_main
)
