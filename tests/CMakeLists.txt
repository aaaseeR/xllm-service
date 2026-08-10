if(NOT BUILD_TESTING)
  return()
endif()

add_subdirectory(xllm_service)
add_subdirectory(e2e)
