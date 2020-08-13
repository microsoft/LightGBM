file(GLOB examples_SRCS "*.cpp")

add_custom_target(unsupported_examples)

include_directories(../../../unsupported ../../../unsupported/test)

foreach(example_src ${examples_SRCS})
  get_filename_component(example ${example_src} NAME_WE)
  add_executable(example_${example} ${example_src})
  if(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO)
    target_link_libraries(example_${example} ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO})
  endif()
  add_custom_command(
    TARGET example_${example}
    POST_BUILD
    COMMAND example_${example}
    ARGS >${CMAKE_CURRENT_BINARY_DIR}/${example}.out
  )
  add_dependencies(unsupported_examples example_${example})
endforeach(example_src)

if(EIGEN_TEST_SYCL)
  add_subdirectory(SYCL)
endif(EIGEN_TEST_SYCL)
