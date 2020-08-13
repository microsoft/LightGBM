
add_custom_target(copy_scripts)

set(script_files go_mean mk_mean_script.sh mk_new_gnuplot.sh
    perlib_plot_settings.txt action_settings.txt gnuplot_common_settings.hh )

foreach(script_file ${script_files})
add_custom_command(
  TARGET copy_scripts
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${script_file} ${CMAKE_CURRENT_BINARY_DIR}/
  ARGS
)
endforeach(script_file)

add_custom_command(
  TARGET copy_scripts
  POST_BUILD
  COMMAND ${CMAKE_CXX_COMPILER} --version | head -n 1 > ${CMAKE_CURRENT_BINARY_DIR}/compiler_version.txt
  ARGS
)
add_custom_command(
  TARGET copy_scripts
  POST_BUILD
  COMMAND echo "${Eigen_SOURCE_DIR}" > ${CMAKE_CURRENT_BINARY_DIR}/eigen_root_dir.txt
  ARGS
)

add_executable(smooth smooth.cxx)
add_executable(regularize regularize.cxx)
add_executable(main mean.cxx)
add_dependencies(main copy_scripts)
