function(assert_submodules_initialized)
    set(submodules compute eigen fast_double_parser fmt)
    foreach(submodule ${submodules})
        if(NOT EXISTS "${CMAKE_SOURCE_DIR}/external_libs/${submodule}/CMakeLists.txt")
            message(
              FATAL_ERROR
              "Required source directory 'external_libs/${submodule}' is empty. "
              "Please run: 'git submodule update --init --recursive' from the root of the repo and re-run cmake"
            )
        endif()
    endforeach()
endfunction()
