function(assert_submodules_initialized use_gpu)
    set(submodules eigen fast_double_parser fmt)
    if (use_gpu)
        list(APPEND submodules compute)
    endif()
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
