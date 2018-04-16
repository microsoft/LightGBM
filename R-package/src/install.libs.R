# User options
use_precompile <- FALSE
use_gpu <- FALSE
use_mingw <- FALSE

if (.Machine$sizeof.pointer != 8){
  stop("Only support 64-bit R, please check your the version of your R and Rtools.")
}

# Check for precompilation
if (!use_precompile) {

  # Check repository content
  source_dir <- file.path(R_PACKAGE_SOURCE, "src", fsep = "/")
  setwd(source_dir)
  
  if (!file.exists("_IS_FULL_PACKAGE")) {
    unlink("./include", recursive = TRUE)
    unlink("./src", recursive = TRUE)
    unlink("./compute", recursive = TRUE)
    unlink("./build", recursive = TRUE)
    if (!file.copy("./../../include", "./", overwrite = TRUE, recursive = TRUE)) {
      stop("Cannot find folder LightGBM/include")
    }
    if (!file.copy("./../../src", "./", overwrite = TRUE, recursive = TRUE)) {
      stop("Cannot find folder LightGBM/src")
    }
    if (!file.copy("./../../compute", "./", overwrite = TRUE, recursive = TRUE)) {
      print("Cannot find folder LightGBM/compute, disabling GPU build.")
      use_gpu <- FALSE
    }
    if (!file.copy("./../../CMakeLists.txt", "./", overwrite = TRUE, recursive = TRUE)) {
      stop("Cannot find file LightGBM/CMakeLists.txt")
    }
  }
  
  # Prepare building package
  build_dir <- file.path(source_dir, "build", fsep = "/")
  dir.create(build_dir, recursive = TRUE, showWarnings = FALSE)
  setwd(build_dir)
  
  # Prepare installation steps
  cmake_cmd <- "cmake "
  build_cmd <- "make _lightgbm"
  lib_folder <- file.path(R_PACKAGE_SOURCE, "src", fsep = "/")
  
  if (use_gpu) {
    cmake_cmd <- paste0(cmake_cmd, " -DUSE_GPU=ON ")
  }

  # Check if Windows installation (for gcc vs Visual Studio)
  if (WINDOWS) {
    if (use_mingw) {
      cmake_cmd <- paste0(cmake_cmd, " -G \"MinGW Makefiles\" ")
      build_cmd <- "mingw32-make.exe _lightgbm"
      system(paste0(cmake_cmd, " ..")) # Must build twice for Windows due sh.exe in Rtools
    } else {
      try_vs <- 0
      local_vs_def <- ""
      vs_versions <- c("Visual Studio 15 2017 Win64", "Visual Studio 14 2015 Win64")
      for(vs in vs_versions){
        vs_def <- paste0(" -G \"", vs, "\"")
        tmp_cmake_cmd <- paste0(cmake_cmd, vs_def)
        try_vs <- system(paste0(tmp_cmake_cmd, " .."))
        if (try_vs == 0) {
          local_vs_def = vs_def
          break
        } else {
          unlink("./*", recursive = TRUE) # Clean up build directory
        }
      }
      if (try_vs == 1) {
        cmake_cmd <- paste0(cmake_cmd, " -G \"MinGW Makefiles\" ") # Switch to MinGW on failure, try build once
        system(paste0(cmake_cmd, " ..")) # Must build twice for Windows due sh.exe in Rtools
        build_cmd <- "mingw32-make.exe _lightgbm"
      } else {
        cmake_cmd <- paste0(cmake_cmd, local_vs_def)
        build_cmd <- "cmake --build . --target _lightgbm  --config Release"
        lib_folder <- file.path(R_PACKAGE_SOURCE, "src/Release", fsep = "/")
      }
    }
  }
  
  # Install
  system(paste0(cmake_cmd, " .."))
  system(build_cmd)
  src <- file.path(lib_folder, paste0("lib_lightgbm", SHLIB_EXT), fsep = "/")
  
} else {

  # Has precompiled package
  lib_folder <- file.path(R_PACKAGE_SOURCE, "../", fsep = "/")
  if (file.exists(file.path(lib_folder, paste0("lib_lightgbm", SHLIB_EXT), fsep = "/"))) {
    src <- file.path(lib_folder, paste0("lib_lightgbm", SHLIB_EXT), fsep = "/")
  } else if (file.exists(file.path(lib_folder, paste0("Release/lib_lightgbm", SHLIB_EXT), fsep = "/"))) {
    src <- file.path(lib_folder, paste0("Release/lib_lightgbm", SHLIB_EXT), fsep = "/") 
  } else {
    src <- file.path(lib_folder, paste0("/windows/x64/DLL/lib_lightgbm", SHLIB_EXT), fsep = "/") # Expected result: installation will fail if it is not here or any other
  }
  
}

# Check installation correctness
dest <- file.path(R_PACKAGE_DIR, paste0("libs", R_ARCH), fsep = "/")
dir.create(dest, recursive = TRUE, showWarnings = FALSE)
if (file.exists(src)) {
  cat("Found library file: ", src, " to move to ", dest, sep = "")
  file.copy(src, dest, overwrite = TRUE)
} else {
  stop(paste0("Cannot find lib_lightgbm", SHLIB_EXT))
}
