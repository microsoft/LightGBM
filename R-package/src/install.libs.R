# User options
use_precompile <- FALSE
use_gpu <- FALSE
use_mingw <- FALSE

# Check for precompilation
if (!use_precompile) {

  # Check repository content
  source_dir <- paste0(R_PACKAGE_SOURCE, "/src")
  setwd(source_dir)
  
  if (!file.exists("_IS_FULL_PACKAGE")) {
    if (!file.copy("./../../include", "./", overwrite = TRUE, recursive = TRUE)) {
      stop("Cannot find folder LightGBM/include")
    }
	  if (!file.copy("./../../src", "./", overwrite = TRUE, recursive = TRUE)) {
		  stop("Cannot find folder LightGBM/src")
	  }
	  if (use_gpu) {
      if (!file.copy("./../../compute", "./", overwrite = TRUE, recursive = TRUE)) {
        print("Cannot find folder LightGBM/compute, will disable GPU build")
        use_gpu <- FALSE
      }
    }
    if (!file.copy("./../../CMakeLists.txt", "./", overwrite = TRUE, recursive = TRUE)) {
  	  stop("Cannot find file LightGBM/CMakeLists.txt")
    }
  }
  
  # Prepare building package
  build_dir <- paste0(source_dir, "/build")
  dir.create(build_dir, recursive = TRUE, showWarnings = FALSE)
  setwd(build_dir)
  
  # Prepare installatio nsteps
  cmake_cmd <- "cmake"
  build_cmd <- "make -j"
  lib_folder <- paste0(R_PACKAGE_SOURCE, "/src")
  
  if (WINDOWS) {
    if (use_mingw) {
      cmake_cmd <- paste0(cmake_cmd, " -G \"MinGW Makefiles\" ")
      build_cmd <- "mingw32-make.exe -j"
    } else {
      cmake_cmd <- paste0(cmake_cmd, " -DCMAKE_GENERATOR_PLATFORM=x64 ")
      build_cmd <- "cmake --build . --target _lightgbm  --config Release"
      lib_folder <- paste0(R_PACKAGE_SOURCE, "/src/Release")
    }
  }
  
  if (use_gpu) {
    cmake_cmd <- paste0(cmake_cmd, " -DUSE_GPU=1 ")
  }
  
  # Install
  system(paste0(cmake_cmd, " .."))
  system(build_cmd)
  dest <- file.path(R_PACKAGE_DIR, paste0("libs", R_ARCH))
  dir.create(dest, recursive = TRUE, showWarnings = FALSE)
  src <- paste0(lib_folder, "/lib_lightgbm", SHLIB_EXT)
  
} else {
  
  lib_folder <- paste0(R_PACKAGE_SOURCE, "/src")
  print(lib_folder)
  dest <- file.path(R_PACKAGE_DIR, paste0("libs", R_ARCH))
  dir.create(dest, recursive = TRUE, showWarnings = FALSE)
  print(dest)
  if (file.exists(paste0(lib_folder, "/../../lib_lightgbm", SHLIB_EXT))) {
    src <- paste0(lib_folder, "/../../lib_lightgbm", SHLIB_EXT)
  } else {
    src <- paste0(lib_folder, "/../../windows/x64/DLL/lib_lightgbm", SHLIB_EXT)
  }
  print(src)
  
}

# Check installation correctness
if (file.exists(src)) {
  print(paste0("Found library file: ", src, " to move to ", dest))
  file.copy(src, dest, overwrite = TRUE)
} else {
  stop("Cannot find lib_lightgbm.dll")
}
