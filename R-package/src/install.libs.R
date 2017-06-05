use_gpu <- FALSE
use_mingw <- FALSE

cmake_cmd <- "cmake"
build_cmd <- "make -j"
lib_folder <- paste0(R_PACKAGE_SOURCE, '/src')

if (WINDOWS) {
  if(use_mingw){
    cmake_cmd <- paste0(cmake_cmd, " -G \"MinGW Makefiles\" ")
    build_cmd <- "mingw32-make.exe -j"
  } else{
    cmake_cmd <- paste0(cmake_cmd, " -DCMAKE_GENERATOR_PLATFORM=x64 ")
    build_cmd <- "cmake --build . --target _lightgbm  --config Release"
    lib_folder <- paste0(R_PACKAGE_SOURCE, '/src/Release')
  }
}

if(use_gpu) {
  cmake_cmd <- paste0(cmake_cmd, " -DUSE_GPU=1 ")
}

system(paste0(cmake_cmd, " ."))
system(build_cmd)
dest <- file.path(R_PACKAGE_DIR, paste0('libs', R_ARCH))
dir.create(dest, recursive = TRUE, showWarnings = FALSE)
src <- paste0(lib_folder, '/lib_lightgbm', SHLIB_EXT)
if(file.exists(src)){
  print(paste0("find library file: ", src))
  file.copy(src, dest, overwrite = TRUE)
} else {
  stop("cannot find lib_lightgbm.dll")
}
