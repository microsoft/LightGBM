# User options
use_precompile <- FALSE
use_gpu <- FALSE
use_mingw <- FALSE

if (.Machine$sizeof.pointer != 8L) {
  stop("Only support 64-bit R, please check your the version of your R and Rtools.")
}

R_int_UUID <- .Internal(internalsID())
R_ver <- as.double(R.Version()$major) + as.double(R.Version()$minor) / 10.0

if (!(R_int_UUID == "0310d4b8-ccb1-4bb8-ba94-d36a55f60262"
    || R_int_UUID == "2fdf6c18-697a-4ba7-b8ef-11c0d92f1327")) {
  print("Warning: unmatched R_INTERNALS_UUID, may cannot run normally.")
}

# Move in CMakeLists.txt
if (!file.copy("../inst/bin/CMakeLists.txt", "CMakeLists.txt", overwrite = TRUE)) {
  stop("Copying CMakeLists failed")
}

# Check for precompilation
if (!use_precompile) {

  # Check repository content
  source_dir <- file.path(R_PACKAGE_SOURCE, "src", fsep = "/")
  setwd(source_dir)

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
  if (R_ver >= 3.5) {
    cmake_cmd <- paste0(cmake_cmd, " -DUSE_R35=ON ")
  }

  # Check if Windows installation (for gcc vs Visual Studio)
  if (WINDOWS) {
    if (use_mingw) {
      cmake_cmd <- paste0(cmake_cmd, " -G \"MinGW Makefiles\" ")
      build_cmd <- "mingw32-make.exe _lightgbm"
      system(paste0(cmake_cmd, " ..")) # Must build twice for Windows due sh.exe in Rtools
    } else {
      try_vs <- 0L
      local_vs_def <- ""
      vs_versions <- c("Visual Studio 16 2019", "Visual Studio 15 2017", "Visual Studio 14 2015")
      for (vs in vs_versions) {
        vs_def <- paste0(" -G \"", vs, "\" -A x64")
        tmp_cmake_cmd <- paste0(cmake_cmd, vs_def)
        try_vs <- system(paste0(tmp_cmake_cmd, " .."))
        if (try_vs == 0L) {
          local_vs_def <- vs_def
          break
        } else {
          unlink("./*", recursive = TRUE) # Clean up build directory
        }
      }
      if (try_vs == 1L) {
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

  # R CMD check complains about the .NOTPARALLEL directive created in the cmake
  # Makefile. We don't need it here anyway since targets are built serially, so trying
  # to remove it with this hack
  generated_makefile <- file.path(
    R_PACKAGE_SOURCE
    , "src"
    , "build"
    , "Makefile"
  )
  if (file.exists(generated_makefile)) {
    makefile_txt <- readLines(
      con = generated_makefile
    )
    makefile_txt <- gsub(
      pattern = ".*NOTPARALLEL.*"
      , replacement = ""
      , x = makefile_txt
    )
    writeLines(
      text = makefile_txt
      , con = generated_makefile
      , sep = "\n"
    )
  }

  system(build_cmd)
  src <- file.path(lib_folder, paste0("lib_lightgbm", SHLIB_EXT), fsep = "/")

} else {

  # Has precompiled package
  lib_folder <- file.path(R_PACKAGE_SOURCE, "../", fsep = "/")
  shared_object_file <- file.path(
    lib_folder
    , paste0("lib_lightgbm", SHLIB_EXT)
    , fsep = "/"
  )
  release_file <- file.path(
    lib_folder
    , paste0("Release/lib_lightgbm", SHLIB_EXT)
    , fsep = "/"
  )
  windows_shared_object_file <- file.path(
    lib_folder
    , paste0("/windows/x64/DLL/lib_lightgbm", SHLIB_EXT)
    , fsep = "/"
  )
  if (file.exists(shared_object_file)) {
    src <- shared_object_file
  } else if (file.exists(release_file)) {
    src <- release_file
  } else {
    # Expected result: installation will fail if it is not here or any other
    src <- windows_shared_object_file
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
