# User options
use_precompile <- FALSE
use_gpu <- FALSE
use_mingw <- FALSE

if (.Machine$sizeof.pointer != 8L) {
  stop("LightGBM only supports 64-bit R, please check the version of R and Rtools.")
}

R_int_UUID <- .Internal(internalsID())
R_ver <- as.double(R.Version()$major) + as.double(R.Version()$minor) / 10.0

if (!(R_int_UUID == "0310d4b8-ccb1-4bb8-ba94-d36a55f60262"
    || R_int_UUID == "2fdf6c18-697a-4ba7-b8ef-11c0d92f1327")) {
  print("Warning: unmatched R_INTERNALS_UUID, may cannot run normally.")
}

# Move in CMakeLists.txt
write_succeeded <- file.copy(
  "../inst/bin/CMakeLists.txt"
  , "CMakeLists.txt"
  , overwrite = TRUE
)
if (!write_succeeded) {
  stop("Copying CMakeLists.txt failed")
}

# Get some paths
source_dir <- file.path(R_PACKAGE_SOURCE, "src", fsep = "/")
build_dir <- file.path(source_dir, "build", fsep = "/")

# Check for precompilation
if (!use_precompile) {

  # Prepare building package
  dir.create(
    build_dir
    , recursive = TRUE
    , showWarnings = FALSE
  )
  setwd(build_dir)

  # Prepare installation steps
  cmake_cmd <- "cmake "
  build_cmd <- "make _lightgbm"
  lib_folder <- file.path(source_dir, fsep = "/")

  if (use_gpu) {
    cmake_cmd <- paste0(cmake_cmd, " -DUSE_GPU=ON ")
  }
  if (R_ver >= 3.5) {
    cmake_cmd <- paste0(cmake_cmd, " -DUSE_R35=ON ")
  }
  cmake_cmd <- paste0(cmake_cmd, " -DBUILD_FOR_R=ON ")

  # Pass in R version, used to help find R executable for linking
  R_version_string <- paste(
    R.Version()[["major"]]
    , R.Version()[["minor"]]
    , sep = "."
  )
  cmake_cmd <- sprintf(
    paste0(cmake_cmd, " -DCMAKE_R_VERSION='%s' ")
    , R_version_string
  )

  # Check if Windows installation (for gcc vs Visual Studio)
  if (WINDOWS) {
    if (use_mingw) {
      print("Trying to build with MinGW")
      cmake_cmd <- paste0(cmake_cmd, " -G \"MinGW Makefiles\" ")
      build_cmd <- "mingw32-make.exe _lightgbm"
      system(paste0(cmake_cmd, " ..")) # Must build twice for Windows due sh.exe in Rtools
    } else {
      local_vs_def <- ""
      vs_versions <- c(
        "Visual Studio 16 2019"
        , "Visual Studio 15 2017"
        , "Visual Studio 14 2015"
      )
      for (vs in vs_versions) {
        print(paste0("Trying to build with: '", vs, "'"))
        vs_def <- paste0(" -G \"", vs, "\" -A x64")
        tmp_cmake_cmd <- paste0(cmake_cmd, vs_def)
        try_vs <- system(paste0(tmp_cmake_cmd, " .."))
        if (try_vs == 0L) {
          local_vs_def <- vs_def
          print(paste0("Building with '", vs, "' succeeded"))
          break
        } else {
          unlink("./*", recursive = TRUE) # Clean up build directory
        }
      }
      if (try_vs == 1L) {
        print("Building with Visual Studio failed. Attempted with MinGW")
        cmake_cmd <- paste0(cmake_cmd, " -G \"MinGW Makefiles\" ")
        system(paste0(cmake_cmd, " ..")) # Must build twice for Windows due sh.exe in Rtools
        build_cmd <- "mingw32-make.exe _lightgbm"
      } else {
        cmake_cmd <- paste0(cmake_cmd, local_vs_def)
        build_cmd <- "cmake --build . --target _lightgbm --config Release"
        lib_folder <- file.path(source_dir, "Release", fsep = "/")
      }
    }
  }

  # Install
  system(paste0(cmake_cmd, " .."))

  # R CMD check complains about the .NOTPARALLEL directive created in the cmake
  # Makefile. We don't need it here anyway since targets are built serially, so trying
  # to remove it with this hack
  generated_makefile <- file.path(
    build_dir
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

# Packages with install.libs.R need to copy some artifacts into the
# expected places in the package structure.
# see https://cran.r-project.org/doc/manuals/r-devel/R-exts.html#Package-subdirectories,
# especially the paragraph on install.libs.R
dest <- file.path(R_PACKAGE_DIR, paste0("libs", R_ARCH), fsep = "/")
dir.create(dest, recursive = TRUE, showWarnings = FALSE)
if (file.exists(src)) {
  print(paste0("Found library file: ", src, " to move to ", dest))
  file.copy(src, dest, overwrite = TRUE)

  symbols_file <- file.path(source_dir, "symbols.rds")
  if (file.exists(symbols_file)) {
    file.copy(symbols_file, dest, overwrite = TRUE)
  }

} else {
  stop(paste0("Cannot find lib_lightgbm", SHLIB_EXT))
}

# clean up the "build" directory
if (dir.exists(build_dir)) {
  print("Removing 'build/' directory")
  unlink(
    x = build_dir
    , recursive = TRUE
    , force = TRUE
  )
}
