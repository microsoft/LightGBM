# User options
use_gpu <- FALSE

# For Windows, the package will be built with Visual Studio
# unless you set one of these to TRUE
use_mingw <- FALSE
use_msys2 <- FALSE

if (use_mingw && use_msys2) {
  stop("Cannot use both MinGW and MSYS2. Please choose only one.")
}

if (.Machine$sizeof.pointer != 8L) {
  stop("LightGBM only supports 64-bit R, please check the version of R and Rtools.")
}

R_int_UUID <- .Internal(internalsID())
R_ver <- as.double(R.Version()$major) + as.double(R.Version()$minor) / 10.0

if (!(R_int_UUID == "0310d4b8-ccb1-4bb8-ba94-d36a55f60262"
    || R_int_UUID == "2fdf6c18-697a-4ba7-b8ef-11c0d92f1327")) {
  warning("Warning: unmatched R_INTERNALS_UUID, may not run normally.")
}

# system() will not raise an R exception if the process called
# fails. Wrapping it here to get that behavior.
#
# system() introduces a lot of overhead, at least on Windows,
# so trying processx if it is available
.run_shell_command <- function(cmd, args, strict = TRUE) {
    on_windows <- .Platform$OS.type == "windows"
    has_processx <- suppressMessages({
      suppressWarnings({
        require("processx")  # nolint
      })
    })
    if (has_processx && on_windows) {
      result <- processx::run(
        command = cmd
        , args = args
        , windows_verbatim_args = TRUE
        , error_on_status = FALSE
        , echo = TRUE
      )
      exit_code <- result$status
    } else {
      if (on_windows) {
        message(paste0(
          "Using system() to run shell commands. Installing "
          , "'processx' with install.packages('processx') might "
          , "make this faster."
        ))
      }
      cmd <- paste0(cmd, " ", paste0(args, collapse = " "))
      exit_code <- system(cmd)
    }

    if (exit_code != 0L && isTRUE(strict)) {
        stop(paste0("Command failed with exit code: ", exit_code))
    }
    return(invisible(exit_code))
}

# try to generate Visual Studio build files
.generate_vs_makefiles <- function(cmake_args) {
  vs_versions <- c(
    "Visual Studio 16 2019"
    , "Visual Studio 15 2017"
    , "Visual Studio 14 2015"
  )
  working_vs_version <- NULL
  for (vs_version in vs_versions) {
    message(sprintf("Trying '%s'", vs_version))
    # if the build directory is not empty, clean it
    if (file.exists("CMakeCache.txt")) {
      file.remove("CMakeCache.txt")
    }
    vs_cmake_args <- c(
      cmake_args
      , "-G"
      , shQuote(vs_version)
      , "-A"
      , "x64"
    )
    exit_code <- .run_shell_command("cmake", c(vs_cmake_args, ".."), strict = FALSE)
    if (exit_code == 0L) {
      message(sprintf("Successfully created build files for '%s'", vs_version))
      return(invisible(TRUE))
    }

  }
  return(invisible(FALSE))
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

# Prepare building package
dir.create(
  build_dir
  , recursive = TRUE
  , showWarnings = FALSE
)
setwd(build_dir)

use_visual_studio <- !(use_mingw || use_msys2)

# If using MSVC to build, pull in the script used
# to create R.def from R.dll
if (WINDOWS && use_visual_studio) {
  write_succeeded <- file.copy(
    "../../inst/make-r-def.R"
    , file.path(build_dir, "make-r-def.R")
    , overwrite = TRUE
  )
  if (!write_succeeded) {
    stop("Copying make-r-def.R failed")
  }
}

# Prepare installation steps
cmake_args <- NULL
build_cmd <- "make"
build_args <- "_lightgbm"
lib_folder <- file.path(source_dir, fsep = "/")

# add in command-line arguments
# NOTE: build_r.R replaces the line below
command_line_args <- NULL
cmake_args <- c(cmake_args, command_line_args)

WINDOWS_BUILD_TOOLS <- list(
  "MinGW" = c(
    build_tool = "mingw32-make.exe"
    , makefile_generator = "MinGW Makefiles"
  )
  , "MSYS2" = c(
    build_tool = "make.exe"
    , makefile_generator = "MSYS Makefiles"
  )
)

if (use_mingw) {
  windows_toolchain <- "MinGW"
} else if (use_msys2) {
  windows_toolchain <- "MSYS2"
} else {
  # Rtools 4.0 moved from MinGW to MSYS toolchain. If user tries
  # Visual Studio install but that fails, fall back to the toolchain
  # supported in Rtools
  if (R_ver >= 4.0) {
    windows_toolchain <- "MSYS2"
  } else {
    windows_toolchain <- "MinGW"
  }
}
windows_build_tool <- WINDOWS_BUILD_TOOLS[[windows_toolchain]][["build_tool"]]
windows_makefile_generator <- WINDOWS_BUILD_TOOLS[[windows_toolchain]][["makefile_generator"]]

if (use_gpu) {
  cmake_args <- c(cmake_args, "-DUSE_GPU=ON")
}
cmake_args <- c(cmake_args, "-D__BUILD_FOR_R=ON")

# Pass in R version, used to help find R executable for linking
R_version_string <- paste(
  R.Version()[["major"]]
  , R.Version()[["minor"]]
  , sep = "."
)
r_version_arg <- sprintf("-DCMAKE_R_VERSION='%s'", R_version_string)
cmake_args <- c(cmake_args, r_version_arg)

# the checks below might already run `cmake -G`. If they do, set this flag
# to TRUE to avoid re-running it later
makefiles_already_generated <- FALSE

# Check if Windows installation (for gcc vs Visual Studio)
if (WINDOWS) {
  if (!use_visual_studio) {
    message(sprintf("Trying to build with %s", windows_toolchain))
    # Must build twice for Windows due sh.exe in Rtools
    cmake_args <- c(cmake_args, "-G", shQuote(windows_makefile_generator))
    .run_shell_command("cmake", c(cmake_args, ".."), strict = FALSE)
    build_cmd <- windows_build_tool
    build_args <- "_lightgbm"
  } else {
    visual_studio_succeeded <- .generate_vs_makefiles(cmake_args)
    if (!isTRUE(visual_studio_succeeded)) {
      warning(sprintf("Building with Visual Studio failed. Attempting with %s", windows_toolchain))
      # Must build twice for Windows due sh.exe in Rtools
      cmake_args <- c(cmake_args, "-G", shQuote(windows_makefile_generator))
      .run_shell_command("cmake", c(cmake_args, ".."), strict = FALSE)
      build_cmd <- windows_build_tool
      build_args <- "_lightgbm"
    } else {
      build_cmd <- "cmake"
      build_args <- c("--build", ".", "--target", "_lightgbm", "--config", "Release")
      lib_folder <- file.path(source_dir, "Release", fsep = "/")
      makefiles_already_generated <- TRUE
    }
  }
} else {
    .run_shell_command("cmake", c(cmake_args, ".."))
    makefiles_already_generated <- TRUE
}

# generate build files
if (!makefiles_already_generated) {
  .run_shell_command("cmake", c(cmake_args, ".."))
}

# build the library
message("Building lib_lightgbm")
.run_shell_command(build_cmd, build_args)
src <- file.path(lib_folder, paste0("lib_lightgbm", SHLIB_EXT), fsep = "/")

# Packages with install.libs.R need to copy some artifacts into the
# expected places in the package structure.
# see https://cran.r-project.org/doc/manuals/r-devel/R-exts.html#Package-subdirectories,
# especially the paragraph on install.libs.R
dest <- file.path(R_PACKAGE_DIR, paste0("libs", R_ARCH), fsep = "/")
dir.create(dest, recursive = TRUE, showWarnings = FALSE)
if (file.exists(src)) {
  message(paste0("Found library file: ", src, " to move to ", dest))
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
  message("Removing 'build/' directory")
  unlink(
    x = build_dir
    , recursive = TRUE
    , force = TRUE
  )
}
