# For macOS users who have decided to use gcc
# (replace 8 with version of gcc installed on your machine)
# NOTE: your gcc / g++ from Homebrew is probably in /usr/local/bin
#export CXX=/usr/local/bin/g++-8 CC=/usr/local/bin/gcc-8
# Sys.setenv("CXX" = "/usr/local/bin/g++-8")
# Sys.setenv("CC" = "/usr/local/bin/gcc-8")

args <- commandArgs(trailingOnly = TRUE)
INSTALL_AFTER_BUILD <- !("--skip-install" %in% args)
TEMP_R_DIR <- file.path(getwd(), "lightgbm_r")
TEMP_SOURCE_DIR <- file.path(TEMP_R_DIR, "src")

# R returns FALSE (not a non-zero exit code) if a file copy operation
# breaks. Let's fix that
.handle_result <- function(res) {
  if (!all(res)) {
    stop("Copying files failed!")
  }
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

# Make a new temporary folder to work in
unlink(x = TEMP_R_DIR, recursive = TRUE)
dir.create(TEMP_R_DIR)

# copy in the relevant files
result <- file.copy(
  from = "R-package/./"
  , to = sprintf("%s/", TEMP_R_DIR)
  , recursive = TRUE
  , overwrite = TRUE
)
.handle_result(result)

# Add blank Makevars files
result <- file.copy(
  from = file.path(TEMP_R_DIR, "inst", "Makevars")
  , to = file.path(TEMP_SOURCE_DIR, "Makevars")
  , overwrite = TRUE
)
.handle_result(result)
result <- file.copy(
  from = file.path(TEMP_R_DIR, "inst", "Makevars.win")
  , to = file.path(TEMP_SOURCE_DIR, "Makevars.win")
  , overwrite = TRUE
)
.handle_result(result)

result <- file.copy(
  from = "include/"
  , to =  sprintf("%s/", TEMP_SOURCE_DIR)
  , recursive = TRUE
  , overwrite = TRUE
)
.handle_result(result)

result <- file.copy(
  from = "src/"
  , to = sprintf("%s/", TEMP_SOURCE_DIR)
  , recursive = TRUE
  , overwrite = TRUE
)
.handle_result(result)

result <- file.copy(
  from = "compute/"
  , to = sprintf("%s/", TEMP_SOURCE_DIR)
  , recursive = TRUE
  , overwrite = TRUE
)
.handle_result(result)

result <- file.copy(
  from = "CMakeLists.txt"
  , to = file.path(TEMP_R_DIR, "inst", "bin/")
  , overwrite = TRUE
)
.handle_result(result)

# remove CRAN-specific files
result <- file.remove(
  file.path(TEMP_R_DIR, "configure")
  , file.path(TEMP_R_DIR, "configure.ac")
  , file.path(TEMP_R_DIR, "configure.win")
  , file.path(TEMP_SOURCE_DIR, "Makevars.in")
  , file.path(TEMP_SOURCE_DIR, "Makevars.win.in")
)
.handle_result(result)

# copy files into the place CMake expects
for (src_file in c("lightgbm_R.cpp", "lightgbm_R.h", "R_object_helper.h")) {
  result <- file.copy(
    from = file.path(TEMP_SOURCE_DIR, src_file)
    , to = file.path(TEMP_SOURCE_DIR, "src", src_file)
    , overwrite = TRUE
  )
  .handle_result(result)
  result <- file.remove(
    file.path(TEMP_SOURCE_DIR, src_file)
  )
  .handle_result(result)
}

result <- file.copy(
  from = file.path("R-package", "inst", "make-r-def.R")
  , to = file.path(TEMP_R_DIR, "inst", "bin/")
  , overwrite = TRUE
)
.handle_result(result)

# NOTE: --keep-empty-dirs is necessary to keep the deep paths expected
#       by CMake while also meeting the CRAN req to create object files
#       on demand
.run_shell_command("R", c("CMD", "build", TEMP_R_DIR, "--keep-empty-dirs"))

# Install the package
version <- gsub(
  "Version: ",
  "",
  grep(
    "Version: "
    , readLines(con = file.path(TEMP_R_DIR, "DESCRIPTION"))
    , value = TRUE
  )
)
tarball <- file.path(getwd(), sprintf("lightgbm_%s.tar.gz", version))

install_cmd <- "R"
install_args <- c("CMD", "INSTALL", "--no-multiarch", "--with-keep.source", tarball)
if (INSTALL_AFTER_BUILD) {
  .run_shell_command(install_cmd, install_args)
} else {
  cmd <- paste0(install_cmd, " ", paste0(install_args, collapse = " "))
  print(sprintf("Skipping installation. Install the package with command '%s'", cmd))
}
