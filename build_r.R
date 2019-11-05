# For macOS users who have decided to use gcc
# (replace 8 with version of gcc installed on your machine)
# NOTE: your gcc / g++ from Homebrew is probably in /usr/local/bin
#export CXX=/usr/local/bin/g++-8 CC=/usr/local/bin/gcc-8
# Sys.setenv("CXX" = "/usr/local/bin/g++-8")
# Sys.setenv("CC" = "/usr/local/bin/gcc-8")

# R returns FALSE (not a non-zero exit code) if a file copy operation
# breaks. Let's fix that
.handle_result <- function(res) {
  if (!res) {
    stop("Copying files failed!")
  }
}

# system() will not raise an R exception if the process called
# fails. Wrapping it here to get that behavior
.run_shell_command <- function(cmd, ...){
    exit_code <- system(cmd, ...)
    if (exit_code != 0){
        stop(paste0("Command failed with exit code: ", exit_code))
    }
}

# Make a new temporary folder to work in
unlink(x = "lightgbm_r", recursive = TRUE)
dir.create("lightgbm_r")

# copy in the relevant files
result <- file.copy(from = "R-package/./",
                    to = "lightgbm_r/",
                    recursive = TRUE,
                    overwrite = TRUE)
.handle_result(result)

result <- file.copy(from = "include/",
                    to = file.path("lightgbm_r", "src/"),
                    recursive = TRUE,
                    overwrite = TRUE)
.handle_result(result)

result <- file.copy(from = "src/",
                    to = file.path("lightgbm_r", "src/"),
                    recursive = TRUE,
                    overwrite = TRUE)
.handle_result(result)

result <- file.copy(from = "compute/",
                    to = file.path("lightgbm_r", "src/"),
                    recursive = TRUE,
                    overwrite = TRUE)
.handle_result(result)

result <- file.copy(from = "CMakeLists.txt",
                    to = file.path("lightgbm_r", "inst", "bin/"),
                    overwrite = TRUE)
.handle_result(result)

# Build the package (do not touch this line!)
# NOTE: --keep-empty-dirs is necessary to keep the deep paths expected
#       by CMake while also meeting the CRAN req to create object files
#       on demand
cmd <- "R CMD build lightgbm_r --keep-empty-dirs"
.run_shell_command(cmd)

# Install the package
version <- gsub(
  "Version: ",
  "",
  grep(
    "Version: ",
    readLines(con = file.path("lightgbm_r", "DESCRIPTION")),
    value = TRUE
  )
)
tarball <- file.path(getwd(), sprintf("lightgbm_%s.tar.gz", version))

cmd <- sprintf("R CMD INSTALL %s --no-multiarch", tarball)
.run_shell_command(cmd)

# Run R CMD CHECK
# R CMD CHECK lightgbm_2.1.2.tar.gz --as-cran | tee check.log | cat
