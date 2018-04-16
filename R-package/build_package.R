unlink("./src/include", recursive = TRUE)
unlink("./src/src", recursive = TRUE)
unlink("./src/compute", recursive = TRUE)
unlink("./src/build", recursive = TRUE)
unlink("./src/Release", recursive = TRUE)
if (!file.copy("./../include", "./src/", overwrite = TRUE, recursive = TRUE)) {
  stop("Cannot find folder LightGBM/include")
}
if (!file.copy("./../src", "./src/", overwrite = TRUE, recursive = TRUE)) {
  stop("Cannot find folder LightGBM/src")
}
if (!file.copy("./../compute", "./src/", overwrite = TRUE, recursive = TRUE)) {
  print("Cannot find folder LightGBM/compute, will disable GPU build")
}
if (!file.copy("./../CMakeLists.txt", "./src/", overwrite = TRUE, recursive = TRUE)) {
  stop("Cannot find file LightGBM/CMakeLists.txt")
}
if (!file.exists("./src/_IS_FULL_PACKAGE")) {
  file.create("./src/_IS_FULL_PACKAGE")
}
system("R CMD build --no-build-vignettes .")
file.remove("./src/_IS_FULL_PACKAGE")
