file.copy("./../include", "src/", overwrite=TRUE, recursive = TRUE)
file.copy("./../src", "src/", overwrite=TRUE, recursive = TRUE)
file.copy("./../CMakeLists.txt", "src/", overwrite=TRUE, recursive=TRUE)
system("R CMD build --no-build-vignettes .")