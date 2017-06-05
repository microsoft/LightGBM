if(!file.copy("./../include", "src/", overwrite=TRUE, recursive = TRUE)){
  stop("cannot find folder LightGBM/include")
}
if(!file.copy("./../src", "src/", overwrite=TRUE, recursive = TRUE)){
  stop("cannot find folder LightGBM/src")
}
if(!file.copy("./../compute", "src/", overwrite=TRUE, recursive = TRUE)){
  print("cannot find folder LightGBM/compute, will disable GPU build")
}
if(!file.copy("./../CMakeLists.txt", "src/", overwrite=TRUE, recursive=TRUE)){
  stop("cannot find file LightGBM/CMakeLists.txt")
}
if(!file.exists("src/_IS_FULL_PACKAGE")){
  file.create("src/_IS_FULL_PACKAGE")
}
system("R CMD build --no-build-vignettes .")