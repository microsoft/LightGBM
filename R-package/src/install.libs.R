folders <- c(R_PACKAGE_SOURCE, paste0(R_PACKAGE_SOURCE, '/inst/'),
             paste0(R_PACKAGE_SOURCE, '/..'), paste0(R_PACKAGE_SOURCE, '/../lib'))
if (WINDOWS) {
  folders <- c(folders, paste0(R_PACKAGE_SOURCE, '/../Release/'), paste0(R_PACKAGE_SOURCE, '/../windows/x64/DLL/'))
}
dest <- file.path(R_PACKAGE_DIR, paste0('libs', R_ARCH))
dir.create(dest, recursive = TRUE, showWarnings = FALSE)
has_dll <- FALSE
for (folder in folders) {
  src <- paste0(folder, '/lib_lightgbm', SHLIB_EXT)
  if(file.exists(src)){
    has_dll <- TRUE
    print(paste0("find library file: ", src))
    file.copy(src, dest, overwrite = TRUE)
    break
  }
}
if (has_dll == FALSE){
  stop("cannot find lib_lightgbm.dll")
}
