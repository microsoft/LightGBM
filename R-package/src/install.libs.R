src <- paste0(R_PACKAGE_SOURCE, '\\lib_lightgbm', SHLIB_EXT)
dest <- file.path(R_PACKAGE_DIR, paste0('libs', R_ARCH))
dir.create(dest, recursive = TRUE, showWarnings = FALSE)
if(file.exists(src)){
    file.copy(src, dest, overwrite = TRUE)
} else {
	stop("cannot find lib_lightgbm.dll")
}
