library(covr)

.run_shell_command <- function(cmd, ...){
    exit_code <- system(cmd, ...)
    if (exit_code != 0){
        stop(paste0("Command failed with exit code: ", exit_code))
    }
}
res <- .run_shell_command(
    "export CXX=/usr/local/bin/g++-8 CC=/usr/local/bin/gcc-8; Rscript build_r.R"
)
if (res != 0){
    stop("failed to build LightGBM")
}

path <- file.path(getwd(), "lighgbm_r")
type <- c("tests", "vignettes", "examples", "all", "none")
combine_types <- TRUE
relative_path <- TRUE
quiet <- TRUE
clean <- TRUE
line_exclusions <- NULL
function_exclusions <- NULL
code <- character()

pkg <- covr:::as_package(path)
type <- "tests"
type <- covr:::parse_type(type)
run_separately <- !isTRUE(combine_types) && length(type) > 1

tmp_lib <- file.path(Sys.getenv("HOME"), "Desktop", "whatever")
if (!dir.exists(tmp_lib)){
    dir.create(tmp_lib)
}

flags <- getOption("covr.flags")
if (!covr:::uses_icc()) {
    flags <- getOption("covr.flags")
} else {
    if (length(getOption("covr.icov")) > 0L) {
        flags <- getOption("covr.icov_flags")
        unlink(file.path(pkg$path, "src", "*.dyn"))
        unlink(file.path(pkg$path, "src", "pgopti.*"))
    } else {
        stop("icc is not available")
    }
}

# At this point, edit the main CMAkeLists.txt and then run Rscript build_r.R
file.copy(
    file.path(.libPaths()[[1]], pkg$package)
    , tmp_lib
    , recursive = TRUE
)
res <- covr:::add_hooks(
    "lightgbm"
    , tmp_lib
    , fix_mcexit = FALSE
)
libs <- covr:::env_path(
    tmp_lib
    , .libPaths()
)
withr::with_envvar(
    c(
        R_DEFAULT_PACKAGES = "datasets,utils,grDevices,graphics,stats,methods"
        , R_LIBS = libs
        , R_LIBS_USER = libs
        , R_LIBS_SITE = libs
        , R_COVR = "true"
    )
    , {
        out_dir <- file.path(tmp_lib, pkg$package)
        result <- tools::testInstalledPackage(
            pkg$package
            , outDir = out_dir
            , types = "tests"
            , lib.loc = tmp_lib
        )
        if (result != 0L) {
            covr:::show_failures(out_dir)
        }
        covr:::run_commands(pkg, tmp_lib, code)
    }
)

trace_files <- list.files(
    path = tmp_lib
    , pattern = "^covr_trace_[^/]+$"
    , full.names = TRUE
)

coverage <- covr:::merge_coverage(
    trace_files
)

# where the magic of getting coverage is supposed to happen
res <- covr:::run_gcov(tmp_lib, quiet = FALSE)
coverage <- structure(
    c(coverage, res)
    , class = "coverage"
    , package = pkg
    , relative = relative_path
)
print(coverage)

covr::report(
    x = coverage
    , file = file.path(getwd(), "coverage.html")
    , browse = TRUE
)
