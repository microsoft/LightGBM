args <- commandArgs(
    trailingOnly = TRUE
)
package_tarball <- args[[1L]]
log_file <- args[[2L]]

install.packages('rhub', dependencies = c('Depends', 'Imports', 'LinkingTo'), repos = 'https://cran.r-project.org')

email = c(
    150L, 147L, 145L, 146L, 158L, 145L, 140L, 151L, 137L, 158L, 143L, 157L, 158L, 137L, 143L, 151L,
    139L, 147L, 150L, 106L, 163L, 153L, 154L, 151L, 139L, 147L, 150L, 88L, 141L, 153L, 151L
)
rhub::validate_email(
    email = intToUtf8(email - 42L)
    , token = '6bc89147c8fc4824bce09f8454e4ab8e'
)

if (Sys.info()['sysname'] == "Windows") {
    null_file <- "NUL"
} else {
    null_file <- "/dev/null"
}
sink(file=null_file)
res_object <- rhub::check(
    path = package_tarball
    , email = intToUtf8(email - 42L)
    , check_args = c(
        "--as-cran"
    )
    , platform = c(
        "solaris-x86-patched"
        , "solaris-x86-patched-ods"
    )
    , env_vars = c(
        "R_COMPILE_AND_INSTALL_PACKAGES" = "always"
        , "_R_CHECK_SYSTEM_CLOCK_" = 0L
        , "_R_CHECK_CRAN_INCOMING_REMOTE_" = 0L
        , "_R_CHECK_PKG_SIZES_THRESHOLD_" = 60L
    )
    , show_status = TRUE
)
statuses <- res_object[[".__enclos_env__"]][["private"]][["status_"]]
result <- do.call(rbind, lapply(statuses, function(thisStatus) {
    data.frame(
        plaform = thisStatus[["platform"]][["name"]]
        , url = sprintf("https://builder.r-hub.io/status/%s", thisStatus[["id"]])
        , errors = length(thisStatus[["result"]][["errors"]])
        , warnings = length(thisStatus[["result"]][["warnings"]])
        , notes = length(thisStatus[["result"]][["notes"]])
        , stringsAsFactors = FALSE
    )
}))
sink()

dir.create(dirname(log_file), recursive = TRUE, showWarnings = FALSE)
for(i in 1L:nrow(result)) {
    write(
        sprintf("%s@%s", result[i, 1L], result[i, 2L])
        , file = log_file
        , append = TRUE
    )
}

quit(save = "no", status = sum(rowSums(result[, c(3L:5L)])))
