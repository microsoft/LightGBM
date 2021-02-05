args <- commandArgs(
    trailingOnly = TRUE
)
package_tarball <- args[[1L]]
log_file <- args[[2L]]
dir.create(dirname(log_file), recursive = TRUE, showWarnings = FALSE)

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
checks_succeeded = TRUE
platforms = c(
    "solaris-x86-patched"
    , "solaris-x86-patched-ods"
)
for (platform in platforms) {
    res_object <- rhub::check(
        path = package_tarball
        , email = intToUtf8(email - 42L)
        , check_args = c(
            "--as-cran"
        )
        , platform = platform
        , env_vars = c(
            "R_COMPILE_AND_INSTALL_PACKAGES" = "always"
            , "_R_CHECK_SYSTEM_CLOCK_" = 0L
            , "_R_CHECK_CRAN_INCOMING_REMOTE_" = 0L
            , "_R_CHECK_PKG_SIZES_THRESHOLD_" = 60L
            , "_R_CHECK_TOPLEVEL_FILES_" = 0L
        )
        , show_status = TRUE
    )
    statuses <- res_object[[".__enclos_env__"]][["private"]][["status_"]]
    plaform_name = names(statuses)[1]
    url = sprintf(
        "https://builder.r-hub.io/status/%s"
        , statuses[[plaform_name]][["id"]]
    )
    errors = statuses[[plaform_name]][["result"]][["errors"]]
    warnings = statuses[[plaform_name]][["result"]][["warnings"]]
    notes = statuses[[plaform_name]][["result"]][["notes"]]
    write(
        sprintf("%s@%s", plaform_name, url)
        , file = log_file
        , append = TRUE
    )
    if (length(errors) > 0L) {
        checks_succeeded = FALSE
    }
    for (warning in warnings) {
        warning = iconv(warning, "UTF-8", "ASCII", sub="")
        # https://github.com/r-hub/rhub/issues/113
        if (!startsWith(warning, "checking top-level files")) {
            checks_succeeded = FALSE
            break
        }
    }
    for (note in notes) {
        note = iconv(note, "UTF-8", "ASCII", sub="")
        # https://github.com/r-hub/rhub/issues/415
        if (!(startsWith(note, "checking CRAN incoming feasibility")
            || note == "checking compilation flags used ... NOTE\nCompilation used the following non-portable flag(s):\n  -march=pentiumpro")) {
            checks_succeeded = FALSE
            break
        }
    }
    if (!checks_succeeded) {
        break
    }
}
sink()

quit(save = "no", status = as.integer(!checks_succeeded))
