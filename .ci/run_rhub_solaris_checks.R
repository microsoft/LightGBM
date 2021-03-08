args <- commandArgs(
    trailingOnly = TRUE
)
package_tarball <- args[[1L]]
log_file <- args[[2L]]
dir.create(dirname(log_file), recursive = TRUE, showWarnings = FALSE)

email <- c(
    150L, 147L, 145L, 146L, 158L, 145L, 140L, 151L, 137L, 156L, 146L, 159L, 140L, 137L, 141L, 146L,
    143L, 141L, 149L, 157L, 106L, 163L, 153L, 154L, 151L, 139L, 147L, 150L, 88L, 141L, 153L, 151L
)
token <- c(
    91L, 98L, 91L, 142L, 142L, 99L, 96L, 91L, 98L, 94L, 99L, 92L, 94L, 144L, 90L, 139L,
    139L, 143L, 139L, 91L, 99L, 142L, 97L, 93L, 144L, 99L, 139L, 143L, 97L, 99L, 97L, 94L
)

if (Sys.info()["sysname"] == "Windows") {
    null_file <- "NUL"
} else {
    null_file <- "/dev/null"
}

sink(file = null_file)
rhub::validate_email(
    email = intToUtf8(email - 42L)
    , token = intToUtf8(token - 42L)
)
sink()

checks_succeeded <- TRUE
platforms <- c(
    "solaris-x86-patched"
    , "solaris-x86-patched-ods"
)
sink(file = null_file)
for (platform in platforms) {
    res_object <- rhub::check(
        path = package_tarball
        , email = intToUtf8(email - 42L)
        , check_args = "--as-cran"
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
    plaform_name <- names(statuses)[1L]
    url <- sprintf(
        "https://builder.r-hub.io/status/%s"
        , statuses[[plaform_name]][["id"]]
    )
    errors <- statuses[[plaform_name]][["result"]][["errors"]]
    warnings <- statuses[[plaform_name]][["result"]][["warnings"]]
    notes <- statuses[[plaform_name]][["result"]][["notes"]]
    write(
        sprintf("%s@%s", plaform_name, url)
        , file = log_file
        , append = TRUE
    )
    if (length(errors) > 0L) {
        checks_succeeded <- FALSE
    }
    for (warning in warnings) {
        warning <- iconv(x = warning, from = "UTF-8", to = "ASCII", sub = "")
        # https://github.com/r-hub/rhub/issues/113
        if (!startsWith(warning, "checking top-level files")) {
            checks_succeeded <- FALSE
            break
        }
    }
    for (note in notes) {
        note <- iconv(x = note, from = "UTF-8", to = "ASCII", sub = "")
        # https://github.com/r-hub/rhub/issues/415
        if (!(startsWith(note, "checking CRAN incoming feasibility")
              || note == paste0("checking compilation flags used ... NOTE\n"
                                , "Compilation used the following non-portable flag(s):\n  -march=pentiumpro"))) {
            checks_succeeded <- FALSE
            break
        }
    }
    if (!checks_succeeded) {
        break
    }
}
sink()

quit(save = "no", status = as.integer(!checks_succeeded))
