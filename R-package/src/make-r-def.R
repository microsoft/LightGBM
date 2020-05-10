# [description]
#     Create a definition file (.def) from a .dll file, using objdump.
#
# [usage]
# 
#     Rscript make-r-def.R something.dll something.def
#
# [references]
#    * https://www.cs.colorado.edu/~main/cs1300/doc/mingwfaq.html

args <- commandArgs(trailingOnly = TRUE)

IN_DLL_FILE <- args[[1]]
OUT_DEF_FILE <- args[[2]]
print(sprintf("Creating '%s' from '%s'", IN_DLL_FILE, OUT_DEF_FILE))

# Creates a .def file from R.dll, using tools bundled with R4.0
#LIBR_CORE_LIBRARY <- "C:/Program Files/R/R-3.6.1/bin/x64/R.dll"

# use objdump to dump all the symbols
OBJDUMP_FILE <- "R.fil"
exit_code <- system2(
    command = "objdump"
    , args = c(
        "-p"
        , shQuote(IN_DLL_FILE)
    )
    , stdout = OBJDUMP_FILE
)

objdump_results <- readLines(OBJDUMP_FILE)
file.remove(OBJDUMP_FILE)

# Name Pointer table start
# https://www.cs.colorado.edu/~main/cs1300/doc/mingwfaq.html
start_index <- which(
    grepl(
        pattern = "[Ordinal/Name Pointer] Table"
        , x = objdump_results
        , fixed = TRUE
    )
)
empty_lines <- which(objdump_results == "")
end_of_table <- empty_lines[empty_lines > start_index][1]

# Read the contents of the table
exported_symbols <- objdump_results[(start_index + 1):end_of_table]
exported_symbols <- gsub("\t", "", exported_symbols)
exported_symbols <- gsub(".*\\] ", "", exported_symbols)
exported_symbols <- gsub(" ", "", exported_symbols)

# Write R.def file
write_succeeded <- writeLines(
    text = c(
        paste0("LIBRARY ", '\"R.dll\"')
        , "EXPORTS"
        , exported_symbols
    )
    , con = OUT_DEF_FILE
    , sep = "\n"
)
if (!isTRUE(write_succeeded)){
    stop("Failed to create R.def")
}
