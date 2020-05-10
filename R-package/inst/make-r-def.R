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
DLL_BASE_NAME <- basename(IN_DLL_FILE)

print(sprintf("Creating '%s' from '%s'", OUT_DEF_FILE, IN_DLL_FILE))

# Creates a .def file from R.dll, using tools bundled with R4.0
#LIBR_CORE_LIBRARY <- "C:/Program Files/R/R-3.6.1/bin/x64/R.dll"

# use objdump to dump all the symbols
OBJDUMP_FILE <- "objdump-out.txt"
exit_code <- system2(
    command = "objdump"
    , args = c(
        "-p"
        , shQuote(IN_DLL_FILE)
    )
    , stdout = OBJDUMP_FILE
)

objdump_results <- readLines(OBJDUMP_FILE)
result <- file.remove(OBJDUMP_FILE)

# Only one table in the objdump results matters for our purposes,
# see https://www.cs.colorado.edu/~main/cs1300/doc/mingwfaq.html
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
writeLines(
    text = c(
        paste0("LIBRARY \"", DLL_BASE_NAME, "\"")
        , "EXPORTS"
        , exported_symbols
    )
    , con = OUT_DEF_FILE
    , sep = "\n"
)
