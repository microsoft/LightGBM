# mirrors that host miktexsetup.zip do so only with explicitly-named
# files like miktexsetup-2.4.5.zip, so hard-coding a link to an archive as a
# way to peg to one mirror does not work
#
# this script will find the specific version of miktexsetup.zip at a given
# mirror
library(httr)
args <- commandArgs(trailingOnly = TRUE)
DESTFILE <- args[[1L]]
MIRROR <- "https://ctan.math.illinois.edu/systems/win32/miktex/setup/windows-x64/"
mirror_contents <- httr::content(
    httr::RETRY("GET", MIRROR)
    , as = "text"
)
content_lines <- strsplit(mirror_contents, "\n")[[1L]]
content_lines <- content_lines[grepl("miktexsetup-.*", content_lines)]
zip_loc <- regexpr(">miktexsetup-[0-9]+.*x64\\.zip", content_lines)
zip_name <- gsub(">", "", regmatches(content_lines, zip_loc))
full_zip_url <- file.path(MIRROR, zip_name)
print(sprintf("downloading %s", full_zip_url))
download.file(
    url = full_zip_url
    , destfile = DESTFILE
)
print(sprintf("MiKTeX setup downloaded to %s", DESTFILE))
