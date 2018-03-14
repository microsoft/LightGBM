# Load useful libraries for development
library(devtools)
library(roxygen2) # devtools::install_github("klutometis/roxygen")
library(pkgdown) # devtools::install_github("Laurae2/pkgdown") # devtools::install_github("hadley/pkgdown")

# Set the working directory to where I am
# setwd("E:/GitHub/LightGBM/R-package")

# Generate documentation
document()

# Check for errors
devtools::check(document = FALSE)

# Build static website
pkgdown::build_site(run_dont_run = TRUE)

# Install package
install()
