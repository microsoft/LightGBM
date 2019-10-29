
library(lintr)

args <- commandArgs(
    trailingOnly = TRUE
)
SOURCE_DIR <- args[[1L]]

FILES_TO_LINT <- list.files(
    path = SOURCE_DIR
    , pattern = "\\.r$"
    , all.files = TRUE
    , ignore.case = TRUE
    , full.names = TRUE
    , recursive = TRUE
    , include.dirs = FALSE
)

LINTERS_TO_USE <- list(
    "assignment" = lintr::assignment_linter
    , "closed_curly" = lintr::closed_curly_linter
    , "equals_na" = lintr::equals_na_linter
    , "function_left" = lintr::function_left_parentheses_linter
    , "commas" = lintr::commas_linter
    , "concatenation" = lintr::unneeded_concatenation_linter
    , "implicit_integers" = lintr::implicit_integer_linter
    , "infix_spaces" = lintr::infix_spaces_linter
    , "long_lines" = lintr::line_length_linter(length = 120L)
    , "tabs" = lintr::no_tab_linter
    , "open_curly" = lintr::open_curly_linter
    , "paren_brace_linter" = lintr::paren_brace_linter
    , "semicolon" = lintr::semicolon_terminator_linter
    , "seq" = lintr::seq_linter
    , "single_quotes" = lintr::single_quotes_linter
    , "spaces_inside" = lintr::spaces_inside_linter
    , "spaces_left_parens" = lintr::spaces_left_parentheses_linter
    , "todo_comments" = lintr::todo_comment_linter
    , "trailing_blank" = lintr::trailing_blank_lines_linter
    , "trailing_white" = lintr::trailing_whitespace_linter
    , "true_false" = lintr::T_and_F_symbol_linter
)

cat(sprintf("Found %i R files to lint\n", length(FILES_TO_LINT)))

results <- NULL

for (r_file in FILES_TO_LINT) {

    this_result <- lintr::lint(
        filename = r_file
        , linters = LINTERS_TO_USE
        , cache = FALSE
    )

    cat(sprintf(
        "Found %i linting errors in %s\n"
        , length(this_result)
        , r_file
    ))

    results <- c(results, this_result)

}

issues_found <- length(results)

if (issues_found > 0L) {
    cat("\n")
    print(results)
}

quit(save = "no", status = issues_found)
