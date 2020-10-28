#!/bin/bash

export PATH="$R_LIB_PATH/R/bin:$PATH"

if [[ $TASK == "r-package-check-docs" ]]; then
    Rscript build_r.R || exit -1
    Rscript --vanilla -e "roxygen2::roxygenize('R-package/', load = 'installed')" || exit -1
    num_doc_files_changed=$(
        git diff --name-only | grep -E "\.Rd|NAMESPACE" | wc -l
    )
    if [[ ${num_doc_files_changed} -gt 0 ]]; then
        echo "Some R documentation files have changed. Please re-generate them and commit those changes."
        echo ""
        echo "    Rscript build_r.R"
        echo "    Rscript -e \"roxygen2::roxygenize('R-package/', load = 'installed')\""
        echo ""
        exit -1
    fi
    exit 0
fi

cd ${BUILD_DIRECTORY}

PKG_TARBALL="lightgbm_*.tar.gz"
LOG_FILE_NAME="lightgbm.Rcheck/00check.log"
if [[ $R_BUILD_TYPE == "cmake" ]]; then
    Rscript build_r.R --skip-install || exit -1
elif [[ $R_BUILD_TYPE == "cran" ]]; then

    # on Linux, we recreate configure in CI to test if
    # a change in a PR has changed configure.ac
    if [[ $OS_NAME == "linux" ]]; then
        ${BUILD_DIRECTORY}/R-package/recreate-configure.sh

        num_files_changed=$(
            git diff --name-only | wc -l
        )
        if [[ ${num_files_changed} -gt 0 ]]; then
            echo "'configure' in the R package has changed. Please recreate it and commit the changes."
            echo "Changed files:"
            git diff --compact-summary
            echo "See R-package/README.md for details on how to recreate this script."
            echo ""
            exit -1
        fi
    fi

    ./build-cran-package.sh || exit -1

    # Test CRAN source .tar.gz in a directory that is not this repo or below it.
    # When people install.packages('lightgbm'), they won't have the LightGBM
    # git repo around. This is to protect against the use of relative paths
    # like ../../CMakeLists.txt that would only work if you are in the repo
    R_CMD_CHECK_DIR="${HOME}/tmp-r-cmd-check/"
    mkdir -p ${R_CMD_CHECK_DIR}
    mv ${PKG_TARBALL} ${R_CMD_CHECK_DIR}
    cd ${R_CMD_CHECK_DIR}
fi

# fails tests if either ERRORs or WARNINGs are thrown by
# R CMD CHECK
check_succeeded="yes"
(
    R CMD check ${PKG_TARBALL} \
        --as-cran \
        --run-donttest \
    || check_succeeded="no"
) &

# R CMD check suppresses output, some CIs kill builds after
# a few minutes with no output. This trick gives R CMD check more time
#     * https://github.com/travis-ci/travis-ci/issues/4190#issuecomment-169987525
#     * https://stackoverflow.com/a/29890106/3986677
CHECK_PID=$!
while kill -0 ${CHECK_PID} >/dev/null 2>&1; do
    echo -n -e " \b"
    sleep 5
done

echo "R CMD check build logs:"
BUILD_LOG_FILE=lightgbm.Rcheck/00install.out
cat ${BUILD_LOG_FILE}

if [[ $check_succeeded == "no" ]]; then
    exit -1
fi

if grep -q -R "WARNING" "$LOG_FILE_NAME"; then
    echo "WARNINGS have been found by R CMD check!"
    exit -1
fi

ALLOWED_CHECK_NOTES=2
NUM_CHECK_NOTES=$(
    cat ${LOG_FILE_NAME} \
        | grep -e '^Status: .* NOTE.*' \
        | sed 's/[^0-9]*//g'
)
if [[ ${NUM_CHECK_NOTES} -gt ${ALLOWED_CHECK_NOTES} ]]; then
    echo "Found ${NUM_CHECK_NOTES} NOTEs from R CMD check. Only ${ALLOWED_CHECK_NOTES} are allowed"
    exit -1
fi

# this check makes sure that CI builds of the CRAN package on Mac
# actually use OpenMP
if [[ $OS_NAME == "macos" ]] && [[ $R_BUILD_TYPE == "cran" ]]; then
    omp_working=$(
        cat $BUILD_LOG_FILE \
        | grep -E "checking whether OpenMP will work .*yes" \
        | wc -l
    )
    if [[ $omp_working -ne 1 ]]; then
        echo "OpenMP was not found, and should be when testing the CRAN package on macOS"
        exit -1
    fi
fi

# this check makes sure that no "warning: unknown pragma ignored" logs
# reach the user leading them to believe that something went wrong
if [[ $R_BUILD_TYPE == "cran" ]]; then
    pragma_warning_present=$(
        cat $BUILD_LOG_FILE \
        | grep -E "warning: unknown pragma ignored" \
        | wc -l
    )
    if [[ $pragma_warning_present -ne 0 ]]; then
        echo "Unknown pragma warning is present, pragmas should have been removed before build"
        exit -1
    fi
fi
