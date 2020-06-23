### configure.ac                    -*- Autoconf -*-
# Template used by Autoconf to generate 'configure' script. For more see:
#   * https://unconj.ca/blog/an-autoconf-primer-for-r-package-authors.html
#   * https://cran.r-project.org/doc/manuals/r-release/R-exts.html#Configure-and-cleanup

AC_PREREQ(2.69)
AC_INIT([lightgbm], [2.3.2], [], [lightgbm], [])

###########################
# find compiler and flags #
###########################

AC_MSG_CHECKING([location of R])
AC_MSG_RESULT([${R_HOME}])

# set up CPP flags
# find the compiler and compiler flags used by R.
: ${R_HOME=`R HOME`}
if test -z "${R_HOME}"; then
    echo "could not determine R_HOME"
    exit 1
fi
CC=`"${R_HOME}/bin/R" CMD config CC`
CXX=`"${R_HOME}/bin/R" CMD config CXX11`
CFLAGS=`"${R_HOME}/bin/R" CMD config CFLAGS`
CPPFLAGS=`"${R_HOME}/bin/R" CMD config CPPFLAGS`

################
# LGB_CPPFLAGS #
################

LGB_CPPFLAGS=""

AC_MSG_CHECKING([whether using R version 3.5.0 or later])
ac_pkg_r35=no
R_35_OR_ABOVE=$(
    ${R_HOME}/bin/Rscript -e "
        R_ver <- as.double(R.version[['major']]) + as.double(R.version[['minor']]) / 10.0;
        cat(as.character(R_ver > 3.5))
    "
)
if test ${R_35_OR_ABOVE} = "TRUE"
then
    ac_pkg_r35=yes
    LGB_CPPFLAGS+=" -DR_VER_ABOVE_35"
fi
AC_MSG_RESULT([${ac_pkg_r35}])

# Compile for GPU?
AC_ARG_ENABLE(
    [gpu],
    AS_HELP_STRING(
        [--enable-gpu],
        [Pass this flag to install the R package for usage with GPU]
    ),
    [
        LGB_CPPFLAGS+=" -DUSE_GPU"
    ]
)

###############
# MM_PREFETCH #
###############

AC_MSG_CHECKING([whether MM_PREFETCH works])
ac_mmprefetch=no
AC_LANG_CONFTEST(
    [
        AC_LANG_PROGRAM(
            [[
                #include <xmmintrin.h>
            ]],
            [[
                int main() {
                    int a = 0;
                    _mm_prefetch(&a, _MM_HINT_NTA);
                    return 0;
                }
            ]]
        )
    ]
)
${CC} -o conftest conftest.c 2>/dev/null && ./conftest && ac_mmprefetch=yes
AC_MSG_RESULT([${ac_mmprefetch}])
if test "${ac_mmprefetch}" = yes; then
    LGB_CPPFLAGS+=" -DMM_PREFETCH=1"
fi

############
# MM_ALLOC #
############

AC_MSG_CHECKING([whether MM_MALLOC works])
ac_mm_malloc=no
AC_LANG_CONFTEST(
    [
        AC_LANG_PROGRAM(
            [[
                #include <mm_malloc.h>
            ]],
            [[
                int main() {
                    char *a = (char*)_mm_malloc(8, 16);
                    _mm_free(a);
                    return 0;
                }
            ]]
        )
    ]
)
${CC} -o conftest conftest.c 2>/dev/null && ./conftest && ac_mm_malloc=yes
AC_MSG_RESULT([${ac_mm_malloc}])
if test "${ac_mm_malloc}" = yes; then
    LGB_CPPFLAGS+=" -DMM_MALLOC=1"
fi

##########
# OpenMP #
##########

OPENMP_CXXFLAGS=""

if test `uname -s` = "Linux"
then
    OPENMP_CXXFLAGS="\$(SHLIB_OPENMP_CXXFLAGS)"
fi

if test `uname -s` = "Darwin"
then
    OPENMP_CXXFLAGS='-Xclang -fopenmp'
    OPENMP_LIB='/usr/local/lib/libomp.dylib'
    ac_pkg_openmp=no
    AC_MSG_CHECKING([whether OpenMP will work in a package])
    AC_LANG_CONFTEST(
        [
            AC_LANG_PROGRAM(
                [[
                    #include <omp.h>
                ]],
                [[
                    return (omp_get_max_threads() <= 1);
                ]]
            )
        ]
    )
    ${CC} -o conftest conftest.c ${OPENMP_LIB} ${OPENMP_CXXFLAGS} 2>/dev/null && ./conftest && ac_pkg_openmp=yes
    AC_MSG_RESULT([${ac_pkg_openmp}])
    if test "${ac_pkg_openmp}" = no; then
        OPENMP_CXXFLAGS=''
        OPENMP_LIB=''
        echo '***********************************************************************************************'
        echo ' OpenMP is unavailable on this Mac OSX system. LightGBM code may be slower as a result'
        echo ' To use all CPU cores for training jobs, you should install OpenMP by running'
        echo ''
        echo '     brew install libomp'
        echo '***********************************************************************************************'
    fi
fi

# substitute variables from this script into Makevars.in
AC_SUBST(OPENMP_CXXFLAGS)
AC_SUBST(OPENMP_LIB)
AC_SUBST(LGB_CPPFLAGS)
AC_CONFIG_FILES([src/Makevars])

# write out Autoconf output
AC_OUTPUT
