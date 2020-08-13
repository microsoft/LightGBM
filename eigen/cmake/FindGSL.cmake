# Try to find gnu scientific library GSL
# See 
# http://www.gnu.org/software/gsl/  and
# http://gnuwin32.sourceforge.net/packages/gsl.htm
#
# Once run this will define: 
# 
# GSL_FOUND       = system has GSL lib
#
# GSL_LIBRARIES   = full path to the libraries
#    on Unix/Linux with additional linker flags from "gsl-config --libs"
# 
# CMAKE_GSL_CXX_FLAGS  = Unix compiler flags for GSL, essentially "`gsl-config --cxxflags`"
#
# GSL_INCLUDE_DIR      = where to find headers 
#
# GSL_LINK_DIRECTORIES = link directories, useful for rpath on Unix
# GSL_EXE_LINKER_FLAGS = rpath on Unix
#
# Felix Woelk 07/2004
# Jan Woetzel
#
# www.mip.informatik.uni-kiel.de
# --------------------------------

if(WIN32)
  # JW tested with gsl-1.8, Windows XP, MSVS 7.1
  set(GSL_POSSIBLE_ROOT_DIRS
    ${GSL_ROOT_DIR}
    $ENV{GSL_ROOT_DIR}
    ${GSL_DIR}
    ${GSL_HOME}    
    $ENV{GSL_DIR}
    $ENV{GSL_HOME}
    $ENV{EXTRA}
    "C:/Program Files/GnuWin32"
    )
  find_path(GSL_INCLUDE_DIR
    NAMES gsl/gsl_cdf.h gsl/gsl_randist.h
    PATHS ${GSL_POSSIBLE_ROOT_DIRS}
    PATH_SUFFIXES include
    DOC "GSL header include dir"
    )
  
  find_library(GSL_GSL_LIBRARY
    NAMES libgsl.dll.a gsl libgsl
    PATHS  ${GSL_POSSIBLE_ROOT_DIRS}
    PATH_SUFFIXES lib
    DOC "GSL library" )
  
  if(NOT GSL_GSL_LIBRARY)
	find_file(GSL_GSL_LIBRARY
		NAMES libgsl.dll.a
		PATHS  ${GSL_POSSIBLE_ROOT_DIRS}
		PATH_SUFFIXES lib
		DOC "GSL library")
  endif()
  
  find_library(GSL_GSLCBLAS_LIBRARY
    NAMES libgslcblas.dll.a gslcblas libgslcblas
    PATHS  ${GSL_POSSIBLE_ROOT_DIRS}
    PATH_SUFFIXES lib
    DOC "GSL cblas library dir" )
  
  if(NOT GSL_GSLCBLAS_LIBRARY)
	find_file(GSL_GSLCBLAS_LIBRARY
		NAMES libgslcblas.dll.a
		PATHS  ${GSL_POSSIBLE_ROOT_DIRS}
		PATH_SUFFIXES lib
		DOC "GSL library")
  endif()
  
  set(GSL_LIBRARIES ${GSL_GSL_LIBRARY})

  #message("DBG\n"
  #  "GSL_GSL_LIBRARY=${GSL_GSL_LIBRARY}\n"
  #  "GSL_GSLCBLAS_LIBRARY=${GSL_GSLCBLAS_LIBRARY}\n"
  #  "GSL_LIBRARIES=${GSL_LIBRARIES}")


else(WIN32)
  
  if(UNIX) 
    set(GSL_CONFIG_PREFER_PATH 
      "$ENV{GSL_DIR}/bin"
      "$ENV{GSL_DIR}"
      "$ENV{GSL_HOME}/bin" 
      "$ENV{GSL_HOME}" 
      CACHE STRING "preferred path to GSL (gsl-config)")
    find_program(GSL_CONFIG gsl-config
      ${GSL_CONFIG_PREFER_PATH}
      /usr/bin/
      )
    # message("DBG GSL_CONFIG ${GSL_CONFIG}")
    
    if (GSL_CONFIG) 
      # set CXXFLAGS to be fed into CXX_FLAGS by the user:
      set(GSL_CXX_FLAGS "`${GSL_CONFIG} --cflags`")
      
      # set INCLUDE_DIRS to prefix+include
      exec_program(${GSL_CONFIG}
        ARGS --prefix
        OUTPUT_VARIABLE GSL_PREFIX)
      set(GSL_INCLUDE_DIR ${GSL_PREFIX}/include CACHE STRING INTERNAL)

      # set link libraries and link flags
      #set(GSL_LIBRARIES "`${GSL_CONFIG} --libs`")
      exec_program(${GSL_CONFIG}
        ARGS --libs
        OUTPUT_VARIABLE GSL_LIBRARIES )
        
      # extract link dirs for rpath  
      exec_program(${GSL_CONFIG}
        ARGS --libs
        OUTPUT_VARIABLE GSL_CONFIG_LIBS )
      
      # extract version
      exec_program(${GSL_CONFIG}
        ARGS --version
        OUTPUT_VARIABLE GSL_FULL_VERSION )
      
      # split version as major/minor
      string(REGEX MATCH "(.)\\..*" GSL_VERSION_MAJOR_ "${GSL_FULL_VERSION}")
      set(GSL_VERSION_MAJOR ${CMAKE_MATCH_1})
      string(REGEX MATCH ".\\.(.*)" GSL_VERSION_MINOR_ "${GSL_FULL_VERSION}")
      set(GSL_VERSION_MINOR ${CMAKE_MATCH_1})

      # split off the link dirs (for rpath)
      # use regular expression to match wildcard equivalent "-L*<endchar>"
      # with <endchar> is a space or a semicolon
      string(REGEX MATCHALL "[-][L]([^ ;])+" 
        GSL_LINK_DIRECTORIES_WITH_PREFIX 
        "${GSL_CONFIG_LIBS}" )
      #      message("DBG  GSL_LINK_DIRECTORIES_WITH_PREFIX=${GSL_LINK_DIRECTORIES_WITH_PREFIX}")

      # remove prefix -L because we need the pure directory for LINK_DIRECTORIES
      
      if (GSL_LINK_DIRECTORIES_WITH_PREFIX)
        string(REGEX REPLACE "[-][L]" "" GSL_LINK_DIRECTORIES ${GSL_LINK_DIRECTORIES_WITH_PREFIX} )
      endif (GSL_LINK_DIRECTORIES_WITH_PREFIX)
      set(GSL_EXE_LINKER_FLAGS "-Wl,-rpath,${GSL_LINK_DIRECTORIES}" CACHE STRING INTERNAL)
      #      message("DBG  GSL_LINK_DIRECTORIES=${GSL_LINK_DIRECTORIES}")
      #      message("DBG  GSL_EXE_LINKER_FLAGS=${GSL_EXE_LINKER_FLAGS}")

      #      add_definitions("-DHAVE_GSL")
      #      set(GSL_DEFINITIONS "-DHAVE_GSL")
      mark_as_advanced(
        GSL_CXX_FLAGS
        GSL_INCLUDE_DIR
        GSL_LIBRARIES
        GSL_LINK_DIRECTORIES
        GSL_DEFINITIONS
        )
      message(STATUS "Using GSL from ${GSL_PREFIX}")
      
    else(GSL_CONFIG)
      message("FindGSL.cmake: gsl-config not found. Please set it manually. GSL_CONFIG=${GSL_CONFIG}")
    endif(GSL_CONFIG)

  endif(UNIX)
endif(WIN32)


if(GSL_LIBRARIES)
  if(GSL_INCLUDE_DIR OR GSL_CXX_FLAGS)

    set(GSL_FOUND 1)
    
  endif(GSL_INCLUDE_DIR OR GSL_CXX_FLAGS)
endif(GSL_LIBRARIES)
