# - MACRO_OPTIONAL_ADD_SUBDIRECTORY() combines add_subdirectory() with an option()
# MACRO_OPTIONAL_ADD_SUBDIRECTORY( <dir> )
# If you use MACRO_OPTIONAL_ADD_SUBDIRECTORY() instead of add_subdirectory(),
# this will have two effects
# 1 - CMake will not complain if the directory doesn't exist
#     This makes sense if you want to distribute just one of the subdirs
#     in a source package, e.g. just one of the subdirs in kdeextragear.
# 2 - If the directory exists, it will offer an option to skip the 
#     subdirectory.
#     This is useful if you want to compile only a subset of all
#     directories.

# Copyright (c) 2007, Alexander Neundorf, <neundorf@kde.org>
#
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.


macro (MACRO_OPTIONAL_ADD_SUBDIRECTORY _dir )
   get_filename_component(_fullPath ${_dir} ABSOLUTE)
   if(EXISTS ${_fullPath})
      if(${ARGC} EQUAL 2)
        option(BUILD_${_dir} "Build directory ${_dir}" ${ARGV1})
      else(${ARGC} EQUAL 2)
        option(BUILD_${_dir} "Build directory ${_dir}" TRUE)
      endif(${ARGC} EQUAL 2)
      if(BUILD_${_dir})
         add_subdirectory(${_dir})
      endif(BUILD_${_dir})
   endif(EXISTS ${_fullPath})
endmacro (MACRO_OPTIONAL_ADD_SUBDIRECTORY)
