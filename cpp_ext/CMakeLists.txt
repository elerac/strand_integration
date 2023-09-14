cmake_minimum_required(VERSION  3.22.4)

project(strandtools LANGUAGES CXX)

if(NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

# We are now ready to compile the actual extension module
nanobind_add_module(

  # Name of the extension
  _strandtools_impl

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC

  # Source code goes here
  src/strandtools.cpp
)

# Install directive for scikit-build-core
install(TARGETS _strandtools_impl LIBRARY DESTINATION strandtools)

# Include directories
include_directories(src/include)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# target_link_libraries(_strandtools_impl PRIVATE ${OpenCV_LIBS})
target_link_libraries(_strandtools_impl PRIVATE opencv_core opencv_imgproc)

# OpenMP
if(APPLE)
  find_package(OpenMP)

  if(NOT OpenMP_FOUND)
    message(STATUS "OpenMP not found, trying again with brew --prefix libomp")

    # Try again with extra path info; required for libomp 15+ from Homebrew
    execute_process(COMMAND brew --prefix libomp
      OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
    set(OpenMP_C_LIB_NAMES omp)
    set(OpenMP_CXX_LIB_NAMES omp)
    set(OpenMP_omp_LIBRARY ${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib)
    find_package(OpenMP REQUIRED)
    
    message(STATUS "OpenMP_C_FLAGS: ${OpenMP_C_FLAGS}")
    message(STATUS "OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP_C_LIB_NAMES: ${OpenMP_C_LIB_NAMES}")
    message(STATUS "OpenMP_CXX_LIB_NAMES: ${OpenMP_CXX_LIB_NAMES}")
    message(STATUS "OpenMP_omp_LIBRARY: ${OpenMP_omp_LIBRARY}")
  endif()
else()
  find_package(OpenMP REQUIRED)
endif()

target_link_libraries(_strandtools_impl PRIVATE OpenMP::OpenMP_CXX)
