# kbref: generators of plain-text NESSi reference data for KadanoffBaym.
#
# Build (from the repository root):
#   cmake -S .nessi/kbref -B .nessi/kbref/build
#   cmake --build .nessi/kbref/build --target <target>
#
# Prerequisites (see .omo/plans/01-phase1-foundation.md, section 3.2):
#   .nessi/install         NESSi libcntr (hdf5/omp/mpi/ness OFF)
#   .nessi/eigen3-install  Eigen3 install prefix
cmake_minimum_required(VERSION 3.10)
project(kbref CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

get_filename_component(KBREF_NESSI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(KBREF_CNTR_PREFIX "${KBREF_NESSI_DIR}/install")
set(KBREF_EIGEN_PREFIX "${KBREF_NESSI_DIR}/eigen3-install")
set(KBREF_CNTR_LIB "${KBREF_CNTR_PREFIX}/lib/libcntr.so")

if(NOT EXISTS "${KBREF_CNTR_LIB}")
    message(FATAL_ERROR
        "kbref: ${KBREF_CNTR_LIB} not found. Build and install NESSi first:\n"
        "  cmake -S .nessi/nessi-2.0.0/libcntr -B .nessi/build ... && "
        "cmake --build .nessi/build && cmake --install .nessi/build")
endif()
if(NOT EXISTS "${KBREF_CNTR_PREFIX}/include/cntr/cntr.hpp")
    message(FATAL_ERROR
        "kbref: ${KBREF_CNTR_PREFIX}/include/cntr/cntr.hpp not found. "
        "Run: cmake --install .nessi/build")
endif()
if(NOT EXISTS "${KBREF_EIGEN_PREFIX}/include/eigen3/Eigen/Dense")
    message(FATAL_ERROR
        "kbref: Eigen3 headers not found in ${KBREF_EIGEN_PREFIX}/include. "
        "Run: cmake --install .nessi/eigen3-build")
endif()

# NESSi uses #include <eigen3/Eigen/Dense>, so the plain include prefix of
# Eigen is required. The NESSi headers include each other without the
# cntr/ prefix, so include/cntr is added as well.
set(KBREF_INCLUDES
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${KBREF_CNTR_PREFIX}/include"
    "${KBREF_CNTR_PREFIX}/include/cntr"
    "${KBREF_EIGEN_PREFIX}/include")

set(CMAKE_BUILD_RPATH "${KBREF_CNTR_PREFIX}/lib")

add_executable(weights weights.cpp)
target_include_directories(weights PRIVATE ${KBREF_INCLUDES})
target_link_libraries(weights PRIVATE "${KBREF_CNTR_LIB}")

add_executable(convolution convolution.cpp)
target_include_directories(convolution PRIVATE ${KBREF_INCLUDES})
target_compile_definitions(convolution PRIVATE CNTR_NO_EXTERN_TEMPLATES)
target_link_libraries(convolution PRIVATE "${KBREF_CNTR_LIB}")

# P2 generators (see .omo/plans/02-phase2-utilities.md, Wave 7)
foreach(target p2_utilities p2_differentiation p2_convolution_utilities p2_response)
    add_executable(${target} ${target}.cpp)
    target_include_directories(${target} PRIVATE ${KBREF_INCLUDES})
    target_compile_definitions(${target} PRIVATE CNTR_NO_EXTERN_TEMPLATES)
    target_link_libraries(${target} PRIVATE "${KBREF_CNTR_LIB}")
endforeach()

# P3 generators (see .omo/plans/03-phase3-dyson.md, Wave 5)
foreach(target p3_dyson_ref p3_equilibrium_ref)
    add_executable(${target} ${target}.cpp)
    target_include_directories(${target} PRIVATE ${KBREF_INCLUDES})
    target_compile_definitions(${target} PRIVATE CNTR_NO_EXTERN_TEMPLATES)
    target_link_libraries(${target} PRIVATE "${KBREF_CNTR_LIB}")
endforeach()

# P4 generators (see .omo/plans/04-phase4-vie2-bubble.md, Wave 4)
foreach(target phase4_vie2 phase4_bubble phase4_observables)
    add_executable(${target} ${target}.cpp)
    target_include_directories(${target} PRIVATE ${KBREF_INCLUDES})
    target_compile_definitions(${target} PRIVATE CNTR_NO_EXTERN_TEMPLATES)
    target_link_libraries(${target} PRIVATE "${KBREF_CNTR_LIB}")
endforeach()

# P6 generator (see .omo/plans/06-phase6-pseudo.md, Wave 5)
add_executable(pseudo_reference pseudo_reference.cpp)
target_include_directories(pseudo_reference PRIVATE ${KBREF_INCLUDES})
target_compile_definitions(pseudo_reference PRIVATE CNTR_NO_EXTERN_TEMPLATES)
target_link_libraries(pseudo_reference PRIVATE "${KBREF_CNTR_LIB}")

# P7 generators (see .omo/plans/07-phase7-fourier-ness.md, Wave 10). They
# require libcntr built with ness=ON and the system FFTW.
find_library(KBREF_FFTW_LIB NAMES fftw3 REQUIRED)
foreach(target phase7_fourier_ref phase7_ness_ref phase7_examples_siam_ref
        phase7_examples_bethe_ref phase7_examples_bethe_prop_ref)
    add_executable(${target} ${target}.cpp)
    target_include_directories(${target} PRIVATE ${KBREF_INCLUDES})
    target_compile_definitions(${target} PRIVATE CNTR_NO_EXTERN_TEMPLATES CNTR_USE_NESS)
    target_link_libraries(${target} PRIVATE "${KBREF_CNTR_LIB}" "${KBREF_FFTW_LIB}")
endforeach()

# P8 generators (see .omo/plans/08-phase8-moving-parallel.md, Wave 7). The
# moving Dyson solver of the installed libcntr is cntr_dyson_moving_impl_org.hpp.
foreach(target p8_trunc_bethe_ref p8_moving_solver_ref p8_moving_ref p8_gw_ref)
    add_executable(${target} ${target}.cpp)
    target_include_directories(${target} PRIVATE ${KBREF_INCLUDES})
    target_compile_definitions(${target} PRIVATE CNTR_NO_EXTERN_TEMPLATES)
    target_link_libraries(${target} PRIVATE "${KBREF_CNTR_LIB}")
endforeach()

# P8 generator with NESS2 (needs ness=ON and FFTW, see the P7 block)
add_executable(p8_ness2_bethe_trunc_ref p8_ness2_bethe_trunc_ref.cpp)
target_include_directories(p8_ness2_bethe_trunc_ref PRIVATE ${KBREF_INCLUDES})
target_compile_definitions(p8_ness2_bethe_trunc_ref PRIVATE CNTR_NO_EXTERN_TEMPLATES CNTR_USE_NESS)
target_link_libraries(p8_ness2_bethe_trunc_ref PRIVATE "${KBREF_CNTR_LIB}" "${KBREF_FFTW_LIB}")

# P5 generators (see .omo/plans/05-phase5-examples.md, P5-W1-T7). They
# link only libcntr and, where needed, the physics implementation of the
# NESSi examples (no HDF5, MPI, or OpenMP). A target whose source file is
# not written yet is reported as pending.
set(KBREF_EXAMPLES "${CMAKE_CURRENT_SOURCE_DIR}/../nessi-2.0.0/examples/programs")
set(P5_TARGETS hubbard_chain_text fkm_bethe_quench_text holstein_bethe_text
    holstein_nambu_text holstein_impurity_text)
set(P5_IMPL_hubbard_chain_text "${KBREF_EXAMPLES}/hubbard_chain_selfen_impl.cpp")
set(P5_IMPL_fkm_bethe_quench_text "")
set(P5_IMPL_holstein_bethe_text "${KBREF_EXAMPLES}/Holstein_impurity_impl.cpp;${KBREF_EXAMPLES}/Holstein_utils_impl.cpp")
set(P5_IMPL_holstein_nambu_text "${P5_IMPL_holstein_bethe_text}")
set(P5_IMPL_holstein_impurity_text "${P5_IMPL_holstein_bethe_text}")
foreach(target ${P5_TARGETS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${target}.cpp")
        add_executable(${target} ${target}.cpp ${P5_IMPL_${target}})
        # The extern templates of libcntr are used, as in the NESSi example
        # build (with CNTR_NO_EXTERN_TEMPLATES cntr_response_convolution_impl.hpp
        # undefines the GREEN macro needed by the example headers)
        target_include_directories(${target} PRIVATE ${KBREF_INCLUDES} "${KBREF_EXAMPLES}"
                                   "${KBREF_EIGEN_PREFIX}/include/eigen3")
        target_link_libraries(${target} PRIVATE "${KBREF_CNTR_LIB}")
    else()
        message(STATUS "kbref: P5 target ${target} pending (no source file)")
    endif()
endforeach()
