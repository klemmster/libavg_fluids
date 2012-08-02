cmake_minimum_required (VERSION 2.6)

set(AVG_PATH $ENV{AVG_PATH})
if(NOT EXISTS "${AVG_PATH}/libavg/src/api.h")
          message(FATAL_ERROR "Please define the environment variable AVG_PATH to the directory containing the main libavg dir.")
endif()

project (libavg_fluids)

INCLUDE(FindCUDA)
IF(CUDA_FOUND)

    find_package(Boost REQUIRED COMPONENTS python thread system)
        set(INCLUDES ${INCLUDES} ${Boost_INCLUDE_DIRS})
        set(LIBS ${LIBS} ${Boost_LIBRARIES})
    find_package(LibXml2 REQUIRED)
        set(INCLUDES ${INCLUDES} ${LIBXML2_INCLUDE_DIR})
        set(LIBS ${LIBS} ${LIBXML2_LIBRARIES})
    find_package(PythonLibs REQUIRED)
        set(INCLUDES ${INCLUDES} ${PYTHON_INCLUDE_DIRS})
        set(LIBS ${LIBS} ${PYTHON_LIBRARIES})

    set(INCLUDES ${INCLUDES} ${AVG_PATH}/libavg/src)

    INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/../shared/inc)
    INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/common/inc)
    INCLUDE_DIRECTORIES (${INCLUDES})

    set(FLUID_SRC src/FluidNode.cpp src/ProcessThread.cpp)

    add_library(fluidnode MODULE ${FLUID_SRC})
    SET(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/plugin/)
    SET(LIBS ${LIBS} ${AVG_PATH}/libavg/src/wrapper/.libs/avg.so)

    message(INFO:  ${LIBS} )

    SET(CMAKE_CXX_COMPILER_VERSION "4.6.3")
    SET(CMAKE_C_COMPILER_VERSION "4.6.3")
    set(CXX_FLAGS ${CXX_FLAGS} "-O0 -g -Wall")

    LINK_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/lib)
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_20)
    target_link_libraries (fluidnode ${LIBS} )

    if(UNIX)
        add_custom_command(
            TARGET fluidnode
            POST_BUILD
            COMMAND ${CMAKE_COMMAND}
            ARGS -E copy ${CMAKE_CFG_INTDIR}/plugin/libfluidnode${CMAKE_SHARED_LIBRARY_SUFFIX}
            ${CMAKE_CFG_INTDIR}/plugin/fluidnode${CMAKE_SHARED_LIBRARY_SUFFIX}
            COMMAND ${CMAKE_COMMAND}
            ARGS -E remove ${CMAKE_CFG_INTDIR}/plugin/libfluidnode${CMAKE_SHARED_LIBRARY_SUFFIX}
            )

        execute_process(COMMAND python2 -c "import os,libavg; print os.path.dirname(libavg.__file__)"
            OUTPUT_VARIABLE AVG_LIB_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            )
    endif()

    install(TARGETS fluidnode
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/plugin/
        )
    set_target_properties(PROPERTIES PREFIX "")

ENDIF(CUDA_FOUND)

