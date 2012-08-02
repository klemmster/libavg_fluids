cmake_minimum_required (VERSION 2.6)

set(AVG_PATH $ENV{AVG_PATH})
if(NOT EXISTS "${AVG_PATH}/libavg/src/api.h")
          message(FATAL_ERROR "Please define the environment variable AVG_PATH to the directory containing the main libavg dir.")
endif()

project (libavg_fluids)

if(UNIX)
    find_package(Boost REQUIRED COMPONENTS python thread)
    set(INCLUDES ${INCLUDES} ${Boost_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${Boost_LIBRARIES})
    find_package(LibXml2 REQUIRED)
    set(INCLUDES ${INCLUDES} ${LIBXML2_INCLUDE_DIR})
    set(LIBS ${LIBS} ${LIBXML2_LIBRARIES})
endif(UNIX)

find_package(PythonLibs REQUIRED)
set(INCLUDES ${INCLUDES} ${PYTHON_INCLUDE_DIRS})
set(LIBS ${LIBS} ${PYTHON_LIBRARIES})

set(INCLUDES ${INCLUDES} ${AVG_PATH}/libavg/src)

include_directories (${INCLUDES})

set(FLUID_SRC src/FluidNode.cpp src/ProcessThread.cpp)

add_library(fluidNode MODULE ${FLUID_SRC})
SET(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/plugin/)

if(UNIX AND NOT APPLE)
    set(CXX_FLAGS ${CXX_FLAGS} "-O0 -g -Wall")
    target_link_libraries (fluidNode ${LIBS} ${AVG_PATH}/libavg/src/wrapper/.libs/avg.so )
	message(INFO ${LIBS} )
endif()

if(WIN32)
    set_target_properties(fluidNode PROPERTIES COMPILE_DEFINITIONS "BOOST_ALL_NO_LIB;AVG_PLUGIN")
    target_link_libraries (fluidNode ${LIBS} ${AVG_PATH}/libavg/win/Release/avg.lib)
endif()

if(UNIX)
	add_custom_command(
		TARGET fluidNode
		POST_BUILD
		COMMAND ${CMAKE_COMMAND}
		ARGS -E copy ${CMAKE_CFG_INTDIR}/plugin/libfluidNode${CMAKE_SHARED_LIBRARY_SUFFIX}
        ${CMAKE_CFG_INTDIR}/plugin/Release/fluidNode${CMAKE_SHARED_LIBRARY_SUFFIX}
		COMMAND ${CMAKE_COMMAND}
        ARGS -E remove ${CMAKE_CFG_INTDIR}/plugin/libfluidNode${CMAKE_SHARED_LIBRARY_SUFFIX}
		)

	execute_process(COMMAND python2 -c "import os,libavg; print os.path.dirname(libavg.__file__)"
		OUTPUT_VARIABLE AVG_LIB_PATH
		OUTPUT_STRIP_TRAILING_WHITESPACE
		)
endif()

install(TARGETS fluidNode
	DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/plugin/
	)
set_target_properties(PROPERTIES PREFIX "")

