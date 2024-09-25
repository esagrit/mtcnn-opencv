cmake_minimum_required(VERSION 3.2)

project (MTCNN_APP)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

find_package( OpenCV REQUIRED )
find_package( Boost REQUIRED COMPONENTS system filesystem timer )
include_directories( ${Boost_INCLUDE_DIRS} )

set(SOURCES
./mtcnn/detector.cpp
./mtcnn/face.h 
./mtcnn/onet.cpp 
./mtcnn/pnet.cpp ./mtcnn/rnet.cpp
./mtcnn/detector.h ./mtcnn/helpers.h ./mtcnn/onet.h ./mtcnn/pnet.h ./mtcnn/rnet.h
./draw.hpp
./helpers.hpp
)

set(APP_SOURCES
# infer_photo.cpp
# infer_photo2.cpp
# mtcnn_lfw.cpp
# mtcnn_rafi.cpp
mtcnn_ibug.cpp
mtcnn_lfw_160.cpp
# mtcnn_mugshot.cpp
# yolo_lfw.cpp
# yolo_mugshot.cpp
# dnn_mugshot.cpp
# infer_cam.cpp
)

foreach( sourcefile ${APP_SOURCES} )
    # Simple string replace, to cut off .cpp.
    string( REPLACE ".cpp" "" name ${sourcefile} )
    add_executable( ${name} ${sourcefile} ${SOURCES} )
    target_link_libraries (${name} LINK_PUBLIC ${OpenCV_LIBS} ${Boost_LIBRARIES} )
endforeach( sourcefile ${APP_SOURCES} )