cmake_minimum_required(VERSION 3.10)
project(AseCoding)

set(SOURCES
    app/app.cc
    app/test.cc
    app/utils.cc
)

set(HEADERS
    app/test.h
    app/utils.h
)

add_executable(app
    ${SOURCES}
    ${HEADERS}
)

# Add sub project
add_subdirectory(lib)
target_include_directories(app PUBLIC lib)
target_link_libraries(app lib)
