set(CMAKE_CXX_COMPILER "dpcpp")
set(CMAKE_C_COMPILE "clang")

cmake_minimum_required(VERSION 3.4.3)

project(atomic_scan)

set(CMAKE_CXX_STANDARD 20)

include_directories(${PROJECT_SOURCE_DIR}/include)

# add_compile_options(-Wall -Wextra -pedantic -Werror)
# add_compile_options(-O2)

add_subdirectory(src)
