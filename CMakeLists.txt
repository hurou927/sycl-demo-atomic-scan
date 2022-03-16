set(CMAKE_CXX_COMPILER "dpcpp")
set(CMAKE_C_COMPILE "clang")

cmake_minimum_required(VERSION 3.4.3)

project(atomic_scan)

set(CMAKE_CXX_STANDARD 20)

include_directories(${PROJECT_SOURCE_DIR}/include)

# add_compile_options(-Wall -Wextra -pedantic -Werror)
add_compile_options(-O3)

# https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html
# ❯ sudo lshw -class processor
#   *-cpu
#        description: CPU
#        product: 11th Gen Intel(R) Core(TM) i7-11700K @ 3.60GHz
set(CMAKE_CXX_FLAGS "-fsycl-targets=spir64_gen -Xs \"-device tgllp\"")
add_subdirectory(src)
