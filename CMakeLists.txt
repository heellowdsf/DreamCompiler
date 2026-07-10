cmake_minimum_required(VERSION 3.15)
project(DreamCompiler)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# MSVC reads source as the system code page (GBK on Chinese Windows) by default,
# which corrupts any non-ASCII byte. Force UTF-8 so the build is locale-independent.
if(MSVC)
    add_compile_options(/utf-8)
endif()

# Place the built executable in bin/ (standard project convention)
# so users don't have to dig into build/Debug/ to find it.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/bin)

find_package(LLVM REQUIRED CONFIG)

include_directories(include)
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(llvm_libs Core Support)

add_executable(DreamCompiler src/main.cpp src/Lexer.cpp src/Parser.cpp src/AST.cpp)
target_link_libraries(DreamCompiler ${llvm_libs})

# CUDA/cuBLAS are loaded at runtime via LoadLibraryA/dlopen in gpu_backend.inc,
# no compile-time linking needed. If CUDA is present the runtime detects it
# automatically; otherwise it falls back to OpenCL.

# Post-build: report the output location so users know where DreamCompiler.exe is
add_custom_command(TARGET DreamCompiler POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "DreamCompiler built at: $<TARGET_FILE:DreamCompiler>"
    COMMAND ${CMAKE_COMMAND} -E echo "Run from project root: bin\\DreamCompiler.exe examples\\<file>.dream"
    COMMAND ${CMAKE_COMMAND} -E echo ""
)
