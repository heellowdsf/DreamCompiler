// dream language runtime     module orchestrator with crash handler

// physical layout (all modules are #included into this single translation
// unit, in dependency order, exactly as before -- the directory grouping is
// organizational only and does not change how the runtime is compiled):
//   core/     tensor + memory + indexing primitives
//   ops/      elementwise math, extra math, autograd
//   nn/       im2col, convolution/pooling, training (RNN/LSTM/attention/augment)
//   io/       strings, structs, closures, image I/O, builtins
//   backend/  GPU (CUDA/OpenCL) dispatch
#include "runtime.h"
#include "backend/gpu.inc"

extern "C" {
#include "core/memory_pool.inc"
#include "core/tensor.inc"
#include "ops/elementwise.inc"
#include "nn/im2col.inc"
#include "ops/autograd.inc"
#include "core/indexing.inc"
#include "io/builtins.inc"
#include "io/strings.inc"
#include "ops/math_extra.inc"
#include "io/structs.inc"
#include "io/closures.inc"
#include "nn/conv.inc"
#include "io/image_io.inc"
#include "nn/training.inc"

Tensor* app();
} //extern "c"

// dream crash handler
// turns cryptic windows exception codes into actionable error messages
// that point users to the likely cause in their Dream code.

static thread_local const char* g_current_op = "unknown";
static thread_local const char* g_current_fn = "app";
static thread_local int g_current_line = 0;

#if defined(_WIN32)

static const char* seh_code_to_name(DWORD code) {
    switch (code) {
        case EXCEPTION_ACCESS_VIOLATION:       return "ACCESS_VIOLATION (0xc0000005)";
        case EXCEPTION_STACK_OVERFLOW:         return "STACK_OVERFLOW (0xc00000fd)";
        case EXCEPTION_INT_DIVIDE_BY_ZERO:     return "INTEGER_DIVIDE_BY_ZERO (0xc0000094)";
        case EXCEPTION_FLT_DIVIDE_BY_ZERO:     return "FLOAT_DIVIDE_BY_ZERO (0xc000008e)";
        case EXCEPTION_FLT_INVALID_OPERATION:  return "FLOAT_INVALID (0xc0000090)";
        case EXCEPTION_ILLEGAL_INSTRUCTION:    return "ILLEGAL_INSTRUCTION (0xc000001d)";
        case 0xc0000409:                       return "STACK_BUFFER_OVERRUN (0xc0000409)";
        default:                               return "UNKNOWN_EXCEPTION";
    }
}

static const char* seh_explanation(DWORD code) {
    switch (code) {
        case EXCEPTION_ACCESS_VIOLATION:
            return "The program tried to read or write an invalid memory address.\n"
                   "  Common causes in Dream:\n"
                   "    * Using a tensor after it was garbage-collected\n"
                   "    * Passing a null tensor to an operation (check if function returned successfully)\n"
                   "    * Tuple indexing: use `let (a, b) = expr` not `let a = expr[0]`\n"
                   "    * Shape mismatch not caught by runtime checks";
        case EXCEPTION_STACK_OVERFLOW:
            return "The program's call stack overflowed.\n"
                   "  Common causes in Dream:\n"
                   "    * Recursive function with no base case\n"
                   "    * Deep computation graph with recursive backward\n"
                   "    * Very large local tensors";
        case 0xc0000409:
            return "Stack buffer overrun / memory corruption detected.\n"
                   "  Common causes in Dream:\n"
                   "    * Writing past the end of a tensor's data buffer\n"
                   "    * Use-after-free: tensor was freed but still referenced\n"
                   "    * Reference count imbalance";
        case EXCEPTION_INT_DIVIDE_BY_ZERO:
        case EXCEPTION_FLT_DIVIDE_BY_ZERO:
            return "Division by zero.\n"
                   "  Check if a tensor contains zeros before dividing.";
        case EXCEPTION_FLT_INVALID_OPERATION:
            return "Invalid floating point operation (often NaN from 0/0 or log of negative).\n"
                   "  Common in Dream: unstable training, exploding gradients, bad initialization.";
        default:
            return "An unexpected error occurred.";
    }
}

static LONG WINAPI dream_crash_handler(EXCEPTION_POINTERS* info) {
    DWORD code = info->ExceptionRecord->ExceptionCode;
    void* addr = info->ExceptionRecord->ExceptionAddress;

    std::cerr << "\n"
              << "============================================================\n"
              << "  DREAM RUNTIME CRASH\n"
              << "============================================================\n"
              << "  Exception: " << seh_code_to_name(code) << "\n"
              << "  Address:   " << addr << "\n"
              << "  Function:  " << g_current_fn << "\n"
              << "  Operation: " << g_current_op << "\n";
    if (g_current_line > 0)
        std::cerr << "  Line:      " << g_current_line << "\n";
    std::cerr << "\n  Why this happened:\n  " << seh_explanation(code) << "\n\n";

    std::cerr << "  Runtime state:\n";
    std::cerr << "    tensors alive:   " << g_arena.size() << "\n";
    std::cerr << "    allocs since gc: " << g_alloc_count << "\n";

    // Diagnostic: dump arena distribution to help find leaks
    extern Tensor* dream_arena_dump();
    dream_arena_dump();

    if (code == EXCEPTION_ACCESS_VIOLATION) {
        std::cerr << "\n  Try:\n"
                  << "    1. Run with a smaller test case to isolate the crash\n"
                  << "    2. Add `print(x)` before the crash point to verify tensor validity\n"
                  << "    3. For tuples, use `let (a, b) = func()` destructuring\n"
                  << "    4. Verify all tensor arguments were initialized before use\n";
    }
    if (code == 0xc0000409) {
        std::cerr << "\n  Try:\n"
                  << "    1. Check for shape mismatches in recent operations\n"
                  << "    2. If crash is after many iterations, likely memory corruption\n"
                  << "    3. Reduce batch size or model size to isolate\n";
    }
    std::cerr << "============================================================\n" << std::flush;
    return EXCEPTION_EXECUTE_HANDLER;
}

static void install_crash_handler() { SetUnhandledExceptionFilter(dream_crash_handler); }
#else
static void install_crash_handler() {}
#endif

extern "C" Tensor* dream_set_context(const char* fn_name, const char* op_name, int line) {
    g_current_fn = fn_name ? fn_name : "app";
    g_current_op = op_name ? op_name : "unknown";
    g_current_line = line;
    return nullptr;
}

// libdream builds (dream lib) exclude the executable entry point so a C
// program can supply its own main(). Normal Dream builds keep it: main()
//HEre is what calls the generated app().
#ifndef DREAM_NO_MAIN
int main() {
    install_crash_handler();
    Tensor* r = app();
    std::cout << std::flush;
    std::cerr << std::flush;
    return (r && r->size > 0 && r->data[0] != 0.0) ? 1 : 0;
}
#endif  // DREAM_NO_MAIN
