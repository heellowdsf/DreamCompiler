#include "Lexer.h"
#include "Parser.h"
#include "AST.h"
#include "llvm/IR/Module.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <system_error>
#include <cstdlib>
#include <set>
#include <cctype>
#include <string>
#include <map>
const std::map<std::string,std::string>& builtinRemap();
#include <clocale>

#if defined(_WIN32)
#include <direct.h>
#ifndef NOMINMAX
#define NOMINMAX  // prevent windows.h from defining min/max as macros,
                  // which conflicts with std::min({...}) below
#endif
#include <windows.h>
#include <io.h>
#define MKDIR(d) _mkdir(d)
#define PATH_SEP "\\"
#else
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>
#define MKDIR(d) mkdir(d, 0755)
#define PATH_SEP "/"
#endif

extern void InitializeLLVM();
extern std::unique_ptr<llvm::Module> TheModule;


static std::vector<std::string> g_import_paths;

// =====================================================================
// Install-directory resolver
//
// Dream needs to find runtime/, include/, stdlib/ regardless of where the
// user runs `dream` from. This lets users install Dream once (e.g. into
// PATH) and then use it from any project directory.
//
// Strategy:
//   1. Find the directory containing the dream executable itself (argv[0]).
//   2. The install root is one level up if exe is in bin/, otherwise the
//      same directory.
//   3. All runtime files are looked up relative to the install root.
//   4. Working files (tmp/) go to the user's CURRENT directory, so multiple
//      Dream invocations in different projects don't trample each other.
// =====================================================================
static std::string g_install_root;   // e.g. "C:\Tools\Dream"
static std::string g_runtime_dir;    // install_root + "/runtime"
static std::string g_include_dir;    // install_root + "/include"
static std::string g_src_dir;        // install_root + "/src"
static std::string g_tmp_dir;        // CWD + "/tmp"  (user-local)

static std::string getExePath() {
#if defined(_WIN32)
    // Use wide-char API to handle non-ASCII paths (e.g. Chinese characters).
    // Then convert to UTF-8 — matches our UTF-8 locale set in main().
    wchar_t wbuf[MAX_PATH];
    DWORD n = GetModuleFileNameW(nullptr, wbuf, MAX_PATH);
    if (n > 0 && n < MAX_PATH) {
        // Convert UTF-16 → UTF-8
        int needed = WideCharToMultiByte(CP_UTF8, 0, wbuf, (int)n, nullptr, 0, nullptr, nullptr);
        if (needed > 0) {
            std::string out(needed, '\0');
            WideCharToMultiByte(CP_UTF8, 0, wbuf, (int)n, &out[0], needed, nullptr, nullptr);
            return out;
        }
    }
#else
    char buf[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf));
    if (n > 0) return std::string(buf, n);
#endif
    return "";
}

static std::string parentDir(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return ".";
    return path.substr(0, pos);
}

static std::string baseName(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

static bool fileExists(const std::string& path) {
#if defined(_WIN32)
    // path is UTF-8. Convert to wchar for the file API.
    int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), (int)path.size(), nullptr, 0);
    if (wlen <= 0) return false;
    std::wstring wpath(wlen, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), (int)path.size(), &wpath[0], wlen);
    DWORD attrs = GetFileAttributesW(wpath.c_str());
    return (attrs != INVALID_FILE_ATTRIBUTES) && !(attrs & FILE_ATTRIBUTE_DIRECTORY);
#else
    std::ifstream f(path);
    return f.good();
#endif
}

// User-level cache root (override with DREAM_CACHE_DIR):
//   Windows: %LOCALAPPDATA%\Dream    Others: $XDG_CACHE_HOME/dream or ~/.cache/dream
// The project directory no longer holds tmp/; all intermediates go here.
static std::string dreamUserCacheDir() {
    if (const char* o = getenv("DREAM_CACHE_DIR"); o && o[0]) return o;
#if defined(_WIN32)
    if (const char* l = getenv("LOCALAPPDATA"); l && l[0])
        return std::string(l) + PATH_SEP + "Dream";
#else
    if (const char* x = getenv("XDG_CACHE_HOME"); x && x[0])
        return std::string(x) + PATH_SEP + "dream";
    if (const char* h = getenv("HOME"); h && h[0])
        return std::string(h) + PATH_SEP + ".cache" + PATH_SEP + "dream";
#endif
    return g_install_root + PATH_SEP + "cache";   // fallback: next to install dir
}

static unsigned long long fnvStr(const std::string& s) {
    unsigned long long h = 1469598103934665603ULL;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h;
}

static void resolveInstallDirs(const char* argv0) {
    // 1. Locate the exe
    std::string exe = getExePath();
    if (exe.empty()) exe = argv0 ? argv0 : "dream";
    std::string exeDir = parentDir(exe);

    // 2. If the exe is in a "bin" subdirectory, install root is one level up.
    //    Otherwise (exe sits in the project root), install root is exeDir.
    std::string parentName = baseName(exeDir);
    // Lowercase compare for "bin"
    std::string lower;
    for (char c : parentName) lower += (char)std::tolower((unsigned char)c);
    if (lower == "bin") {
        g_install_root = parentDir(exeDir);
    } else {
        g_install_root = exeDir;
    }

    g_runtime_dir = g_install_root + PATH_SEP + "runtime";
    g_include_dir = g_install_root + PATH_SEP + "include";
    g_src_dir     = g_install_root + PATH_SEP + "src";

    // 3. Intermediates (output.ll / temp exe / gpu_kernel.cl) stay out of the project dir:
    //    they go to the user cache, bucketed by a hash of the script path (the real
    //    g_tmp_dir is set once main() resolves the script path). Exported artifacts
    //    (.c/.h/.drck/models) still land in the cwd -- those are outputs, not junk.
    g_tmp_dir.clear();

    // 4. Sanity check: did we find runtime/?
    std::string sentinel = g_runtime_dir + PATH_SEP + "runtime.cpp";
    if (!fileExists(sentinel)) {
        std::cerr << "WARNING: Dream runtime not found at expected location:\n"
                  << "  Looking for: " << sentinel << "\n"
                  << "  Install root inferred as: " << g_install_root << "\n"
                  << "  Compilation will likely fail.\n"
                  << "  Make sure dream.exe is in <install>/bin/ and runtime/ is at <install>/runtime/\n"
                  << std::flush;
    }
}

static std::string findFile(const std::string& filename) {
    if (fileExists(filename)) return filename;
    
    for (auto& dir : g_import_paths) {
        std::string full = dir + "/" + filename;
        if (fileExists(full)) return full;
    }
    return filename;
}

static std::string resolveImports(const std::string& filename,
                                   std::set<std::string>& visited) {
    std::string resolved = findFile(filename);
    if (visited.count(resolved)) return "";
    visited.insert(resolved);

#if defined(_WIN32)
    // Windows: convert UTF-8 path to wchar so ifstream finds files in
    // directories with non-ASCII names (e.g. Chinese folders).
    int wlen = MultiByteToWideChar(CP_UTF8, 0, resolved.c_str(), (int)resolved.size(), nullptr, 0);
    std::wstring wpath(wlen, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, resolved.c_str(), (int)resolved.size(), &wpath[0], wlen);
    std::ifstream file(wpath.c_str());
#else
    std::ifstream file(resolved);
#endif
    if (!file.is_open()) {
        std::cerr << "\n"
                  << "------------------------------------------------------------\n"
                  << "  Cannot open source file\n"
                  << "------------------------------------------------------------\n"
                  << "  Requested: " << filename << "\n";
        if (resolved != filename)
            std::cerr << "  Resolved:  " << resolved << "\n";
        std::cerr << "\n  Searched in:\n";
        for (auto& d : g_import_paths)
            std::cerr << "    " << d << "/" << filename << "\n";
        std::cerr << "\n  Hints:\n"
                  << "    * Check the file path is correct\n"
                  << "    * For imports, put shared libraries in lib/ (e.g. lib/nn.dream)\n"
                  << "    * Examples go in examples/ and are found automatically\n"
                  << "------------------------------------------------------------\n"
                  << std::flush;
        exit(1);
    }
    
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    std::string result;
    size_t i = 0;
    
    while (i < content.size()) {
        bool isImport = (content.compare(i, 7, "import ") == 0) ||
                        (content.compare(i, 8, "#import ") == 0);
        if (isImport) {
            size_t start = i;
            i += (content[i] == '#' ? 8 : 7);
            while (i < content.size() && std::isspace(content[i])) ++i;
            
            if (i < content.size() && content[i] == '"') {
                ++i;
                std::string inc;
                while (i < content.size() && content[i] != '"') inc += content[i++];
                if (i < content.size()) ++i;
                while (i < content.size() && content[i] != '\n') ++i;
                result += resolveImports(inc, visited);
                continue;
            }
            i = start + 1;
        }
        result += content[i++];
    }
    return result;
}

static const char* interpretExitCode(unsigned code) {
    switch (code) {
        case 0xc0000005: return "ACCESS_VIOLATION - tried to read/write invalid memory";
        case 0xc00000fd: return "STACK_OVERFLOW - recursive call too deep";
        case 0xc0000094: return "INTEGER_DIVIDE_BY_ZERO";
        case 0xc000008e: return "FLOAT_DIVIDE_BY_ZERO";
        case 0xc0000090: return "FLOAT_INVALID - usually NaN from 0/0 or log(negative)";
        case 0xc000001d: return "ILLEGAL_INSTRUCTION - corrupted code";
        case 0xc0000409: return "STACK_BUFFER_OVERRUN - memory corruption";
        case 0xc0000374: return "HEAP_CORRUPTION - double-free or bad pointer";
        case 3:          return "segmentation fault";
        default:         return nullptr;
    }
}

static int runCommand(const std::string& cmd) {
#if defined(_WIN32)
    // Convert UTF-8 → UTF-16 and use _wsystem for non-ASCII paths
    int wlen = MultiByteToWideChar(CP_UTF8, 0, cmd.c_str(), (int)cmd.size(), nullptr, 0);
    std::wstring wcmd(wlen, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, cmd.c_str(), (int)cmd.size(), &wcmd[0], wlen);
    int code = _wsystem(wcmd.c_str());
#else
    int code = std::system(cmd.c_str());
#endif
#if defined(_WIN32)
    if (code != 0) {
        unsigned uc = (unsigned)code;
        const char* meaning = interpretExitCode(uc);
        if (uc > 0xFFFF) {
            std::cerr << "\n"
                      << "------------------------------------------------------------\n"
                      << "  Program crashed\n"
                      << "------------------------------------------------------------\n"
                      << "  Exit code: 0x" << std::hex << uc << std::dec;
            if (meaning) std::cerr << "  (" << meaning << ")";
            std::cerr << "\n";
            if (!meaning) {
                std::cerr << "  Look up: https://learn.microsoft.com/cpp/c-runtime-library/errno-constants\n";
            }
            std::cerr << "  The runtime should have printed detailed info above this message.\n"
                      << "  If not, the crash happened before the crash handler was installed.\n"
                      << "------------------------------------------------------------\n"
                      << std::flush;
        }
        else {
            std::cerr << "note: program exited with code " << code << "\n" << std::flush;
        }
    }
#else
    if (code != 0)
        std::cerr << "note: program exited with code " << code << "\n" << std::flush;
#endif
    return code;
}

// Run a command and capture stdout+stderr while echoing live.
// Returns exit code; full text goes into 'out'.
static int runCommandCapture(const std::string& cmd, std::string& out) {
    out.clear();
#if defined(_WIN32)
    // UTF-8 → UTF-16 for non-ASCII paths in command
    int wlen = MultiByteToWideChar(CP_UTF8, 0, cmd.c_str(), (int)cmd.size(), nullptr, 0);
    std::wstring wcmd(wlen, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, cmd.c_str(), (int)cmd.size(), &wcmd[0], wlen);
    FILE* p = _wpopen(wcmd.c_str(), L"r");
#else
    FILE* p = popen(cmd.c_str(), "r");
#endif
    if (!p) {
        return runCommand(cmd);
    }
    char buf[1024];
    while (fgets(buf, sizeof(buf), p)) {
        std::cerr << buf;
        out += buf;
    }
#if defined(_WIN32)
    int code = _pclose(p);
#else
    int code = pclose(p);
#endif
    return code;
}

// Levenshtein-distance "did you mean" suggester.
// Returns the closest name in 'candidates' to 'target', or "" if none close enough.
static int editDistance(const std::string& a, const std::string& b) {
    size_t m = a.size(), n = b.size();
    std::vector<std::vector<int>> d(m+1, std::vector<int>(n+1));
    for (size_t i = 0; i <= m; ++i) d[i][0] = (int)i;
    for (size_t j = 0; j <= n; ++j) d[0][j] = (int)j;
    for (size_t i = 1; i <= m; ++i) {
        for (size_t j = 1; j <= n; ++j) {
            int cost = (std::tolower((unsigned char)a[i-1]) ==
                        std::tolower((unsigned char)b[j-1])) ? 0 : 1;
            d[i][j] = std::min({ d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost });
        }
    }
    return d[m][n];
}

static std::string findClosestName(const std::string& target,
                                    const std::vector<std::string>& candidates) {
    if (candidates.empty()) return "";
    int best = 1000;
    std::string bestName;
    for (const auto& c : candidates) {
        int d = editDistance(target, c);
        if (d < best) { best = d; bestName = c; }
    }
    int threshold = std::max(2, (int)target.size() / 3);
    return (best <= threshold) ? bestName : "";
}

static void ensureDir(const char* path) { 
    MKDIR(path); 
}

// =====================================================================
//  Runtime precompile cache
//  runtime.cpp (with all its #included .inc/.h) is recompiled to .o only when
//  its content or the compile flags change; afterwards each dream build only
//  processes output.ll and links. Measured: ~17s -> <0.5s per build.
//  Set DREAM_NO_CACHE=1 to disable.
// =====================================================================
#include <filesystem>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

static unsigned long long fnv1a(const char* p, size_t n, unsigned long long h) {
    for (size_t i = 0; i < n; ++i) { h ^= (unsigned char)p[i]; h *= 1099511628211ULL; }
    return h;
}
static unsigned long long hashFile(const std::string& path, unsigned long long h) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return h;
    char buf[65536];
    while (f) { f.read(buf, sizeof(buf)); h = fnv1a(buf, (size_t)f.gcount(), h); }
    return h;
}

// Fingerprint over all runtime source files + compile flags.
static std::string runtimeFingerprint(const std::string& flags) {
    unsigned long long h = 1469598103934665603ULL;
    h = fnv1a(flags.c_str(), flags.size(), h);
    std::vector<std::string> files;
    std::error_code ec;
    // recursive: the runtime is organized into core/ ops/ nn/ io/ backend/
    // subdirectories, so the fingerprint must descend into them.
    for (auto& e : std::filesystem::recursive_directory_iterator(g_runtime_dir, ec)) {
        if (!e.is_regular_file()) continue;
        auto ext = e.path().extension().string();
        if (ext == ".cpp" || ext == ".inc" || ext == ".h" || ext == ".hpp")
            files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());
    for (auto& f : files) { h = fnv1a(f.c_str(), f.size(), h); h = hashFile(f, h); }
    char buf[24];
    snprintf(buf, sizeof(buf), "%016llx", h);
    return buf;
}

// Returns the cached .o path, compiling it once if absent. Empty string on failure (falls back).
static std::string ensureRuntimeObject(const std::string& flags) {
    if (const char* nc = getenv("DREAM_NO_CACHE"); nc && nc[0]=='1') return "";
    std::string cacheDir = dreamUserCacheDir() + PATH_SEP + "obj";
    std::error_code ec;
    std::filesystem::create_directories(cacheDir, ec);
    if (ec) return "";
    std::string obj = cacheDir + PATH_SEP + "runtime-" + runtimeFingerprint(flags) + ".o";
    if (std::filesystem::exists(obj)) return obj;

    std::cout << "[dream] first run with this runtime — precompiling cache ("
              << "one-time, subsequent builds are instant)...\n" << std::flush;
    std::string runtimeCpp = g_runtime_dir + PATH_SEP + "runtime.cpp";
    std::string tmpObj = obj + ".tmp";
    std::string cmd = std::string("clang++ ") + flags + " -c \"" + runtimeCpp + "\" -o \""
        + tmpObj + "\" -Wno-override-module -D_CRT_SECURE_NO_WARNINGS "
        + "-I \"" + g_runtime_dir + "\" -I \"" + g_src_dir + "\" -I \"" + g_include_dir + "\" 2>&1";
    std::string out;
    if (runCommandCapture(cmd, out) != 0) {
        std::cerr << out << "\n[dream] runtime precompile failed, falling back to direct build\n";
        return "";
    }
    std::filesystem::rename(tmpObj, obj, ec);
    if (ec) { std::filesystem::copy_file(tmpObj, obj, std::filesystem::copy_options::overwrite_existing, ec); }
    // Prune stale cache objects (keep the 4 most recent).
    std::vector<std::filesystem::directory_entry> olds;
    for (auto& e : std::filesystem::directory_iterator(cacheDir, ec))
        if (e.path().extension()==".o") olds.push_back(e);
    if (olds.size() > 4) {
        std::sort(olds.begin(), olds.end(), [](auto&a, auto&b){
            std::error_code e1,e2;
            return std::filesystem::last_write_time(a,e1) < std::filesystem::last_write_time(b,e2); });
        for (size_t i = 0; i + 4 < olds.size(); ++i) std::filesystem::remove(olds[i], ec);
    }
    return obj;
}

// Scan the runtime dir for all C-linkable function names (manifest is fingerprint-cached).
// runtime.cpp wraps every .inc in extern "C", so recognizing a line as
// "<return-type> name(" suffices; kept loose (over-collecting is harmless).
static void loadRuntimeManifest() {
    std::string dir = dreamUserCacheDir() + PATH_SEP + "obj";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    std::string symPath = dir + PATH_SEP + "runtime-" + runtimeFingerprint("syms-v1") + ".syms";

    auto loadFrom = [&](const std::string& p)->bool{
        std::ifstream f(p);
        if (!f) return false;
        std::string s;
        while (std::getline(f, s)) if (!s.empty()) g_runtime_syms.insert(s);
        return !g_runtime_syms.empty();
    };
    if (loadFrom(symPath)) return;

    static const char* kTypes[] = {"Tensor*","Tensor *","double","float","int","void",
                                   "bool","char*","char *","const char*","const char *",
                                   "size_t","long","unsigned","uintptr_t"};
    for (auto& e : std::filesystem::recursive_directory_iterator(g_runtime_dir, ec)) {
        auto ext = e.path().extension().string();
        if (ext != ".inc" && ext != ".cpp" && ext != ".h") continue;
        std::ifstream f(e.path());
        std::string line;
        while (std::getline(f, line)) {
            size_t b = line.find_first_not_of(" \t");
            if (b == std::string::npos) continue;
            std::string s = line.substr(b);
            if (s.rfind("extern \"C\"", 0) == 0) {
                size_t q = s.find('"', 8);
                s = (q != std::string::npos) ? s.substr(q+1) : s;
                b = s.find_first_not_of(" \t");
                s = (b == std::string::npos) ? "" : s.substr(b);
            }
            if (s.empty()) continue;
            // Macro-defined ops: UNARY(name,...) / BINOP(name,...) expand to same-named functions,
            // which the type-prefix scan misses. Parse the macro first arg as the name.
            // (Without this, relu/tensor_add would be misreported as unknown functions.)
            if (s.rfind("UNARY(", 0) == 0 || s.rfind("BINOP(", 0) == 0) {
                size_t p = s.find('(') + 1;
                while (p < s.size() && (s[p]==' '||s[p]=='\t')) ++p;
                size_t ns = p;
                while (p < s.size() && (isalnum((unsigned char)s[p]) || s[p]=='_')) ++p;
                if (p > ns) g_runtime_syms.insert(s.substr(ns, p - ns));
                continue;
            }
            if (s.rfind("static",0)==0 || s.rfind("inline",0)==0 || s.rfind("template",0)==0 ||
                s.rfind("//",0)==0 || s.rfind("/*",0)==0 || s.rfind("#",0)==0 ||
                s.rfind("typedef",0)==0 || s.rfind("using",0)==0 || s.rfind("struct",0)==0 ||
                s.rfind("class",0)==0 || s.rfind("return",0)==0) continue;
            const char* matched = nullptr;
            for (const char* t : kTypes)
                if (s.rfind(t, 0) == 0) { matched = t; break; }
            if (!matched) continue;
            size_t pos = strlen(matched);
            while (pos < s.size() && (s[pos]==' '||s[pos]=='\t'||s[pos]=='*')) ++pos;
            size_t nameStart = pos;
            while (pos < s.size() && (isalnum((unsigned char)s[pos]) || s[pos]=='_')) ++pos;
            if (pos == nameStart || pos >= s.size() || s[pos] != '(') continue;
            g_runtime_syms.insert(s.substr(nameStart, pos - nameStart));
        }
    }
    // Include builtin-remap targets too (keeps the trusted symbol set equivalent).
    for (const auto& kv : builtinRemap()) g_runtime_syms.insert(kv.second);

    std::ofstream out(symPath);
    for (const auto& s : g_runtime_syms) out << s << "\n";
}

int main(int argc, char* argv[]) {
#if defined(_WIN32)
    // Tell Windows to use UTF-8 for both console output and the C runtime's
    // multibyte/path handling. Without this, non-ASCII paths in std::system()
    // get encoded as GBK (legacy Windows ANSI on Chinese locales), which
    // clang then misinterprets as UTF-8, producing mojibake.
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    // Keep the console open when dream.exe itself was double-clicked or had a
    // .dream file drag-dropped onto it (fresh console vanishes at exit).
    // From cmd/PowerShell the console has >= 2 attached processes -> no pause.
    std::atexit([]{
        DWORD pids[2];
        if (GetConsoleProcessList(pids, 2) > 1) return;
        if (std::getenv("DREAM_NO_PAUSE")) return;
        if (!_isatty(_fileno(stdin)) || !_isatty(_fileno(stdout))) return;
        std::fputs("\n[dream] finished - press Enter to close...", stdout);
        std::fflush(stdout);
        (void)std::getchar();
    });
    // _MSC_VER builds: tell the C runtime to treat narrow strings as UTF-8.
    // (No-op on MinGW; safe on MSVC.)
    setlocale(LC_ALL, ".UTF-8");

    // On Windows, the default 'argv' from int main is ANSI/GBK.
    // To get UTF-8 args (for e.g. Chinese filenames), pull the wide
    // command line and re-encode each arg as UTF-8.
    int wargc = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    std::vector<std::string> argv_storage;
    std::vector<char*> argv_ptrs;
    if (wargv) {
        argv_storage.reserve(wargc);
        argv_ptrs.reserve(wargc + 1);
        for (int i = 0; i < wargc; ++i) {
            int n = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
            std::string s(n > 0 ? n - 1 : 0, '\0');
            if (n > 0) WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, &s[0], n, nullptr, nullptr);
            argv_storage.push_back(std::move(s));
        }
        for (auto& s : argv_storage) argv_ptrs.push_back(&s[0]);
        argv_ptrs.push_back(nullptr);
        argc = wargc;
        argv = argv_ptrs.data();
        LocalFree(wargv);
    }
#endif
    // Resolve install dirs first - needed even for error messages so users know
    // where Dream is looking for runtime files.
    resolveInstallDirs(argc > 0 ? argv[0] : nullptr);

    // ---------- Subcommand and argument parsing ----------
    //   dream <src.dream>              compile and run (legacy usage)
    //   dream run   <src.dream>        same as above
    //   dream build <src.dream> [-o out]   compile only, produce a standalone exe
    //   common: --emit-ll  also write the generated LLVM IR to <script>.ll in cwd
    bool buildOnly = false, emitLL = false, libMode = false;
    std::string srcArg, outArg;

    // ---- `dream test [dir]` : run every tests/*.dream and match expectations ----
    // Each test declares its expected output inline:
    //     // EXPECT: some substring that must appear in the output
    // (multiple EXPECT lines allowed; all must match). The runner spawns this
    // same binary with `run` per test, so it exercises the full pipeline
    // exactly as a user would. Cross-platform: uses std::system + redirection.
    if (argc >= 2 && std::string(argv[1]) == "test") {
        std::string dir = (argc >= 3) ? argv[2] : "tests";
        namespace fs = std::filesystem;
        if (!fs::exists(dir)) {
            std::cerr << "error: test directory '" << dir << "' not found\n";
            return 1;
        }
        std::vector<std::string> files;
        for (auto& e : fs::directory_iterator(dir))
            if (e.path().extension() == ".dream") files.push_back(e.path().string());
        std::sort(files.begin(), files.end());
        if (files.empty()) {
            std::cerr << "error: no .dream tests in '" << dir << "'\n";
            return 1;
        }
        std::string tmpOut = dreamUserCacheDir() + PATH_SEP + "test_output.txt";
        {
            std::error_code ec;
            std::filesystem::create_directories(dreamUserCacheDir(), ec);
        }
        int passCnt = 0, failCnt = 0;
        std::printf("=== dream test: %d test(s) in %s ===\n", (int)files.size(), dir.c_str());
        for (const auto& f : files) {
            // collect EXPECT lines
            std::vector<std::string> expects;
            {
                std::ifstream in(f);
                std::string line;
                const std::string tag = "// EXPECT:";
                while (std::getline(in, line)) {
                    size_t p = line.find(tag);
                    if (p != std::string::npos) {
                        std::string e = line.substr(p + tag.size());
                        while (!e.empty() && (e.front()==' '||e.front()=='\t')) e.erase(e.begin());
                        while (!e.empty() && (e.back()=='\r'||e.back()==' ')) e.pop_back();
                        if (!e.empty()) expects.push_back(e);
                    }
                }
            }
            std::string cmd = std::string("\"") + argv[0] + "\" run \"" + f + "\" > \""
                            + tmpOut + "\" 2>&1";
            int rc = std::system(cmd.c_str());
            std::string out;
            {
                std::ifstream in(tmpOut);
                std::stringstream ss; ss << in.rdbuf(); out = ss.str();
            }
            std::vector<std::string> missing;
            for (const auto& e : expects)
                if (out.find(e) == std::string::npos) missing.push_back(e);
            bool ok = missing.empty() && !expects.empty();
            std::string base = fs::path(f).filename().string();
            if (ok) { std::printf("  PASS  %s\n", base.c_str()); ++passCnt; }
            else {
                ++failCnt;
                if (expects.empty())
                    std::printf("  FAIL  %s  (no // EXPECT: lines in test)\n", base.c_str());
                else
                    std::printf("  FAIL  %s  (missing: \"%s\"%s)\n", base.c_str(),
                                missing[0].c_str(), missing.size()>1 ? ", ..." : "");
                (void)rc;
            }
        }
        std::printf("=== %d passed, %d failed ===\n", passCnt, failCnt);
        return failCnt == 0 ? 0 : 1;
    }

    // ---- `dream lib [-o out.a]` : build libdream, the C-callable runtime ----
    // Produces a static archive of the whole runtime plus dream.h next to it.
    // Compiled with -ffunction-sections/-fdata-sections so the FINAL link of
    // a C program (--gc-sections / /OPT:REF) strips every op it does not
    // use -- linking libdream must not bloat the C executable.
    if (argc >= 2 && std::string(argv[1]) == "lib") {
        std::string out = "libdream.a";
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "-o" && i + 1 < argc) out = argv[++i];
        }
        std::string cacheDir = dreamUserCacheDir();
        { std::error_code ec; std::filesystem::create_directories(cacheDir, ec); }
        std::string obj = cacheDir + PATH_SEP + "libdream_runtime.o";
        std::string runtimeCpp = g_runtime_dir + PATH_SEP + "runtime.cpp";
#if defined(_WIN32)
        // Windows clang links its own libomp automatically via -fopenmp.
        const char* ompFlag = "-fopenmp ";
#else
        // Emit GNU OpenMP ABI so a plain `gcc ... -lgomp` link works --
        // clang's default libomp symbols (__kmpc_*) are not in libgomp.
        const char* ompFlag = "-fopenmp=libgomp ";
#endif
        std::string cc =
            std::string("clang++ -c -O2 -ffast-math -mavx2 -mfma -fno-math-errno -DNDEBUG -DDREAM_NO_MAIN ")
            + ompFlag
            + "-ffunction-sections -fdata-sections -D_CRT_SECURE_NO_WARNINGS "
            + "-Wno-unused-command-line-argument "
            + "\"" + runtimeCpp + "\" -o \"" + obj + "\" "
            + "-I \"" + g_runtime_dir + "\" -I \"" + g_src_dir + "\" -I \"" + g_include_dir + "\"";
        std::printf("[dream] compiling runtime for libdream...\n");
        if (std::system(cc.c_str()) != 0 || !std::filesystem::exists(obj)) {
            std::cerr << "error: runtime compilation for libdream failed\n";
            return 1;
        }
        std::string ar = "llvm-ar rcs \"" + out + "\" \"" + obj + "\" 2>/dev/null";
        if (std::system(ar.c_str()) != 0) {
            ar = "ar rcs \"" + out + "\" \"" + obj + "\"";
            if (std::system(ar.c_str()) != 0) {
                std::cerr << "error: neither llvm-ar nor ar could create the archive\n";
                return 1;
            }
        }
        // Ship dream.h next to the archive.
        std::string hdrSrc = g_install_root + PATH_SEP + "interop" + PATH_SEP + "dream.h";
        std::string outDir = std::filesystem::path(out).has_parent_path()
                           ? std::filesystem::path(out).parent_path().string() : ".";
        std::error_code ec;
        std::filesystem::copy_file(hdrSrc, outDir + PATH_SEP + "dream.h",
                                   std::filesystem::copy_options::overwrite_existing, ec);
        std::printf("[dream] library: %s (+ dream.h)\n", out.c_str());
        std::printf("[dream] link a C program:\n");
        std::printf("  Linux:   cc app.c %s -o app -lstdc++ -lm -ldl -lgomp -Wl,--gc-sections\n", out.c_str());
        std::printf("  Windows: clang app.c %s -o app.exe -fopenmp -Xlinker /OPT:REF\n", out.c_str());
        return 0;
    }

    {
        int i = 1;
        if (i < argc && (std::string(argv[i]) == "run" || std::string(argv[i]) == "build")) {
            buildOnly = (std::string(argv[i]) == "build");
            ++i;
        }
        for (; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "-o" && i + 1 < argc) { outArg = argv[++i]; }
            else if (a == "--emit-ll") { emitLL = true; }
            else if (a == "--lib") { libMode = true; buildOnly = true; }
            else if (!a.empty() && a[0] == '-') {
                std::cerr << "unknown option: " << a << "\n"; return 1;
            }
            else if (srcArg.empty()) { srcArg = a; }
        }
    }

    if (srcArg.empty()) {
        std::cerr << "Dream Compiler\n"
                  << "\n"
                  << "Usage:\n"
                  << "  dream <source.dream>              compile & run\n"
                  << "  dream run <source.dream>          compile & run\n"
                  << "  dream build <source.dream> [-o a.exe]   compile only\n"
                  << "  options: --emit-ll   dump LLVM IR next to your script\n"
                  << "\n"
                  << "Install root: " << g_install_root << "\n"
                  << "  runtime:    " << g_runtime_dir << "\n"
                  << "  cache:      " << dreamUserCacheDir() << "\n"
                  << std::flush;
        return 1;
    }

    // Work dir: <user-cache>/work/<hash of script absolute path>
    {
        std::error_code ec;
        std::string absSrc = std::filesystem::absolute(srcArg, ec).string();
        if (ec) absSrc = srcArg;
        char hb[24]; snprintf(hb, sizeof(hb), "%016llx", fnvStr(absSrc));
        g_tmp_dir = dreamUserCacheDir() + PATH_SEP + "work" + PATH_SEP + hb;
        std::filesystem::create_directories(g_tmp_dir, ec);
        if (ec) {           // if the cache is unwritable, fall back to old behavior
            g_tmp_dir = "tmp";
            ensureDir(g_tmp_dir.c_str());
        }
    }

    // Search paths for `import` statements.
    // Order: current dir first, then user's lib/, then install stdlib/, then install examples/.
    g_import_paths.push_back(".");
    g_import_paths.push_back("lib");
    g_import_paths.push_back("examples");
    g_import_paths.push_back(g_install_root + PATH_SEP + "stdlib");
    g_import_paths.push_back(g_install_root + PATH_SEP + "examples");
    
    std::string srcFile = srcArg;
    size_t lastSlash = srcFile.find_last_of("/\\");
    if (lastSlash != std::string::npos) {
        g_import_paths.push_back(srcFile.substr(0, lastSlash));
    }

    std::set<std::string> visited;
    std::string source = resolveImports(srcFile, visited);
    
    InitializeLLVM();
    Lexer  lex(source);
    Parser parser(lex);
    auto   fns = parser.parseAllFunctions();

    // Diagnostic context + user-function pre-scan (for forward-call checks) + runtime symbol manifest
    g_diag_lexer = &lex;
    g_diag_file  = srcFile;
    for (auto& fn : fns) if (fn) g_user_fns.insert(fn->getName());
    loadRuntimeManifest();

    // Syntax errors were already reported during parsing (with location). The AST
    // is incomplete here; continuing to codegen would crash, so exit cleanly.
    if (parser.hadError()) {
        std::cerr << "\n------------------------------------------------------------\n"
                  << "  Compilation stopped due to a syntax error\n"
                  << "------------------------------------------------------------\n"
                  << std::flush;
        return 1;
    }


    if (fns.empty()) {
        std::cerr << "\n"
                  << "------------------------------------------------------------\n"
                  << "  No functions found in source file\n"
                  << "------------------------------------------------------------\n"
                  << "  Dream requires at least a `fn app()` function.\n"
                  << "  Example:\n"
                  << "      fn app() {\n"
                  << "          println(\"Hello, Dream!\");\n"
                  << "          return 0.0;\n"
                  << "      }\n"
                  << "------------------------------------------------------------\n"
                  << std::flush;
        return 1;
    }
    
    std::string gpuCode;
    bool hasApp = false;
    bool ok     = true;

    // Collect all defined function names (user-defined). We use these later
    // to suggest "did you mean X?" when the linker reports an undefined symbol.
    std::vector<std::string> userFunctionNames;

    for (auto& fn : fns) {
        if (!fn) continue;
        if (fn->getName() == "app") hasApp = true;
        userFunctionNames.push_back(fn->getName());
        if (!fn->codegen()) { ok = false; break; }
        gpuCode += fn->genOpenCL() + "\n";
    }

    if (!ok) {
        std::cerr << "\n"
                  << "------------------------------------------------------------\n"
                  << "  Compilation failed\n"
                  << "------------------------------------------------------------\n"
                  << "  See the error messages above for details.\n"
                  << "  Common fixes:\n"
                  << "    * Undefined variable: declare with `let` before use\n"
                  << "    * Wrong argument type: check function signatures\n"
                  << "    * Missing `return` in a function expected to return a value\n"
                  << "    * For tuples: use `let (a, b) = func()` not `let a = func()[0]`\n"
                  << "------------------------------------------------------------\n"
                  << std::flush;
        return 1;
    }

    if (!hasApp && !libMode) {   // a library has no entry point requirement
        std::cerr << "\n"
                  << "------------------------------------------------------------\n"
                  << "  Missing entry point: no `fn app()` found\n"
                  << "------------------------------------------------------------\n"
                  << "  Every Dream program needs a function named 'app' as its entry point.\n"
                  << "  Add this to your file:\n"
                  << "      fn app() {\n"
                  << "          // your code here\n"
                  << "          return 0.0;\n"
                  << "      }\n"
                  << "------------------------------------------------------------\n"
                  << std::flush;
        return 1;
    }

    std::string llPath = g_tmp_dir + PATH_SEP + "output.ll";
    std::string clPath = g_tmp_dir + PATH_SEP + "gpu_kernel.cl";
    // Script stem (no dir, no extension); used for build default output and --emit-ll.
    std::string stem = srcFile;
    if (size_t sl = stem.find_last_of("/\\"); sl != std::string::npos) stem = stem.substr(sl+1);
    if (size_t dot = stem.rfind(".dream"); dot != std::string::npos) stem = stem.substr(0, dot);
    std::string exePath;
    if (buildOnly) {
        exePath = !outArg.empty() ? outArg : (stem +
#if defined(_WIN32)
            std::string(".exe")
#else
            std::string("")
#endif
        );
    } else {
        exePath = g_tmp_dir + PATH_SEP + "__dream_tmp.exe";
    }

    {
        std::error_code ec;
        llvm::raw_fd_ostream dest(llPath, ec);
        if (ec) {
            std::cerr << "error: cannot write " << llPath << ": " << ec.message() << "\n" << std::flush;
            return 1;
        }
        TheModule->print(dest, nullptr);
    }
    if (emitLL) {
        std::error_code ec;
        std::filesystem::copy_file(llPath, stem + ".ll",
            std::filesystem::copy_options::overwrite_existing, ec);
        if (!ec) std::cout << "[dream] IR written to " << stem << ".ll\n";
    }

    {
#if defined(_WIN32)
        int wlen = MultiByteToWideChar(CP_UTF8, 0, clPath.c_str(), (int)clPath.size(), nullptr, 0);
        std::wstring wclPath(wlen, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, clPath.c_str(), (int)clPath.size(), &wclPath[0], wlen);
        std::ofstream gpuFile(wclPath.c_str());
#else
        std::ofstream gpuFile(clPath);
#endif
        gpuFile << gpuCode;
    }

    // Build link command using absolute install paths so Dream works from any CWD.
    // The runtime.cpp lives in the install dir; tmp output goes in the user's CWD.
    //
    // Quoting: paths may contain spaces (e.g. "C:\Program Files\Dream"),
    // so we wrap each path in double quotes.
    // Release the LLVM module: output.ll is on disk, the parent no longer needs it.
    // On low-memory hosts this avoids pushing the clang child into page thrashing
    // (measured: parent residency degrades the child build from 2s to 17s in a 4GB box).
    TheModule.reset();

    // ---- `dream build --lib f.dream [-o f.o]` ----
    // Compile the Dream module to a plain object file plus an auto-generated
    // C header declaring every user function (they are all Tensor* -> Tensor*
    // in the IR, exactly matching the dream.h convention). A C program then
    // links: f.o + libdream.a. Dream stays fully independent; this is an
    // additional export surface, not a dependency.
    if (libMode) {
        std::string objPath = !outArg.empty() ? outArg : (stem + ".o");
        std::string hdrPath = objPath;
        if (size_t d = hdrPath.rfind('.'); d != std::string::npos) hdrPath = hdrPath.substr(0, d);
        hdrPath += ".h";
        std::string cc = std::string("clang -c -O2 -Wno-override-module \"")
                       + llPath + "\" -o \"" + objPath + "\" 2>&1";
        if (std::system(cc.c_str()) != 0 || !std::filesystem::exists(objPath)) {
            std::cerr << "error: object compilation failed\n";
            return 1;
        }
        std::ofstream h(hdrPath);
        h << "/* Generated by `dream build --lib` from " << stem << ".dream\n"
          << " * Every Dream function takes and returns Tensor* (see dream.h for\n"
          << " * the tensor API and ownership rules: returns are +1, release with\n"
          << " * dream_release). Link: cc app.c " << objPath << " libdream.a ... */\n"
          << "#ifndef DREAM_LIB_" << stem << "_H\n#define DREAM_LIB_" << stem << "_H\n\n"
          << "typedef struct Tensor Tensor;\n\n"
          << "#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n";
        for (auto& fn : fns) {
            if (!fn) continue;
            h << "Tensor* " << fn->getName() << "(";
            const auto& as = fn->getArgs();
            if (as.empty()) h << "void";
            else for (size_t ai = 0; ai < as.size(); ++ai)
                h << (ai ? ", " : "") << "Tensor* " << as[ai].first;
            h << ");\n";
        }
        h << "\n#ifdef __cplusplus\n}\n#endif\n#endif\n";
        h.close();
        std::printf("[dream] library object: %s\n", objPath.c_str());
        std::printf("[dream] header:         %s\n", hdrPath.c_str());
        std::printf("[dream] link a C program:\n");
        std::printf("  cc app.c %s libdream.a -o app -lstdc++ -lm -ldl -lgomp -Wl,--gc-sections\n",
                    objPath.c_str());
        return 0;
    }

    std::string linkCmd;
    if (buildOnly) {
        // build: produce a standalone exe for distribution; optimize for size.
        //  - compile together with runtime source (not the prebuilt .o) so the linker
        //    sees the full call graph and can strip unused ops (helloworld 673KB -> ~48KB)
        //  - -ffunction-sections + --gc-sections for dead-code elimination
        //  - -Os for size, -s to strip the symbol table
        std::string runtimeCpp = g_runtime_dir + PATH_SEP + "runtime.cpp";
#if defined(_WIN32)
        // COFF/PE linking (lld-link / MSVC link.exe): the GNU flags
        // --gc-sections and -s are SILENTLY IGNORED here -- that is exactly
        // why Windows builds came out ~300KB while Linux was ~48KB.
        // COFF equivalents:
        //   /OPT:REF  drop unreferenced COMDAT sections (pairs with
        //             -ffunction-sections/-fdata-sections below)
        //   /OPT:ICF  fold identical sections
        const char* sizeFlags =
            "-ffunction-sections -fdata-sections "
            "-Xlinker /OPT:REF -Xlinker /OPT:ICF ";
#else
        const char* sizeFlags =
            "-ffunction-sections -fdata-sections -Wl,--gc-sections -Wl,-s ";
#endif
        linkCmd =
            std::string("clang++ -Os -ffast-math -mavx2 -mfma -fno-math-errno -DNDEBUG -fopenmp ")
            + sizeFlags
            + "\"" + llPath + "\" \"" + runtimeCpp + "\" -o \"" + exePath + "\" "
            + "-Wno-override-module -Wno-unused-command-line-argument -D_CRT_SECURE_NO_WARNINGS "
            + "-I \"" + g_runtime_dir + "\" -I \"" + g_src_dir + "\" -I \"" + g_include_dir + "\" 2>&1";
    } else {
        // run: dev iteration; optimize for speed. Use the prebuilt runtime cache + -O3.
        const std::string kFlags = "-O3 -ffast-math -mavx2 -mfma -fno-math-errno -DNDEBUG -fopenmp";
        std::string runtimeObj = ensureRuntimeObject(kFlags);
        std::string runtimeInput = runtimeObj.empty()
            ? std::string("\"") + g_runtime_dir + PATH_SEP + "runtime.cpp" + "\""
            : std::string("\"") + runtimeObj + "\"";
        linkCmd =
            std::string("clang++ ") + kFlags + " "
            + "\"" + llPath + "\" " + runtimeInput + " -o \"" + exePath + "\" "
            + "-Wno-override-module -Wno-unused-command-line-argument -D_CRT_SECURE_NO_WARNINGS "
            + "-I \"" + g_runtime_dir + "\" -I \"" + g_src_dir + "\" -I \"" + g_include_dir + "\" 2>&1";
    }
        
    std::string linkOutput;
    if (runCommandCapture(linkCmd, linkOutput) != 0) {
        std::cerr << "\n"
                  << "------------------------------------------------------------\n"
                  << "  Compilation failed: undefined function or runtime issue\n"
                  << "------------------------------------------------------------\n";

        // Parse linker output to find undefined symbols.
        // Forms seen across linkers:
        //   lld-link: error: undefined symbol: train_ai
        //   undefined reference to `train_ai'
        //   error LNK2019: unresolved external symbol "train_ai"
        std::set<std::string> undefSymbols;
        size_t pos = 0;
        const std::vector<std::string> markers = {
            "undefined symbol: ",
            "undefined reference to `",
            "unresolved external symbol \"",
            "unresolved external symbol "
        };
        for (const auto& marker : markers) {
            pos = 0;
            while ((pos = linkOutput.find(marker, pos)) != std::string::npos) {
                pos += marker.size();
                size_t end = pos;
                while (end < linkOutput.size() &&
                       linkOutput[end] != '\n' && linkOutput[end] != '\r' &&
                       linkOutput[end] != '\'' && linkOutput[end] != '"' &&
                       linkOutput[end] != ' ' && linkOutput[end] != '(' &&
                       linkOutput[end] != ')') {
                    ++end;
                }
                std::string sym = linkOutput.substr(pos, end - pos);
                // Strip trailing punctuation
                while (!sym.empty() && (sym.back() == ',' || sym.back() == ';'))
                    sym.pop_back();
                if (!sym.empty()) undefSymbols.insert(sym);
            }
        }

        // Build the candidate list: user functions + Dream-side names from builtinRemap.
        // The keys in builtinRemap are the names users write in Dream (e.g. "linear"),
        // mapped to the C++ runtime symbol (e.g. "dream_linear").
        extern const std::map<std::string,std::string>& builtinRemap();
        std::vector<std::string> allKnown = userFunctionNames;
        for (auto& kv : builtinRemap()) allKnown.push_back(kv.first);

        if (!undefSymbols.empty()) {
            std::cerr << "\n  Your program uses these names that the runtime can't find:\n\n";
            for (const auto& sym : undefSymbols) {
                std::cerr << "    " << sym;
                std::string hint = findClosestName(sym, allKnown);
                if (!hint.empty() && hint != sym) {
                    std::cerr << "    -> did you mean '" << hint << "' ?";
                }
                std::cerr << "\n";
            }
            std::cerr << "\n  What this means:\n"
                      << "    Dream looked for '" << *undefSymbols.begin() << "' as either:\n"
                      << "      1. A function YOU defined (`fn " << *undefSymbols.begin() << "(...)` in your file)\n"
                      << "      2. A built-in operation (like 'relu', 'linear', 'matmul', etc.)\n"
                      << "    But neither was found.\n"
                      << "\n  How to fix:\n"
                      << "    * Check spelling. Names are case-sensitive.\n"
                      << "    * If you meant your own function, make sure it's defined in this file.\n"
                      << "    * If it should be a built-in, run `dream` with no args to see usage info.\n";
        } else {
            std::cerr << "\n  The C++ runtime failed to compile. This is usually a Dream\n"
                      << "  compiler bug, not your code's fault. The clang++ errors above\n"
                      << "  show what went wrong.\n"
                      << "\n  Common causes:\n"
                      << "    * Missing runtime function (an op not yet implemented)\n"
                      << "    * Include order issue in runtime.cpp\n"
                      << "    * clang++ not in PATH or version too old\n";
        }

        std::cerr << "\n  Paths used:\n"
                  << "    runtime: " << g_runtime_dir << "\n"
                  << "    output:  " << llPath << "\n"
                  << "------------------------------------------------------------\n"
                  << std::flush;
        return 1;
    }

if (buildOnly) {
        std::cout << "[dream] built: " << exePath << "\n" << std::flush;
        return 0;
    }

    std::string runCmd = "\"" + exePath + "\"";
    int exitCode = runCommand(runCmd);
    // Crash messages are printed by runCommand / the runtime's SEH handler.
    // No need to add more noise here.
    return exitCode;
}