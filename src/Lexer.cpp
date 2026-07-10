#include "Lexer.h"
#include <cctype>
#include <sstream>

Lexer::Lexer(const std::string& src)
    : Source(src), Pos(0), LastChar(' '), CurrentLine(1), CurrentCol(0)
{
    std::istringstream iss(src);
    std::string line;
    while (std::getline(iss, line)) Lines.push_back(line);
}

std::string Lexer::getLine(int lineNum) const {
    if (lineNum > 0 && static_cast<size_t>(lineNum) <= Lines.size())
        return Lines[lineNum - 1];
    return "";
}

int Lexer::advance() {
    if (Pos >= Source.length()) return EOF;
    int c = static_cast<unsigned char>(Source[Pos++]);
    if (c == '\n') { CurrentLine++; CurrentCol = 0; }
    else           { CurrentCol++; }
    return c;
}

static bool skipUnicodeSpace(int lastChar, size_t pos, const std::string& src, int& outBytes) {
    unsigned char c  = static_cast<unsigned char>(lastChar);
    unsigned char n1 = (pos   < src.size()) ? static_cast<unsigned char>(src[pos])   : 0;
    unsigned char n2 = (pos+1 < src.size()) ? static_cast<unsigned char>(src[pos+1]) : 0;
    if (c == 0xC2 && n1 == 0xA0)               { outBytes = 2; return true; }
    if (c == 0xE3 && n1 == 0x80 && n2 == 0x80) { outBytes = 3; return true; }
    return false;
}

TokenData Lexer::getNextToken() {
    // --- Whitespace ---
    while (LastChar != EOF) {
        if (isspace(LastChar)) { LastChar = advance(); continue; }
        int skip = 0;
        if (skipUnicodeSpace(LastChar, Pos, Source, skip)) {
            for (int i = 1; i < skip; ++i) advance();
            LastChar = advance();
            continue;
        }
        break;
    }

    // --- Line comment //  ---
    if (LastChar == '/' && Pos < Source.length() && Source[Pos] == '/') {
        while (LastChar != EOF && LastChar != '\n' && LastChar != '\r') LastChar = advance();
        return getNextToken();
    }

    // --- Block comment /* ... */  ---
    if (LastChar == '/' && Pos < Source.length() && Source[Pos] == '*') {
        advance(); // consume *
        LastChar = advance();
        while (LastChar != EOF) {
            if (LastChar == '*' && Pos < Source.length() && Source[Pos] == '/') {
                advance(); // consume /
                LastChar = advance();
                break;
            }
            LastChar = advance();
        }
        return getNextToken();
    }

    int startLine = CurrentLine;
    int startCol  = (CurrentCol == 0) ? 1 : CurrentCol;

    // --- String literals ---
    if (LastChar == '"') {
        std::string str;
        LastChar = advance();
        while (LastChar != EOF && LastChar != '"') {
            if (LastChar == '\\') {
                LastChar = advance();
                switch (LastChar) {
                    case 'n':  str += '\n'; break;
                    case 't':  str += '\t'; break;
                    case 'r':  str += '\r'; break;
                    case '0':  str += '\0'; break;
                    case '"':  str += '"';  break;
                    case '\\': str += '\\'; break;
                    case 'x': {
                        // \xHH hex escape
                        char hex[3] = {0, 0, 0};
                        LastChar = advance();
                        if (LastChar != EOF && isxdigit(LastChar)) { hex[0] = (char)LastChar; LastChar = advance(); }
                        if (LastChar != EOF && isxdigit(LastChar)) { hex[1] = (char)LastChar; }
                        else { str += static_cast<char>(strtol(hex, nullptr, 16)); continue; }
                        str += static_cast<char>(strtol(hex, nullptr, 16));
                        break;
                    }
                    default:   str += static_cast<char>(LastChar); break;
                }
            } else {
                str += static_cast<char>(LastChar);
            }
            LastChar = advance();
        }
        if (LastChar == '"') LastChar = advance();
        return {(int)Token::tok_string_lit, str, startLine, startCol};
    }

    // Identifiers and keywords
    if (LastChar != EOF && (isalpha(LastChar) || LastChar == '_')) {
        std::string id;
        id += static_cast<char>(LastChar);
        while ((LastChar = advance()) != EOF && (isalnum(LastChar) || LastChar == '_'))
            id += static_cast<char>(LastChar);

        if (id == "fn")       return {(int)Token::tok_fn,       id, startLine, startCol};
        if (id == "let")      return {(int)Token::tok_let,       id, startLine, startCol};
        if (id == "return")   return {(int)Token::tok_return,    id, startLine, startCol};
        if (id == "grad")     return {(int)Token::tok_grad,      id, startLine, startCol};
        if (id == "while")    return {(int)Token::tok_while,     id, startLine, startCol};
        if (id == "for")      return {(int)Token::tok_for,       id, startLine, startCol};
        if (id == "if")       return {(int)Token::tok_if,        id, startLine, startCol};
        if (id == "else")     return {(int)Token::tok_else,      id, startLine, startCol};
        if (id == "true")     return {(int)Token::tok_true,      id, startLine, startCol};
        if (id == "false")    return {(int)Token::tok_false,     id, startLine, startCol};
        if (id == "break")    return {(int)Token::tok_break,     id, startLine, startCol};
        if (id == "continue") return {(int)Token::tok_continue,  id, startLine, startCol};
        if (id == "in")       return {(int)Token::tok_in,        id, startLine, startCol};
        if (id == "assert")   return {(int)Token::tok_assert,    id, startLine, startCol};
        if (id == "struct")   return {(int)Token::tok_struct,    id, startLine, startCol};
        if (id == "Tensor" || id == "Int" || id == "Float" ||
            id == "Char"   || id == "Bool"|| id == "String")
            return {(int)Token::tok_type, id, startLine, startCol};

        return {(int)Token::tok_identifier, id, startLine, startCol};
    }

    // --- char literals ---
    if (LastChar == '\'') {
        LastChar = advance();
        char c = static_cast<char>(LastChar);
        LastChar = advance();
        if (LastChar == '\'') LastChar = advance();
        return {(int)Token::tok_char_lit, std::string(1, c), startLine, startCol};
    }

    //  Numeric literals
    // Only treat leading '.' as numeric if followed by a digit (e.g. .5),
    // NOT when it's part of '..' range operator or a standalone dot.
    if (LastChar != EOF &&
        (isdigit(LastChar) ||
         (LastChar == '.' && Pos < Source.length() && isdigit(static_cast<unsigned char>(Source[Pos]))))) {
        std::string num;
        bool isFloat = false;
        do {
            if (LastChar == '.') {
                // Check for `..` range operator: stop before consuming second dot
                if (Pos < Source.length() && Source[Pos] == '.') break;
                isFloat = true;
            }
            num += static_cast<char>(LastChar);
            LastChar = advance();
        } while (LastChar != EOF && (isdigit(LastChar) || LastChar == '.'));

        // Scientific notation: e.g. 1e-8, 3.14E+2, 2e10
        if (LastChar == 'e' || LastChar == 'E') {
            isFloat = true;
            num += static_cast<char>(LastChar);
            LastChar = advance();
            if (LastChar == '+' || LastChar == '-') {
                num += static_cast<char>(LastChar);
                LastChar = advance();
            }
            while (LastChar != EOF && isdigit(LastChar)) {
                num += static_cast<char>(LastChar);
                LastChar = advance();
            }
        }

        return {isFloat ? (int)Token::tok_float_lit : (int)Token::tok_int_lit,
                num, startLine, startCol};
    }

    if (LastChar == EOF) return {(int)Token::tok_eof, "", startLine, startCol};

    // symbols
    int c = LastChar;
    LastChar = advance();

    if (c == '&' && LastChar == '&') { LastChar = advance(); return {(int)Token::tok_symbol, "&&",  startLine, startCol}; }
    if (c == '|' && LastChar == '|') { LastChar = advance(); return {(int)Token::tok_symbol, "||",  startLine, startCol}; }
    if (c == '=' && LastChar == '=') { LastChar = advance(); return {(int)Token::tok_symbol, "==",  startLine, startCol}; }
    if (c == '!' && LastChar == '=') { LastChar = advance(); return {(int)Token::tok_symbol, "!=",  startLine, startCol}; }
    if (c == '<' && LastChar == '=') { LastChar = advance(); return {(int)Token::tok_symbol, "<=",  startLine, startCol}; }
    if (c == '>' && LastChar == '=') { LastChar = advance(); return {(int)Token::tok_symbol, ">=",  startLine, startCol}; }
    if (c == '+' && LastChar == '=') { LastChar = advance(); return {(int)Token::tok_symbol, "+=",  startLine, startCol}; }
    if (c == '-' && LastChar == '=') { LastChar = advance(); return {(int)Token::tok_symbol, "-=",  startLine, startCol}; }
    if (c == '*' && LastChar == '=') { LastChar = advance(); return {(int)Token::tok_symbol, "*=",  startLine, startCol}; }
    if (c == '/' && LastChar == '=') { LastChar = advance(); return {(int)Token::tok_symbol, "/=",  startLine, startCol}; }
    if (c == '*' && LastChar == '*') { LastChar = advance(); return {(int)Token::tok_symbol, "**",  startLine, startCol}; }
    if (c == '.' && LastChar == '.') { LastChar = advance(); return {(int)Token::tok_symbol, "..",  startLine, startCol}; }
    if (c == '+' && LastChar == '+') { LastChar = advance(); return {(int)Token::tok_symbol, "++",  startLine, startCol}; }
    if (c == '-' && LastChar == '-') { LastChar = advance(); return {(int)Token::tok_symbol, "--",  startLine, startCol}; }
    if (c == '-' && LastChar == '>') { LastChar = advance(); return {(int)Token::tok_symbol, "->",  startLine, startCol}; }

    return {(int)Token::tok_symbol, std::string(1, static_cast<char>(c)), startLine, startCol};
}