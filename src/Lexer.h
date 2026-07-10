#pragma once
#include <string>
#include <vector>

enum class Token {
    tok_eof        = -1,
    tok_fn         = -2,
    tok_let        = -3,
    tok_return     = -4,
    tok_grad       = -5,
    tok_identifier = -6,
    tok_type       = -7,
    tok_symbol     = -8,
    tok_while      = -9,
    tok_if         = -10,
    tok_else       = -11,
    tok_int_lit    = -12,
    tok_float_lit  = -13,
    tok_char_lit   = -14,
    tok_for        = -15,
    tok_true       = -16,
    tok_false      = -17,
    tok_break      = -18,
    tok_continue   = -19,
    tok_string_lit = -20,
    tok_in         = -21,
    tok_assert     = -22,
    tok_struct     = -23,
};

struct TokenData {
    int         type;
    std::string value;
    int         line;
    int         col;
};

class Lexer {
    std::string              Source;
    size_t                   Pos;
    int                      LastChar;
    int                      CurrentLine;
    int                      CurrentCol;
    std::vector<std::string> Lines;

    int advance();

public:
    explicit Lexer(const std::string& src);
    TokenData   getNextToken();
    std::string getLine(int lineNum) const;
};