#include "Parser.h"
#include <iostream>

namespace KernelGen {
static std::string buildFwd(const std::string& name, const std::vector<std::string>& vars,
                             const std::string& body) {
    std::string s="__kernel void k(__global float* out";
    for(size_t i=0;i<vars.size();++i) s+=", __global const float* v"+std::to_string(i);
    s+=", int sz";
    for(size_t i=0;i<vars.size();++i) s+=", int sz"+std::to_string(i);
    return s+") { int i=get_global_id(0); if(i<sz){out[i]="+body+";} }";
}
static std::string buildBwd(const std::string& name, const std::vector<std::string>& vars,
                             ASTNode* expr) {
    std::string s="__kernel void k(__global const float* grad_out";
    for(size_t i=0;i<vars.size();++i) s+=", __global float* g"+std::to_string(i);
    for(size_t i=0;i<vars.size();++i) s+=", __global const float* v"+std::to_string(i);
    s+=", int sz";
    for(size_t i=0;i<vars.size();++i) s+=", int sz"+std::to_string(i);
    s+=") { int i=get_global_id(0); if(i<sz){ float go=grad_out[i]; ";
    for(size_t i=0;i<vars.size();++i)
        s+="g"+std::to_string(i)+"[i%sz"+std::to_string(i)+"]+=go*("+expr->genFusedGradCode(vars[i],vars)+"); ";
    return s+"} }";
}
}
std::map<std::string, std::vector<std::string>> Parser::StructDefs;

Parser::Parser(Lexer& lexer) : lex(lexer) {
    BinopPrecedence["||"]  = 4;
    BinopPrecedence["&&"]  = 5;
    BinopPrecedence["<"]   = 10;
    BinopPrecedence[">"]   = 10;
    BinopPrecedence["<="]  = 10;
    BinopPrecedence[">="]  = 10;
    BinopPrecedence["=="]  = 10;
    BinopPrecedence["!="]  = 10;
    BinopPrecedence["+"]   = 20;
    BinopPrecedence["-"]   = 20;
    BinopPrecedence["*"]   = 40;
    BinopPrecedence["/"]   = 40;
    BinopPrecedence["%"]   = 40;
    BinopPrecedence["**"]  = 50;
    BinopPrecedence["@"]   = 60;
    getNextToken();
}

TokenData Parser::getNextToken() { return CurTok = lex.getNextToken(); }

int Parser::getTokPrecedence() const {
    if (CurTok.type != (int)Token::tok_symbol) return -1;
    auto it = BinopPrecedence.find(CurTok.value);
    return (it != BinopPrecedence.end()) ? it->second : -1;
}

std::unique_ptr<ASTNode> Parser::logError(const std::string& msg) {
    HadError = true;
    std::cerr << "\nerror: " << msg << "\n";
    std::cerr << "  --> line " << CurTok.line << ":" << CurTok.col << "\n";
    std::string line = lex.getLine(CurTok.line);
    if (!line.empty()) {
        std::cerr << "   |\n" << CurTok.line << "  | " << line << "\n   | ";
        for (int i = 1; i < CurTok.col; ++i) std::cerr << ' ';
        std::cerr << "^--- " << msg << "\n\n";
    }
    return nullptr;
}

std::vector<std::unique_ptr<ASTNode>> Parser::parseArgList() {
    std::vector<std::unique_ptr<ASTNode>> args;
    if (CurTok.type == (int)Token::tok_symbol && CurTok.value == ")") {
        getNextToken(); return args;
    }
    while (true) {
        if (auto arg = parseExpression()) args.push_back(std::move(arg));
        else return {};
        if (CurTok.type == (int)Token::tok_symbol && CurTok.value == ")") break;
        if (CurTok.type != (int)Token::tok_symbol || CurTok.value != ",") {
            logError("expected ',' or ')' in argument list"); return {};
        }
        getNextToken();
    }
    getNextToken();
    return args;
}

std::unique_ptr<ASTNode> Parser::parsePrintStmt(bool newline) {
    getNextToken();
    if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "(")
        return logError("expected '(' after print");
    getNextToken();
    auto args = parseArgList();
    if (CurTok.type == (int)Token::tok_symbol && CurTok.value == ";") getNextToken();
    return std::make_unique<PrintStmtAST>(std::move(args), newline);
}

std::unique_ptr<ASTNode> Parser::parseIntLit() {
    // stoi throws out_of_range and aborts on integers beyond int range. Guard with
    // stoll; integer literals beyond int32 become floats (Dream numbers are double
    // internally, exact up to 2^53). Also defends against invalid numeric text.
    std::unique_ptr<ASTNode> n;
    try {
        long long v = std::stoll(CurTok.value);
        if (v >= INT32_MIN && v <= INT32_MAX)
            n = std::make_unique<IntExprAST>((int)v);
        else
            n = std::make_unique<FloatExprAST>((double)v);
    } catch (...) {
        logError("invalid integer literal '" + CurTok.value + "'");
        return nullptr;
    }
    getNextToken(); return n;
}
std::unique_ptr<ASTNode> Parser::parseFloatLit() {
    auto n = std::make_unique<FloatExprAST>(std::stod(CurTok.value));
    getNextToken(); return n;
}
std::unique_ptr<ASTNode> Parser::parseCharLit() {
    auto n = std::make_unique<CharExprAST>(CurTok.value[0]);
    getNextToken(); return n;
}
std::unique_ptr<ASTNode> Parser::parseParenExpr() {
    getNextToken(); // consume '('
    auto first = parseExpression(); if (!first) return nullptr;

    // If comma follows, it's a tuple: (a, b, ...)
    if (CurTok.type == (int)Token::tok_symbol && CurTok.value == ",") {
        std::vector<std::unique_ptr<ASTNode>> elems;
        elems.push_back(std::move(first));
        while (CurTok.type == (int)Token::tok_symbol && CurTok.value == ",") {
            getNextToken();
            auto e = parseExpression(); if (!e) return nullptr;
            elems.push_back(std::move(e));
        }
        if (CurTok.type != (int)Token::tok_symbol || CurTok.value != ")")
            return logError("expected ')' in tuple");
        getNextToken();
        return std::make_unique<TupleLitExprAST>(std::move(elems));
    }

    if (CurTok.type != (int)Token::tok_symbol || CurTok.value != ")")
        return logError("expected ')'");
    getNextToken();
    return first;
}

std::unique_ptr<ASTNode> Parser::parseLoadExpr() {
    getNextToken(); // consume "load"
    if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "(")
        return logError("expected (filename) after load");
    getNextToken(); // consume '('
    if (CurTok.type != (int)Token::tok_string_lit)
        return logError("load requires a string literal filename");
    std::string filename = CurTok.value;
    getNextToken();
    if (CurTok.type != (int)Token::tok_symbol || CurTok.value != ")")
        return logError("expected ) after load filename");
    getNextToken();
    return std::make_unique<LoadExprAST>(filename);
}

std::unique_ptr<ASTNode> Parser::parseIdentifierExpr() {
    std::string name = CurTok.value;
    int callLine = CurTok.line;   // record the callee line for unknown-function diagnostics
    getNextToken();
    // load("file") is handled specially so the filename stays as a C string
    if (name == "load") {
        if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "(")
            return logError("expected (filename) after load");
        getNextToken();
        if (CurTok.type != (int)Token::tok_string_lit)
            return logError("load requires a string literal filename");
        std::string filename = CurTok.value;
        getNextToken();
        if (CurTok.type != (int)Token::tok_symbol || CurTok.value != ")")
            return logError("expected ) after load filename");
        getNextToken();
        return std::make_unique<LoadExprAST>(filename);
    }
    if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "(") {
        auto v = std::make_unique<VariableExprAST>(name);
        v->Line = callLine;
        return v;
    }
    getNextToken(); // consume '('
    auto args = parseArgList();
    auto call = std::make_unique<CallExprAST>(name, std::move(args));
    call->Line = callLine;
    return call;
}

std::unique_ptr<ASTNode> Parser::parseArrayLit() {
    getNextToken(); // consume '['
    std::vector<std::unique_ptr<ASTNode>> elems;
    if (CurTok.type == (int)Token::tok_symbol && CurTok.value == "]") {
        getNextToken();
        return std::make_unique<ArrayLitExprAST>(std::move(elems));
    }
    while (true) {
        if (auto e = parseExpression()) elems.push_back(std::move(e));
        else return nullptr;
        if (CurTok.type == (int)Token::tok_symbol && CurTok.value == "]") break;
        if (CurTok.type != (int)Token::tok_symbol || CurTok.value != ",")
            return logError("expected ',' or ']' in array literal");
        getNextToken();
    }
    getNextToken(); // consume ']'
    return std::make_unique<ArrayLitExprAST>(std::move(elems));
}

std::unique_ptr<ASTNode> Parser::parsePostfix(std::unique_ptr<ASTNode> base) {
    while (true) {
        // .field access
        if (CurTok.type == (int)Token::tok_symbol && CurTok.value == ".") {
            getNextToken();
            if (CurTok.type == (int)Token::tok_symbol && CurTok.value == ".") {
                // It's '..' range operator, put the first '.' back by breaking
                break;
            }
            if (CurTok.type != (int)Token::tok_identifier)
                return logError("expected field name after '.'");
            std::string field = CurTok.value;
            getNextToken();
            // check for .field = value (handled in parseBlock, not here)
            base = std::make_unique<FieldAccessExprAST>(std::move(base), field);
            continue;
        }

        // [index] or [slice]
        if (!(CurTok.type == (int)Token::tok_symbol && CurTok.value == "["))
            break;
        getNextToken(); // consume '['

        // [:end] slice from beginning
        if (CurTok.type == (int)Token::tok_symbol && CurTok.value == ":") {
            getNextToken();
            std::unique_ptr<ASTNode> endExpr;
            if (!(CurTok.type == (int)Token::tok_symbol && CurTok.value == "]"))
                endExpr = parseExpression();
            if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "]")
                return logError("expected ']'");
            getNextToken();
            base = std::make_unique<SliceExprAST>(std::move(base),
                std::make_unique<IntExprAST>(0), std::move(endExpr));
            continue;
        }

        auto first = parseExpression();
        if (!first) return nullptr;

        // [start:end] slice
        if (CurTok.type == (int)Token::tok_symbol && CurTok.value == ":") {
            getNextToken();
            std::unique_ptr<ASTNode> endExpr;
            if (!(CurTok.type == (int)Token::tok_symbol && CurTok.value == "]"))
                endExpr = parseExpression();
            if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "]")
                return logError("expected ']'");
            getNextToken();
            base = std::make_unique<SliceExprAST>(std::move(base),
                std::move(first), std::move(endExpr));
            continue;
        }

        // [i, j] or [i, j, k]
        if (CurTok.type == (int)Token::tok_symbol && CurTok.value == ",") {
            getNextToken();
            auto idx2 = parseExpression();
            if (!idx2) return nullptr;
            if (CurTok.type == (int)Token::tok_symbol && CurTok.value == ",") {
                getNextToken();
                auto idx3 = parseExpression();
                if (!idx3) return nullptr;
                if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "]")
                    return logError("expected ']'");
                getNextToken();
                std::vector<std::unique_ptr<ASTNode>> args;
                args.push_back(std::move(base));
                args.push_back(std::move(first));
                args.push_back(std::move(idx2));
                base = std::make_unique<CallExprAST>("tensor_index_2d", std::move(args));
                base = std::make_unique<IndexExprAST>(std::move(base), std::move(idx3));
                continue;
            }
            if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "]")
                return logError("expected ']'");
            getNextToken();
            std::vector<std::unique_ptr<ASTNode>> args;
            args.push_back(std::move(base));
            args.push_back(std::move(first));
            args.push_back(std::move(idx2));
            base = std::make_unique<CallExprAST>("tensor_index_2d", std::move(args));
            continue;
        }

        // [i] simple index
        if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "]")
            return logError("expected ']'");
        getNextToken();
        base = std::make_unique<IndexExprAST>(std::move(base), std::move(first));
    }
    return base;
}

std::unique_ptr<ASTNode> Parser::parsePrimary() {
    std::unique_ptr<ASTNode> result;

    if (CurTok.type == (int)Token::tok_int_lit)         result = parseIntLit();
    else if (CurTok.type == (int)Token::tok_float_lit)  result = parseFloatLit();
    else if (CurTok.type == (int)Token::tok_char_lit)   result = parseCharLit();
    else if (CurTok.type == (int)Token::tok_identifier) result = parseIdentifierExpr();
    else if (CurTok.type == (int)Token::tok_type) {
        std::string typeName = CurTok.value;
        getNextToken();
        if (CurTok.type == (int)Token::tok_symbol && CurTok.value == "(") {
            getNextToken();
            auto args = parseArgList();
            result = std::make_unique<CallExprAST>(typeName, std::move(args));
        } else {
            result = std::make_unique<VariableExprAST>(typeName);
        }
    }
    else if (CurTok.type == (int)Token::tok_string_lit) {
        result = std::make_unique<StringExprAST>(CurTok.value);
        getNextToken();
    }
    else if (CurTok.type == (int)Token::tok_true)  { getNextToken(); result = std::make_unique<FloatExprAST>(1.0); }
    else if (CurTok.type == (int)Token::tok_false) { getNextToken(); result = std::make_unique<FloatExprAST>(0.0); }
    else if (CurTok.type == (int)Token::tok_symbol && CurTok.value == "(") result = parseParenExpr();
    else if (CurTok.type == (int)Token::tok_symbol && CurTok.value == "[") result = parseArrayLit();
    else if (CurTok.type == (int)Token::tok_symbol && CurTok.value == "@") {
        // @function_name ?? first-class function reference
        getNextToken();
        if (CurTok.type != (int)Token::tok_identifier)
            return logError("expected function name after '@'");
        std::string fnName = CurTok.value;
        getNextToken();
        result = std::make_unique<FnRefExprAST>(fnName);
    }
    else if (CurTok.type == (int)Token::tok_symbol && CurTok.value == "-") {
        getNextToken();
        if (CurTok.type == (int)Token::tok_int_lit)   {
            try {
                long long v = std::stoll(CurTok.value);
                if (v >= INT32_MIN && v <= INT32_MAX)
                    result = std::make_unique<IntExprAST>((int)(-v));
                else
                    result = std::make_unique<FloatExprAST>(-(double)v);
            } catch (...) { logError("invalid integer literal '" + CurTok.value + "'"); return nullptr; }
            getNextToken();
        }
        else if (CurTok.type == (int)Token::tok_float_lit) { result=std::make_unique<FloatExprAST>(-std::stod(CurTok.value)); getNextToken(); }
        else {
            auto operand = parsePrimary(); if (!operand) return nullptr;
            result = std::make_unique<UnaryExprAST>("-", std::move(operand));
        }
    }
    else if (CurTok.type == (int)Token::tok_symbol && CurTok.value == "!") {
        getNextToken();
        auto operand = parsePrimary(); if (!operand) return nullptr;
        result = std::make_unique<UnaryExprAST>("!", std::move(operand));
    }
    else if (CurTok.type == (int)Token::tok_grad) {
        std::string kw = CurTok.value; getNextToken();
        if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "(") {
            result = std::make_unique<VariableExprAST>(kw);
        } else {
            getNextToken();
            auto expr = parseExpression();
            if (CurTok.type != (int)Token::tok_symbol || CurTok.value != ")")
                return logError("expected ')' after grad");
            getNextToken();
            result = std::make_unique<GradExprAST>(std::move(expr));
        }
    }
    else {
        return logError("expected an expression");
    }

    if (!result) return nullptr;
    return parsePostfix(std::move(result));
}

std::unique_ptr<ASTNode> Parser::parseBinOpRHS(int minPrec, std::unique_ptr<ASTNode> lhs) {
    while (true) {
        int prec = getTokPrecedence();
        if (prec < minPrec) return lhs;
        std::string op = CurTok.value;
        getNextToken();
        auto rhs = parsePrimary(); if (!rhs) return nullptr;
        if (prec < getTokPrecedence()) {
            rhs = parseBinOpRHS(prec+1, std::move(rhs)); if (!rhs) return nullptr;
        }
        lhs = std::make_unique<BinaryExprAST>(op, std::move(lhs), std::move(rhs));
    }
}

std::unique_ptr<ASTNode> Parser::parseExpression() {
    auto lhs  = parsePrimary(); if (!lhs) return nullptr;
    auto expr = parseBinOpRHS(0, std::move(lhs)); if (!expr) return nullptr;
    if (CurTok.type == (int)Token::tok_symbol && CurTok.value == "?") {
        getNextToken();
        auto thenExpr = parseExpression(); if (!thenExpr) return nullptr;
        if (CurTok.type != (int)Token::tok_symbol || CurTok.value != ":")
            return logError("expected ':' in ternary expression");
        getNextToken();
        auto elseExpr = parseExpression(); if (!elseExpr) return nullptr;
        return std::make_unique<TernaryExprAST>(
            std::move(expr), std::move(thenExpr), std::move(elseExpr));
    }

    // `expr as f32|f64|exact` precision suffix
    // `expr as f32|f64|exact` precision suffix. while (not if) so chained
    // annotations like `x as f32 as f64` parse cleanly instead of treating the
    // second `as` as a variable.
    while (CurTok.type == (int)Token::tok_identifier && CurTok.value == "as") {
        getNextToken();
        if (CurTok.type != (int)Token::tok_identifier)
            return logError("expected precision (f32, f64, or exact) after 'as'");
        DreamPrec p;
        if      (CurTok.value == "f32")   p = DreamPrec::F32;
        else if (CurTok.value == "f64")   p = DreamPrec::F64;
        else if (CurTok.value == "exact") p = DreamPrec::Exact;
        else return logError("unknown precision '" + CurTok.value +
                             "' (expected f32, f64, or exact)");
        getNextToken();
        expr = std::make_unique<PrecisionExprAST>(std::move(expr), p);
    }

    if (expr->isOpNode() && expr->isFuseable()) {
        std::vector<std::string> vars;
        expr->collectVars(vars);
        if (!vars.empty()) {
            static int ctr = 0;
            std::string kn  = "fused_k_"+std::to_string(ctr);
            std::string bwdn= kn+"_bwd";
            std::string ks  = KernelGen::buildFwd(kn, vars, expr->genFusedCode(vars));
            std::string bwds= KernelGen::buildBwd(bwdn, vars, expr.get());
            ++ctr;
            return std::make_unique<FusedExprAST>(
                std::move(expr), std::move(vars), kn, ks, bwdn, bwds);
        }
    }
    return expr;
}

std::unique_ptr<ASTNode> Parser::parseCompoundAssign(const std::string& varName,
                                                       const std::string& op) {
    getNextToken(); // consume 'op='
    auto rhs = parseExpression(); if (!rhs) return nullptr;
    auto lhsVar = std::make_unique<VariableExprAST>(varName);
    auto newRhs = std::make_unique<BinaryExprAST>(op, std::move(lhsVar), std::move(rhs));
    if (CurTok.type == (int)Token::tok_symbol && CurTok.value == ";") getNextToken();
    return std::make_unique<AssignStmtAST>(varName, std::move(newRhs));
}

std::unique_ptr<ASTNode> Parser::parseAssertStmt() {
    getNextToken(); // consume 'assert'
    if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "(")
        return logError("expected ( after assert");
    getNextToken();
    auto cond = parseExpression();
    if (!cond) return nullptr;
    std::string msg;
    if (CurTok.type == (int)Token::tok_symbol && CurTok.value == ",") {
        getNextToken();
        if (CurTok.type == (int)Token::tok_string_lit) {
            msg = CurTok.value;
            getNextToken();
        }
    }
    if (CurTok.type != (int)Token::tok_symbol || CurTok.value != ")")
        return logError("expected ) after assert");
    getNextToken();
    if (CurTok.type == (int)Token::tok_symbol && CurTok.value == ";") getNextToken();
    return std::make_unique<AssertStmtAST>(std::move(cond), msg);
}

std::unique_ptr<ASTNode> Parser::parseSaveStmt() {
    getNextToken(); // consume '('
    auto tensorExpr = parseExpression();
    if (!tensorExpr) return nullptr;
    if (CurTok.type != (int)Token::tok_symbol || CurTok.value != ",")
        return logError("expected , between tensor and filename in save");
    getNextToken();
    if (CurTok.type != (int)Token::tok_string_lit)
        return logError("save requires a string literal filename");
    std::string filename = CurTok.value;
    getNextToken();
    if (CurTok.type != (int)Token::tok_symbol || CurTok.value != ")")
        return logError("expected ) after save filename");
    getNextToken();
    if (CurTok.type == (int)Token::tok_symbol && CurTok.value == ";") getNextToken();
    return std::make_unique<SaveStmtAST>(std::move(tensorExpr), filename);
}

std::unique_ptr<ASTNode> Parser::parseDestructLet() {
    getNextToken(); // consume '('
    std::vector<std::string> names;
    while (CurTok.type == (int)Token::tok_identifier) {
        names.push_back(CurTok.value);
        getNextToken();
        if (CurTok.type == (int)Token::tok_symbol && CurTok.value == ",") getNextToken();
    }
    if (CurTok.type != (int)Token::tok_symbol || CurTok.value != ")")
        return logError("expected ')' in destructuring let");
    getNextToken();
    if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "=")
        return logError("expected '=' in destructuring let");
    getNextToken();
    auto expr = parseExpression();
    if (CurTok.type == (int)Token::tok_symbol && CurTok.value == ";") getNextToken();
    if (!expr) return nullptr;
    return std::make_unique<DestructLetStmtAST>(std::move(names), std::move(expr));
}

std::unique_ptr<ASTNode> Parser::parseLetStmt() {
    getNextToken(); // consume 'let'
    // Destructuring: let (a, b, ...) = expr
    if (CurTok.type == (int)Token::tok_symbol && CurTok.value == "(")
        return parseDestructLet();
    if (CurTok.type != (int)Token::tok_identifier)
        return logError("expected variable name after 'let'");
    std::string vn = CurTok.value; getNextToken();
    std::string vt = "auto";
    if (CurTok.type == (int)Token::tok_symbol && CurTok.value == ":") {
        getNextToken();
        if (CurTok.type != (int)Token::tok_type && CurTok.type != (int)Token::tok_identifier)
            return logError("expected type after ':'");
        vt = CurTok.value; getNextToken();
    }
    if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "=")
        return logError("expected '=' in let");
    getNextToken();
    auto expr = parseExpression();
    if (CurTok.type == (int)Token::tok_symbol && CurTok.value == ";") getNextToken();
    if (!expr) return nullptr;
    return std::make_unique<LetStmtAST>(vn, vt, std::move(expr));
}

std::unique_ptr<ASTNode> Parser::parseReturnStmt() {
    getNextToken();
    if (CurTok.type == (int)Token::tok_symbol && CurTok.value == ";") {
        getNextToken(); return std::make_unique<ReturnStmtAST>(nullptr);
    }
    auto expr = parseExpression();
    if (CurTok.type == (int)Token::tok_symbol && CurTok.value == ";") getNextToken();
    return std::make_unique<ReturnStmtAST>(std::move(expr));
}

std::unique_ptr<ASTNode> Parser::parseWhileStmt() {
    getNextToken();
    if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "(") return logError("expected '(' after while");
    getNextToken();
    auto cond = parseExpression();
    if (CurTok.type != (int)Token::tok_symbol || CurTok.value != ")") return logError("expected ')'");
    getNextToken();
    if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "{") return logError("expected '{'");
    getNextToken();
    auto body = parseBlock(); getNextToken();
    return std::make_unique<WhileStmtAST>(std::move(cond), std::move(body));
}

std::unique_ptr<ASTNode> Parser::parseForStmt() {
    getNextToken(); // consume 'for'
    if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "(")
        return logError("expected '(' after 'for'");
    getNextToken();
    if (CurTok.type == (int)Token::tok_identifier) {
        std::string varName = CurTok.value;
        TokenData saved = CurTok;
        getNextToken();
        if (CurTok.type == (int)Token::tok_in) {
            // Range-based for
            getNextToken();
            auto startExpr = parseExpression(); if (!startExpr) return nullptr;
            if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "..")
                return logError("expected '..' in range-for");
            getNextToken();
            auto endExpr = parseExpression(); if (!endExpr) return nullptr;
            std::unique_ptr<ASTNode> stepExpr;
            if (CurTok.type == (int)Token::tok_symbol && CurTok.value == "..") {
                getNextToken();
                stepExpr = parseExpression(); if (!stepExpr) return nullptr;
            }
            if (CurTok.type != (int)Token::tok_symbol || CurTok.value != ")")
                return logError("expected ')' after range-for");
            getNextToken();
            if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "{")
                return logError("expected '{' before range-for body");
            getNextToken();
            auto body = parseBlock(); getNextToken();
            return std::make_unique<ForRangeStmtAST>(
                varName, std::move(startExpr), std::move(endExpr),
                std::move(stepExpr), std::move(body));
        }
        // Not range-based. Try C-style without let: for (i = 0; ...)
        if (CurTok.type == (int)Token::tok_symbol && CurTok.value == "=") {
            getNextToken();
            auto initExpr = parseExpression(); if (!initExpr) return nullptr;
            // Wrap as assignment: i = expr (the variable must already exist)
            auto init = std::make_unique<AssignStmtAST>(varName, std::move(initExpr));
            if (CurTok.type == (int)Token::tok_symbol && CurTok.value == ";") getNextToken();
            auto cond = parseExpression(); if (!cond) return nullptr;
            if (CurTok.type != (int)Token::tok_symbol || CurTok.value != ";")
                return logError("expected ';' after for condition");
            getNextToken();
            if (CurTok.type != (int)Token::tok_identifier)
                return logError("expected identifier in for step");
            std::string stepVar = CurTok.value; getNextToken();
            std::unique_ptr<ASTNode> stepExpr;
            if (CurTok.type == (int)Token::tok_symbol && CurTok.value == "=") {
                getNextToken(); stepExpr = parseExpression(); if (!stepExpr) return nullptr;
            } else if (CurTok.type == (int)Token::tok_symbol && CurTok.value == "++") {
                getNextToken();
                stepExpr = std::make_unique<BinaryExprAST>("+",
                    std::make_unique<VariableExprAST>(stepVar), std::make_unique<IntExprAST>(1));
            } else if (CurTok.type == (int)Token::tok_symbol && CurTok.value == "--") {
                getNextToken();
                stepExpr = std::make_unique<BinaryExprAST>("-",
                    std::make_unique<VariableExprAST>(stepVar), std::make_unique<IntExprAST>(1));
            } else if (CurTok.type == (int)Token::tok_symbol &&
                       (CurTok.value == "+=" || CurTok.value == "-=" ||
                        CurTok.value == "*=" || CurTok.value == "/=")) {
                std::string op = CurTok.value.substr(0,1); getNextToken();
                auto rhs = parseExpression(); if (!rhs) return nullptr;
                stepExpr = std::make_unique<BinaryExprAST>(op,
                    std::make_unique<VariableExprAST>(stepVar), std::move(rhs));
            } else {
                return logError("expected '=', '++', '--', or '+=' in for step");
            }
            if (CurTok.type != (int)Token::tok_symbol || CurTok.value != ")")
                return logError("expected ')'");
            getNextToken();
            if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "{")
                return logError("expected '{'");
            getNextToken();
            auto body = parseBlock(); getNextToken();
            return std::make_unique<ForStmtAST>(
                std::move(init), std::move(cond), stepVar, std::move(stepExpr), std::move(body));
        }
        return logError("expected 'in' or '=' after identifier in for loop");
    }

    // --- C-style for ---
    if (CurTok.type != (int)Token::tok_let)
        return logError("expected 'let' initialization in for loop");
    auto init = parseLetStmt(); if (!init) return nullptr;
    auto cond = parseExpression(); if (!cond) return nullptr;
    if (CurTok.type != (int)Token::tok_symbol || CurTok.value != ";")
        return logError("expected ';' after for condition");
    getNextToken();
    if (CurTok.type != (int)Token::tok_identifier)
        return logError("expected identifier in for step");
    std::string stepVar = CurTok.value; getNextToken();

    std::unique_ptr<ASTNode> stepExpr;
    if (CurTok.type == (int)Token::tok_symbol && CurTok.value == "=") {
        // i = expr
        getNextToken();
        stepExpr = parseExpression(); if (!stepExpr) return nullptr;
    } else if (CurTok.type == (int)Token::tok_symbol && CurTok.value == "++") {
        // i++
        getNextToken();
        stepExpr = std::make_unique<BinaryExprAST>("+",
            std::make_unique<VariableExprAST>(stepVar),
            std::make_unique<IntExprAST>(1));
    } else if (CurTok.type == (int)Token::tok_symbol && CurTok.value == "--") {
        // i--
        getNextToken();
        stepExpr = std::make_unique<BinaryExprAST>("-",
            std::make_unique<VariableExprAST>(stepVar),
            std::make_unique<IntExprAST>(1));
    } else if (CurTok.type == (int)Token::tok_symbol &&
               (CurTok.value == "+=" || CurTok.value == "-=" ||
                CurTok.value == "*=" || CurTok.value == "/=")) {
        // i += expr, i -= expr, etc.
        std::string op = CurTok.value.substr(0, 1);
        getNextToken();
        auto rhs = parseExpression(); if (!rhs) return nullptr;
        stepExpr = std::make_unique<BinaryExprAST>(op,
            std::make_unique<VariableExprAST>(stepVar), std::move(rhs));
    } else {
        return logError("expected '=', '++', '--', or '+=' in for step");
    }
    if (CurTok.type != (int)Token::tok_symbol || CurTok.value != ")")
        return logError("expected ')'");
    getNextToken();
    if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "{")
        return logError("expected '{'");
    getNextToken();
    auto body = parseBlock(); getNextToken();
    return std::make_unique<ForStmtAST>(
        std::move(init), std::move(cond), stepVar, std::move(stepExpr), std::move(body));
}

std::unique_ptr<ASTNode> Parser::parseIfStmt() {
    getNextToken();
    if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "(") return logError("expected '('");
    getNextToken();
    auto cond = parseExpression();
    if (CurTok.type != (int)Token::tok_symbol || CurTok.value != ")") return logError("expected ')'");
    getNextToken();
    if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "{") return logError("expected '{'");
    getNextToken();
    auto thenBody = parseBlock(); getNextToken();

    std::vector<std::unique_ptr<ASTNode>> elseBody;
    if (CurTok.type == (int)Token::tok_else) {
        getNextToken();
        if (CurTok.type == (int)Token::tok_if) {
            if (auto elseIf = parseIfStmt()) {
                elseBody.push_back(std::move(elseIf));
                return std::make_unique<IfStmtAST>(std::move(cond),std::move(thenBody),std::move(elseBody));
            }
            return nullptr;
        }
        if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "{") return logError("expected '{'");
        getNextToken();
        elseBody = parseBlock(); getNextToken();
    }
    return std::make_unique<IfStmtAST>(std::move(cond),std::move(thenBody),std::move(elseBody));
}

std::vector<std::unique_ptr<ASTNode>> Parser::parseBlock() {
    std::vector<std::unique_ptr<ASTNode>> body;

    while (CurTok.type != (int)Token::tok_eof &&
           !(CurTok.type == (int)Token::tok_symbol && CurTok.value == "}"))
    {
        // Skip stray semicolons
        if (CurTok.type == (int)Token::tok_symbol && CurTok.value == ";") {
            getNextToken(); continue;
        }

        // Keywords with dedicated parsers
        if (CurTok.type == (int)Token::tok_let)    { if (auto s=parseLetStmt())    body.push_back(std::move(s)); continue; }
        if (CurTok.type == (int)Token::tok_return) { if (auto s=parseReturnStmt()) body.push_back(std::move(s)); continue; }
        if (CurTok.type == (int)Token::tok_while)  { if (auto s=parseWhileStmt())  body.push_back(std::move(s)); continue; }
        if (CurTok.type == (int)Token::tok_for)    { if (auto s=parseForStmt())    body.push_back(std::move(s)); continue; }
        if (CurTok.type == (int)Token::tok_if)     { if (auto s=parseIfStmt())     body.push_back(std::move(s)); continue; }
        if (CurTok.type == (int)Token::tok_assert) { if (auto s=parseAssertStmt()) body.push_back(std::move(s)); continue; }

        // no_grad { ... } block: no graph is built inside. For inference, long
        // accumulation loops, and stats -- none need gradients, and building a
        if (CurTok.type == (int)Token::tok_identifier && CurTok.value == "no_grad") {
            getNextToken();
            if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "{") {
                logError("expected '{' after no_grad"); continue;
            }
            getNextToken();  // consume '{'
            auto inner = parseBlock();
            if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "}") {
                logError("expected '}' to close no_grad block"); continue;
            }
            getNextToken();  // consume '}'
            body.push_back(std::make_unique<NoGradStmtAST>(std::move(inner)));
            continue;
        }

        // Nested function definition: fn name [captures] (params) { body }
        // Becomes a top-level function with captures as extra leading params.
        // Emits: let name = closure(@mangled_name, cap1, cap2, ...)
        if (CurTok.type == (int)Token::tok_fn) {
            getNextToken(); // consume 'fn'
            if (CurTok.type != (int)Token::tok_identifier) { logError("expected function name"); continue; }
            std::string localName = CurTok.value;
            static int closureCounter = 0;
            std::string mangledName = "__closure_" + localName + "_" + std::to_string(closureCounter++);
            getNextToken();

            // Parse optional captures: [x, y, z]
            std::vector<std::string> captures;
            if (CurTok.type == (int)Token::tok_symbol && CurTok.value == "[") {
                getNextToken();
                while (CurTok.type == (int)Token::tok_identifier) {
                    captures.push_back(CurTok.value);
                    getNextToken();
                    if (CurTok.type == (int)Token::tok_symbol && CurTok.value == ",") getNextToken();
                }
                if (CurTok.type == (int)Token::tok_symbol && CurTok.value == "]") getNextToken();
                else { logError("expected ']' after captures"); continue; }
            }

            // Parse params: (a, b, c)
            if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "(") { logError("expected '('"); continue; }
            getNextToken();
            std::vector<std::pair<std::string,std::string>> params;
            // Captures become leading params
            for (auto& cap : captures) params.push_back({cap, "Tensor"});
            // User-visible params
            while (CurTok.type == (int)Token::tok_identifier) {
                std::string pn = CurTok.value, pt = "Tensor"; getNextToken();
                if (CurTok.type==(int)Token::tok_symbol && CurTok.value==":") {
                    getNextToken(); pt=CurTok.value; getNextToken();
                }
                params.push_back({pn, pt});
                if (CurTok.type==(int)Token::tok_symbol && CurTok.value==",") getNextToken();
            }
            if (CurTok.type != (int)Token::tok_symbol || CurTok.value != ")") { logError("expected ')'"); continue; }
            getNextToken();

            // Parse body
            if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "{") { logError("expected '{'"); continue; }
            getNextToken();
            auto fnBody = parseBlock();
            getNextToken(); // consume '}'

            // Create top-level function AST and codegen it immediately
            auto fnAST = std::make_unique<FunctionAST>(mangledName, std::move(params), std::move(fnBody));
            fnAST->codegen();

            // Emit: let localName = dream_make_closure("mangled_name", ncaptures, cap1, cap2, ...)
            // For now, create a fn ref and pack captures
            std::vector<std::unique_ptr<ASTNode>> closureArgs;
            closureArgs.push_back(std::make_unique<StringExprAST>(mangledName));
            for (auto& cap : captures)
                closureArgs.push_back(std::make_unique<VariableExprAST>(cap));
            auto closureExpr = std::make_unique<CallExprAST>("dream_make_closure", std::move(closureArgs));
            auto letStmt = std::make_unique<LetStmtAST>(localName, "auto", std::move(closureExpr));
            body.push_back(std::move(letStmt));
            continue;
        }

        if (CurTok.type == (int)Token::tok_break) {
            getNextToken();
            if (CurTok.type==(int)Token::tok_symbol&&CurTok.value==";") getNextToken();
            body.push_back(std::make_unique<BreakStmtAST>()); continue;
        }
        if (CurTok.type == (int)Token::tok_continue) {
            getNextToken();
            if (CurTok.type==(int)Token::tok_symbol&&CurTok.value==";") getNextToken();
            body.push_back(std::make_unique<ContinueStmtAST>()); continue;
        }

        // Bare string literal -> print it
        if (CurTok.type == (int)Token::tok_string_lit) {
            std::vector<std::unique_ptr<ASTNode>> args;
            args.push_back(std::make_unique<StringExprAST>(CurTok.value));
            getNextToken();
            if (CurTok.type==(int)Token::tok_symbol&&CurTok.value==";") getNextToken();
            body.push_back(std::make_unique<PrintStmtAST>(std::move(args), false));
            continue;
        }

        // Identifier/type/grad: might be assignment, call, index-assign, etc.
        if (CurTok.type == (int)Token::tok_identifier ||
            CurTok.type == (int)Token::tok_grad ||
            CurTok.type == (int)Token::tok_type)
        {
            std::string name = CurTok.value;

            // print/println
            if (name == "print" || name == "println") {
                body.push_back(parsePrintStmt(name == "println"));
                continue;
            }
            // save(tensor, "file")
            if (name == "save") {
                getNextToken();
                if (CurTok.type == (int)Token::tok_symbol && CurTok.value == "(") {
                    if (auto s = parseSaveStmt()) body.push_back(std::move(s));
                } else { logError("expected '(' after save"); }
                continue;
            }

            getNextToken();

            // Field assignment: name.field = expr
            if (CurTok.type==(int)Token::tok_symbol && CurTok.value==".") {
                getNextToken();
                if (CurTok.type != (int)Token::tok_identifier) {
                    logError("expected field name after '.'"); continue;
                }
                std::string field = CurTok.value;
                getNextToken();
                if (CurTok.type==(int)Token::tok_symbol && CurTok.value=="=") {
                    getNextToken();
                    auto val = parseExpression();
                    if (CurTok.type==(int)Token::tok_symbol&&CurTok.value==";") getNextToken();
                    if (val) body.push_back(std::make_unique<FieldAssignStmtAST>(
                        name, field, std::move(val)));
                } else {
                    // Field read as expression statement (name.field ...)
                    std::unique_ptr<ASTNode> base = std::make_unique<FieldAccessExprAST>(
                        std::make_unique<VariableExprAST>(name), field);
                    base = parsePostfix(std::move(base));
                    auto full = parseBinOpRHS(0, std::move(base));
                    if (CurTok.type==(int)Token::tok_symbol&&CurTok.value==";") getNextToken();
                    if (full) body.push_back(std::move(full));
                }
                continue;
            }

            // Simple assignment: name = expr
            if (CurTok.type==(int)Token::tok_symbol && CurTok.value=="=") {
                getNextToken();
                auto expr = parseExpression();
                if (CurTok.type==(int)Token::tok_symbol&&CurTok.value==";") getNextToken();
                if (expr) body.push_back(std::make_unique<AssignStmtAST>(name, std::move(expr)));
                continue;
            }
            // Compound assignment: name += expr, name -= expr, etc.
            if (CurTok.type==(int)Token::tok_symbol &&
                (CurTok.value=="+="||CurTok.value=="-="||
                 CurTok.value=="*="||CurTok.value=="/=")) {
                std::string op = CurTok.value.substr(0,1);
                if (auto s = parseCompoundAssign(name, op)) body.push_back(std::move(s));
                continue;
            }
            // Increment/decrement: name++, name--
            if (CurTok.type==(int)Token::tok_symbol && CurTok.value=="++") {
                getNextToken();
                if (CurTok.type==(int)Token::tok_symbol&&CurTok.value==";") getNextToken();
                auto lhs = std::make_unique<VariableExprAST>(name);
                auto one = std::make_unique<IntExprAST>(1);
                body.push_back(std::make_unique<AssignStmtAST>(name,
                    std::make_unique<BinaryExprAST>("+", std::move(lhs), std::move(one))));
                continue;
            }
            if (CurTok.type==(int)Token::tok_symbol && CurTok.value=="--") {
                getNextToken();
                if (CurTok.type==(int)Token::tok_symbol&&CurTok.value==";") getNextToken();
                auto lhs = std::make_unique<VariableExprAST>(name);
                auto one = std::make_unique<IntExprAST>(1);
                body.push_back(std::make_unique<AssignStmtAST>(name,
                    std::make_unique<BinaryExprAST>("-", std::move(lhs), std::move(one))));
                continue;
            }
            // Index access/assignment: name[idx]... = expr
            if (CurTok.type==(int)Token::tok_symbol && CurTok.value=="[") {
                std::vector<std::unique_ptr<ASTNode>> indices;
                while (CurTok.type==(int)Token::tok_symbol && CurTok.value=="[") {
                    getNextToken();
                    auto idx = parseExpression();
                    if (!idx) break;
                    // Support name[i, j] multi-dimensional
                    if (CurTok.type==(int)Token::tok_symbol && CurTok.value==",") {
                        indices.push_back(std::move(idx));
                        getNextToken();
                        auto idx2 = parseExpression();
                        if (idx2) indices.push_back(std::move(idx2));
                        if (CurTok.type==(int)Token::tok_symbol && CurTok.value=="]") getNextToken();
                        break;
                    }
                    if (CurTok.type==(int)Token::tok_symbol && CurTok.value=="]") getNextToken();
                    else { logError("expected ']'"); break; }
                    indices.push_back(std::move(idx));
                }
                if (CurTok.type==(int)Token::tok_symbol && CurTok.value=="=") {
                    // Index assignment
                    getNextToken();
                    auto val = parseExpression();
                    if (CurTok.type==(int)Token::tok_symbol&&CurTok.value==";") getNextToken();
                    if (val) body.push_back(std::make_unique<IndexAssignStmtAST>(
                        name, std::move(indices), std::move(val)));
                } else {
                    // Indexed expression used as statement
                    std::unique_ptr<ASTNode> base = std::make_unique<VariableExprAST>(name);
                    if (indices.size() == 2) {
                        std::vector<std::unique_ptr<ASTNode>> args;
                        args.push_back(std::move(base));
                        args.push_back(std::move(indices[0]));
                        args.push_back(std::move(indices[1]));
                        base = std::make_unique<CallExprAST>("tensor_index_2d", std::move(args));
                    } else {
                        for (auto& idx : indices)
                            base = std::make_unique<IndexExprAST>(std::move(base), std::move(idx));
                    }
                    auto expr = parseBinOpRHS(0, std::move(base));
                    if (CurTok.type==(int)Token::tok_symbol&&CurTok.value==";") getNextToken();
                    if (expr) body.push_back(std::move(expr));
                }
                continue;
            }
            // Function call: name(args)
            if (CurTok.type==(int)Token::tok_symbol && CurTok.value=="(") {
                getNextToken();
                auto args = parseArgList();
                if (CurTok.type==(int)Token::tok_symbol&&CurTok.value==";") getNextToken();
                body.push_back(std::make_unique<CallExprAST>(name, std::move(args)));
                continue;
            }

            // Anything else starting with this identifier: treat as expression statement
            // Reconstruct the expression from the variable and parse the rest
            auto varExpr = std::make_unique<VariableExprAST>(name);
            auto full = parsePostfix(std::move(varExpr));
            if (full) full = parseBinOpRHS(0, std::move(full));
            if (CurTok.type==(int)Token::tok_symbol&&CurTok.value==";") getNextToken();
            if (full) body.push_back(std::move(full));
            continue;
        }

        // Catch-all: try parsing as an expression statement
        // This handles things like: -x; [1,2,3]; (a+b); etc.
        auto expr = parseExpression();
        if (expr) {
            if (CurTok.type==(int)Token::tok_symbol&&CurTok.value==";") getNextToken();
            body.push_back(std::move(expr));
        } else {
            // Error recovery: skip tokens until we find ; or }
            while (CurTok.type != (int)Token::tok_eof &&
                   !(CurTok.type == (int)Token::tok_symbol && CurTok.value == "}") &&
                   !(CurTok.type == (int)Token::tok_symbol && CurTok.value == ";"))
                getNextToken();
            if (CurTok.type==(int)Token::tok_symbol&&CurTok.value==";") getNextToken();
        }
    }
    return body;
}

std::unique_ptr<FunctionAST> Parser::parseFunction() {
    getNextToken(); // consume 'fn'
    if (CurTok.type != (int)Token::tok_identifier) { logError("expected function name"); return nullptr; }
    std::string fn = CurTok.value; getNextToken();
    if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "(") { logError("expected '('"); return nullptr; }
    getNextToken();

    std::vector<std::pair<std::string,std::string>> args;
    while (CurTok.type == (int)Token::tok_identifier) {
        std::string an = CurTok.value, at = "Tensor"; getNextToken();
        if (CurTok.type==(int)Token::tok_symbol && CurTok.value==":") {
            getNextToken(); at=CurTok.value; getNextToken();
        }
        args.push_back({an,at});
        if (CurTok.type==(int)Token::tok_symbol&&CurTok.value==",") getNextToken();
    }
    if (CurTok.type!=(int)Token::tok_symbol||CurTok.value!=")") { logError("expected ')'"); return nullptr; }
    // Skip to '{' ?? allows optional `-> Type` return annotation
    while (CurTok.type!=(int)Token::tok_eof&&!(CurTok.type==(int)Token::tok_symbol&&CurTok.value=="{")) getNextToken();
    if (CurTok.type!=(int)Token::tok_symbol||CurTok.value!="{") { logError("expected '{'"); return nullptr; }
    getNextToken();
    auto body=parseBlock();
    // parseBlock stops at '}' or EOF. Stopping at EOF means a missing closing brace.
    if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "}") {
        logError("expected '}' to close function '" + fn + "'");
        return nullptr;
    }
    getNextToken(); // consume '}'
    return std::make_unique<FunctionAST>(fn,std::move(args),std::move(body));
}

void Parser::parseStructDef() {
    getNextToken(); // consume 'struct'
    if (CurTok.type != (int)Token::tok_identifier) {
        logError("expected struct name"); return;
    }
    std::string name = CurTok.value;
    getNextToken();
    if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "{") {
        logError("expected '{' after struct name"); return;
    }
    getNextToken();
    std::vector<std::string> fields;
    while (CurTok.type == (int)Token::tok_identifier) {
        fields.push_back(CurTok.value);
        getNextToken();
        if (CurTok.type == (int)Token::tok_symbol && CurTok.value == ",") getNextToken();
    }
    if (CurTok.type != (int)Token::tok_symbol || CurTok.value != "}") {
        logError("expected '}' after struct fields"); return;
    }
    getNextToken();
    StructDefs[name] = fields;
    // Also register in global map for codegen
    extern std::map<std::string, std::vector<std::string>> g_struct_defs;
    g_struct_defs[name] = fields;
}

std::vector<std::unique_ptr<FunctionAST>> Parser::parseAllFunctions() {
    std::vector<std::unique_ptr<FunctionAST>> fns;
    while (CurTok.type != (int)Token::tok_eof) {
        if (CurTok.type == (int)Token::tok_fn) {
            if (auto f=parseFunction()) fns.push_back(std::move(f));
        } else if (CurTok.type == (int)Token::tok_struct) {
            parseStructDef();
        } else {
            getNextToken();
        }
        // Stop immediately after a syntax error: parsing a malformed token stream
        // yields an incomplete AST that crashes codegen. Report the first error, not segfault.
        if (HadError) break;
    }
    return fns;
}