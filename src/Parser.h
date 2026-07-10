#pragma once
#include "Lexer.h"
#include "AST.h"
#include <memory>
#include <map>
#include <vector>
#include <string>
class Parser {
    Lexer&                     lex;
    TokenData                  CurTok;
    bool                       HadError = false;   // set on syntax error to block codegen
    std::map<std::string, int> BinopPrecedence;
    TokenData getNextToken();
    int       getTokPrecedence() const;
    std::unique_ptr<ASTNode> logError(const std::string& msg);
    std::unique_ptr<ASTNode> parsePrimary();
    std::unique_ptr<ASTNode> parseExpression();
    std::unique_ptr<ASTNode> parseBinOpRHS(int minPrec, std::unique_ptr<ASTNode> lhs);
    std::unique_ptr<ASTNode> parseIdentifierExpr();
    std::unique_ptr<ASTNode> parseIntLit();
    std::unique_ptr<ASTNode> parseFloatLit();
    std::unique_ptr<ASTNode> parseCharLit();
    std::unique_ptr<ASTNode> parseParenExpr();
    std::unique_ptr<ASTNode> parseLetStmt();
    std::unique_ptr<ASTNode> parseReturnStmt();
    std::unique_ptr<ASTNode> parseWhileStmt();
    std::unique_ptr<ASTNode> parseForStmt();
    std::unique_ptr<ASTNode> parseIfStmt();
    std::vector<std::unique_ptr<ASTNode>> parseBlock();
    std::unique_ptr<ASTNode> parseCompoundAssign(const std::string& varName, const std::string& op);
    std::vector<std::unique_ptr<ASTNode>> parseArgList();
    std::unique_ptr<ASTNode> parsePrintStmt(bool newline);
    std::unique_ptr<ASTNode> parseAssertStmt();
    std::unique_ptr<ASTNode> parseSaveStmt();
    std::unique_ptr<ASTNode> parseLoadExpr();
    std::unique_ptr<ASTNode> parseArrayLit();
    std::unique_ptr<ASTNode> parsePostfix(std::unique_ptr<ASTNode> base);
    std::unique_ptr<ASTNode> parseDestructLet();
    void parseStructDef();
    static std::map<std::string, std::vector<std::string>> StructDefs;
public:
    explicit Parser(Lexer& lexer);
    bool hadError() const { return HadError; }
    std::unique_ptr<FunctionAST>              parseFunction();
    std::vector<std::unique_ptr<FunctionAST>> parseAllFunctions();
};