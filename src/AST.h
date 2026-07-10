#pragma once
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <set>
#include <map>
#include "llvm/IR/Value.h"
#include "llvm/IR/Function.h"

class Lexer;
// Compile-time diagnostic shared state (populated by main.cpp).
extern std::set<std::string> g_runtime_syms;   // callable runtime symbols
extern std::set<std::string> g_user_fns;       // user function names in this build
extern Lexer*      g_diag_lexer;               // source-line access (for did-you-mean)
extern std::string g_diag_file;

class ASTNode {
public:
    int Line = 0;                              // source line (0 = unknown)
    virtual ~ASTNode() = default;
    virtual llvm::Value* codegen() = 0;
    virtual std::string  genOpenCL() { return ""; }
    virtual bool         isFuseable()  const { return false; }
    virtual bool         isOpNode()    const { return false; }
    virtual void         collectVars(std::vector<std::string>&) const {}
    virtual std::string  genFusedCode(const std::vector<std::string>&) const { return ""; }
    virtual std::string  genFusedGradCode(const std::string&,
                                          const std::vector<std::string>&) const { return "0.0f"; }
};

class IntExprAST : public ASTNode {
    int Val;
public:
    explicit IntExprAST(int Val);
    int getVal() const { return Val; }
    llvm::Value* codegen() override;
    std::string  genOpenCL() override;
    bool         isFuseable() const override;
    std::string  genFusedCode(const std::vector<std::string>&) const override;
    std::string  genFusedGradCode(const std::string&, const std::vector<std::string>&) const override;
};

class FloatExprAST : public ASTNode {
    double Val;
public:
    explicit FloatExprAST(double Val);
    double getVal() const { return Val; }
    llvm::Value* codegen() override;
    std::string  genOpenCL() override;
    bool         isFuseable() const override;
    std::string  genFusedCode(const std::vector<std::string>&) const override;
    std::string  genFusedGradCode(const std::string&, const std::vector<std::string>&) const override;
};

class CharExprAST : public ASTNode {
    char Val;
public:
    explicit CharExprAST(char Val);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override;
    bool         isFuseable() const override;
    std::string  genFusedCode(const std::vector<std::string>&) const override;
    std::string  genFusedGradCode(const std::string&, const std::vector<std::string>&) const override;
};

class StringExprAST : public ASTNode {
    std::string Val;
public:
    explicit StringExprAST(const std::string& Val);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override;
    const std::string& getVal() const { return Val; }
};

class VariableExprAST : public ASTNode {
    std::string Name;
public:
    explicit VariableExprAST(const std::string& Name);
    llvm::Value*       codegen() override;
    std::string        genOpenCL() override;
    const std::string& getName() const;
    bool               isFuseable() const override;
    void               collectVars(std::vector<std::string>&) const override;
    std::string        genFusedCode(const std::vector<std::string>&) const override;
    std::string        genFusedGradCode(const std::string&, const std::vector<std::string>&) const override;
};

class CallExprAST : public ASTNode {
    std::string                           Callee;
    std::vector<std::unique_ptr<ASTNode>> Args;
public:
    int PrecHint = 0;   // 0=default, 3=exact (set by PrecisionExprAST)
    const std::string& getCallee() const { return Callee; }
    CallExprAST(const std::string& Callee, std::vector<std::unique_ptr<ASTNode>> Args);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override;
    bool         isFuseable() const override;
    std::string  genFusedCode(const std::vector<std::string>&) const override;
    std::string  genFusedGradCode(const std::string&, const std::vector<std::string>&) const override;
};

class UnaryExprAST : public ASTNode {
    std::string              Op;
    std::unique_ptr<ASTNode> Operand;
public:
    UnaryExprAST(std::string Op, std::unique_ptr<ASTNode> Operand);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override;
    bool         isFuseable() const override;
    std::string  genFusedCode(const std::vector<std::string>&) const override;
    std::string  genFusedGradCode(const std::string&, const std::vector<std::string>&) const override;
};

// `expr as f32|f64|exact` -- precision annotation.
//   exact: reductions (sum/mean/dot/var...) redirect to compensated-sum variants
//   f32/f64: set the result tensor storage precision (dtype flag)
// Does not change the inner expression structure; only adjusts codegen by precision.
enum class DreamPrec { Default, F32, F64, Exact };
class PrecisionExprAST : public ASTNode {
    std::unique_ptr<ASTNode> Inner;
    DreamPrec                Prec;
public:
    PrecisionExprAST(std::unique_ptr<ASTNode> Inner, DreamPrec Prec)
        : Inner(std::move(Inner)), Prec(Prec) {}
    llvm::Value* codegen() override;
};

// no_grad { ... } -- disables autograd graph building inside the block.
class NoGradStmtAST : public ASTNode {
    std::vector<std::unique_ptr<ASTNode>> Body;
public:
    explicit NoGradStmtAST(std::vector<std::unique_ptr<ASTNode>> Body)
        : Body(std::move(Body)) {}
    llvm::Value* codegen() override;
};

class BinaryExprAST : public ASTNode {
    std::string              Op;
    std::unique_ptr<ASTNode> LHS, RHS;
public:
    BinaryExprAST(std::string Op, std::unique_ptr<ASTNode> LHS, std::unique_ptr<ASTNode> RHS);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override;
    bool         isFuseable() const override;
    bool         isOpNode()   const override;
    void         collectVars(std::vector<std::string>&) const override;
    std::string  genFusedCode(const std::vector<std::string>&) const override;
    std::string  genFusedGradCode(const std::string&, const std::vector<std::string>&) const override;
};

class TernaryExprAST : public ASTNode {
    std::unique_ptr<ASTNode> Cond, Then, Else;
public:
    TernaryExprAST(std::unique_ptr<ASTNode> Cond,
                   std::unique_ptr<ASTNode> Then,
                   std::unique_ptr<ASTNode> Else);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override;
    bool         isFuseable() const override;
    std::string  genFusedCode(const std::vector<std::string>&) const override;
    std::string  genFusedGradCode(const std::string&, const std::vector<std::string>&) const override;
};

class FusedExprAST : public ASTNode {
    std::unique_ptr<ASTNode> Root;
    std::vector<std::string> Vars;
    std::string KernelName, KernelSrc, BwdKernelName, BwdKernelSrc;
public:
    FusedExprAST(std::unique_ptr<ASTNode> Root, std::vector<std::string> Vars,
                 std::string KN, std::string KS, std::string BN, std::string BS);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override;
};

class LetStmtAST : public ASTNode {
    std::string              VarName, VarType;
    std::unique_ptr<ASTNode> Expr;
public:
    LetStmtAST(const std::string& VarName, const std::string& VarType,
               std::unique_ptr<ASTNode> Expr);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override;
};

class AssignStmtAST : public ASTNode {
    std::string              VarName;
    std::unique_ptr<ASTNode> Expr;
public:
    AssignStmtAST(const std::string& VarName, std::unique_ptr<ASTNode> Expr);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override;
};

class WhileStmtAST : public ASTNode {
    std::unique_ptr<ASTNode>              Cond;
    std::vector<std::unique_ptr<ASTNode>> Body;
public:
    WhileStmtAST(std::unique_ptr<ASTNode> Cond, std::vector<std::unique_ptr<ASTNode>> Body);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override;
};

class ForStmtAST : public ASTNode {
    std::unique_ptr<ASTNode>              Init, Cond, StepExpr;
    std::string                           StepVar;
    std::vector<std::unique_ptr<ASTNode>> Body;
public:
    ForStmtAST(std::unique_ptr<ASTNode> Init, std::unique_ptr<ASTNode> Cond,
               std::string StepVar, std::unique_ptr<ASTNode> StepExpr,
               std::vector<std::unique_ptr<ASTNode>> Body);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override;
};

class ForRangeStmtAST : public ASTNode {
    std::string                           VarName;
    std::unique_ptr<ASTNode>              Start, End, Step;
    std::vector<std::unique_ptr<ASTNode>> Body;
public:
    ForRangeStmtAST(std::string VarName,
                    std::unique_ptr<ASTNode> Start,
                    std::unique_ptr<ASTNode> End,
                    std::unique_ptr<ASTNode> Step,
                    std::vector<std::unique_ptr<ASTNode>> Body);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override;
};

class IfStmtAST : public ASTNode {
    std::unique_ptr<ASTNode>              Cond;
    std::vector<std::unique_ptr<ASTNode>> ThenBody, ElseBody;
public:
    IfStmtAST(std::unique_ptr<ASTNode> Cond,
              std::vector<std::unique_ptr<ASTNode>> ThenBody,
              std::vector<std::unique_ptr<ASTNode>> ElseBody);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override;
};

class ReturnStmtAST : public ASTNode {
    std::unique_ptr<ASTNode> Expr;
public:
    explicit ReturnStmtAST(std::unique_ptr<ASTNode> Expr);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override;
};

class BreakStmtAST : public ASTNode {
public:
    llvm::Value* codegen() override;
};

class ContinueStmtAST : public ASTNode {
public:
    llvm::Value* codegen() override;
};

class PrintStmtAST : public ASTNode {
    std::vector<std::unique_ptr<ASTNode>> Args;
    bool Newline; // true for println
public:
    PrintStmtAST(std::vector<std::unique_ptr<ASTNode>> Args, bool Newline);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override;
};

class FunctionAST : public ASTNode {
    std::string                                     Name;
    std::vector<std::pair<std::string,std::string>> Args;
    std::vector<std::unique_ptr<ASTNode>>           Body;
public:
    FunctionAST(const std::string& Name,
                std::vector<std::pair<std::string,std::string>> Args,
                std::vector<std::unique_ptr<ASTNode>> Body);
    const std::string& getName() const { return Name; }
    const std::vector<std::pair<std::string,std::string>>& getArgs() const { return Args; }
    llvm::Function*    codegen();
    std::string        genOpenCL() override;
};

class GradExprAST : public ASTNode {
    std::unique_ptr<ASTNode> Expr;
public:
    explicit GradExprAST(std::unique_ptr<ASTNode> Expr);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override;
};
// assert(cond) or assert(cond, "message")
class AssertStmtAST : public ASTNode {
    std::unique_ptr<ASTNode> Cond;
    std::string              Msg;
public:
    AssertStmtAST(std::unique_ptr<ASTNode> Cond, std::string Msg);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override { return ""; }
};

// save(tensor, "filename.bin")
class SaveStmtAST : public ASTNode {
    std::unique_ptr<ASTNode> TensorExpr;
    std::string              Filename;
public:
    SaveStmtAST(std::unique_ptr<ASTNode> TensorExpr, std::string Filename);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override { return ""; }
};

// load("filename.bin")  ?? used as an expression
class LoadExprAST : public ASTNode {
    std::string Filename;
public:
    explicit LoadExprAST(std::string Filename);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override { return ""; }
};

// tensor[index] or tensor[i, j]
class IndexExprAST : public ASTNode {
    std::unique_ptr<ASTNode> Base;
    std::unique_ptr<ASTNode> Index;
public:
    IndexExprAST(std::unique_ptr<ASTNode> Base, std::unique_ptr<ASTNode> Index);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override { return ""; }
};

// tensor[index] = value  (supports multi-dimensional: tensor[i][j] = v)
class IndexAssignStmtAST : public ASTNode {
    std::string VarName;
    std::vector<std::unique_ptr<ASTNode>> Indices;
    std::unique_ptr<ASTNode> Value;
public:
    IndexAssignStmtAST(std::string VarName,
                        std::vector<std::unique_ptr<ASTNode>> Indices,
                        std::unique_ptr<ASTNode> Value);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override { return ""; }
};

// [expr, expr, ...] array/tensor literal
class ArrayLitExprAST : public ASTNode {
    std::vector<std::unique_ptr<ASTNode>> Elements;
public:
    explicit ArrayLitExprAST(std::vector<std::unique_ptr<ASTNode>> Elements);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override { return ""; }
};

// (a, b, c) tuple literal ?? packs multiple values for multi-return
class TupleLitExprAST : public ASTNode {
    std::vector<std::unique_ptr<ASTNode>> Elements;
public:
    explicit TupleLitExprAST(std::vector<std::unique_ptr<ASTNode>> Elements);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override { return ""; }
};

// let (x, y, z) = expr ?? destructuring bind from tuple
class DestructLetStmtAST : public ASTNode {
    std::vector<std::string>  Names;
    std::unique_ptr<ASTNode>  Expr;
public:
    DestructLetStmtAST(std::vector<std::string> Names, std::unique_ptr<ASTNode> Expr);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override { return ""; }
};

// t[start:end] slice expression
class SliceExprAST : public ASTNode {
    std::unique_ptr<ASTNode> Base;
    std::unique_ptr<ASTNode> Start;
    std::unique_ptr<ASTNode> End;
public:
    SliceExprAST(std::unique_ptr<ASTNode> Base,
                 std::unique_ptr<ASTNode> Start,
                 std::unique_ptr<ASTNode> End);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override { return ""; }
};

// expr.field_name ?? struct field access
class FieldAccessExprAST : public ASTNode {
    std::unique_ptr<ASTNode> Base;
    std::string FieldName;
public:
    FieldAccessExprAST(std::unique_ptr<ASTNode> Base, std::string FieldName);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override { return ""; }
};

// name.field = expr ?? struct field assignment
class FieldAssignStmtAST : public ASTNode {
    std::string VarName;
    std::string FieldName;
    std::unique_ptr<ASTNode> Value;
public:
    FieldAssignStmtAST(std::string VarName, std::string FieldName,
                        std::unique_ptr<ASTNode> Value);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override { return ""; }
};

// @function_name ?? first-class function reference
class FnRefExprAST : public ASTNode {
    std::string FnName;
public:
    explicit FnRefExprAST(std::string FnName);
    llvm::Value* codegen() override;
    std::string  genOpenCL() override { return ""; }
};