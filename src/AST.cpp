#include "AST.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Constants.h"
#include <map>
#include <set>
#include <algorithm>
#include <iostream>

llvm::LLVMContext                   TheContext;
llvm::IRBuilder<>                   Builder(TheContext);
#include "Lexer.h"
const std::map<std::string,std::string>& builtinRemap();   // defined below

std::set<std::string> g_runtime_syms;
std::set<std::string> g_user_fns;
Lexer*      g_diag_lexer = nullptr;
std::string g_diag_file;

// Bounded Levenshtein (gives up past cut); used for did-you-mean.
static int levBounded(const std::string& a, const std::string& b, int cut) {
    int n=(int)a.size(), m=(int)b.size();
    if (std::abs(n-m) > cut) return cut+1;
    std::vector<int> prev(m+1), cur(m+1);
    for (int j=0;j<=m;++j) prev[j]=j;
    for (int i=1;i<=n;++i) {
        cur[0]=i; int rowMin=cur[0];
        for (int j=1;j<=m;++j) {
            int c = (a[i-1]==b[j-1])?0:1;
            cur[j] = std::min({prev[j]+1, cur[j-1]+1, prev[j-1]+c});
            rowMin = std::min(rowMin, cur[j]);
        }
        if (rowMin > cut) return cut+1;
        std::swap(prev,cur);
    }
    return prev[m];
}

// Generic compile-time diagnostic: title + source location + ^ highlight + optional hint.
// All codegen-stage semantic errors go through here for a uniform format.
//   title    error summary (without the "error:" prefix)
//   token    identifier to highlight (locates column / draws ^); empty = no arrow
//   line     source line number (<=0 = unknown, title only)
//   hint     "did you mean" suggestion; empty = none
static void diagAtLine(const std::string& title, const std::string& token,
                       int line, const std::string& hint = "") {
    std::cerr << "\nerror: " << title << "\n";
    if (line > 0) {
        std::cerr << "  --> " << (g_diag_file.empty() ? "<source>" : g_diag_file)
                  << ":" << line << "\n";
        if (g_diag_lexer) {
            std::string src = g_diag_lexer->getLine(line);
            std::cerr << "   |\n " << line << " | " << src << "\n   | ";
            if (!token.empty()) {
                size_t col = src.find(token + "(");
                if (col == std::string::npos) col = src.find(token);
                for (size_t i = 0; col != std::string::npos && i < col; ++i) std::cerr << ' ';
                for (size_t i = 0; i < token.size(); ++i) std::cerr << '^';
                std::cerr << "\n";
            }
        }
    }
    if (!hint.empty()) std::cerr << "  did you mean '" << hint << "'?\n";
    std::cerr << "\n";
}

// Known variable names (for undefined-variable spelling suggestions).
std::set<std::string> g_known_vars;

// Unknown function: compile-time error (with location and suggestion), not leaked to linker.
static void diagUnknownFn(const std::string& name, int line) {
    std::string best; int bestD = 3;
    auto consider = [&](const std::string& c){
        int d = levBounded(name, c, 2);
        if (d < bestD) { bestD = d; best = c; }
    };
    for (const auto& s : g_user_fns) consider(s);
    for (const auto& kv : builtinRemap()) consider(kv.first);
    for (const auto& s : g_runtime_syms) consider(s);
    diagAtLine("unknown function '" + name + "'", name, line, best);
}

std::unique_ptr<llvm::Module>       TheModule;
std::map<std::string, llvm::Value*> NamedValues;
llvm::StructType*                   TensorTy = nullptr;
std::vector<std::string>            GPUKernelArgs;
// Global struct definition registry (populated by Parser, used by codegen)
std::map<std::string, std::vector<std::string>> g_struct_defs;

static std::vector<llvm::BasicBlock*> BreakTargets;
static std::vector<llvm::BasicBlock*> ContinueTargets;
static std::vector<std::string> CurrentFnLocals;

void InitializeLLVM() {
    TheModule = std::make_unique<llvm::Module>("DreamCompiler", TheContext);
    TensorTy  = llvm::StructType::create(TheContext, "struct.Tensor");
    TensorTy->setBody({
        llvm::PointerType::getUnqual(llvm::Type::getDoubleTy(TheContext)),
        llvm::PointerType::getUnqual(llvm::Type::getDoubleTy(TheContext)),
        llvm::Type::getInt32Ty(TheContext),
        llvm::Type::getInt32Ty(TheContext),
        llvm::Type::getInt32Ty(TheContext),
        llvm::Type::getInt32Ty(TheContext),
        llvm::Type::getInt32Ty(TheContext),
    });
}

static llvm::Type* Int32Ty()  { return llvm::Type::getInt32Ty(TheContext); }
static llvm::Type* DoubleTy() { return llvm::Type::getDoubleTy(TheContext); }
static llvm::Type* Int8Ty()   { return llvm::Type::getInt8Ty(TheContext); }
static llvm::Type* TPtrTy()   { return llvm::PointerType::getUnqual(TensorTy); }
static llvm::Type* DblPtrTy() { return llvm::PointerType::getUnqual(DoubleTy()); }
static llvm::Type* CharPtrTy(){ return llvm::PointerType::getUnqual(Int8Ty()); }

static llvm::Function* getOrDeclare(const std::string& name,
                                     llvm::Type* retTy,
                                     llvm::ArrayRef<llvm::Type*> argTys,
                                     bool varArgs = false) {
    if (auto* F = TheModule->getFunction(name)) return F;
    return llvm::Function::Create(
        llvm::FunctionType::get(retTy, argTys, varArgs),
        llvm::Function::ExternalLinkage, name, TheModule.get());
}

static llvm::Value* castToTensor(llvm::Value* V) {
    if (V->getType()->isIntegerTy(32))
        V = Builder.CreateSIToFP(V, DoubleTy());
    if (V->getType()->isIntegerTy(8))
        V = Builder.CreateSIToFP(Builder.CreateSExt(V, Int32Ty()), DoubleTy());
    if (V->getType()->isDoubleTy()) {
        auto* F = getOrDeclare("create_temp_scalar", TPtrTy(), {DoubleTy()});
        return Builder.CreateCall(F, {V});
    }
    return V;
}

static llvm::AllocaInst* createEntryAlloca(llvm::Function* fn,
                                            const std::string& name,
                                            llvm::Type* ty) {
    llvm::IRBuilder<> tmp(&fn->getEntryBlock(), fn->getEntryBlock().begin());
    auto* alloca = tmp.CreateAlloca(ty, nullptr, name);
    // Null-init pointer allocas so tensor_decref(null) is safe on first loop iteration
    if (ty->isPointerTy())
        tmp.CreateStore(llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(ty)), alloca);
    return alloca;
}

static llvm::Value* buildCondBool(llvm::Value* V) {
    llvm::Type* ty = V->getType();
    if (ty->isIntegerTy(1))  return V;
    if (ty->isIntegerTy(32)) return Builder.CreateICmpNE(V, Builder.getInt32(0));
    if (ty->isIntegerTy(8))
        return Builder.CreateICmpNE(V, llvm::ConstantInt::get(Int8Ty(), 0));
    if (ty->isDoubleTy())
        return Builder.CreateFCmpONE(V, llvm::ConstantFP::get(DoubleTy(), 0.0));
    // Tensor*: first element
    auto* dp    = Builder.CreateLoad(DblPtrTy(), Builder.CreateStructGEP(TensorTy, V, 0));
    auto* first = Builder.CreateLoad(DoubleTy(), dp);
    return Builder.CreateFCmpONE(first, llvm::ConstantFP::get(DoubleTy(), 0.0));
}

static llvm::Value* coerceValue(llvm::Value* val, llvm::Type* targetTy,
                                 const std::string& vn = "") {
    if (val->getType() == targetTy) return val;

    // Helper: extract the first element from a Tensor* as double
    auto extractScalarFromTensor = [&](llvm::Value* tensorPtr) -> llvm::Value* {
        auto* dp = Builder.CreateLoad(DblPtrTy(),
                       Builder.CreateStructGEP(TensorTy, tensorPtr, 0));
        return Builder.CreateLoad(DoubleTy(), dp);
    };

    if (targetTy == Int32Ty()) {
        if (val->getType()->isDoubleTy())   return Builder.CreateFPToSI(val, Int32Ty());
        if (val->getType()->isIntegerTy(8)) return Builder.CreateSExt(val, Int32Ty());
        // Tensor* ?? Int: extract first element and truncate to int
        if (val->getType()->isPointerTy()) {
            auto* fv = extractScalarFromTensor(val);
            return Builder.CreateFPToSI(fv, Int32Ty());
        }
        std::cerr << "\nerror: Cannot convert to Int '" << vn << "'\n"; return nullptr;
    }
    if (targetTy == DoubleTy()) {
        if (val->getType()->isIntegerTy(32)) return Builder.CreateSIToFP(val, DoubleTy());
        if (val->getType()->isIntegerTy(8))
            return Builder.CreateSIToFP(Builder.CreateSExt(val, Int32Ty()), DoubleTy());
        // Tensor* ?? Float: extract first element
        if (val->getType()->isPointerTy()) {
            return extractScalarFromTensor(val);
        }
        std::cerr << "\nerror: Cannot convert to Float '" << vn << "'\n"; return nullptr;
    }
    if (targetTy == Int8Ty()) {
        if (val->getType()->isIntegerTy(32))
            return Builder.CreateTrunc(val, Int8Ty());
        if (val->getType()->isDoubleTy())
            return Builder.CreateTrunc(Builder.CreateFPToSI(val, Int32Ty()), Int8Ty());
        if (val->getType()->isPointerTy()) {
            auto* fv = extractScalarFromTensor(val);
            return Builder.CreateTrunc(Builder.CreateFPToSI(fv, Int32Ty()), Int8Ty());
        }
        std::cerr << "\nerror: Type mismatch for Char '" << vn << "'\n"; return nullptr;
    }
    if (targetTy->isPointerTy()) return castToTensor(val);
    std::cerr << "\nerror: Unknown target type for '" << vn << "'\n"; return nullptr;
}

static llvm::Value* performAssign(llvm::Value* val, llvm::AllocaInst* dest,
                                   const std::string& vn, bool source_is_borrow = true) {
    llvm::Type* destTy = dest->getAllocatedType();
    if (destTy->isPointerTy()) {
        val = castToTensor(val);
        // Refcount: decref old; incref new ONLY if source was a borrow
        // (VariableExpr/IndexExpr/FieldAccess). If source produced a fresh
        // tensor (CallExpr/BinaryExpr/etc), it already gave us +1 gift.
        auto* old = Builder.CreateLoad(TPtrTy(), dest);
        auto* decF = getOrDeclare("tensor_decref", TPtrTy(), {TPtrTy()});
        Builder.CreateCall(decF, {old});
        if (source_is_borrow) {
            auto* incF = getOrDeclare("tensor_incref", TPtrTy(), {TPtrTy()});
            Builder.CreateCall(incF, {val});
        }
        auto* markF = getOrDeclare("tensor_mark_var", TPtrTy(), {TPtrTy()});
        val = Builder.CreateCall(markF, {val});
        Builder.CreateStore(val, dest);
        return val;
    }
    val = coerceValue(val, destTy, vn);
    if (!val) return nullptr;
    Builder.CreateStore(val, dest);
    return val;
}

static llvm::BasicBlock* makeDeadBlock(const std::string& nm = "dead") {
    auto* fn = Builder.GetInsertBlock()->getParent();
    auto* bb = llvm::BasicBlock::Create(TheContext, nm, fn);
    Builder.SetInsertPoint(bb);
    return bb;
}

// =====================================================================
// OWNERSHIP CLASSIFICATION (single source of truth)
//
// Under the pure-refcount contract, EVERY runtime function reachable as
// an expression returns a +1 owned reference. Therefore an expression
// node "produces fresh" (caller owns +1 and must release it) unless it
// is a plain variable read, which is a borrow of the alloca's reference.
//
// Fresh producers: CallExpr, BinaryExpr, UnaryExpr, TernaryExpr,
//   FusedExpr, GradExpr, IndexExpr (tensor_index copies), SliceExpr,
//   FieldAccess (dream_struct_get increfs), StringExpr (dream_str_create),
//   ArrayLit/TupleLit, LoadExpr.
// Borrows: VariableExpr only.
// =====================================================================
static bool producesFreshTensor(ASTNode* node) {
    if (!node) return false;
    return dynamic_cast<CallExprAST*>(node)        != nullptr ||
           dynamic_cast<BinaryExprAST*>(node)      != nullptr ||
           dynamic_cast<UnaryExprAST*>(node)       != nullptr ||
           dynamic_cast<TernaryExprAST*>(node)     != nullptr ||
           dynamic_cast<FusedExprAST*>(node)       != nullptr ||
           dynamic_cast<GradExprAST*>(node)        != nullptr ||
           dynamic_cast<IndexExprAST*>(node)       != nullptr ||
           dynamic_cast<SliceExprAST*>(node)       != nullptr ||
           dynamic_cast<FieldAccessExprAST*>(node) != nullptr ||
           dynamic_cast<StringExprAST*>(node)      != nullptr ||
           dynamic_cast<ArrayLitExprAST*>(node)    != nullptr ||
           dynamic_cast<TupleLitExprAST*>(node)    != nullptr ||
           dynamic_cast<LoadExprAST*>(node)        != nullptr;
}

static bool isBorrowExpr(ASTNode* node) {
    return dynamic_cast<VariableExprAST*>(node) != nullptr;
}

// Decref the result of an expression-statement if it's a tensor pointer.
// Used after evaluating things like `param_update(W1, ...)` at statement
// level: the call returns a +1 ref to the tensor, but no let-binding
// captures it, so without this decref the tensor leaks.
static void decrefIfTensor(llvm::Value* v, ASTNode* node) {
    if (!v) return;
    if (!v->getType()->isPointerTy()) return;
    if (!producesFreshTensor(node)) return;
    auto* decF = getOrDeclare("tensor_decref", TPtrTy(), {TPtrTy()});
    Builder.CreateCall(decF, {v});
}

// Release a sub-expression temporary after its value has been consumed
// (copied into a result tensor, linked as an owned graph edge, or read
// as a scalar). `origV` is the value BEFORE castToTensor, `tensorV` the
// value actually passed to the runtime. A temp needs releasing when:
//   - the AST node produced a fresh +1 tensor, OR
//   - a non-pointer scalar was boxed via create_temp_scalar (castToTensor).
static void decrefTemp(llvm::Value* tensorV, llvm::Value* origV, ASTNode* node) {
    if (!tensorV || !tensorV->getType()->isPointerTy()) return;
    bool boxed = origV && !origV->getType()->isPointerTy();
    if (!boxed && !producesFreshTensor(node)) return;
    auto* decF = getOrDeclare("tensor_decref", TPtrTy(), {TPtrTy()});
    Builder.CreateCall(decF, {tensorV});
}

static void emitScopeCleanup() {
    auto* decF = getOrDeclare("tensor_decref", TPtrTy(), {TPtrTy()});
    for (auto& name : CurrentFnLocals) {
        auto it = NamedValues.find(name);
        if (it == NamedValues.end()) continue;
        auto* alloca = llvm::dyn_cast<llvm::AllocaInst>(it->second);
        if (!alloca || !alloca->getAllocatedType()->isPointerTy()) continue;
        auto* val = Builder.CreateLoad(TPtrTy(), alloca);
        Builder.CreateCall(decF, {val});
    }
}

const std::map<std::string,std::string>& builtinRemap() {
    static const std::map<std::string,std::string> m = {
        {"exp","dream_exp"},{"log","dream_log"},{"sqrt","dream_sqrt"},{"abs","dream_abs"},
        {"sigmoid","dream_sigmoid"},{"tanh_t","dream_tanh"},{"tanh","dream_tanh"},
        {"softmax","dream_softmax"},
        {"transpose","dream_transpose"},{"concat","dream_concat"},{"clip","dream_clip"},
        {"randn","dream_randn"},{"cross_entropy","dream_cross_entropy"},
        {"argmax","dream_argmax"},
        {"save_matrix","dream_save_matrix"},
        {"augment","dream_augment_x"},{"augment_y","dream_augment_y"},
        {"augment_count","dream_augment_count"},
        {"gpu_selftest","gpu_selftest"},
        {"gpu_warmup","dream_gpu_warmup"},
        {"sum_dim","dream_sum_dim"},{"mean_dim","dream_mean_dim"},{"max_dim","dream_max_dim"},
        {"attention_causal","dream_attention_causal"},{"print_shape","dream_print_shape"},
        {"l2_loss","dream_l2_loss"},{"pow","dream_pow"},
        {"logical_not","dream_logical_not"},
        {"dropout","dream_dropout"},{"layer_norm","dream_layer_norm"},
        {"max_val","dream_max_val"},{"min_val","dream_min_val"},
        {"min","dream_minimum"},{"max","dream_maximum"},
        {"norm","dream_norm"},{"linspace","dream_linspace"},
        {"arange","dream_arange"},{"flatten","dream_flatten"},
        {"clip_grad","dream_clip_grad"},{"cumsum","dream_cumsum"},
        {"var","dream_var"},{"std_dev","dream_std"},
        {"floor","dream_floor"},{"ceil","dream_ceil"},{"round","dream_round"},
        {"sgd_step","dream_sgd_step"},{"adam_step","dream_adam_step"},
        {"time","dream_time"},
        {"assert_fn","dream_assert_fn"},
        {"len","dream_len"},{"shape_of","dream_shape_of"},
        {"fill","dream_fill"},{"copy","dream_copy"},
        {"slice","dream_slice"},{"eye","dream_eye"},
        {"sin","dream_sin"},{"cos","dream_cos"},{"tan","dream_tan"},
        {"atan","dream_atan"},{"atan2","dream_atan2"},
        {"where","dream_where"},
        {"element_min","dream_element_min"},{"element_max","dream_element_max"},
        {"str_concat","dream_str_concat"},{"str_len","dream_str_len"},
        {"to_string","dream_to_string"},
        {"int_cast","dream_int_cast"},{"float_cast","dream_float_cast"},
        {"sum_axis","dream_sum_axis"},{"mean_axis","dream_mean_axis"},
        {"gc","dream_gc"},{"unroot","dream_unroot"},
        {"create_scalar","dream_scalar"},
        {"call","dream_call"},{"struct_new","dream_struct_new"},
        {"map","dream_map"},{"filter","dream_filter"},{"reduce","dream_reduce"},
        {"load_image","dream_load_image"},{"save_bmp","dream_save_bmp"},
        {"gen_digits","dream_gen_digits"},{"predict","dream_predict"},
        {"accuracy","dream_accuracy"},{"print_digit","dream_print_digit"},
        {"get_row","dream_get_row"},
        {"conv2d","dream_conv2d"},{"maxpool2d","dream_maxpool2d"},
        {"resize","dream_resize"},{"batch_norm","dream_batch_norm"},
        {"gen_shapes","dream_gen_shapes"},{"print_image","dream_print_image_16"},
        {"flatten_2d","dream_flatten_2d"},
        {"pool_stats","dream_pool_stats"},
        {"transpose_view","dream_transpose_view"},
        {"slice_view","dream_slice_view"},
        {"inplace_add_","dream_inplace_add_"},
        {"inplace_sub_","dream_inplace_sub_"},
        {"inplace_mul_","dream_inplace_mul_"},
        {"no_grad_assign","dream_no_grad_assign"},
        {"param_update","dream_param_update"},
        {"bmm","dream_bmm"},
        {"avgpool2d","dream_avgpool2d"},
        {"batch_iter","dream_batch_iter"},
        {"has_next","dream_has_next"},
        {"next_batch","dream_next_batch"},
        {"reset_iter","dream_reset_iter"},
        {"iter_len","dream_iter_len"},
        {"load_bmp_list","dream_load_bmp_list"},
        {"to_float32","dream_to_float32"},
        {"to_float64","dream_to_float64"},
        {"grad_scale","dream_grad_scale"},
        {"grad_accum_update","dream_grad_accum_update"},
        {"loss_scale","dream_loss_scale"},
        {"async_iter","dream_async_iter"},
        {"async_has_next","dream_async_has_next"},
        {"async_next","dream_async_next"},
        {"async_reset","dream_async_reset"},
        {"async_stop","dream_async_stop"},
        {"enable_fp16","dream_enable_fp16"},
        {"disable_fp16","dream_disable_fp16"},
        {"enable_checkpointing","dream_enable_checkpointing"},
        {"disable_checkpointing","dream_disable_checkpointing"},
        {"residual_add","dream_residual_add"},
        {"kaiming_init","dream_kaiming_init"},
        {"xavier_init","dream_xavier_init"},
        {"leaky_relu","dream_leaky_relu"},
        {"gelu","dream_gelu"},
        {"batchnorm2d","dream_batchnorm2d"},
        {"disk_loader","dream_disk_loader"},
        {"disk_has_next","dream_disk_has_next"},
        {"disk_next","dream_disk_next"},
        {"disk_reset","dream_disk_reset"},
        {"disk_stop","dream_disk_stop"},
        {"tensor_stats","dream_tensor_stats"},
        {"check_nan","dream_check_nan"},
        {"dump_graph","dream_dump_graph"},
        {"grad_histogram","dream_grad_histogram"},
        {"detach","dream_detach"},
        {"rnn_cell","dream_rnn_cell"},
        {"lstm_cell","dream_lstm_cell"},
        {"attention","dream_attention"},
        {"mse","dream_mse"},
        {"bce_loss","dream_bce_loss"},
        {"huber_loss","dream_huber_loss"},
        {"cosine_sim","dream_cosine_sim"},
        {"random_flip_h","dream_random_flip_h"},
        {"normalize","dream_normalize"},
        {"random_noise","dream_random_noise"},
        {"random_crop","dream_random_crop"},
        {"adamw_update","dream_adamw_update"},
        {"adamw_reset","dream_adamw_reset"},
        {"lr_cosine","dream_lr_cosine"},
        {"lr_step","dream_lr_step"},
        {"lr_exp","dream_lr_exp"},
        {"lr_linear","dream_lr_linear"},
        {"global_clip_2","dream_global_clip_2"},
        {"global_clip_3","dream_global_clip_3"},
        {"global_clip_4","dream_global_clip_4"},
        {"silu","dream_silu"},
        {"mish","dream_mish"},
        {"elu","dream_elu"},
        {"embedding","dream_embedding"},
        {"embedding_init","dream_embedding_init"},
        {"save_model","dream_save_model"},
        {"load_model","dream_load_model"},
        {"count_params","dream_count_params"},
        {"logger_new","dream_logger_new"},
        {"logger_log","dream_logger_log"},
        {"logger_epoch_end","dream_logger_epoch_end"},
        {"one_hot","dream_one_hot"},
        {"shuffle_rows","dream_shuffle_rows"},
        {"conv2d_padded","dream_conv2d_padded"},
        {"global_avg_pool","dream_global_avg_pool"},
        {"pad2d","dream_pad2d"},
        {"reshape_nd","dream_reshape_nd"},
        {"squeeze","dream_squeeze"},
        {"unsqueeze","dream_unsqueeze"},
        {"stack","dream_stack"},
        {"cat_0","dream_cat_0"},
        {"repeat","dream_repeat"},
        {"gather","dream_gather"},
        {"topk","dream_topk"},
        {"clamp","dream_clamp"},
        {"masked_fill","dream_masked_fill"},
        {"topk_accuracy","dream_topk_accuracy"},
        {"confusion_matrix","dream_confusion_matrix"},
        {"classification_report","dream_classification_report"},
        {"early_stop_new","dream_early_stop_new"},
        {"early_stop_check","dream_early_stop_check"},
        {"early_stop_reset","dream_early_stop_reset"},
        {"ema_update","dream_ema_update"},
        {"moving_avg","dream_moving_avg"},
        {"linear","dream_linear"},
        {"linear_relu","dream_linear_relu"},
        {"tic","dream_tic"},
        {"toc","dream_toc"},
        {"op_stats","dream_op_stats"},
        {"set_seed","dream_set_seed"},
        {"log_softmax","dream_log_softmax"},
        {"nll_loss","dream_nll_loss"},
        {"has_nan","dream_has_nan"},
        {"nan_to_num","dream_nan_to_num"},
        {"clamp","dream_clamp"},
        {"read_csv","dream_read_csv"},
        {"write_csv","dream_write_csv"},
        {"ckpt_new","dream_ckpt_new"},
        {"ckpt_add","dream_ckpt_add"},
        {"ckpt_set_meta","dream_ckpt_set_meta"},
        {"ckpt_save","dream_ckpt_save"},
        {"ckpt_load","dream_ckpt_load"},
        {"ckpt_get","dream_ckpt_get"},
        {"ckpt_epoch","dream_ckpt_epoch"},
        {"ckpt_lr","dream_ckpt_lr"},
        {"load_safetensors","dream_load_safetensors"},
        {"st_get","dream_st_get"},
        {"st_list","dream_st_list"},
        {"save_safetensors","dream_save_safetensors"},
        {"export_c_header","dream_export_c_header"},
        {"export_c_header_int8","dream_export_c_header_int8"},
        {"hessian_diag","dream_hessian_diag"},
        {"grad_check","dream_grad_check"},
        {"batch_dot","dream_batch_dot"},
        {"batch_norm_l2","dream_batch_norm_l2"},
        {"batch_softmax","dream_batch_softmax"},
        {"all_reduce_avg","dream_all_reduce_avg"},
        {"data_shard","dream_data_shard"},
        {"sync_gradients","dream_sync_gradients"},
        {"broadcast","dream_broadcast"},
        {"barrier","dream_barrier"},
        {"dist_info","dream_dist_info"},
        {"dist_init","dream_dist_init"},
        {"all_reduce_sum","dream_all_reduce_sum"},
        {"dist_selftest","dist_selftest"},
        {"comm_selftest","comm_selftest"},
        {"tp_matmul_column","dream_tp_matmul_column"},
        {"tp_matmul_row","dream_tp_matmul_row"},
        {"tensor_parallel_selftest","tensor_parallel_selftest"},
        {"pipeline_parallel_selftest","pipeline_parallel_selftest"},
        {"checkpoint_selftest","checkpoint_selftest"},
        {"microbatch_selftest","microbatch_selftest"},
        {"bf16_selftest","bf16_selftest"},
        {"moe_selftest","moe_selftest"},
        {"dataloader_selftest","dataloader_selftest"},
        {"flash_attention","dream_flash_attention"},
        {"flash_attention_selftest","flash_attention_selftest"},
        {"zero_adamw_selftest","zero_adamw_selftest"},
        {"scaling_report","dream_scaling_report"},
        {"ring_optimality_selftest","ring_optimality_selftest"},
        {"speedup_report","dream_speedup_report"},
        {"speedup_selftest","speedup_selftest"},
        {"quantize_calibrate","dream_quantize_calibrate"},
        {"quantize","dream_quantize"},
        {"dequantize","dream_dequantize"},
        {"fake_quantize","dream_fake_quantize"},
        {"quant_error","dream_quant_error"},
        {"prune_magnitude","dream_prune_magnitude"},
        {"prune_structured","dream_prune_structured"},
        {"prune_mask","dream_prune_mask"},
        {"apply_mask","dream_apply_mask"},
        {"sparsity","dream_sparsity"},
        {"profile_start","dream_profile_start"},
        {"profile_mark","dream_profile_mark"},
        {"profile_end_op","dream_profile_end_op"},
        {"profile_report","dream_profile_report"},
        {"profile_op","dream_profile_op"},
        {"model_size","dream_model_size"},
        {"export_flat","dream_export_flat"},
        {"distill_loss","dream_distill_loss"},
        {"lr_find_start","dream_lr_find_start"},
        {"lr_find_get","dream_lr_find_get"},
        {"lr_find_step","dream_lr_find_step"},
        {"watchdog_new","dream_watchdog_new"},
        {"watchdog_check","dream_watchdog_check"},
        {"watchdog_summary","dream_watchdog_summary"},
    };
    return m;
}

IntExprAST::IntExprAST(int Val) : Val(Val) {}
llvm::Value* IntExprAST::codegen() { return llvm::ConstantInt::get(TheContext, llvm::APInt(32,Val,true)); }
bool        IntExprAST::isFuseable() const { return true; }
std::string IntExprAST::genFusedCode(const std::vector<std::string>&) const { return std::to_string(Val)+".0f"; }
std::string IntExprAST::genFusedGradCode(const std::string&,const std::vector<std::string>&) const { return "0.0f"; }

FloatExprAST::FloatExprAST(double Val) : Val(Val) {}
llvm::Value* FloatExprAST::codegen() { return llvm::ConstantFP::get(TheContext,llvm::APFloat(Val)); }
bool        FloatExprAST::isFuseable() const { return true; }
std::string FloatExprAST::genFusedCode(const std::vector<std::string>&) const { return std::to_string(Val)+"f"; }
std::string FloatExprAST::genFusedGradCode(const std::string&,const std::vector<std::string>&) const { return "0.0f"; }

CharExprAST::CharExprAST(char Val) : Val(Val) {}
llvm::Value* CharExprAST::codegen() { return llvm::ConstantInt::get(TheContext,llvm::APInt(8,Val,true)); }
bool        CharExprAST::isFuseable() const { return true; }
std::string CharExprAST::genFusedCode(const std::vector<std::string>&) const { return std::to_string((int)Val)+".0f"; }
std::string CharExprAST::genFusedGradCode(const std::string&,const std::vector<std::string>&) const { return "0.0f"; }
StringExprAST::StringExprAST(const std::string& Val) : Val(Val) {}
llvm::Value* StringExprAST::codegen() {
    auto* strPtr = Builder.CreateGlobalStringPtr(Val, ".str");
    auto* F = getOrDeclare("dream_str_create", TPtrTy(), {CharPtrTy()});
    return Builder.CreateCall(F, {strPtr}, "strtmp");
}
std::string StringExprAST::genOpenCL() { return "\"" + Val + "\""; }
VariableExprAST::VariableExprAST(const std::string& Name) : Name(Name) {}
const std::string& VariableExprAST::getName() const { return Name; }
llvm::Value* VariableExprAST::codegen() {
    auto it = NamedValues.find(Name);
    if (it == NamedValues.end()) {
        // Find the closest-spelled name in the current scope.
        std::string best; int bestD = 3;
        for (const auto& kv : NamedValues) {
            int d = levBounded(Name, kv.first, 2);
            if (d < bestD) { bestD = d; best = kv.first; }
        }
        diagAtLine("undefined variable '" + Name + "'", Name, Line, best);
        return nullptr;
    }
    auto* alloca = llvm::cast<llvm::AllocaInst>(it->second);
    return Builder.CreateLoad(alloca->getAllocatedType(), alloca, Name);
}
bool VariableExprAST::isFuseable() const { return true; }
void VariableExprAST::collectVars(std::vector<std::string>& vars) const {
    if (std::find(vars.begin(),vars.end(),Name)==vars.end()) vars.push_back(Name);
}
std::string VariableExprAST::genFusedCode(const std::vector<std::string>& vars) const {
    auto it=std::find(vars.begin(),vars.end(),Name);
    int idx=std::distance(vars.begin(),it);
    return "v"+std::to_string(idx)+"[i%sz"+std::to_string(idx)+"]";
}
std::string VariableExprAST::genFusedGradCode(const std::string& tv,const std::vector<std::string>&) const {
    return (Name==tv)?"1.0f":"0.0f";
}

UnaryExprAST::UnaryExprAST(std::string Op, std::unique_ptr<ASTNode> Operand)
    : Op(std::move(Op)), Operand(std::move(Operand)) {}

llvm::Value* UnaryExprAST::codegen() {
    auto* V = Operand->codegen(); if (!V) return nullptr;
    if (Op == "!") {
        if (V->getType()->isIntegerTy(32))
            return Builder.CreateZExt(Builder.CreateICmpEQ(V,Builder.getInt32(0)),Int32Ty());
        if (V->getType()->isDoubleTy())
            return Builder.CreateUIToFP(Builder.CreateFCmpOEQ(V,llvm::ConstantFP::get(DoubleTy(),0.0)),DoubleTy());
        auto* origV = V;
        V = castToTensor(V);
        auto* notRes = Builder.CreateCall(getOrDeclare("dream_logical_not",TPtrTy(),{TPtrTy()}),{V});
        decrefTemp(V, origV, Operand.get());
        return notRes;
    }
    if (Op == "-") {
        if (V->getType()->isIntegerTy(32)) return Builder.CreateNeg(V,"negtmp");
        if (V->getType()->isDoubleTy())    return Builder.CreateFNeg(V,"fnegtmp");
        auto* origV = V;
        V = castToTensor(V);
        auto* negOne = castToTensor(llvm::ConstantFP::get(DoubleTy(),-1.0));
        auto* negRes = Builder.CreateCall(getOrDeclare("tensor_mul",TPtrTy(),{TPtrTy(),TPtrTy()}),{V,negOne});
        decrefTemp(V, origV, Operand.get());
        auto* decF = getOrDeclare("tensor_decref", TPtrTy(), {TPtrTy()});
        Builder.CreateCall(decF, {negOne});   // boxed -1.0 temp
        return negRes;
    }
    std::cerr << "\nerror: Unknown unary op '" << Op << "'\n"; return nullptr;
}

// no_grad { ... } codegen: disable graph building in the block, restore after.
llvm::Value* NoGradStmtAST::codegen() {
    auto* setF = getOrDeclare("dream_set_autograd", TPtrTy(), {TPtrTy()});
    auto* getF = getOrDeclare("dream_autograd_enabled", TPtrTy(), {});
    auto* decF = getOrDeclare("tensor_decref", TPtrTy(), {TPtrTy()});

    // Save the current state (returned scalar tensor), then disable autograd.
    auto* prev = Builder.CreateCall(getF, {});
    auto* zero = castToTensor(llvm::ConstantFP::get(DoubleTy(), 0.0));
    auto* r1 = Builder.CreateCall(setF, {zero});
    Builder.CreateCall(decF, {r1});
    Builder.CreateCall(decF, {zero});

    for (auto& stmt : Body)
        if (!stmt->codegen()) return nullptr;

    // Restore the previous autograd state.
    auto* r2 = Builder.CreateCall(setF, {prev});
    Builder.CreateCall(decF, {r2});
    Builder.CreateCall(decF, {prev});
    return llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(TPtrTy()));
}

// Codegen for `expr as f32|f64|exact`.
llvm::Value* PrecisionExprAST::codegen() {
    // exact: if inner is an exact-capable reduction, set the hint then codegen.
    if (Prec == DreamPrec::Exact) {
        if (auto* call = dynamic_cast<CallExprAST*>(Inner.get()))
            call->PrecHint = 3;
        return Inner->codegen();
    }
    // f32/f64: evaluate normally, then set the result storage-precision flag.
    // The compute store is always double (correctness); f32 only affects storage/export,
    // handled by runtime dream_set_precision (a safe no-op if absent).
    auto* v = Inner->codegen();
    if (!v) return nullptr;
    if (!v->getType()->isPointerTy()) return v;   // scalar int/float, no tensor to annotate
    int code = (Prec == DreamPrec::F32) ? 1 : 0;
    auto* setF = getOrDeclare("dream_set_precision", TPtrTy(), {TPtrTy(), Int32Ty()});
    return Builder.CreateCall(setF, {v, Builder.getInt32(code)});
}

std::string UnaryExprAST::genOpenCL() { return "("+Op+Operand->genOpenCL()+")"; }
bool UnaryExprAST::isFuseable() const { return Operand->isFuseable(); }
std::string UnaryExprAST::genFusedCode(const std::vector<std::string>& vars) const {
    if (Op=="!") return "("+Operand->genFusedCode(vars)+"==0.0f?1.0f:0.0f)";
    if (Op=="-") return "(-"+Operand->genFusedCode(vars)+")";
    return "0.0f";
}
std::string UnaryExprAST::genFusedGradCode(const std::string& tv,const std::vector<std::string>& vars) const {
    if (Op=="-") return "(-(" +Operand->genFusedGradCode(tv,vars)+"))";
    return "0.0f";
}

TernaryExprAST::TernaryExprAST(std::unique_ptr<ASTNode> C,
                                std::unique_ptr<ASTNode> T,
                                std::unique_ptr<ASTNode> E)
    : Cond(std::move(C)), Then(std::move(T)), Else(std::move(E)) {}

llvm::Value* TernaryExprAST::codegen() {
    auto* condV = Cond->codegen(); if (!condV) return nullptr;
    auto* fn    = Builder.GetInsertBlock()->getParent();
    auto* thenBB  = llvm::BasicBlock::Create(TheContext,"tern.then",fn);
    auto* elseBB  = llvm::BasicBlock::Create(TheContext,"tern.else");
    auto* mergeBB = llvm::BasicBlock::Create(TheContext,"tern.merge");
    auto* condBool = buildCondBool(condV);
    decrefTemp(condV, condV, Cond.get());
    Builder.CreateCondBr(condBool,thenBB,elseBB);

    auto* incF = getOrDeclare("tensor_incref", TPtrTy(), {TPtrTy()});
    Builder.SetInsertPoint(thenBB);
    auto* thenV = Then->codegen(); if (!thenV) return nullptr;
    thenV = castToTensor(thenV);
    // Normalize both branches to +1 so the merged phi is uniformly fresh.
    if (isBorrowExpr(Then.get())) Builder.CreateCall(incF, {thenV});
    auto* thenBlock = Builder.GetInsertBlock();
    Builder.CreateBr(mergeBB);

    fn->insert(fn->end(),elseBB);
    Builder.SetInsertPoint(elseBB);
    auto* elseV = Else->codegen(); if (!elseV) return nullptr;
    elseV = castToTensor(elseV);
    if (isBorrowExpr(Else.get())) Builder.CreateCall(incF, {elseV});
    auto* elseBlock = Builder.GetInsertBlock();
    Builder.CreateBr(mergeBB);

    fn->insert(fn->end(),mergeBB);
    Builder.SetInsertPoint(mergeBB);
    auto* phi = Builder.CreatePHI(TPtrTy(),2,"ternphi");
    phi->addIncoming(thenV,thenBlock);
    phi->addIncoming(elseV,elseBlock);
    return phi;
}
std::string TernaryExprAST::genOpenCL() {
    return "("+Cond->genOpenCL()+"?"+Then->genOpenCL()+":"+Else->genOpenCL()+")";
}
bool TernaryExprAST::isFuseable() const {
    return Cond->isFuseable()&&Then->isFuseable()&&Else->isFuseable();
}
std::string TernaryExprAST::genFusedCode(const std::vector<std::string>& v) const {
    return "("+Cond->genFusedCode(v)+"?"+Then->genFusedCode(v)+":"+Else->genFusedCode(v)+")";
}
std::string TernaryExprAST::genFusedGradCode(const std::string& tv,const std::vector<std::string>& v) const {
    return "("+Cond->genFusedCode(v)+"?"+Then->genFusedGradCode(tv,v)+":"+Else->genFusedGradCode(tv,v)+")";
}

CallExprAST::CallExprAST(const std::string& Callee,
                          std::vector<std::unique_ptr<ASTNode>> Args)
    : Callee(Callee), Args(std::move(Args)) {}

llvm::Value* CallExprAST::codegen() {
    // Compile-time guardrails for augment(X, Y, when_below, up_to).
    // These fire at build time so mistakes are caught before any run.
    if ((Callee == "augment" || Callee == "augment_y") && Args.size() >= 4) {
        auto litVal = [](ASTNode* n, double& out) -> bool {
            if (auto* f = dynamic_cast<FloatExprAST*>(n)) { out = f->getVal(); return true; }
            if (auto* i = dynamic_cast<IntExprAST*>(n))   { out = (double)i->getVal(); return true; }
            return false;
        };
        double wb, up;
        bool haveWb = litVal(Args[2].get(), wb);
        bool haveUp = litVal(Args[3].get(), up);
        if (haveWb && wb < 1.0) {
            std::cerr << "\nwarning: augment when_below=" << wb
                      << " is too small -- with fewer than one real sample there is\n"
                      << "         nothing to interpolate from, so this would produce garbage.\n"
                      << "         Use a threshold of at least a few dozen real samples.\n";
        }
        if (haveWb && haveUp && up <= wb) {
            std::cerr << "\nwarning: augment up_to=" << up << " <= when_below=" << wb
                      << " -- augmentation would never add rows (target below trigger).\n";
        }
        if (haveUp && haveWb && up > wb * 100.0) {
            std::cerr << "\nwarning: augment up_to=" << up << " is over 100x when_below="
                      << wb << " -- synthesizing that many rows from so few real ones\n"
                      << "         risks a dataset dominated by interpolated points.\n";
        }
    }

    // Check if this is a struct constructor: Point(x, y)
    extern std::map<std::string, std::vector<std::string>> g_struct_defs;
    auto sIt = g_struct_defs.find(Callee);
    if (sIt != g_struct_defs.end()) {
        const auto& fields = sIt->second;
        if (Args.size() != fields.size()) {
            std::cerr << "\nerror: struct '" << Callee << "' expects " << fields.size()
                      << " fields, got " << Args.size() << "\n";
            return nullptr;
        }
        // Build field name string "x,y,z"
        std::string fieldStr;
        for (size_t i = 0; i < fields.size(); ++i) {
            if (i > 0) fieldStr += ",";
            fieldStr += fields[i];
        }
        // Create the struct tensor
        auto* nfields = Builder.getInt32((int)fields.size());
        auto* F = getOrDeclare("dream_struct_create", TPtrTy(),
                               {CharPtrTy(), Int32Ty()});
        auto* structT = Builder.CreateCall(F,
            {Builder.CreateGlobalStringPtr(fieldStr, ".sfields"), nfields}, "struct_tmp");
        // Set each field
        auto* setF = getOrDeclare("dream_struct_set", TPtrTy(),
                                   {TPtrTy(), CharPtrTy(), TPtrTy()});
        for (size_t i = 0; i < Args.size(); ++i) {
            auto* ov = Args[i]->codegen(); if (!ov) return nullptr;
            auto* v = castToTensor(ov);
            Builder.CreateCall(setF, {structT,
                Builder.CreateGlobalStringPtr(fields[i], ".sf"), v});
            // struct_set took its own +1 on the member; release our temp.
            decrefTemp(v, ov, Args[i].get());
        }
        return structT;
    }

    const auto& rm = builtinRemap();
    auto rmIt = rm.find(Callee);
    std::string actualName = (rmIt != rm.end()) ? rmIt->second : Callee;

    // exact precision: redirect reduction calls to compensated-sum variants.
    // Only listed reductions have exact versions; `as exact` on others is a no-op.
    if (PrecHint == 3) {
        static const std::map<std::string,std::string> exactMap = {
            {"sum","dream_sum_exact"}, {"dream_sum","dream_sum_exact"},
            {"mean","dream_mean_exact"}, {"dream_mean","dream_mean_exact"},
        };
        auto eIt = exactMap.find(actualName);
        if (eIt == exactMap.end()) eIt = exactMap.find(Callee);
        if (eIt != exactMap.end()) actualName = eIt->second;
    }

    // Compile-time undefined-function check.
    // Allowed: user functions, module-defined/declared, runtime manifest hits.
    // Empty manifest means scan failed; fall back to old behavior (no false positives).
    if (!g_runtime_syms.empty() &&
        !g_user_fns.count(Callee) &&
        !g_runtime_syms.count(actualName) &&
        !TheModule->getFunction(actualName)) {
        diagUnknownFn(Callee, Line);
        return nullptr;
    }

    std::vector<llvm::Value*> argVals, argOrig;
    std::vector<llvm::Type*>  argTys;
    for (size_t i = 0; i < Args.size(); ++i) {
        auto* ov = Args[i]->codegen();
        if (!ov) { std::cerr << "\nerror: arg " << i+1 << " failed in call to '" << Callee << "'\n"; return nullptr; }
        auto* v = castToTensor(ov);
        argVals.push_back(v); argOrig.push_back(ov); argTys.push_back(v->getType());
    }

    // Arguments are passed as borrows: the callee never releases them.
    // If the callee stores an argument anywhere lasting (graph edge,
    // struct field, checkpoint), it takes its own +1. So once the call
    // returns, any fresh/boxed argument temporary is ours to release.
    auto* F = getOrDeclare(actualName, TPtrTy(), argTys);
    auto* callRes = Builder.CreateCall(F, argVals, "calltmp");
    for (size_t i = 0; i < argVals.size(); ++i)
        decrefTemp(argVals[i], argOrig[i], Args[i].get());
    return callRes;
}

static llvm::Value* generateTensorLoop(const std::string& op, llvm::Value* L, llvm::Value* R) {
    auto* fn = Builder.GetInsertBlock()->getParent();
    auto ld32 = [&](llvm::Value* p,int i){ return Builder.CreateLoad(Int32Ty(),Builder.CreateStructGEP(TensorTy,p,i)); };
    auto maxDim=[&](llvm::Value* a,llvm::Value* b){ return Builder.CreateSelect(Builder.CreateICmpSGT(a,b),a,b); };
    auto* allocF=getOrDeclare("alloc_tensor",TPtrTy(),{Int32Ty(),Int32Ty(),Int32Ty(),Int32Ty()});
    auto* res=Builder.CreateCall(allocF,{maxDim(ld32(L,3),ld32(R,3)),maxDim(ld32(L,4),ld32(R,4)),maxDim(ld32(L,5),ld32(R,5)),maxDim(ld32(L,6),ld32(R,6))},"res");
    auto* lData=Builder.CreateLoad(DblPtrTy(),Builder.CreateStructGEP(TensorTy,L,0));
    auto* rData=Builder.CreateLoad(DblPtrTy(),Builder.CreateStructGEP(TensorTy,R,0));
    auto* resData=Builder.CreateLoad(DblPtrTy(),Builder.CreateStructGEP(TensorTy,res,0));
    auto* resSize=Builder.CreateLoad(Int32Ty(),Builder.CreateStructGEP(TensorTy,res,2));
    auto* lSize=ld32(L,2); auto* rSize=ld32(R,2);
    auto* ia=Builder.CreateAlloca(Int32Ty(),nullptr,"li");
    Builder.CreateStore(Builder.getInt32(0),ia);
    auto*c=llvm::BasicBlock::Create(TheContext,"tl.c",fn);
    auto*b=llvm::BasicBlock::Create(TheContext,"tl.b",fn);
    auto*e=llvm::BasicBlock::Create(TheContext,"tl.e",fn);
    Builder.CreateBr(c); Builder.SetInsertPoint(c);
    auto* idx=Builder.CreateLoad(Int32Ty(),ia);
    Builder.CreateCondBr(Builder.CreateICmpSLT(idx,resSize),b,e);
    Builder.SetInsertPoint(b);
    auto* le=Builder.CreateLoad(DoubleTy(),Builder.CreateInBoundsGEP(DoubleTy(),lData,Builder.CreateSRem(idx,lSize)));
    auto* re=Builder.CreateLoad(DoubleTy(),Builder.CreateInBoundsGEP(DoubleTy(),rData,Builder.CreateSRem(idx,rSize)));
    auto z=llvm::ConstantFP::get(DoubleTy(),0.0);
    auto b2f=[&](llvm::Value* cmp){return Builder.CreateUIToFP(cmp,DoubleTy());};
    llvm::Value* opRes=nullptr;
    if(op=="+")  opRes=Builder.CreateFAdd(le,re);
    else if(op=="-")  opRes=Builder.CreateFSub(le,re);
    else if(op=="*")  opRes=Builder.CreateFMul(le,re);
    else if(op=="/")  opRes=Builder.CreateFDiv(le,re);
    else if(op=="%")  opRes=Builder.CreateFRem(le,re);
    else if(op==">")  opRes=b2f(Builder.CreateFCmpOGT(le,re));
    else if(op=="<")  opRes=b2f(Builder.CreateFCmpOLT(le,re));
    else if(op==">=") opRes=b2f(Builder.CreateFCmpOGE(le,re));
    else if(op=="<=") opRes=b2f(Builder.CreateFCmpOLE(le,re));
    else if(op=="==") opRes=b2f(Builder.CreateFCmpOEQ(le,re));
    else if(op=="!=") opRes=b2f(Builder.CreateFCmpONE(le,re));
    else if(op=="&&") opRes=b2f(Builder.CreateAnd(Builder.CreateFCmpOGT(le,z),Builder.CreateFCmpOGT(re,z)));
    else if(op=="||") opRes=b2f(Builder.CreateOr(Builder.CreateFCmpOGT(le,z),Builder.CreateFCmpOGT(re,z)));
    Builder.CreateStore(opRes,Builder.CreateInBoundsGEP(DoubleTy(),resData,idx));
    Builder.CreateStore(Builder.CreateAdd(idx,Builder.getInt32(1)),ia);
    Builder.CreateBr(c); Builder.SetInsertPoint(e);
    return res;
}

BinaryExprAST::BinaryExprAST(std::string Op, std::unique_ptr<ASTNode> LHS, std::unique_ptr<ASTNode> RHS)
    : Op(std::move(Op)), LHS(std::move(LHS)), RHS(std::move(RHS)) {}

llvm::Value* BinaryExprAST::codegen() {
    auto* L=LHS->codegen(); auto* R=RHS->codegen();
    if(!L||!R){std::cerr<<"\nerror: operands for '"<<Op<<"' failed\n";return nullptr;}

    if(L->getType()->isIntegerTy(8)) L=Builder.CreateSExt(L,Int32Ty());
    if(R->getType()->isIntegerTy(8)) R=Builder.CreateSExt(R,Int32Ty());
    if(L->getType()->isIntegerTy(32)&&R->getType()->isDoubleTy()) L=Builder.CreateSIToFP(L,DoubleTy());
    if(R->getType()->isIntegerTy(32)&&L->getType()->isDoubleTy()) R=Builder.CreateSIToFP(R,DoubleTy());

    // --- Int ?? Int ---
    if(L->getType()->isIntegerTy(32)&&R->getType()->isIntegerTy(32)){
        auto b2i=[&](llvm::Value* c){return Builder.CreateZExt(c,Int32Ty());};
        if(Op=="+")  return Builder.CreateAdd(L,R,"addtmp");
        if(Op=="-")  return Builder.CreateSub(L,R,"subtmp");
        if(Op=="*")  return Builder.CreateMul(L,R,"multmp");
        if(Op=="/")  return Builder.CreateSDiv(L,R,"divtmp");
        if(Op=="%")  return Builder.CreateSRem(L,R,"modtmp");
        if(Op=="<")  return b2i(Builder.CreateICmpSLT(L,R));
        if(Op==">")  return b2i(Builder.CreateICmpSGT(L,R));
        if(Op=="<=") return b2i(Builder.CreateICmpSLE(L,R));
        if(Op==">=") return b2i(Builder.CreateICmpSGE(L,R));
        if(Op=="==") return b2i(Builder.CreateICmpEQ(L,R));
        if(Op=="!=") return b2i(Builder.CreateICmpNE(L,R));
        if(Op=="**") { // integer power via double
            auto* lf=Builder.CreateSIToFP(L,DoubleTy());
            auto* rf=Builder.CreateSIToFP(R,DoubleTy());
            auto* F=getOrDeclare("dream_pow",TPtrTy(),{TPtrTy(),TPtrTy()});
            auto* lt=castToTensor(lf); auto* rt=castToTensor(rf);
            auto* res=Builder.CreateCall(F,{lt,rt});
            // extract scalar back
            auto* dp=Builder.CreateLoad(DblPtrTy(),Builder.CreateStructGEP(TensorTy,res,0));
            auto* fv=Builder.CreateLoad(DoubleTy(),dp);
            // release the boxed operands and the pow result tensor
            auto* decF=getOrDeclare("tensor_decref",TPtrTy(),{TPtrTy()});
            Builder.CreateCall(decF,{lt}); Builder.CreateCall(decF,{rt}); Builder.CreateCall(decF,{res});
            return Builder.CreateFPToSI(fv,Int32Ty());
        }
        if(Op=="&&"){auto* c1=Builder.CreateICmpNE(L,Builder.getInt32(0));auto* c2=Builder.CreateICmpNE(R,Builder.getInt32(0));return b2i(Builder.CreateAnd(c1,c2));}
        if(Op=="||"){auto* c1=Builder.CreateICmpNE(L,Builder.getInt32(0));auto* c2=Builder.CreateICmpNE(R,Builder.getInt32(0));return b2i(Builder.CreateOr(c1,c2));}
        std::cerr<<"\nerror: Invalid op '"<<Op<<"' for Int\n";return nullptr;
    }

    // --- Float ?? Float ---
    if(L->getType()->isDoubleTy()&&R->getType()->isDoubleTy()){
        auto b2f=[&](llvm::Value* c){return Builder.CreateUIToFP(c,DoubleTy());};
        auto z=llvm::ConstantFP::get(DoubleTy(),0.0);
        if(Op=="+")  return Builder.CreateFAdd(L,R,"addtmp");
        if(Op=="-")  return Builder.CreateFSub(L,R,"subtmp");
        if(Op=="*")  return Builder.CreateFMul(L,R,"multmp");
        if(Op=="/")  return Builder.CreateFDiv(L,R,"divtmp");
        if(Op=="%")  return Builder.CreateFRem(L,R,"modtmp");
        if(Op=="<")  return b2f(Builder.CreateFCmpOLT(L,R));
        if(Op==">")  return b2f(Builder.CreateFCmpOGT(L,R));
        if(Op=="<=") return b2f(Builder.CreateFCmpOLE(L,R));
        if(Op==">=") return b2f(Builder.CreateFCmpOGE(L,R));
        if(Op=="==") return b2f(Builder.CreateFCmpOEQ(L,R));
        if(Op=="!=") return b2f(Builder.CreateFCmpONE(L,R));
        if(Op=="**") { // pow via runtime
            auto* F=getOrDeclare("dream_pow",TPtrTy(),{TPtrTy(),TPtrTy()});
            auto* lt=castToTensor(L); auto* rt=castToTensor(R);
            auto* res=Builder.CreateCall(F,{lt,rt},"powtmp");
            // extract scalar
            auto* dp=Builder.CreateLoad(DblPtrTy(),Builder.CreateStructGEP(TensorTy,res,0));
            auto* fv=Builder.CreateLoad(DoubleTy(),dp);
            auto* decF=getOrDeclare("tensor_decref",TPtrTy(),{TPtrTy()});
            Builder.CreateCall(decF,{lt}); Builder.CreateCall(decF,{rt}); Builder.CreateCall(decF,{res});
            return fv;
        }
        if(Op=="&&"){auto* c1=Builder.CreateFCmpONE(L,z);auto* c2=Builder.CreateFCmpONE(R,z);return b2f(Builder.CreateAnd(c1,c2));}
        if(Op=="||"){auto* c1=Builder.CreateFCmpONE(L,z);auto* c2=Builder.CreateFCmpONE(R,z);return b2f(Builder.CreateOr(c1,c2));}
        std::cerr<<"\nerror: Invalid op '"<<Op<<"' for Float\n";return nullptr;
    }

    // --- Tensor ?? Tensor ---
    auto* origL=L; auto* origR=R;
    L=castToTensor(L); R=castToTensor(R);
    // The op links operands as owned graph edges (tensor_set_lhs/rhs)
    // and returns a fresh +1 result, so our operand temporaries can be
    // released as soon as the op call has been emitted.
    auto finishBin=[&](llvm::Value* res)->llvm::Value*{
        decrefTemp(L, origL, LHS.get());
        decrefTemp(R, origR, RHS.get());
        return res;
    };
    if(Op=="@"){auto* F=getOrDeclare("tensor_matmul",TPtrTy(),{TPtrTy(),TPtrTy()});return finishBin(Builder.CreateCall(F,{L,R},"mm_tmp"));}
    if(Op=="**"){auto* F=getOrDeclare("dream_pow",TPtrTy(),{TPtrTy(),TPtrTy()});return finishBin(Builder.CreateCall(F,{L,R},"pow_tmp"));}
    static const std::map<std::string,std::string> opMap={
        {"+","tensor_add"},{"-","tensor_sub"},{"*","tensor_mul"},{"/","tensor_div"},
        {"%","tensor_mod"},{">","tensor_gt"},{"<","tensor_lt"},{">=","tensor_ge"},
        {"<=","tensor_le"},{"==","tensor_eq"},{"!=","tensor_ne"},{"&&","tensor_and"},{"||","tensor_or"}
    };
    auto it=opMap.find(Op);
    if(it!=opMap.end()){auto* F=getOrDeclare(it->second,TPtrTy(),{TPtrTy(),TPtrTy()});return finishBin(Builder.CreateCall(F,{L,R},"op_tmp"));}
    return finishBin(generateTensorLoop(Op,L,R));
}

bool BinaryExprAST::isFuseable() const { return Op!="@"&&Op!="**"&&LHS->isFuseable()&&RHS->isFuseable(); }
bool BinaryExprAST::isOpNode() const { return true; }
void BinaryExprAST::collectVars(std::vector<std::string>& v) const { LHS->collectVars(v); RHS->collectVars(v); }
std::string BinaryExprAST::genFusedCode(const std::vector<std::string>& v) const {
    std::string l=LHS->genFusedCode(v),r=RHS->genFusedCode(v);
    if(Op==">"||Op=="<"||Op==">="||Op=="<="||Op=="=="||Op=="!=") return "(("+l+" "+Op+" "+r+")?1.0f:0.0f)";
    if(Op=="&&") return "(("+l+">0.0f&&"+r+">0.0f)?1.0f:0.0f)";
    if(Op=="||") return "(("+l+">0.0f||"+r+">0.0f)?1.0f:0.0f)";
    if(Op=="%")  return "fmod("+l+","+r+")";
    if(Op=="**") return "pow("+l+","+r+")";
    return "("+l+" "+Op+" "+r+")";
}
std::string BinaryExprAST::genFusedGradCode(const std::string& tv,const std::vector<std::string>& v) const {
    if(Op==">"||Op==">="||Op=="<"||Op=="<="||Op=="=="||Op=="!="||Op=="&&"||Op=="||"||Op=="%") return "0.0f";
    std::string l=LHS->genFusedCode(v),r=RHS->genFusedCode(v);
    std::string dl=LHS->genFusedGradCode(tv,v),dr=RHS->genFusedGradCode(tv,v);
    if(Op=="+") return "("+dl+"+"+dr+")";
    if(Op=="-") return "("+dl+"-"+dr+")";
    if(Op=="*") return "(("+dl+"*"+r+")+("+l+"*"+dr+"))";
    if(Op=="/") return "(("+dl+"*"+r+"-"+l+"*"+dr+")/("+r+"*"+r+"))";
    if(Op=="**") return "("+l+"!=0.0f?"+r+"*pow("+l+","+r+"-1.0f)*"+dl+":0.0f)";
    return "0.0f";
}
FusedExprAST::FusedExprAST(std::unique_ptr<ASTNode> Root, std::vector<std::string> Vars,
                             std::string KN,std::string KS,std::string BN,std::string BS)
    : Root(std::move(Root)),Vars(std::move(Vars)),KernelName(std::move(KN)),KernelSrc(std::move(KS)),
      BwdKernelName(std::move(BN)),BwdKernelSrc(std::move(BS)) {}

llvm::Value* FusedExprAST::codegen() {
    bool hasTensor=false;
    for(const auto& var:Vars){
        auto it=NamedValues.find(var); if(it==NamedValues.end()) continue;
        if(llvm::cast<llvm::AllocaInst>(it->second)->getAllocatedType()->isPointerTy()){hasTensor=true;break;}
    }
    if(!hasTensor) return Root->codegen();

    std::vector<llvm::Value*> varVals;
    for(const auto& var:Vars){
        auto* V=NamedValues[var];
        auto* a=llvm::cast<llvm::AllocaInst>(V);
        varVals.push_back(castToTensor(Builder.CreateLoad(a->getAllocatedType(),V)));
    }
    auto* arrSz=Builder.getInt32((int)std::max(Vars.size(),size_t(1)));
    auto* arr=Builder.CreateAlloca(TPtrTy(),arrSz,"vars_arr");
    for(size_t i=0;i<Vars.size();++i)
        Builder.CreateStore(varVals[i],Builder.CreateInBoundsGEP(TPtrTy(),arr,Builder.getInt32(i)));
    auto* F=getOrDeclare("run_fused_with_grad",TPtrTy(),
        {CharPtrTy(),CharPtrTy(),CharPtrTy(),CharPtrTy(),llvm::PointerType::getUnqual(TPtrTy()),Int32Ty()});

    // Dual path: use the fused GPU kernel only when a device exists;
    // otherwise fall back to the original (unfused) expression tree,
    // which runs on the CPU tensor ops. Previously the runtime called
    // exit(1) here on machines without OpenCL.
    auto* availF = getOrDeclare("dream_gpu_available", Int32Ty(), {});
    auto* avail  = Builder.CreateCall(availF, {}, "gpu_ok");
    auto* fnP    = Builder.GetInsertBlock()->getParent();
    auto* gpuBB   = llvm::BasicBlock::Create(TheContext, "fused.gpu", fnP);
    auto* cpuBB   = llvm::BasicBlock::Create(TheContext, "fused.cpu");
    auto* mergeBB = llvm::BasicBlock::Create(TheContext, "fused.merge");
    Builder.CreateCondBr(Builder.CreateICmpNE(avail, Builder.getInt32(0)), gpuBB, cpuBB);

    Builder.SetInsertPoint(gpuBB);
    auto* fusedRes = Builder.CreateCall(F,{Builder.CreateGlobalStringPtr(KernelName),
                                  Builder.CreateGlobalStringPtr(KernelSrc),
                                  Builder.CreateGlobalStringPtr(BwdKernelName),
                                  Builder.CreateGlobalStringPtr(BwdKernelSrc),
                                  arr,Builder.getInt32((int)Vars.size())},"fused_tmp");
    auto* gpuEnd = Builder.GetInsertBlock();
    Builder.CreateBr(mergeBB);

    fnP->insert(fnP->end(), cpuBB);
    Builder.SetInsertPoint(cpuBB);
    auto* cpuRes = Root->codegen();
    if (!cpuRes) return nullptr;
    cpuRes = castToTensor(cpuRes);
    auto* cpuEnd = Builder.GetInsertBlock();
    Builder.CreateBr(mergeBB);

    fnP->insert(fnP->end(), mergeBB);
    Builder.SetInsertPoint(mergeBB);
    auto* phi = Builder.CreatePHI(TPtrTy(), 2, "fusedphi");
    phi->addIncoming(fusedRes, gpuEnd);
    phi->addIncoming(cpuRes, cpuEnd);
    return phi;
}
std::string FusedExprAST::genOpenCL() { return Root->genOpenCL(); }
LetStmtAST::LetStmtAST(const std::string& VN,const std::string& VT,std::unique_ptr<ASTNode> Expr)
    : VarName(VN),VarType(VT),Expr(std::move(Expr)) {}

llvm::Value* LetStmtAST::codegen() {
    auto* val=Expr->codegen(); if(!val) return nullptr;
    llvm::Type* allocTy=nullptr;
    if     (VarType=="Int")    {val=coerceValue(val,Int32Ty(),VarName);allocTy=Int32Ty();}
    else if(VarType=="Float")  {val=coerceValue(val,DoubleTy(),VarName);allocTy=DoubleTy();}
    else if(VarType=="Char")   {val=coerceValue(val,Int8Ty(),VarName);allocTy=Int8Ty();}
    else if(VarType=="Bool")   {val=Builder.CreateZExt(buildCondBool(val),Int32Ty());allocTy=Int32Ty();}
    else if(VarType=="Tensor") {val=castToTensor(val);allocTy=TPtrTy();}
    else if(VarType=="String") {val=castToTensor(val);allocTy=TPtrTy();}
    else if(VarType=="auto")   {
        allocTy=val->getType();
        // Scalar double/int keeps its native type, not boxed into a Tensor.
        // Boxing makes total = total + 1.0 go through tensor_add and build a graph,
        // whose chain grows unbounded over billions of iterations until it crashes.
        // Native scalars use pure IR float math: O(1) memory, no crash. Tensor
        // contexts (x @ W etc.) box on demand. (Keep double: isDoubleTy no longer boxes.)
    }
    else{std::cerr<<"\nerror: Unknown type '"<<VarType<<"'\n";return nullptr;}
    if(!val) return nullptr;

    auto* fn = Builder.GetInsertBlock()->getParent();

    // Reuse existing alloca if variable was already declared (e.g., let inside a loop).
    // This prevents duplicate allocas and duplicate CurrentFnLocals entries.
    llvm::AllocaInst* alloca = nullptr;
    bool reusing = false;
    auto existIt = NamedValues.find(VarName);
    if (existIt != NamedValues.end()) {
        alloca = llvm::dyn_cast<llvm::AllocaInst>(existIt->second);
        if (alloca && alloca->getAllocatedType() == allocTy) reusing = true;
        else alloca = nullptr;
    }
    if (!alloca) alloca = createEntryAlloca(fn, VarName, allocTy);

    if(allocTy->isPointerTy()){
        auto* decF = getOrDeclare("tensor_decref", TPtrTy(), {TPtrTy()});
        auto* incF = getOrDeclare("tensor_incref", TPtrTy(), {TPtrTy()});
        // Decref old value before overwriting (null-safe: alloca is null-initialized).
        auto* oldVal = Builder.CreateLoad(TPtrTy(), alloca);
        Builder.CreateCall(decF, {oldVal});
        // PURE REFCOUNT MODEL â€” incref only when borrowing.
        //
        // Functions/operators that produce fresh tensors (CallExpr, BinaryExpr,
        // UnaryExpr, GradExpr, etc.) already incref'd the result before
        // returning to us â€” that's the caller's +1 gift. We just store it
        // into the let alloca, which is now the new owner.
        //
        // But if the source is a "borrow" (VariableExpr, IndexExpr,
        // FieldAccess), we received a raw pointer with no incref, so we must
        // incref to claim ownership â€” same tensor lives in two allocas.
        bool source_is_borrow = isBorrowExpr(Expr.get());
        if (source_is_borrow) {
            Builder.CreateCall(incF, {val});
        }
        val=Builder.CreateCall(getOrDeclare("tensor_mark_var",TPtrTy(),{TPtrTy()}),{val});
    }
    Builder.CreateStore(val,alloca);
    NamedValues[VarName]=alloca;
    if (!reusing && allocTy->isPointerTy()) CurrentFnLocals.push_back(VarName);
    return val;
}

AssignStmtAST::AssignStmtAST(const std::string& VN,std::unique_ptr<ASTNode> Expr)
    : VarName(VN),Expr(std::move(Expr)) {}

llvm::Value* AssignStmtAST::codegen() {
    auto* val=Expr->codegen(); if(!val) return nullptr;
    auto it=NamedValues.find(VarName);
    if(it==NamedValues.end()){
        std::string best; int bestD = 3;
        for (const auto& kv : NamedValues) {
            int d = levBounded(VarName, kv.first, 2);
            if (d < bestD) { bestD = d; best = kv.first; }
        }
        diagAtLine("undefined variable '" + VarName + "'", VarName, Line, best);
        return nullptr;
    }
    bool source_is_borrow = isBorrowExpr(Expr.get());
    return performAssign(val,llvm::cast<llvm::AllocaInst>(it->second),VarName,source_is_borrow);
}

WhileStmtAST::WhileStmtAST(std::unique_ptr<ASTNode> Cond,std::vector<std::unique_ptr<ASTNode>> Body)
    : Cond(std::move(Cond)),Body(std::move(Body)) {}

llvm::Value* WhileStmtAST::codegen() {
    auto* fn=Builder.GetInsertBlock()->getParent();
    auto* condBB=llvm::BasicBlock::Create(TheContext,"while.cond",fn);
    auto* bodyBB=llvm::BasicBlock::Create(TheContext,"while.body",fn);
    auto* afterBB=llvm::BasicBlock::Create(TheContext,"while.after",fn);
    BreakTargets.push_back(afterBB); ContinueTargets.push_back(condBB);
    Builder.CreateBr(condBB); Builder.SetInsertPoint(condBB);
    auto* cv=Cond->codegen(); if(!cv){std::cerr<<"\nerror: while cond\n";return nullptr;}
    auto* condBool = buildCondBool(cv);
    decrefTemp(cv, cv, Cond.get());
    Builder.CreateCondBr(condBool,bodyBB,afterBB);
    Builder.SetInsertPoint(bodyBB);
    for(auto& s:Body){
        auto* sv=s->codegen();
        if(!sv){std::cerr<<"\nerror: while body\n";return nullptr;}
        decrefIfTensor(sv, s.get());
        if(Builder.GetInsertBlock()->getTerminator())break;
    }
    if(!Builder.GetInsertBlock()->getTerminator()) Builder.CreateBr(condBB);
    BreakTargets.pop_back(); ContinueTargets.pop_back();
    Builder.SetInsertPoint(afterBB);
    return llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(TPtrTy()));
}

ForStmtAST::ForStmtAST(std::unique_ptr<ASTNode> Init,std::unique_ptr<ASTNode> Cond,
                         std::string SV,std::unique_ptr<ASTNode> SE,
                         std::vector<std::unique_ptr<ASTNode>> Body)
    : Init(std::move(Init)),Cond(std::move(Cond)),StepVar(std::move(SV)),StepExpr(std::move(SE)),Body(std::move(Body)) {}

llvm::Value* ForStmtAST::codegen() {
    auto* fn=Builder.GetInsertBlock()->getParent();
    if(!Init->codegen()) return nullptr;
    auto* condBB=llvm::BasicBlock::Create(TheContext,"for.cond",fn);
    auto* bodyBB=llvm::BasicBlock::Create(TheContext,"for.body",fn);
    auto* stepBB=llvm::BasicBlock::Create(TheContext,"for.step",fn);
    auto* afterBB=llvm::BasicBlock::Create(TheContext,"for.after",fn);
    BreakTargets.push_back(afterBB); ContinueTargets.push_back(stepBB);
    Builder.CreateBr(condBB); Builder.SetInsertPoint(condBB);
    auto* cv=Cond->codegen(); if(!cv){std::cerr<<"\nerror: for cond\n";return nullptr;}
    auto* condBool = buildCondBool(cv);
    decrefTemp(cv, cv, Cond.get());
    Builder.CreateCondBr(condBool,bodyBB,afterBB);
    Builder.SetInsertPoint(bodyBB);
    for(auto& s:Body){
        auto* sv=s->codegen();
        if(!sv){std::cerr<<"\nerror: for body\n";return nullptr;}
        decrefIfTensor(sv, s.get());
        if(Builder.GetInsertBlock()->getTerminator())break;
    }
    if(!Builder.GetInsertBlock()->getTerminator()) Builder.CreateBr(stepBB);
    Builder.SetInsertPoint(stepBB);
    auto* sv=StepExpr->codegen(); if(!sv) return nullptr;
    auto it=NamedValues.find(StepVar);
    if(it==NamedValues.end()){std::cerr<<"\nerror: step var '"<<StepVar<<"'\n";return nullptr;}
    if(!performAssign(sv,llvm::cast<llvm::AllocaInst>(it->second),StepVar,isBorrowExpr(StepExpr.get()))) return nullptr;
    Builder.CreateBr(condBB);
    BreakTargets.pop_back(); ContinueTargets.pop_back();
    Builder.SetInsertPoint(afterBB);
    return llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(TPtrTy()));
}

ForRangeStmtAST::ForRangeStmtAST(std::string VN,
                                   std::unique_ptr<ASTNode> Start,
                                   std::unique_ptr<ASTNode> End,
                                   std::unique_ptr<ASTNode> Step,
                                   std::vector<std::unique_ptr<ASTNode>> Body)
    : VarName(std::move(VN)),Start(std::move(Start)),End(std::move(End)),
      Step(std::move(Step)),Body(std::move(Body)) {}

llvm::Value* ForRangeStmtAST::codegen() {
    auto* fn = Builder.GetInsertBlock()->getParent();

    // Helper: extract scalar from Tensor* (reads first element as double)
    auto extractScalar = [&](llvm::Value* v) -> llvm::Value* {
        if (!v->getType()->isPointerTy()) return v;
        auto* dp = Builder.CreateLoad(DblPtrTy(),
                       Builder.CreateStructGEP(TensorTy, v, 0));
        return Builder.CreateLoad(DoubleTy(), dp);
    };

    // Evaluate start, end, step once before the loop
    auto* ostartV = Start->codegen(); if (!ostartV) return nullptr;
    auto* oendV   = End->codegen();   if (!oendV)   return nullptr;

    // Extract scalars from Tensor* if needed, then release fresh temps
    auto* startV = extractScalar(ostartV);
    auto* endV   = extractScalar(oendV);
    decrefTemp(ostartV, ostartV, Start.get());
    decrefTemp(oendV,   oendV,   End.get());

    // Determine type: only use int loop if BOTH start and end are int32
    bool isInt = startV->getType()->isIntegerTy(32) && endV->getType()->isIntegerTy(32);
    llvm::Type* varTy = isInt ? Int32Ty() : DoubleTy();

    // Coerce to same type
    if (!isInt) {
        if (startV->getType()->isIntegerTy(32))
            startV = Builder.CreateSIToFP(startV, DoubleTy());
        if (endV->getType()->isIntegerTy(32))
            endV = Builder.CreateSIToFP(endV, DoubleTy());
    }

    // Step value
    llvm::Value* stepV = nullptr;
    if (Step) {
        auto* ostepV = Step->codegen(); if (!ostepV) return nullptr;
        stepV = extractScalar(ostepV);
        decrefTemp(ostepV, ostepV, Step.get());
        if (isInt && stepV->getType()->isDoubleTy()) stepV = Builder.CreateFPToSI(stepV, Int32Ty());
        if (!isInt && stepV->getType()->isIntegerTy(32)) stepV = Builder.CreateSIToFP(stepV, DoubleTy());
    } else {
        stepV = isInt ? (llvm::Value*)Builder.getInt32(1)
                      : (llvm::Value*)llvm::ConstantFP::get(DoubleTy(), 1.0);
    }

    // Alloca for the loop variable
    auto* varAlloca = createEntryAlloca(fn, VarName, varTy);
    Builder.CreateStore(startV, varAlloca);
    NamedValues[VarName] = varAlloca;

    // Alloca for cached end/step (evaluated once)
    auto* endAlloca  = createEntryAlloca(fn, VarName+"_end",  varTy);
    auto* stepAlloca = createEntryAlloca(fn, VarName+"_step", varTy);
    Builder.CreateStore(endV,  endAlloca);
    Builder.CreateStore(stepV, stepAlloca);

    auto* condBB  = llvm::BasicBlock::Create(TheContext, "range.cond",  fn);
    auto* bodyBB  = llvm::BasicBlock::Create(TheContext, "range.body",  fn);
    auto* stepBB  = llvm::BasicBlock::Create(TheContext, "range.step",  fn);
    auto* afterBB = llvm::BasicBlock::Create(TheContext, "range.after", fn);

    BreakTargets.push_back(afterBB);
    ContinueTargets.push_back(stepBB);

    Builder.CreateBr(condBB);
    Builder.SetInsertPoint(condBB);
    auto* curV = Builder.CreateLoad(varTy, varAlloca, VarName);
    auto* endC = Builder.CreateLoad(varTy, endAlloca);
    llvm::Value* condV = isInt
        ? Builder.CreateICmpSLT(curV, endC)
        : Builder.CreateFCmpOLT(curV, endC);
    Builder.CreateCondBr(condV, bodyBB, afterBB);

    Builder.SetInsertPoint(bodyBB);
    for (auto& s : Body) {
        auto* sv = s->codegen();
        if (!sv) { std::cerr << "\nerror: range-for body\n"; return nullptr; }
        decrefIfTensor(sv, s.get());
        if (Builder.GetInsertBlock()->getTerminator()) break;
    }
    if (!Builder.GetInsertBlock()->getTerminator()) Builder.CreateBr(stepBB);

    Builder.SetInsertPoint(stepBB);
    auto* loopVar = Builder.CreateLoad(varTy, varAlloca);
    auto* stepC   = Builder.CreateLoad(varTy, stepAlloca);
    llvm::Value* next = isInt
        ? Builder.CreateAdd(loopVar, stepC)
        : Builder.CreateFAdd(loopVar, stepC);
    Builder.CreateStore(next, varAlloca);
    Builder.CreateBr(condBB);

    BreakTargets.pop_back();
    ContinueTargets.pop_back();

    Builder.SetInsertPoint(afterBB);
    return llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(TPtrTy()));
}
std::string ForRangeStmtAST::genOpenCL() { return "    // range-for\n"; }
IfStmtAST::IfStmtAST(std::unique_ptr<ASTNode> C,
                       std::vector<std::unique_ptr<ASTNode>> T,
                       std::vector<std::unique_ptr<ASTNode>> E)
    : Cond(std::move(C)),ThenBody(std::move(T)),ElseBody(std::move(E)) {}

llvm::Value* IfStmtAST::codegen() {
    auto* cv=Cond->codegen(); if(!cv){std::cerr<<"\nerror: if cond\n";return nullptr;}
    auto* fn=Builder.GetInsertBlock()->getParent();
    auto* thenBB=llvm::BasicBlock::Create(TheContext,"if.then",fn);
    auto* elseBB=llvm::BasicBlock::Create(TheContext,"if.else");
    auto* mergeBB=llvm::BasicBlock::Create(TheContext,"if.merge");
    auto* condBool = buildCondBool(cv);
    decrefTemp(cv, cv, Cond.get());
    Builder.CreateCondBr(condBool,thenBB,elseBB);
    Builder.SetInsertPoint(thenBB);
    for(auto& s:ThenBody){
        auto* sv=s->codegen();
        if(!sv){std::cerr<<"\nerror: if then\n";return nullptr;}
        decrefIfTensor(sv, s.get());
        if(Builder.GetInsertBlock()->getTerminator())break;
    }
    if(!Builder.GetInsertBlock()->getTerminator()) Builder.CreateBr(mergeBB);
    fn->insert(fn->end(),elseBB); Builder.SetInsertPoint(elseBB);
    for(auto& s:ElseBody){
        auto* sv=s->codegen();
        if(!sv){std::cerr<<"\nerror: if else\n";return nullptr;}
        decrefIfTensor(sv, s.get());
        if(Builder.GetInsertBlock()->getTerminator())break;
    }
    if(!Builder.GetInsertBlock()->getTerminator()) Builder.CreateBr(mergeBB);
    fn->insert(fn->end(),mergeBB); Builder.SetInsertPoint(mergeBB);
    return llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(TPtrTy()));
}

ReturnStmtAST::ReturnStmtAST(std::unique_ptr<ASTNode> Expr) : Expr(std::move(Expr)) {}
llvm::Value* ReturnStmtAST::codegen() {
    llvm::Value* rv=nullptr;
    if(Expr){rv=Expr->codegen();if(!rv)return nullptr;rv=castToTensor(rv);}
    if(!rv) rv=llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(TPtrTy()));
    // PURE REFCOUNT MODEL â€” UNIFIED RULE:
    //
    // Every function returning a Tensor* gives caller +1 ref. ALWAYS.
    // Caller is responsible for releasing that +1 (via let-binding cleanup
    // at scope end, or via decrefIfTensor at expr-stmt boundary).
    //
    // We always incref(rv) here, before scope cleanup. Then scope cleanup
    // decrefs locals. If rv is a local, the +1 we just added survives the
    // cleanup decref. If rv is a non-local fresh tensor (e.g. a freshly
    // allocated scalar from `return 0.0`), the cleanup doesn't touch it,
    // and our +1 is still the caller's gift.
    // Borrowed sources (a local or a parameter read) need +1 so the value
    // survives scope cleanup / reaches the caller as an owned reference.
    // Fresh sources (calls, ops, boxed literals) already carry the +1 gift;
    // incref-ing them again would leak one reference per return.
    if (Expr && isBorrowExpr(Expr.get())) {
        auto* incF = getOrDeclare("tensor_incref", TPtrTy(), {TPtrTy()});
        Builder.CreateCall(incF, {rv});
    }
    emitScopeCleanup();
    Builder.CreateRet(rv);
    makeDeadBlock("ret.dead");
    return rv;
}

llvm::Value* BreakStmtAST::codegen() {
    if(BreakTargets.empty()){std::cerr<<"\nerror: break outside loop\n";return nullptr;}
    Builder.CreateBr(BreakTargets.back());
    makeDeadBlock("break.dead");
    return llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(TPtrTy()));
}
llvm::Value* ContinueStmtAST::codegen() {
    if(ContinueTargets.empty()){std::cerr<<"\nerror: continue outside loop\n";return nullptr;}
    Builder.CreateBr(ContinueTargets.back());
    makeDeadBlock("cont.dead");
    return llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(TPtrTy()));
}

PrintStmtAST::PrintStmtAST(std::vector<std::unique_ptr<ASTNode>> Args, bool Newline)
    : Args(std::move(Args)), Newline(Newline) {}

llvm::Value* PrintStmtAST::codegen() {
    auto* printF = getOrDeclare("dream_print_one", TPtrTy(), {TPtrTy()});
    auto* strF   = getOrDeclare("dream_print_string", TPtrTy(), {CharPtrTy()});

    auto* decF = getOrDeclare("tensor_decref", TPtrTy(), {TPtrTy()});
    llvm::Value* last = llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(TPtrTy()));
    for (auto& arg : Args) {
        if (auto* strNode = dynamic_cast<StringExprAST*>(arg.get())) {
            auto* strPtr = Builder.CreateGlobalStringPtr(strNode->getVal(), ".ps");
            last = Builder.CreateCall(strF, {strPtr});
            Builder.CreateCall(decF, {last});   // print_string returns fresh +1
        } else {
            auto* ov = arg->codegen(); if (!ov) return nullptr;
            auto* v = castToTensor(ov);
            last = Builder.CreateCall(printF, {v});
            Builder.CreateCall(decF, {last});   // print returns +1 on its arg
            decrefTemp(v, ov, arg.get());       // release fresh/boxed arg temp
        }
    }
    if (Newline) {
        auto* nl = Builder.CreateGlobalStringPtr("\n", ".nl");
        last = Builder.CreateCall(strF, {nl});
        Builder.CreateCall(decF, {last});
    }
    return llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(TPtrTy()));
}
std::string PrintStmtAST::genOpenCL() { return "    // print\n"; }
FunctionAST::FunctionAST(const std::string& Name,
                          std::vector<std::pair<std::string,std::string>> Args,
                          std::vector<std::unique_ptr<ASTNode>> Body)
    : Name(Name),Args(std::move(Args)),Body(std::move(Body)) {}

llvm::Function* FunctionAST::codegen() {
    // Save/restore locals for nested function support
    auto savedLocals = CurrentFnLocals;
    CurrentFnLocals.clear();

    std::vector<llvm::Type*> argTypes;
    for(auto& [n,t]:Args){
        if(t=="Int") argTypes.push_back(Int32Ty());
        else if(t=="Float") argTypes.push_back(DoubleTy());
        else if(t=="Char")  argTypes.push_back(Int8Ty());
        else argTypes.push_back(TPtrTy());
    }

    // Forward-reference fix: if app() called train_ai() before train_ai is
    // defined, getOrDeclare at the call site already created an empty
    // declaration. We must REUSE that declaration here so the linker resolves
    // the call. Without this, Function::Create would create "train_ai.1"
    // and leave the original "train_ai" as an unresolved external symbol.
    auto* existing = TheModule->getFunction(Name);
    llvm::Function* F = nullptr;
    if (existing && existing->empty()) {
        // Forward-declared but never defined â€” reuse it.
        // Verify the declared signature matches what we're about to define.
        // If the call site inferred different arg types (rare), bail out
        // gracefully rather than creating a name conflict.
        if (existing->arg_size() == argTypes.size()) {
            F = existing;
        }
    }
    if (!F) {
        F = llvm::Function::Create(
            llvm::FunctionType::get(TPtrTy(), argTypes, false),
            llvm::Function::ExternalLinkage, Name, TheModule.get());
    }

    unsigned idx=0; for(auto& a:F->args()) a.setName(Args[idx++].first);
    Builder.SetInsertPoint(llvm::BasicBlock::Create(TheContext,"entry",F));
    NamedValues.clear();
    for(auto& a:F->args()){
        auto* alloca=createEntryAlloca(F,std::string(a.getName()),a.getType());
        // Function parameters are BORROWED references from the caller.
        // The caller owns the tensor (via its own let binding with is_variable=true).
        // We do NOT incref/decref/mark_var here because:
        // 1. GC only runs at backward() entry, not mid-expression
        // 2. The caller's reference keeps the tensor alive
        // 3. incref+decref in scope cleanup was corrupting is_variable state,
        //    causing W1 to get GC'd after returning from sgd_step(W1, lr)
        // DO NOT add params to CurrentFnLocals â€” they're not owned by this function.
        Builder.CreateStore(&a, alloca);
        NamedValues[std::string(a.getName())]=alloca;
    }
    for(auto& s:Body){
        auto* sv=s->codegen();
        if(!sv){std::cerr<<"\nerror: abort in '"<<Name<<"'\n";F->eraseFromParent();CurrentFnLocals=savedLocals;return nullptr;}
        decrefIfTensor(sv, s.get());
        if(Builder.GetInsertBlock()->getTerminator()) break;
    }
    if(!Builder.GetInsertBlock()->getTerminator()) {
        // Emit scope cleanup before implicit return
        emitScopeCleanup();
        Builder.CreateRet(llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(TPtrTy())));
    }
    llvm::verifyFunction(*F);
    CurrentFnLocals = savedLocals;
    return F;
}

GradExprAST::GradExprAST(std::unique_ptr<ASTNode> Expr) : Expr(std::move(Expr)) {}
llvm::Value* GradExprAST::codegen() {
    auto* ov=Expr->codegen(); if(!ov) return nullptr;
    auto* v=castToTensor(ov);
    auto* res=Builder.CreateCall(getOrDeclare("tensor_get_grad",TPtrTy(),{TPtrTy()}),{v},"gradtmp");
    decrefTemp(v, ov, Expr.get());
    return res;
}

std::string IntExprAST::genOpenCL()   { return std::to_string(Val); }
std::string FloatExprAST::genOpenCL() { return std::to_string(Val)+"f"; }
std::string CharExprAST::genOpenCL()  { return std::to_string((int)Val); }
std::string VariableExprAST::genOpenCL() {
    bool isArg=std::find(GPUKernelArgs.begin(),GPUKernelArgs.end(),Name)!=GPUKernelArgs.end();
    return isArg?Name+"[gid]":Name;
}
std::string CallExprAST::genOpenCL() {
    std::string s=Callee+"(";
    for(size_t i=0;i<Args.size();++i){s+=Args[i]->genOpenCL();if(i+1<Args.size())s+=", ";}
    return s+")";
}

static const std::set<std::string>& fuseableFns() {
    static const std::set<std::string> s = {
        "relu","exp","log","sqrt","abs","sigmoid","tanh","sin","cos","tan",
        "dream_exp","dream_log","dream_sqrt","dream_abs","dream_sigmoid","dream_tanh",
        "dream_sin","dream_cos","dream_tan"
    };
    return s;
}

bool CallExprAST::isFuseable() const {
    if (fuseableFns().count(Callee) && Args.size() == 1)
        return Args[0]->isFuseable();
    return false;
}

std::string CallExprAST::genFusedCode(const std::vector<std::string>& vars) const {
    if (Args.size() != 1) return "0.0f";
    std::string inner = Args[0]->genFusedCode(vars);
    if (Callee == "relu" || Callee == "dream_relu") return "("+inner+">0.0f?"+inner+":0.0f)";
    if (Callee == "exp" || Callee == "dream_exp") return "exp("+inner+")";
    if (Callee == "log" || Callee == "dream_log") return "log("+inner+")";
    if (Callee == "sqrt" || Callee == "dream_sqrt") return "sqrt("+inner+")";
    if (Callee == "abs" || Callee == "dream_abs") return "fabs("+inner+")";
    if (Callee == "sigmoid" || Callee == "dream_sigmoid") return "(1.0f/(1.0f+exp(-"+inner+")))";
    if (Callee == "tanh" || Callee == "dream_tanh") return "tanh("+inner+")";
    if (Callee == "sin" || Callee == "dream_sin") return "sin("+inner+")";
    if (Callee == "cos" || Callee == "dream_cos") return "cos("+inner+")";
    if (Callee == "tan" || Callee == "dream_tan") return "tan("+inner+")";
    return "0.0f";
}

std::string CallExprAST::genFusedGradCode(const std::string& tv, const std::vector<std::string>& vars) const {
    if (Args.size() != 1) return "0.0f";
    std::string x = Args[0]->genFusedCode(vars);
    std::string dx = Args[0]->genFusedGradCode(tv, vars);
    if (Callee == "relu" || Callee == "dream_relu")
        return "("+x+">0.0f?"+dx+":0.0f)";
    if (Callee == "exp" || Callee == "dream_exp")
        return "(exp("+x+")*"+dx+")";
    if (Callee == "log" || Callee == "dream_log")
        return "("+dx+"/"+x+")";
    if (Callee == "sqrt" || Callee == "dream_sqrt")
        return "("+dx+"/(2.0f*sqrt("+x+")))";
    if (Callee == "abs" || Callee == "dream_abs")
        return "("+x+">=0.0f?"+dx+":-"+dx+")";
    if (Callee == "sigmoid" || Callee == "dream_sigmoid") {
        std::string s = "(1.0f/(1.0f+exp(-"+x+")))";
        return "("+s+"*(1.0f-"+s+")*"+dx+")";
    }
    if (Callee == "tanh" || Callee == "dream_tanh") {
        std::string th = "tanh("+x+")";
        return "((1.0f-"+th+"*"+th+")*"+dx+")";
    }
    if (Callee == "sin" || Callee == "dream_sin")
        return "(cos("+x+")*"+dx+")";
    if (Callee == "cos" || Callee == "dream_cos")
        return "(-sin("+x+")*"+dx+")";
    if (Callee == "tan" || Callee == "dream_tan") {
        std::string c = "cos("+x+")";
        return "("+dx+"/("+c+"*"+c+"))";
    }
    return "0.0f";
}
std::string BinaryExprAST::genOpenCL() { return "("+LHS->genOpenCL()+" "+Op+" "+RHS->genOpenCL()+")"; }
std::string LetStmtAST::genOpenCL()    { return "    float "+VarName+" = "+Expr->genOpenCL()+";\n"; }
std::string AssignStmtAST::genOpenCL() {
    bool isArg=std::find(GPUKernelArgs.begin(),GPUKernelArgs.end(),VarName)!=GPUKernelArgs.end();
    return "    "+(isArg?VarName+"[gid]":VarName)+" = "+Expr->genOpenCL()+";\n";
}
std::string IfStmtAST::genOpenCL() {
    std::string s="    if ("+Cond->genOpenCL()+") {\n";
    for(auto& st:ThenBody) s+="    "+st->genOpenCL();
    s+="    }";
    if(!ElseBody.empty()){s+=" else {\n";for(auto& st:ElseBody) s+="    "+st->genOpenCL();s+="    }\n";}
    else s+="\n";
    return s;
}
std::string WhileStmtAST::genOpenCL() {
    std::string s="    while ("+Cond->genOpenCL()+") {\n";
    for(auto& st:Body) s+="    "+st->genOpenCL();
    return s+"    }\n";
}
std::string ForStmtAST::genOpenCL()    { return "    // for\n"; }
std::string ReturnStmtAST::genOpenCL() { return Expr?"    return "+Expr->genOpenCL()+";\n":"    return;\n"; }
std::string FunctionAST::genOpenCL() {
    GPUKernelArgs.clear(); for(auto& a:Args) GPUKernelArgs.push_back(a.first);
    std::string s="__kernel void "+Name+"_kernel(";
    for(size_t i=0;i<Args.size();++i){s+="__global float* "+Args[i].first;if(i+1<Args.size())s+=", ";}
    s+=") {\n    int gid=get_global_id(0);\n\n";
    for(auto& st:Body) s+=st->genOpenCL();
    return s+"}\n";
}
std::string GradExprAST::genOpenCL() { return "grad("+Expr->genOpenCL()+")"; }
// AssertStmtAST
AssertStmtAST::AssertStmtAST(std::unique_ptr<ASTNode> Cond, std::string Msg)
    : Cond(std::move(Cond)), Msg(std::move(Msg)) {}

llvm::Value* AssertStmtAST::codegen() {
    auto* ocondV = Cond->codegen(); if (!ocondV) return nullptr;
    auto* condV = castToTensor(ocondV);
    auto* msgPtr = Builder.CreateGlobalStringPtr(Msg.empty() ? "assertion failed" : Msg, ".assertmsg");
    auto* F = getOrDeclare("dream_assert_fn", TPtrTy(), {TPtrTy(), CharPtrTy()});
    auto* res = Builder.CreateCall(F, {condV, msgPtr});
    auto* decF = getOrDeclare("tensor_decref", TPtrTy(), {TPtrTy()});
    Builder.CreateCall(decF, {res});          // assert_fn returns +1 on its arg
    decrefTemp(condV, ocondV, Cond.get());
    return llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(TPtrTy()));
}

// SaveStmtAST
SaveStmtAST::SaveStmtAST(std::unique_ptr<ASTNode> TensorExpr, std::string Filename)
    : TensorExpr(std::move(TensorExpr)), Filename(std::move(Filename)) {}

llvm::Value* SaveStmtAST::codegen() {
    auto* otv = TensorExpr->codegen(); if (!otv) return nullptr;
    auto* tv = castToTensor(otv);
    auto* filePtr = Builder.CreateGlobalStringPtr(Filename, ".savefile");
    auto* F = getOrDeclare("dream_save", TPtrTy(), {TPtrTy(), CharPtrTy()});
    auto* res = Builder.CreateCall(F, {tv, filePtr});
    auto* decF = getOrDeclare("tensor_decref", TPtrTy(), {TPtrTy()});
    Builder.CreateCall(decF, {res});          // dream_save returns +1 on its arg
    decrefTemp(tv, otv, TensorExpr.get());
    return llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(TPtrTy()));
}

// LoadExprAST
LoadExprAST::LoadExprAST(std::string Filename) : Filename(std::move(Filename)) {}

llvm::Value* LoadExprAST::codegen() {
    auto* filePtr = Builder.CreateGlobalStringPtr(Filename, ".loadfile");
    auto* F = getOrDeclare("dream_load", TPtrTy(), {CharPtrTy()});
    return Builder.CreateCall(F, {filePtr}, "loadtmp");
}

// tensor[index] -> calls tensor_index(tensor, index_as_tensor)
IndexExprAST::IndexExprAST(std::unique_ptr<ASTNode> Base, std::unique_ptr<ASTNode> Index)
    : Base(std::move(Base)), Index(std::move(Index)) {}

llvm::Value* IndexExprAST::codegen() {
    auto* obaseV = Base->codegen(); if (!obaseV) return nullptr;
    auto* baseV = castToTensor(obaseV);
    auto* oidxV = Index->codegen(); if (!oidxV) return nullptr;
    auto* idxV = castToTensor(oidxV);
    auto* F = getOrDeclare("tensor_index", TPtrTy(), {TPtrTy(), TPtrTy()});
    auto* res = Builder.CreateCall(F, {baseV, idxV}, "idxtmp");
    decrefTemp(baseV, obaseV, Base.get());
    decrefTemp(idxV, oidxV, Index.get());
    return res;
}

// tensor[i] = val / tensor[i][j] = val
IndexAssignStmtAST::IndexAssignStmtAST(std::string VarName,
                                         std::vector<std::unique_ptr<ASTNode>> Indices,
                                         std::unique_ptr<ASTNode> Value)
    : VarName(std::move(VarName)), Indices(std::move(Indices)), Value(std::move(Value)) {}

llvm::Value* IndexAssignStmtAST::codegen() {
    auto it = NamedValues.find(VarName);
    if (it == NamedValues.end()) {
        std::cerr << "\nerror: undefined variable '" << VarName << "'\n";
        return nullptr;
    }
    auto* alloca = llvm::cast<llvm::AllocaInst>(it->second);
    llvm::Value* base = Builder.CreateLoad(alloca->getAllocatedType(), alloca);
    base = castToTensor(base);

    auto* oval = Value->codegen(); if (!oval) return nullptr;
    auto* val = castToTensor(oval);

    if (Indices.size() == 1) {
        auto* oidx = Indices[0]->codegen(); if (!oidx) return nullptr;
        auto* idx = castToTensor(oidx);
        auto* F = getOrDeclare("tensor_set_element", TPtrTy(), {TPtrTy(), TPtrTy(), TPtrTy()});
        auto* r = Builder.CreateCall(F, {base, idx, val});
        decrefTemp(idx, oidx, Indices[0].get());
        decrefTemp(val, oval, Value.get());
        return r;
    } else if (Indices.size() == 2) {
        auto* oidx0 = Indices[0]->codegen(); if (!oidx0) return nullptr;
        auto* oidx1 = Indices[1]->codegen(); if (!oidx1) return nullptr;
        auto* idx0 = castToTensor(oidx0);
        auto* idx1 = castToTensor(oidx1);
        auto* F = getOrDeclare("tensor_set_element_2d", TPtrTy(),
                               {TPtrTy(), TPtrTy(), TPtrTy(), TPtrTy()});
        auto* r = Builder.CreateCall(F, {base, idx0, idx1, val});
        decrefTemp(idx0, oidx0, Indices[0].get());
        decrefTemp(idx1, oidx1, Indices[1].get());
        decrefTemp(val, oval, Value.get());
        return r;
    }
    std::cerr << "\nerror: too many indices (max 2)\n";
    return nullptr;
}

// [expr, expr, ...] -> tensor_from_list
ArrayLitExprAST::ArrayLitExprAST(std::vector<std::unique_ptr<ASTNode>> Elements)
    : Elements(std::move(Elements)) {}

llvm::Value* ArrayLitExprAST::codegen() {
    int n = (int)Elements.size();
    if (n == 0) {
        return Builder.CreateCall(
            getOrDeclare("alloc_tensor", TPtrTy(),
                         {Int32Ty(), Int32Ty(), Int32Ty(), Int32Ty()}),
            {Builder.getInt32(1), Builder.getInt32(1),
             Builder.getInt32(1), Builder.getInt32(1)});
    }
    // Allocate a stack array of doubles and fill it
    auto* arrPtr = Builder.CreateAlloca(DoubleTy(), Builder.getInt32(n), "arr_data");
    for (int i = 0; i < n; ++i) {
        auto* v = Elements[i]->codegen(); if (!v) return nullptr;
        // Convert to double
        if (v->getType()->isIntegerTy(32))
            v = Builder.CreateSIToFP(v, DoubleTy());
        else if (v->getType()->isIntegerTy(8))
            v = Builder.CreateSIToFP(Builder.CreateSExt(v, Int32Ty()), DoubleTy());
        else if (v->getType()->isPointerTy()) {
            // Extract first element from tensor
            auto* tv = v;
            auto* dp = Builder.CreateLoad(DblPtrTy(),
                           Builder.CreateStructGEP(TensorTy, tv, 0));
            v = Builder.CreateLoad(DoubleTy(), dp);
            decrefTemp(tv, tv, Elements[i].get());  // scalar copied out; drop fresh temp
        }
        Builder.CreateStore(v, Builder.CreateInBoundsGEP(DoubleTy(), arrPtr,
                                                          Builder.getInt32(i)));
    }
    auto* F = getOrDeclare("tensor_from_list", TPtrTy(), {DblPtrTy(), Int32Ty()});
    return Builder.CreateCall(F, {arrPtr, Builder.getInt32(n)}, "arrtmp");
}

// (a, b, c) tuple literal
TupleLitExprAST::TupleLitExprAST(std::vector<std::unique_ptr<ASTNode>> Elements)
    : Elements(std::move(Elements)) {}

llvm::Value* TupleLitExprAST::codegen() {
    int n = (int)Elements.size();
    auto* arrPtr = Builder.CreateAlloca(TPtrTy(), Builder.getInt32(n), "tuple_arr");
    std::vector<llvm::Value*> elemVals, elemOrig;
    for (int i = 0; i < n; ++i) {
        auto* ov = Elements[i]->codegen(); if (!ov) return nullptr;
        auto* v = castToTensor(ov);
        elemVals.push_back(v); elemOrig.push_back(ov);
        Builder.CreateStore(v, Builder.CreateInBoundsGEP(TPtrTy(), arrPtr,
                                                          Builder.getInt32(i)));
    }
    auto* packTy = llvm::PointerType::getUnqual(TPtrTy());
    auto* F = getOrDeclare("dream_pack", TPtrTy(), {packTy, Int32Ty()});
    auto* res = Builder.CreateCall(F, {arrPtr, Builder.getInt32(n)}, "tupletmp");
    // dream_pack incref'd every member as an owned parent; drop our temps.
    for (int i = 0; i < n; ++i)
        decrefTemp(elemVals[i], elemOrig[i], Elements[i].get());
    return res;
}

// let (x, y, z) = expr
DestructLetStmtAST::DestructLetStmtAST(std::vector<std::string> Names,
                                         std::unique_ptr<ASTNode> Expr)
    : Names(std::move(Names)), Expr(std::move(Expr)) {}

llvm::Value* DestructLetStmtAST::codegen() {
    auto* tupleV = Expr->codegen(); if (!tupleV) return nullptr;
    tupleV = castToTensor(tupleV);
    auto* unpackF = getOrDeclare("dream_unpack", TPtrTy(), {TPtrTy(), Int32Ty()});
    auto* incF    = getOrDeclare("tensor_incref", TPtrTy(), {TPtrTy()});
    auto* decF    = getOrDeclare("tensor_decref", TPtrTy(), {TPtrTy()});
    auto* markF   = getOrDeclare("tensor_mark_var", TPtrTy(), {TPtrTy()});
    auto* fn = Builder.GetInsertBlock()->getParent();
    for (int i = 0; i < (int)Names.size(); ++i) {
        auto* val = Builder.CreateCall(unpackF,
            {tupleV, Builder.getInt32(i)}, "unpack_" + Names[i]);

        // Loop-safe: reuse existing alloca if variable already declared
        llvm::AllocaInst* alloca = nullptr;
        bool reusing = false;
        auto existIt = NamedValues.find(Names[i]);
        if (existIt != NamedValues.end()) {
            alloca = llvm::dyn_cast<llvm::AllocaInst>(existIt->second);
            if (alloca && alloca->getAllocatedType() == TPtrTy()) reusing = true;
            else alloca = nullptr;
        }
        if (!alloca) alloca = createEntryAlloca(fn, Names[i], TPtrTy());

        // Decref old value, incref new value
        auto* oldVal = Builder.CreateLoad(TPtrTy(), alloca);
        Builder.CreateCall(decF, {oldVal});
        Builder.CreateCall(incF, {val});
        val = Builder.CreateCall(markF, {val});

        Builder.CreateStore(val, alloca);
        NamedValues[Names[i]] = alloca;
        if (!reusing) CurrentFnLocals.push_back(Names[i]);
    }
    // Release the tuple wrapper itself â€” but ONLY if this statement owns
    // it (fresh +1 from a call like next_batch()). If the source is a
    // borrowed variable (`let (a,b) = t`), the alloca still owns it.
    if (!isBorrowExpr(Expr.get()))
        Builder.CreateCall(decF, {tupleV});
    return tupleV;
}

// t[start:end] slice
SliceExprAST::SliceExprAST(std::unique_ptr<ASTNode> Base,
                            std::unique_ptr<ASTNode> Start,
                            std::unique_ptr<ASTNode> End)
    : Base(std::move(Base)), Start(std::move(Start)), End(std::move(End)) {}

llvm::Value* SliceExprAST::codegen() {
    auto* baseV = Base->codegen(); if (!baseV) return nullptr;
    auto* obaseV = baseV;
    baseV = castToTensor(baseV);
    auto* ostartV = Start->codegen(); if (!ostartV) return nullptr;
    auto* startV = castToTensor(ostartV);
    llvm::Value* endV; llvm::Value* oendV = nullptr; bool endAlwaysFresh = false;
    if (End) {
        oendV = End->codegen(); if (!oendV) return nullptr;
        endV = castToTensor(oendV);
    } else {
        // t[start:] means slice to the end â€” dream_len returns a fresh +1
        auto* lenF = getOrDeclare("dream_len", TPtrTy(), {TPtrTy()});
        endV = Builder.CreateCall(lenF, {baseV}, "lentmp");
        endAlwaysFresh = true;
    }
    auto* F = getOrDeclare("dream_slice", TPtrTy(), {TPtrTy(), TPtrTy(), TPtrTy()});
    auto* res = Builder.CreateCall(F, {baseV, startV, endV}, "slicetmp");
    decrefTemp(baseV, obaseV, Base.get());
    decrefTemp(startV, ostartV, Start.get());
    if (endAlwaysFresh) {
        auto* decF = getOrDeclare("tensor_decref", TPtrTy(), {TPtrTy()});
        Builder.CreateCall(decF, {endV});
    } else {
        decrefTemp(endV, oendV, End.get());
    }
    return res;
}

// struct.field ¡ú dream_struct_get(struct, field_index)
FieldAccessExprAST::FieldAccessExprAST(std::unique_ptr<ASTNode> Base, std::string FieldName)
    : Base(std::move(Base)), FieldName(std::move(FieldName)) {}

llvm::Value* FieldAccessExprAST::codegen() {
    auto* baseV = Base->codegen(); if (!baseV) return nullptr;
    baseV = castToTensor(baseV);
    auto* namePtr = Builder.CreateGlobalStringPtr(FieldName, ".fname");
    auto* F = getOrDeclare("dream_struct_get", TPtrTy(), {TPtrTy(), CharPtrTy()});
    return Builder.CreateCall(F, {baseV, namePtr}, "fldtmp");
}

// name.field = value ¡ú dream_struct_set(struct, field_name, value)
FieldAssignStmtAST::FieldAssignStmtAST(std::string VarName, std::string FieldName,
                                         std::unique_ptr<ASTNode> Value)
    : VarName(std::move(VarName)), FieldName(std::move(FieldName)), Value(std::move(Value)) {}

llvm::Value* FieldAssignStmtAST::codegen() {
    auto it = NamedValues.find(VarName);
    if (it == NamedValues.end()) {
        std::cerr << "\nerror: undefined variable '" << VarName << "'\n";
        return nullptr;
    }
    auto* alloca = llvm::cast<llvm::AllocaInst>(it->second);
    llvm::Value* base = Builder.CreateLoad(alloca->getAllocatedType(), alloca);
    base = castToTensor(base);
    auto* oval = Value->codegen(); if (!oval) return nullptr;
    auto* val = castToTensor(oval);
    auto* namePtr = Builder.CreateGlobalStringPtr(FieldName, ".fname");
    auto* F = getOrDeclare("dream_struct_set", TPtrTy(), {TPtrTy(), CharPtrTy(), TPtrTy()});
    auto* r = Builder.CreateCall(F, {base, namePtr, val});
    decrefTemp(val, oval, Value.get());
    return r;
}

// @function_name ¡ú creates fn ref and registers the function pointer
FnRefExprAST::FnRefExprAST(std::string FnName) : FnName(std::move(FnName)) {}

llvm::Value* FnRefExprAST::codegen() {
    auto* namePtr = Builder.CreateGlobalStringPtr(FnName, ".fnref");
    auto* i8PtrTy = CharPtrTy(); // i8* / ptr

    // If the function exists in the module, register its pointer
    // so resolve_dream_fn works without OS symbol table lookup
    auto* targetFn = TheModule->getFunction(FnName);
    if (targetFn) {
        auto* regF = getOrDeclare("dream_register_fn",
            llvm::Type::getVoidTy(TheContext), {i8PtrTy, i8PtrTy});
        Builder.CreateCall(regF, {namePtr, targetFn});
    }

    auto* F = getOrDeclare("dream_fn_ref", TPtrTy(), {i8PtrTy});
    return Builder.CreateCall(F, {namePtr}, "fnreftmp");
}