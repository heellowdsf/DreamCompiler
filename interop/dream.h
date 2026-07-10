/* =====================================================================
 *  dream.h -- C API for the Dream tensor / autograd runtime (libdream)
 *
 *  Build the library once:      dream lib            (-> libdream.a + dream.h)
 *  Link a C program (Linux):    cc app.c libdream.a -o app \
 *                                  -lstdc++ -lm -ldl -lgomp -Wl,--gc-sections
 *  Link (Windows, clang):       clang app.c libdream.a -o app.exe \
 *                                  -fopenmp -Xlinker /OPT:REF
 *  The runtime is compiled with -ffunction-sections, so --gc-sections /
 *  /OPT:REF strips every op your program does not use -- linking libdream
 *  does NOT bloat your executable.
 *
 *  OWNERSHIP: every function returning Tensor* returns a reference the
 *  caller owns; release it with dream_release(t). Functions never steal
 *  their arguments. Elementwise/matmul/nn ops build the autograd graph;
 *  call backward(loss) then read gradients via tensor_get_grad(param).
 *
 *  Dimension arguments follow the runtime convention of scalar tensors;
 *  use the drm_* inline helpers below for plain int/double ergonomics.
 * ===================================================================== */
#ifndef DREAM_H
#define DREAM_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Tensor Tensor;   /* opaque to C */

/* ---- lifetime & creation ---- */
Tensor* alloc_tensor(int d0, int d1, int d2, int d3);
Tensor* create_scalar(double v);
Tensor* tensor_incref(Tensor* t);
Tensor* tensor_decref(Tensor* t);
Tensor* dream_retain(Tensor* t);
Tensor* dream_release(Tensor* t);
Tensor* zeros(Tensor* d0, Tensor* d1);
Tensor* ones(Tensor* d0, Tensor* d1);
Tensor* dream_randn(Tensor* d0, Tensor* d1);
Tensor* dream_kaiming_init(Tensor* rows_t, Tensor* cols_t);
Tensor* dream_set_seed(Tensor* seed_t);
Tensor* dream_set_precision(Tensor* t, int code);

/* ---- element & shape access (C-side) ---- */
double* dream_data(Tensor* t);
int dream_size(Tensor* t);
int dream_dim0(Tensor* t);
int dream_dim1(Tensor* t);
int dream_dim2(Tensor* t);
int dream_dim3(Tensor* t);
double dream_get(Tensor* t, int i);
void dream_set(Tensor* t, int i, double v);

/* ---- elementwise & linear algebra ---- */
Tensor* tensor_add(Tensor* a, Tensor* b);
Tensor* tensor_sub(Tensor* a, Tensor* b);
Tensor* tensor_mul(Tensor* a, Tensor* b);
Tensor* tensor_div(Tensor* a, Tensor* b);
Tensor* tensor_matmul(Tensor* a, Tensor* b);
Tensor* dream_pow(Tensor* base, Tensor* exp_t);
Tensor* dream_clamp(Tensor* x, Tensor* lo_t, Tensor* hi_t);
Tensor* dream_transpose(Tensor* a);
Tensor* reshape(Tensor* t, Tensor* d0, Tensor* d1);
Tensor* dream_concat(Tensor* a, Tensor* b);
Tensor* dream_cumsum(Tensor* t);
Tensor* dream_argmax(Tensor* a);
Tensor* sum(Tensor* a);
Tensor* mean(Tensor* a);
Tensor* dream_sum_exact(Tensor* a);
Tensor* dream_mean_exact(Tensor* a);

/* ---- activations ---- */
Tensor* relu(Tensor* a);
Tensor* dream_abs(Tensor* a);
Tensor* dream_exp(Tensor* a);
Tensor* dream_log(Tensor* a);
Tensor* dream_sqrt(Tensor* a);
Tensor* dream_sigmoid(Tensor* a);
Tensor* dream_tanh(Tensor* a);
Tensor* dream_gelu(Tensor* x);
Tensor* dream_silu(Tensor* x);
Tensor* dream_mish(Tensor* x);
Tensor* dream_elu(Tensor* x, Tensor* alpha_t);
Tensor* dream_leaky_relu(Tensor* x, Tensor* alpha_t);
Tensor* dream_sin(Tensor* a);
Tensor* dream_cos(Tensor* a);
Tensor* dream_softmax(Tensor* a);
Tensor* dream_log_softmax(Tensor* x);

/* ---- neural network ops ---- */
Tensor* dream_linear(Tensor* x, Tensor* W, Tensor* b);
Tensor* dream_embedding(Tensor* indices, Tensor* weight);
Tensor* dream_attention(Tensor* Q, Tensor* K, Tensor* V, Tensor* batch_t, Tensor* seq_t, Tensor* dk_t);
Tensor* dream_bmm(Tensor* a, Tensor* b, Tensor* batch_t, Tensor* m_t, Tensor* k_t, Tensor* n_t);
Tensor* dream_layer_norm(Tensor* t);
Tensor* dream_batchnorm2d(Tensor* input, Tensor* h_t, Tensor* w_t, Tensor* c_t);
Tensor* dream_conv2d(Tensor* input, Tensor* kernel, Tensor* bias, Tensor* h_t, Tensor* w_t, Tensor* kh_t, Tensor* kw_t);
Tensor* dream_maxpool2d(Tensor* input, Tensor* h_t, Tensor* w_t, Tensor* channels_t, Tensor* pool_t);
Tensor* dream_avgpool2d(Tensor* input, Tensor* h_t, Tensor* w_t, Tensor* channels_t, Tensor* pool_t);
Tensor* dream_dropout(Tensor* t, Tensor* keep_rate);

/* ---- losses ---- */
Tensor* mse_loss(Tensor* pred, Tensor* target);
Tensor* dream_nll_loss(Tensor* log_probs, Tensor* target);
Tensor* dream_cross_entropy(Tensor* logits, Tensor* target);
Tensor* dream_bce_loss(Tensor* pred, Tensor* target);
Tensor* dream_huber_loss(Tensor* pred, Tensor* target, Tensor* delta_t);

/* ---- autograd & optimizers ---- */
Tensor* backward(Tensor* root);
Tensor* zero_grad(Tensor* t);
Tensor* tensor_get_grad(Tensor* t);
Tensor* dream_param_update(Tensor* param, Tensor* grad_t, Tensor* lr_t);
Tensor* dream_adamw_update(Tensor* param, Tensor* lr_t, Tensor* b1_t, Tensor* b2_t, Tensor* eps_t, Tensor* wd_t);
Tensor* dream_adamw_reset(Tensor* param);

/* ---- io & verification ---- */
Tensor* dream_load_csv(const char* path, int skip_header);
Tensor* dream_save(Tensor* t, const char* filename);
Tensor* print(Tensor* t);
Tensor* dream_print_shape(Tensor* t);
Tensor* gradcheck();
Tensor* gemm_selftest();

#ifdef __cplusplus
}  /* extern "C" */
#endif

/* ---- drm_* convenience wrappers (header-only, plain C types) ---- */
static inline Tensor* drm_scalar(double v) { return create_scalar(v); }
static inline Tensor* drm_zeros(int r, int c) {
    Tensor *a = create_scalar((double)r), *b = create_scalar((double)c);
    Tensor *t = zeros(a, b); dream_release(a); dream_release(b); return t;
}
static inline Tensor* drm_ones(int r, int c) {
    Tensor *a = create_scalar((double)r), *b = create_scalar((double)c);
    Tensor *t = ones(a, b); dream_release(a); dream_release(b); return t;
}
static inline Tensor* drm_randn(int r, int c) {
    Tensor *a = create_scalar((double)r), *b = create_scalar((double)c);
    Tensor *t = dream_randn(a, b); dream_release(a); dream_release(b); return t;
}
static inline Tensor* drm_kaiming(int rows, int cols) {
    Tensor *a = create_scalar((double)rows), *b = create_scalar((double)cols);
    Tensor *t = dream_kaiming_init(a, b); dream_release(a); dream_release(b); return t;
}
static inline void drm_seed(unsigned s) {
    Tensor* t = create_scalar((double)s);
    Tensor* r = dream_set_seed(t);
    dream_release(t); dream_release(r);
}
static inline void drm_adamw(Tensor* param, double lr, double b1, double b2,
                             double eps, double wd) {
    Tensor *a = create_scalar(lr),  *b = create_scalar(b1), *c = create_scalar(b2);
    Tensor *d = create_scalar(eps), *e = create_scalar(wd);
    Tensor *r = dream_adamw_update(param, a, b, c, d, e);
    dream_release(a); dream_release(b); dream_release(c);
    dream_release(d); dream_release(e); dream_release(r);
    Tensor *z = zero_grad(param); dream_release(z);
}

#endif /* DREAM_H */
