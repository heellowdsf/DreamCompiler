static std::map<std::string, void*> g_fn_registry;

void dream_register_fn(const char* name, void* ptr) {
    if (name && ptr) g_fn_registry[name] = ptr;
}

static void* resolve_dream_fn(const std::string& name) {
    auto it = g_fn_registry.find(name);
    if (it != g_fn_registry.end()) return it->second;
#if defined(_WIN32)
    HMODULE self = GetModuleHandleA(NULL);
    if (self) {
        void* p = (void*)GetProcAddress(self, name.c_str());
        if (p) return p;
    }
#elif defined(__linux__) || defined(__APPLE__)
    void* p = dlsym(RTLD_DEFAULT, name.c_str());
    if (p) return p;
#endif
    return nullptr;
}

// Function references and closures
Tensor* dream_fn_ref(const char* name) {
    if (!name) return create_scalar(0.0);
    int len = (int)std::strlen(name);
    Tensor* t = alloc_tensor(1, len, 1, 1);
    t->grad_fn = "fn";
    for (int i = 0; i < len; ++i) t->data[i] = (double)(unsigned char)name[i];
    return t;
}

typedef Tensor* (*DreamFn1)(Tensor*);
typedef Tensor* (*DreamFn2)(Tensor*, Tensor*);
typedef Tensor* (*DreamFn3)(Tensor*, Tensor*, Tensor*);
typedef Tensor* (*DreamFn4)(Tensor*, Tensor*, Tensor*, Tensor*);

// Create a closure: fn name + captured variables
Tensor* dream_make_closure(Tensor* name_t) {
    Tensor* closure = alloc_tensor(1, 1, 1, 1);
    closure->grad_fn = "closure";
    uintptr_t ptr = reinterpret_cast<uintptr_t>(name_t);
    std::memcpy(&closure->data[0], &ptr, sizeof(ptr));
    closure->num_parents = 0;
    closure->parents = nullptr;
    // Register the function if found
    if (name_t && name_t->grad_fn == "string") {
        std::string nm;
        for (int i = 0; i < name_t->size; ++i) nm += (char)(int)name_t->data[i];
        // Try to resolve and register
        void* fp = resolve_dream_fn(nm);
        if (fp) dream_register_fn(nm.c_str(), fp);
    }
    return closure;
}
Tensor* dream_make_closure2(Tensor* name_t, Tensor* c1) {
    Tensor* cl = dream_make_closure(name_t);
    if (c1) c1->refcount++;
    { Tensor** par = new Tensor*[1]{c1}; tensor_set_parents(cl, par, 1); } return cl;
}
Tensor* dream_make_closure3(Tensor* name_t, Tensor* c1, Tensor* c2) {
    Tensor* cl = dream_make_closure(name_t);
    if (c1) c1->refcount++;
    if (c2) c2->refcount++;
    { Tensor** par = new Tensor*[2]{c1, c2}; tensor_set_parents(cl, par, 2); } return cl;
}
Tensor* dream_make_closure4(Tensor* name_t, Tensor* c1, Tensor* c2, Tensor* c3) {
    Tensor* cl = dream_make_closure(name_t);
    if (c1) c1->refcount++;
    if (c2) c2->refcount++;
    if (c3) c3->refcount++;
    { Tensor** par = new Tensor*[3]{c1, c2, c3}; tensor_set_parents(cl, par, 3); } return cl;
}

static std::string extract_fn_name(Tensor* ref) {
    if (!ref) return "";
    if (ref->grad_fn == "fn") {
        std::string n; for (int i=0; i<ref->size; ++i) n+=(char)(int)ref->data[i]; return n;
    }
    if (ref->grad_fn == "closure") {
        uintptr_t ptr=0; std::memcpy(&ptr, &ref->data[0], sizeof(ptr));
        Tensor* nt = reinterpret_cast<Tensor*>(ptr);
        if (nt && nt->grad_fn == "string") {
            std::string n; for (int i=0; i<nt->size; ++i) n+=(char)(int)nt->data[i]; return n;
        }
    }
    return "";
}

Tensor* dream_call(Tensor* fn_ref, Tensor* a1) {
    if (!fn_ref) { std::cerr << "error: call() on null\n" << std::flush; return create_scalar(0.0); }
    std::string name = extract_fn_name(fn_ref);
    void* fp = resolve_dream_fn(name);
    if (!fp) { std::cerr << "error: function '" << name << "' not found\n" << std::flush; return create_scalar(0.0); }
    if (fn_ref->grad_fn == "closure" && fn_ref->num_parents > 0) {
        int nc = fn_ref->num_parents;
        if (nc==1) return ((DreamFn2)fp)(fn_ref->parents[0], a1);
        if (nc==2) return ((DreamFn3)fp)(fn_ref->parents[0], fn_ref->parents[1], a1);
        if (nc==3) return ((DreamFn4)fp)(fn_ref->parents[0], fn_ref->parents[1], fn_ref->parents[2], a1);
    }
    return ((DreamFn1)fp)(a1);
}

Tensor* dream_map(Tensor* fn_ref, Tensor* data) {
    if (!fn_ref || !data) return create_scalar(0.0);
    std::string name = extract_fn_name(fn_ref);
    void* fp = resolve_dream_fn(name);
    if (!fp) { std::cerr << "error: map: '" << name << "' not found\n" << std::flush; return create_scalar(0.0); }
    bool has_c = fn_ref->grad_fn=="closure" && fn_ref->num_parents>0;
    Tensor* result = alloc_tensor(data->d0, data->d1, 1, 1);
    for (int i=0; i<data->size; ++i) {
        Tensor* e = create_scalar(data->data[i]);
        Tensor* r;
        if (has_c && fn_ref->num_parents==1) r = ((DreamFn2)fp)(fn_ref->parents[0], e);
        else if (has_c && fn_ref->num_parents==2) r = ((DreamFn3)fp)(fn_ref->parents[0], fn_ref->parents[1], e);
        else r = ((DreamFn1)fp)(e);
        result->data[i] = r->data[0];
    }
    return result;
}

Tensor* dream_filter(Tensor* fn_ref, Tensor* data) {
    if (!fn_ref || !data) return create_scalar(0.0);
    std::string name = extract_fn_name(fn_ref);
    void* fp = resolve_dream_fn(name);
    if (!fp) return create_scalar(0.0);
    bool has_c = fn_ref->grad_fn=="closure" && fn_ref->num_parents>0;
    std::vector<double> kept;
    for (int i=0; i<data->size; ++i) {
        Tensor* e = create_scalar(data->data[i]);
        Tensor* r;
        if (has_c && fn_ref->num_parents==1) r = ((DreamFn2)fp)(fn_ref->parents[0], e);
        else r = ((DreamFn1)fp)(e);
        if (r->data[0] != 0.0) kept.push_back(data->data[i]);
    }
    if (kept.empty()) return alloc_tensor(1, 1, 1, 1);
    Tensor* result = alloc_tensor(1, (int)kept.size(), 1, 1);
    for (int i=0; i<(int)kept.size(); ++i) result->data[i] = kept[i];
    return result;
}

Tensor* dream_reduce(Tensor* fn_ref, Tensor* data, Tensor* init) {
    if (!fn_ref || !data) return init ? init : create_scalar(0.0);
    std::string name = extract_fn_name(fn_ref);
    void* fp = resolve_dream_fn(name);
    if (!fp) return init ? init : create_scalar(0.0);
    Tensor* acc = init ? init : create_scalar(0.0);
    for (int i=0; i<data->size; ++i) {
        Tensor* e = create_scalar(data->data[i]);
        acc = ((DreamFn2)fp)(acc, e);
    }
    return acc;
}

