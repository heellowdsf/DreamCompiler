Tensor* dream_str_create(const char* s) {
    int len = s ? (int)std::strlen(s) : 0;
    if (len == 0) {
        Tensor* t = alloc_tensor(1, 1, 1, 1);
        t->grad_fn = "string";
        t->data[0] = 0;
        return t;
    }
    Tensor* t = alloc_tensor(1, len, 1, 1);
    t->grad_fn = "string";
    for (int i = 0; i < len; ++i) t->data[i] = (double)(unsigned char)s[i];
    return t;
}

// string concatenation
Tensor* dream_str_concat(Tensor* a, Tensor* b) {
    int na = a ? a->size : 0, nb = b ? b->size : 0;
    Tensor* t = alloc_tensor(1, na + nb, 1, 1);
    t->grad_fn = "string";
    for (int i = 0; i < na; ++i) t->data[i] = a->data[i];
    for (int i = 0; i < nb; ++i) t->data[na + i] = b->data[i];
    return t;
}

// string length
Tensor* dream_str_len(Tensor* t) {
    return create_scalar(t ? (double)t->size : 0.0);
}

// number to string
Tensor* dream_to_string(Tensor* t) {
    if (!t) return dream_str_create("null");
    std::string s = std::to_string(t->data[0]);
    // Trim trailing zeros for clean output
    size_t dot = s.find('.');
    if (dot != std::string::npos) {
        size_t last = s.find_last_not_of('0');
        if (last == dot) last++;
        s = s.substr(0, last + 1);
    }
    return dream_str_create(s.c_str());
}

// trig functions
