Tensor* dream_sin(Tensor* a) { Tensor* c=makeUnary(a,"sin"); runGPUUnary(a,c,"sin(a_val)"); return c; }
Tensor* dream_cos(Tensor* a) { Tensor* c=makeUnary(a,"cos"); runGPUUnary(a,c,"cos(a_val)"); return c; }
Tensor* dream_tan(Tensor* a) { Tensor* c=makeUnary(a,"tan"); runGPUUnary(a,c,"tan(a_val)"); return c; }
Tensor* dream_atan(Tensor* a) { Tensor* c=makeUnary(a,"atan"); runGPUUnary(a,c,"atan(a_val)"); return c; }

Tensor* dream_atan2(Tensor* y, Tensor* x) {
    checkShape(y, x, "atan2");
    Tensor* c = makeBinop(y, x, "atan2");
    runGPU(y, x, c, "atan2(a_val,b_val)");
    return c;
}

// Conditional select: where(cond, a, b) = cond > 0 ? a : b
Tensor* dream_where(Tensor* cond, Tensor* a, Tensor* b) {
    int n = std::max({cond->size, a->size, b->size});
    int rd0 = std::max({cond->d0, a->d0, b->d0});
    int rd1 = std::max({cond->d1, a->d1, b->d1});
    Tensor* c = alloc_tensor(rd0, rd1, 1, 1);
    c->is_leaf = false;
    for (int i = 0; i < c->size; ++i)
        c->data[i] = (cond->data[i % cond->size] > 0.0)
                    ? a->data[i % a->size] : b->data[i % b->size];
    return c;
}

// Element-wise min/max of two tensors
Tensor* dream_element_min(Tensor* a, Tensor* b) {
    checkShape(a, b, "element_min");
    Tensor* c = makeBinop(a, b, "emin");
    for (int i = 0; i < c->size; ++i)
        c->data[i] = std::min(a->data[i % a->size], b->data[i % b->size]);
    return c;
}
Tensor* dream_element_max(Tensor* a, Tensor* b) {
    checkShape(a, b, "element_max");
    Tensor* c = makeBinop(a, b, "emax");
    for (int i = 0; i < c->size; ++i)
        c->data[i] = std::max(a->data[i % a->size], b->data[i % b->size]);
    return c;
}

// Type casting helpers
Tensor* dream_int_cast(Tensor* t) {
    Tensor* c = alloc_tensor(t->d0, t->d1, t->d2, t->d3);
    for (int i = 0; i < t->size; ++i) c->data[i] = (double)(int)t->data[i];
    return c;
}
Tensor* dream_float_cast(Tensor* t) {
    Tensor* c = alloc_tensor(t->d0, t->d1, t->d2, t->d3);
    std::copy(t->data, t->data + t->size, c->data);
    return c;
}

// Axis reduction: sum_axis(tensor, axis) / mean_axis(tensor, axis)
Tensor* dream_sum_axis(Tensor* t, Tensor* axis_t) {
    int axis = (int)axis_t->data[0];
    if (axis == 0) {
        // Sum along rows -> 1×d1
        Tensor* c = alloc_tensor(1, t->d1, 1, 1);
        for (int j = 0; j < t->d1; ++j) {
            double s = 0;
            for (int i = 0; i < t->d0; ++i) s += t->data[i * t->d1 + j];
            c->data[j] = s;
        }
        return c;
    } else {
        // Sum along cols -> d0×1
        Tensor* c = alloc_tensor(t->d0, 1, 1, 1);
        for (int i = 0; i < t->d0; ++i) {
            double s = 0;
            for (int j = 0; j < t->d1; ++j) s += t->data[i * t->d1 + j];
            c->data[i] = s;
        }
        return c;
    }
}

Tensor* dream_mean_axis(Tensor* t, Tensor* axis_t) {
    int axis = (int)axis_t->data[0];
    Tensor* s = dream_sum_axis(t, axis_t);
    double div = (axis == 0) ? (double)t->d0 : (double)t->d1;
    for (int i = 0; i < s->size; ++i) s->data[i] /= div;
    return s;
}

// Struct support: uses tensor with grad_fn="struct:field1,field2,..."
// Fields are stored as embedded tensor pointers (like tuples)

// Create a struct with named fields: struct_new("Point", "x,y", val_x, val_y, ...)
// Called with tensor args packed by codegen
static std::map<std::string, std::vector<std::string>> g_struct_defs;

// ccccccccccccccccccccccccccccreate a struct tensor with named fields
