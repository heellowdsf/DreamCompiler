Tensor* tensor_index(Tensor* t, Tensor* idx_t) {
    if (!t || !idx_t) { std::cerr << "error: null in tensor_index\n" << std::flush; exit(1); }
    sync_f32_to_data(t);   // flush pending f32 result before reading data
    int idx = (int)idx_t->data[0];

    // For 1×N tensors (vectors from linspace, arange, slice etc.),
    // always treat as flat element access
    if (t->d0 == 1) {
        int total = t->d1 * t->d2 * t->d3;
        if (idx < 0) idx += total;
        if (idx < 0 || idx >= total) {
            std::cerr << "runtime error: index " << (int)idx_t->data[0]
                      << " out of bounds (size " << total << ")\n" << std::flush;
            return create_temp_scalar(0.0);
        }
        return create_temp_scalar(t->data[idx]);
    }

    // For N×1×1×1 flat vectors, direct element access
    if (t->d1 == 1 && t->d2 == 1 && t->d3 == 1) {
        if (idx < 0) idx += t->d0;
        if (idx < 0 || idx >= t->d0) {
            std::cerr << "runtime error: index " << (int)idx_t->data[0]
                      << " out of bounds (size " << t->d0 << ")\n" << std::flush;
            return create_temp_scalar(0.0);
        }
        return create_temp_scalar(t->data[idx]);
    }

    // For N×M multi-dimensional: index along d0, return row as flat vector
    int slice = t->d1 * t->d2 * t->d3;
    if (idx < 0) idx += t->d0;
    if (idx < 0 || idx >= t->d0) {
        std::cerr << "runtime error: index " << (int)idx_t->data[0]
                  << " out of bounds for dim0 (size " << t->d0 << ")\n" << std::flush;
        return create_temp_scalar(0.0);
    }
    if (slice == 1) return create_temp_scalar(t->data[idx]);
    if (t->d2 == 1 && t->d3 == 1) {
        Tensor* r = alloc_tensor(t->d1, 1, 1, 1);
        std::copy(t->data + idx * slice, t->data + (idx + 1) * slice, r->data);
        return r;
    }
    Tensor* r = alloc_tensor(t->d1, t->d2, t->d3, 1);
    std::copy(t->data + idx * slice, t->data + (idx + 1) * slice, r->data);
    return r;
}

// 1D index write: t[i] = val (writes scalar or copies slice)
Tensor* tensor_set_element(Tensor* t, Tensor* idx_t, Tensor* val) {
    if (!t || !idx_t || !val) return t;
    int idx = (int)idx_t->data[0];
    bool has_inner = (t->d1 > 1 || t->d2 > 1 || t->d3 > 1);

    if (!has_inner) {
        // Flat vector
        if (idx < 0) idx += t->d0;
        if (idx < 0 || idx >= t->d0) {
            std::cerr << "runtime error: set index " << (int)idx_t->data[0]
                      << " out of bounds (size " << t->d0 << ")\n" << std::flush;
            return t;
        }
        t->data[idx] = val->data[0];
        return t;
    }

    // Multi-dimensional: write along d0
    int slice = t->d1 * t->d2 * t->d3;
    int dim_size = t->d0;

    if (dim_size == 1 && idx != 0) {
        // 1×N convenience: flat write
        int total = slice;
        if (idx < 0) idx += total;
        if (idx < 0 || idx >= total) {
            std::cerr << "runtime error: set index " << (int)idx_t->data[0]
                      << " out of bounds (size " << total << ")\n" << std::flush;
            return t;
        }
        t->data[idx] = val->data[0];
        return t;
    }

    if (idx < 0) idx += dim_size;
    if (idx < 0 || idx >= dim_size) {
        std::cerr << "runtime error: set index " << (int)idx_t->data[0]
                  << " out of bounds (size " << dim_size << ")\n" << std::flush;
        return t;
    }
    int offset = idx * slice;
    int copy_n = std::min(val->size, slice);
    std::copy(val->data, val->data + copy_n, t->data + offset);
    return t;
}

// 2D index write: t[row, col] = val
Tensor* tensor_set_element_2d(Tensor* t, Tensor* row_t, Tensor* col_t, Tensor* val) {
    if (!t || !row_t || !col_t || !val) return t;
    int row = (int)row_t->data[0];
    int col = (int)col_t->data[0];
    int cols = t->d1;
    if (t->d0 <= 1 && t->d1 > 1) { cols = t->d1; }
    if (row < 0) row += t->d0;
    if (col < 0) col += cols;
    int flat = row * cols + col;
    if (flat < 0 || flat >= t->size) {
        std::cerr << "runtime error: index [" << (int)row_t->data[0] << ", "
                  << (int)col_t->data[0] << "] out of bounds\n" << std::flush;
        return t;
    }
    t->data[flat] = val->data[0];
    return t;
}

// Create tensor from an array of doubles (for [1, 2, 3] literal)
Tensor* tensor_from_list(double* data, int n) {
    if (n <= 0) return alloc_tensor(1, 1, 1, 1);
    Tensor* t = alloc_tensor(1, n, 1, 1);
    for (int i = 0; i < n; ++i) t->data[i] = data[i];
    return t;
}

// len(tensor) -> number of elements
Tensor* dream_len(Tensor* t) {
    return create_scalar(t ? (double)t->size : 0.0);
}

// shape_of(tensor) -> [d0, d1, d2, d3] tensor
Tensor* dream_shape_of(Tensor* t) {
    Tensor* r = alloc_tensor(1, 4, 1, 1);
    if (t) {
        r->data[0] = t->d0; r->data[1] = t->d1;
        r->data[2] = t->d2; r->data[3] = t->d3;
    }
    return r;
}

// fill(rows, cols, value) -> tensor of given shape filled with value
Tensor* dream_fill(Tensor* d0, Tensor* d1, Tensor* val) {
    int r = (int)d0->data[0], c = (int)d1->data[0];
    Tensor* t = alloc_tensor(r, c, 1, 1);
    double v = val->data[0];
    std::fill(t->data, t->data + t->size, v);
    return t;
}

// copy(tensor) -> deep copy
Tensor* dream_copy(Tensor* t) {
    if (!t) return nullptr;
    Tensor* c = alloc_tensor(t->d0, t->d1, t->d2, t->d3);
    std::copy(t->data, t->data + t->size, c->data);
    return c;
}

// slice(tensor, start, end) -> sub-tensor from flat index start to end
Tensor* dream_slice(Tensor* t, Tensor* start_t, Tensor* end_t) {
    if (!t) return nullptr;
    int s = (int)start_t->data[0], e = (int)end_t->data[0];
    if (s < 0) s += t->size;
    if (e < 0) e += t->size;
    if (s < 0) s = 0;
    if (e > t->size) e = t->size;
    int n = e - s;
    if (n <= 0) return alloc_tensor(1, 1, 1, 1);
    Tensor* r = alloc_tensor(1, n, 1, 1);
    std::copy(t->data + s, t->data + e, r->data);
    return r;
}

// eye(n) -> n×n identity matrix
Tensor* dream_eye(Tensor* n_t) {
    int n = (int)n_t->data[0];
    Tensor* t = alloc_tensor(n, n, 1, 1);
    for (int i = 0; i < n; ++i) t->data[i * n + i] = 1.0;
    return t;
}

// Tuple pack: stores N tensor pointers in the wrapper's parents array.
// Pure refcount: each member is incref'd; when the wrapper is destroyed,
// free_tensor decrefs the parents automatically.
//
// We ALSO store pointers in data[] (as bit patterns) so dream_unpack
// can extract them by index without walking parents — this is just a
// fast path; the authoritative ownership is via parents[].
Tensor* dream_pack(Tensor** arr, int n) {
    Tensor* t = alloc_tensor(1, n, 1, 1);
    t->grad_fn = "tuple";
    t->is_leaf = false;
    // Set parents to take ownership (each member incref'd by the helper)
    Tensor** par = new Tensor*[n];
    for (int i = 0; i < n; ++i) par[i] = arr[i];
    tensor_set_parents(t, par, n);
    // Also store pointers in data[] for fast unpack by index
    for (int i = 0; i < n; ++i) {
        uintptr_t ptr = reinterpret_cast<uintptr_t>(arr[i]);
        std::memcpy(&t->data[i], &ptr, sizeof(ptr));
    }
    return t;
}

// Tuple unpack: extract the i-th tensor pointer (no incref — caller does that)
Tensor* dream_unpack(Tensor* t, int idx) {
    if (!t || idx < 0 || idx >= t->size) {
        std::cerr << "error: unpack index " << idx << " out of bounds (tuple size "
                  << (t ? t->size : 0) << ")\n" << std::flush;
        return create_scalar(0.0);
    }
    uintptr_t ptr = 0;
    std::memcpy(&ptr, &t->data[idx], sizeof(ptr));
    return reinterpret_cast<Tensor*>(ptr);
}
