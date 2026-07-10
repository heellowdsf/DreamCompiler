Tensor* dream_struct_create(const char* fields_str, int nfields) {
    Tensor* t = alloc_tensor(1, nfields, 1, 1);
    t->grad_fn = std::string("struct:") + fields_str;
    for (int i = 0; i < nfields; ++i) t->data[i] = 0.0;
    return t;
}

Tensor* dream_struct_new(Tensor* name_t, Tensor* fields_t) {
    // Extract name and field names from string tensors
    std::string name, fields_str;
    if (name_t && name_t->grad_fn == "string")
        for (int i = 0; i < name_t->size; ++i) name += (char)(int)name_t->data[i];
    if (fields_t && fields_t->grad_fn == "string")
        for (int i = 0; i < fields_t->size; ++i) fields_str += (char)(int)fields_t->data[i];

    // Parse comma-separated field names
    std::vector<std::string> flds;
    std::string cur;
    for (char c : fields_str) {
        if (c == ',') { if (!cur.empty()) flds.push_back(cur); cur.clear(); }
        else cur += c;
    }
    if (!cur.empty()) flds.push_back(cur);
    g_struct_defs[name] = flds;

    // Return a template tensor (no values yet)
    Tensor* t = alloc_tensor(1, (int)flds.size(), 1, 1);
    t->grad_fn = "struct:" + fields_str;
    return t;
}

// Get field by name from struct tensor
Tensor* dream_struct_get(Tensor* s, const char* field_name) {
    if (!s || !field_name) return create_scalar(0.0);
    std::string fn = field_name;

    // Parse field names from grad_fn "struct:x,y,z"
    std::string meta = s->grad_fn;
    if (meta.substr(0, 7) != "struct:") {
        // Not a struct, try numeric access (tensor property)
        if (fn == "size") return create_scalar((double)s->size);
        if (fn == "d0") return create_scalar((double)s->d0);
        if (fn == "d1") return create_scalar((double)s->d1);
        if (fn == "d2") return create_scalar((double)s->d2);
        if (fn == "d3") return create_scalar((double)s->d3);
        std::cerr << "error: not a struct, cannot access field '" << fn << "'\n" << std::flush;
        return create_scalar(0.0);
    }

    // Find field index
    std::string flist = meta.substr(7);
    std::vector<std::string> fields;
    std::string cur;
    for (char c : flist) {
        if (c == ',') { fields.push_back(cur); cur.clear(); }
        else cur += c;
    }
    if (!cur.empty()) fields.push_back(cur);

    for (int i = 0; i < (int)fields.size(); ++i) {
        if (fields[i] == fn) {
            // Retrieve embedded tensor pointer
            uintptr_t ptr = 0;
            std::memcpy(&ptr, &s->data[i], sizeof(ptr));
            Tensor* val = reinterpret_cast<Tensor*>(ptr);
            // OWNERSHIP: return +1 like every other runtime expression.
            return val ? tensor_incref(val) : create_scalar(0.0);
        }
    }
    std::cerr << "error: struct has no field '" << fn << "'\n" << std::flush;
    return create_scalar(0.0);
}

// Set field by name on struct tensor
Tensor* dream_struct_set(Tensor* s, const char* field_name, Tensor* val) {
    if (!s || !field_name) return s;
    std::string fn = field_name;
    std::string meta = s->grad_fn;
    if (meta.substr(0, 7) != "struct:") {
        std::cerr << "error: not a struct, cannot set field '" << fn << "'\n" << std::flush;
        return s;
    }
    std::string flist = meta.substr(7);
    std::vector<std::string> fields;
    std::string cur;
    for (char c : flist) {
        if (c == ',') { fields.push_back(cur); cur.clear(); }
        else cur += c;
    }
    if (!cur.empty()) fields.push_back(cur);

    for (int i = 0; i < (int)fields.size(); ++i) {
        if (fields[i] == fn) {
            // OWNERSHIP: struct fields hold +1 owning references.
            // incref the new member, decref whatever was stored before.
            uintptr_t old_bits = 0;
            std::memcpy(&old_bits, &s->data[i], sizeof(old_bits));
            Tensor* old_member = reinterpret_cast<Tensor*>(old_bits);
            if (val) tensor_incref(val);
            uintptr_t ptr = reinterpret_cast<uintptr_t>(val);
            std::memcpy(&s->data[i], &ptr, sizeof(ptr));
            if (old_member) tensor_decref(old_member);
            return s;
        }
    }
    std::cerr << "error: struct has no field '" << fn << "'\n" << std::flush;
    return s;
}

// Function registry (cross-platform, no OS symbol table dependency)
