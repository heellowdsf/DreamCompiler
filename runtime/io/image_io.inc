// Dream Runtime - Image I/O and training data generation
// Supports: BMP loading/saving, synthetic pattern generation

// Load a 24-bit BMP file into a Tensor (flattened, normalized 0-1)
// Returns shape (1, height*width*channels) for grayscale or RGB
Tensor* dream_load_bmp(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        std::cerr << "error: cannot open image '" << path << "'\n" << std::flush;
        return alloc_tensor(1, 1, 1, 1);
    }
    unsigned char header[54];
    if (fread(header, 1, 54, f) != 54) { fclose(f); return alloc_tensor(1, 1, 1, 1); }
    if (header[0] != 'B' || header[1] != 'M') {
        std::cerr << "error: not a BMP file\n" << std::flush;
        fclose(f); return alloc_tensor(1, 1, 1, 1);
    }
    int width  = *(int*)&header[18];
    int height = *(int*)&header[22];
    int bpp    = *(short*)&header[28];
    int offset = *(int*)&header[10];
    if (bpp != 24 && bpp != 8) {
        std::cerr << "error: only 24-bit and 8-bit BMP supported (got " << bpp << ")\n" << std::flush;
        fclose(f); return alloc_tensor(1, 1, 1, 1);
    }
    fseek(f, offset, SEEK_SET);
    int channels = (bpp == 24) ? 3 : 1;
    int rowSize = ((width * channels + 3) / 4) * 4; // BMP rows are 4-byte aligned
    int pixels = width * height;
    // Output: grayscale flattened (convert RGB to gray if needed)
    Tensor* t = alloc_tensor(height, width, 1, 1);
    std::vector<unsigned char> row(rowSize);
    for (int y = height - 1; y >= 0; --y) { // BMP is bottom-up
        if (fread(row.data(), 1, rowSize, f) != (size_t)rowSize) break;
        for (int x = 0; x < width; ++x) {
            if (channels == 3) {
                double b = row[x*3+0] / 255.0;
                double g = row[x*3+1] / 255.0;
                double r = row[x*3+2] / 255.0;
                t->data[y * width + x] = 0.299*r + 0.587*g + 0.114*b; // grayscale
            } else {
                t->data[y * width + x] = row[x] / 255.0;
            }
        }
    }
    fclose(f);
    return t;
}

// Load image and return shape info as tuple (image_tensor, width, height)
Tensor* dream_load_image(Tensor* path_t) {
    if (!path_t) return alloc_tensor(1, 1, 1, 1);
    std::string path;
    if (path_t->grad_fn == "string")
        for (int i = 0; i < path_t->size; ++i) path += (char)(int)path_t->data[i];
    else
        return alloc_tensor(1, 1, 1, 1);
    return dream_load_bmp(path.c_str());
}

// Save tensor as 8-bit grayscale BMP
Tensor* dream_save_bmp(Tensor* img, Tensor* path_t) {
    if (!img || !path_t) return create_scalar(0.0);
    std::string path;
    if (path_t->grad_fn == "string")
        for (int i = 0; i < path_t->size; ++i) path += (char)(int)path_t->data[i];
    else return create_scalar(0.0);

    int h = img->d0, w = img->d1;
    int rowSize = ((w + 3) / 4) * 4;
    int imgSize = rowSize * h;
    int fileSize = 54 + 256*4 + imgSize; // header + palette + pixels

    dream_mkdirs_for(path);
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { std::cerr << "error: cannot write '" << path << "'\n"; return create_scalar(0.0); }

    // BMP header
    unsigned char header[54] = {};
    header[0]='B'; header[1]='M';
    *(int*)&header[2] = fileSize;
    *(int*)&header[10] = 54 + 256*4; // pixel offset (after palette)
    *(int*)&header[14] = 40; // DIB header size
    *(int*)&header[18] = w;
    *(int*)&header[22] = h;
    *(short*)&header[26] = 1; // planes
    *(short*)&header[28] = 8; // 8-bit
    *(int*)&header[34] = imgSize;
    fwrite(header, 1, 54, f);

    // Grayscale palette
    for (int i = 0; i < 256; ++i) {
        unsigned char c[4] = {(unsigned char)i, (unsigned char)i, (unsigned char)i, 0};
        fwrite(c, 1, 4, f);
    }

    // Pixel data (bottom-up)
    std::vector<unsigned char> row(rowSize, 0);
    for (int y = h - 1; y >= 0; --y) {
        for (int x = 0; x < w; ++x) {
            double v = img->data[y * w + x];
            v = std::max(0.0, std::min(1.0, v));
            row[x] = (unsigned char)(v * 255.0);
        }
        fwrite(row.data(), 1, rowSize, f);
    }
    fclose(f);
    return create_scalar(1.0);
}

// Generate synthetic 8x8 digit patterns (0-9) for training
// Returns a tensor of shape (n_samples, 64) and labels (n_samples, 10) one-hot
static void draw_digit(double* pixels, int digit) {
    // 8x8 pixel art for digits 0-9
    static const char* patterns[10] = {
        // 0
        "01111100"
        "11000110"
        "11001110"
        "11011110"
        "11110110"
        "11100110"
        "01111100"
        "00000000",
        // 1
        "00011000"
        "00111000"
        "01111000"
        "00011000"
        "00011000"
        "00011000"
        "01111110"
        "00000000",
        // 2
        "01111100"
        "11000110"
        "00001110"
        "00111100"
        "01110000"
        "11000000"
        "11111110"
        "00000000",
        // 3
        "01111100"
        "11000110"
        "00000110"
        "00111100"
        "00000110"
        "11000110"
        "01111100"
        "00000000",
        // 4
        "00001100"
        "00011100"
        "00111100"
        "01101100"
        "11001100"
        "11111110"
        "00001100"
        "00000000",
        // 5
        "11111110"
        "11000000"
        "11111100"
        "00000110"
        "00000110"
        "11000110"
        "01111100"
        "00000000",
        // 6
        "00111100"
        "01100000"
        "11000000"
        "11111100"
        "11000110"
        "11000110"
        "01111100"
        "00000000",
        // 7
        "11111110"
        "00000110"
        "00001100"
        "00011000"
        "00110000"
        "00110000"
        "00110000"
        "00000000",
        // 8
        "01111100"
        "11000110"
        "11000110"
        "01111100"
        "11000110"
        "11000110"
        "01111100"
        "00000000",
        // 9
        "01111100"
        "11000110"
        "11000110"
        "01111110"
        "00000110"
        "00001100"
        "01111000"
        "00000000"
    };
    for (int i = 0; i < 64; ++i)
        pixels[i] = (patterns[digit][i] == '1') ? 1.0 : 0.0;
}

// Generate noisy training samples: dream_gen_digits(n_per_digit)
// Returns tuple: (data[n*10, 64], labels[n*10, 10])
Tensor* dream_gen_digits(Tensor* n_per_t) {
    int n_per = (int)n_per_t->data[0];
    if (n_per < 1) n_per = 1;
    int total = n_per * 10;

    Tensor* data   = alloc_tensor(total, 64, 1, 1);
    Tensor* labels = alloc_tensor(total, 10, 1, 1);
    std::fill(labels->data, labels->data + total * 10, 0.0);

    static std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.15);

    for (int digit = 0; digit < 10; ++digit) {
        double clean[64];
        draw_digit(clean, digit);
        for (int s = 0; s < n_per; ++s) {
            int idx = digit * n_per + s;
            for (int p = 0; p < 64; ++p) {
                double v = clean[p] + noise(rng);
                data->data[idx * 64 + p] = std::max(0.0, std::min(1.0, v));
            }
            labels->data[idx * 10 + digit] = 1.0; // one-hot
        }
    }

    // Pack as tuple
    Tensor** arr = new Tensor*[2]{data, labels};
    Tensor* result = dream_pack(arr, 2);
    delete[] arr;
    // Release alloc's +1 from data/labels; dream_pack increfed them as parents.
    tensor_decref(data);
    tensor_decref(labels);
    return result;
}

// Predict: return index of max element (argmax)
Tensor* dream_predict(Tensor* output) {
    if (!output || output->size == 0) return create_scalar(-1.0);
    int best = 0;
    for (int i = 1; i < output->size; ++i)
        if (output->data[i] > output->data[best]) best = i;
    return create_scalar((double)best);
}

// Compute accuracy: compare predictions vs one-hot labels
// predictions: (n, 10), labels: (n, 10)
Tensor* dream_accuracy(Tensor* preds, Tensor* labels) {
    if (!preds || !labels) return create_scalar(0.0);
    int n = preds->d0;
    int classes = preds->d1;
    // Labels may be class indices (Nx1) or one-hot (NxC).
    bool index_labels = (labels->d1 == 1 && classes > 1);
    int rows = index_labels ? std::min(n, labels->d0) : n;
    int correct = 0;
    for (int i = 0; i < rows; ++i) {
        int pred_class = 0;
        for (int c = 1; c < classes; ++c)
            if (preds->data[i*classes+c] > preds->data[i*classes+pred_class]) pred_class = c;
        int true_class = 0;
        if (index_labels) {
            true_class = (int)labels->data[i];
        } else {
            for (int c = 1; c < classes; ++c) {
                int k = i*classes+c;
                if (k < labels->size && labels->data[k] > labels->data[i*classes+true_class]) true_class = c;
            }
        }
        if (pred_class == true_class) correct++;
    }
    return create_scalar((double)correct / std::max(1, rows));
}

// Print a digit pattern (8x8) to console
Tensor* dream_print_digit(Tensor* img) {
    if (!img || img->size < 64) return create_scalar(0.0);
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            double v = img->data[y * 8 + x];
            if (v > 0.5) std::cout << "##";
            else std::cout << "  ";
        }
        std::cout << "\n";
    }
    std::cout << std::flush;
    return create_scalar(1.0);
}

// Extract row i from a matrix (returns 1D tensor)
Tensor* dream_get_row(Tensor* mat, Tensor* idx_t) {
    if (!mat || !idx_t) return alloc_tensor(1, 1, 1, 1);
    int i = (int)idx_t->data[0];
    int cols = mat->d1;
    if (i < 0 || i >= mat->d0) return alloc_tensor(1, cols, 1, 1);
    Tensor* row = alloc_tensor(1, cols, 1, 1);
    std::copy(mat->data + i*cols, mat->data + (i+1)*cols, row->data);
    return row;
}

// Batch iterator: splits in-memory data into shuffled mini-batches
// Usage in Dream:
//   let iter = batch_iter(data, labels, 32)
//   while (has_next(iter) > 0.5) {
//       let (bx, by) = next_batch(iter)
//       ...train on bx, by...
//   }
//   reset_iter(iter)  // shuffle + restart for next epoch

struct BatchIterState {
    double* data_ptr;    int data_rows, data_cols;
    double* label_ptr;   int label_rows, label_cols;
    int batch_size;
    int current;
    std::vector<int> order;
    // We hold owning references to the source data and label tensors,
    // so the raw pointers data_ptr/label_ptr stay valid as long as the
    // iterator is alive. dream_batch_iter increfs these.
    Tensor* src_data = nullptr;
    Tensor* src_labels = nullptr;

    void shuffle() {
        static std::mt19937 rng(std::random_device{}());
        for (int i = (int)order.size()-1; i > 0; --i) {
            int j = std::uniform_int_distribution<int>(0, i)(rng);
            std::swap(order[i], order[j]);
        }
    }
};

static std::vector<BatchIterState*> g_batch_iters;

// Create a batch iterator over (data, labels) with given batch_size
// data: (N, features), labels: (N, classes)
Tensor* dream_batch_iter(Tensor* data, Tensor* labels, Tensor* bs_t) {
    if (!data || !labels || !bs_t) return create_scalar(-1.0);
    data = materialize(data); labels = materialize(labels);
    int N = data->d0;
    int bs = (int)bs_t->data[0];
    if (bs <= 0) bs = 1;

    auto* st = new BatchIterState();
    st->data_ptr = data->data;   st->data_rows = N; st->data_cols = data->d1;
    st->label_ptr = labels->data; st->label_rows = N; st->label_cols = labels->d1;
    st->batch_size = bs;
    st->current = 0;
    st->order.resize(N);
    for (int i = 0; i < N; ++i) st->order[i] = i;
    st->shuffle();

    int id = (int)g_batch_iters.size();
    g_batch_iters.push_back(st);
    // Pure refcount: incref data and labels so their data/data buffers
    // stay valid for the lifetime of the iterator. The iterator holds
    // owning references; they're released when the iterator is destroyed.
    data->refcount++;   st->src_data = data;
    labels->refcount++; st->src_labels = labels;
    return create_scalar((double)id);
}

// Returns 1.0 if more batches remain, 0.0 otherwise
Tensor* dream_has_next(Tensor* iter_t) {
    if (!iter_t) return create_scalar(0.0);
    int id = (int)iter_t->data[0];
    if (id < 0 || id >= (int)g_batch_iters.size()) return create_scalar(0.0);
    auto* st = g_batch_iters[id];
    return create_scalar(st->current < st->data_rows ? 1.0 : 0.0);
}

// Get the next batch as a (batch_data, batch_labels) tuple
Tensor* dream_next_batch(Tensor* iter_t) {
    if (!iter_t) return create_scalar(0.0);
    int id = (int)iter_t->data[0];
    if (id < 0 || id >= (int)g_batch_iters.size()) return create_scalar(0.0);
    auto* st = g_batch_iters[id];

    int start = st->current;
    int end = std::min(start + st->batch_size, st->data_rows);
    int actual_bs = end - start;
    if (actual_bs <= 0) return create_scalar(0.0);
    st->current = end;

    // Pure refcount mode: alloc_tensor returns rc=0. We pass bd/bl into
    // dream_pack, which calls tensor_set_parents — that increfs them as
    // owning parents of the tuple wrapper (rc=1). When the tuple wrapper
    // eventually gets decref'd to 0, it will decref bd/bl back to 0.
    Tensor* bd = alloc_tensor(actual_bs, st->data_cols, 1, 1);
    Tensor* bl = alloc_tensor(actual_bs, st->label_cols, 1, 1);
    for (int i = 0; i < actual_bs; ++i) {
        int src = st->order[start + i];
        std::copy(st->data_ptr + src * st->data_cols,
                  st->data_ptr + (src+1) * st->data_cols,
                  bd->data + i * st->data_cols);
        std::copy(st->label_ptr + src * st->label_cols,
                  st->label_ptr + (src+1) * st->label_cols,
                  bl->data + i * st->label_cols);
    }

    Tensor** arr = new Tensor*[2]{bd, bl};
    Tensor* result = dream_pack(arr, 2);
    delete[] arr;
    // Release alloc's +1 from bd/bl; dream_pack increfed them as parents.
    tensor_decref(bd);
    tensor_decref(bl);
    return result;
}

// Shuffle and reset to beginning for the next epoch
Tensor* dream_reset_iter(Tensor* iter_t) {
    if (!iter_t) return create_scalar(0.0);
    int id = (int)iter_t->data[0];
    if (id < 0 || id >= (int)g_batch_iters.size()) return create_scalar(0.0);
    auto* st = g_batch_iters[id];
    st->current = 0;
    st->shuffle();
    return create_scalar(1.0);
}

// Return total number of samples in the dataset
Tensor* dream_iter_len(Tensor* iter_t) {
    if (!iter_t) return create_scalar(0.0);
    int id = (int)iter_t->data[0];
    if (id < 0 || id >= (int)g_batch_iters.size()) return create_scalar(0.0);
    return create_scalar((double)g_batch_iters[id]->data_rows);
}

// Load a list of BMP files into a single data tensor
// list_file: text file with one BMP path per line
// Returns (data_tensor, count) where data is (N, H*W)
Tensor* dream_load_bmp_list(Tensor* list_path_t, Tensor* h_t, Tensor* w_t) {
    if (!list_path_t || !h_t || !w_t) return create_scalar(0.0);
    std::string path;
    if (list_path_t->grad_fn == "string")
        for (int i = 0; i < list_path_t->size; ++i) path += (char)(int)list_path_t->data[i];
    else return create_scalar(0.0);

    int H = (int)h_t->data[0], W = (int)w_t->data[0];

    // Read file list
    std::vector<std::string> files;
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (!f) {
        std::cerr << "error: cannot open list file '" << path << "'\n" << std::flush;
        return create_scalar(0.0);
    }
    char line[1024];
    while (std::fgets(line, sizeof(line), f)) {
        std::string s(line);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        if (!s.empty()) files.push_back(s);
    }
    std::fclose(f);

    int N = (int)files.size();
    if (N == 0) return create_scalar(0.0);

    // Load all images, resize to H x W
    Tensor* data = alloc_tensor(N, H * W, 1, 1);
    int loaded = 0;
    for (int i = 0; i < N; ++i) {
        Tensor* img = dream_load_bmp(files[i].c_str());
        if (!img || img->size <= 1) continue;
        // Resize if needed
        if (img->d0 != H || img->d1 != W) {
            Tensor* nh = create_scalar((double)H);
            Tensor* nw = create_scalar((double)W);
            img = dream_resize(img, nh, nw);
        }
        int copy_n = std::min(img->size, H * W);
        std::copy(img->data, img->data + copy_n, data->data + loaded * H * W);
        loaded++;
    }
    std::cerr << "loaded " << loaded << "/" << N << " images (" << H << "x" << W << ")\n" << std::flush;

    // Resize data if some failed
    if (loaded < N) {
        Tensor* trimmed = alloc_tensor(loaded, H * W, 1, 1);
        std::copy(data->data, data->data + loaded * H * W, trimmed->data);
        data = trimmed;
    }

    Tensor* loaded_scalar = create_scalar((double)loaded);
    Tensor** arr = new Tensor*[2]{data, loaded_scalar};
    Tensor* result = dream_pack(arr, 2);
    delete[] arr;
    // Release alloc's +1 from data/loaded_scalar; dream_pack increfed them.
    tensor_decref(data);
    tensor_decref(loaded_scalar);
    return result;
}

// Async DataLoader: background thread pre-fetches batches into a ring buffer.
// While GPU computes batch N, the worker has already prepared N+1 and N+2.
// Uses producer-consumer pattern with std::thread + mutex + condition_variable.

struct AsyncBatch {
    Tensor* data;
    Tensor* labels;
};

struct AsyncBatchLoader {
    // Source data (kept alive externally)
    double* src_data;   int data_rows, data_cols;
    double* src_labels; int label_rows, label_cols;
    int batch_size;

    // Shuffle order
    std::vector<int> order;
    int cursor;

    // Ring buffer of pre-fetched batches
    static const int PREFETCH_DEPTH = 3;
    std::queue<AsyncBatch> ready;
    std::mutex mtx;
    std::condition_variable cv_has_space;  // producer waits when buffer full
    std::condition_variable cv_has_data;   // consumer waits when buffer empty
    std::thread worker;
    std::atomic<bool> stop_flag{false};
    std::atomic<bool> epoch_done{false};

    void shuffle() {
        static std::mt19937 rng(std::random_device{}());
        for (int i = (int)order.size()-1; i > 0; --i) {
            int j = std::uniform_int_distribution<int>(0, i)(rng);
            std::swap(order[i], order[j]);
        }
    }

    // Gather one batch starting at position `from` in the shuffle order
    AsyncBatch make_batch(int from) {
        int end = std::min(from + batch_size, data_rows);
        int bs = end - from;
        // Allocate tensors (this runs on worker thread, alloc_tensor is NOT thread-safe,
        // so we use raw new[] and wrap into Tensor on the main thread in next())
        AsyncBatch b;
        b.data = nullptr; b.labels = nullptr;
        if (bs <= 0) return b;

        double* bd = new double[bs * data_cols]();
        double* bl = new double[bs * label_cols]();
        for (int i = 0; i < bs; ++i) {
            int src = order[from + i];
            std::copy(src_data  + src * data_cols,
                      src_data  + (src+1) * data_cols,
                      bd + i * data_cols);
            std::copy(src_labels + src * label_cols,
                      src_labels + (src+1) * label_cols,
                      bl + i * label_cols);
        }
        // Store raw pointers; the consumer will wrap them into Tensors
        // Use a trick: store pointer and size in a minimal Tensor-like struct
        // Actually, allocate Tensor on main thread. Store raw buffers here.
        auto* dt = new Tensor();
        dt->data = bd; dt->size = bs * data_cols;
        dt->d0 = bs; dt->d1 = data_cols; dt->d2 = 1; dt->d3 = 1;
        dt->owns_data = false; // we manage this memory ourselves
        dt->grad = nullptr; dt->is_leaf = true; dt->grad_fn = "";
        dt->lhs = nullptr; dt->rhs = nullptr; tensor_set_parents(dt, nullptr, 0); dt->is_variable = false; dt->refcount = 0;
        dt->ndim = 2; dt->data_offset = 0; dt->version = 0; dt->base = nullptr;
        dt->dtype = 0; dt->fdata = nullptr; dt->fgrad = nullptr;
        dt->bwd_name = ""; dt->bwd_src = "";
        for(int i=0;i<8;++i){dt->dims[i]=1;dt->strides[i]=0;}
        dt->dims[0]=bs; dt->dims[1]=data_cols;
        dt->strides[0]=data_cols; dt->strides[1]=1;

        auto* lt = new Tensor();
        *lt = *dt; // copy fields
        lt->data = bl; lt->size = bs * label_cols;
        lt->d1 = label_cols; lt->dims[1] = label_cols;
        lt->strides[0] = label_cols;

        b.data = dt; b.labels = lt;
        return b;
    }

    void worker_fn() {
        int pos = 0;
        while (!stop_flag.load()) {
            if (pos >= data_rows) {
                epoch_done.store(true);
                cv_has_data.notify_one();
                return;
            }
            AsyncBatch b = make_batch(pos);
            pos += batch_size;

            std::unique_lock<std::mutex> lock(mtx);
            cv_has_space.wait(lock, [this]{ return (int)ready.size() < PREFETCH_DEPTH || stop_flag.load(); });
            if (stop_flag.load()) return;
            ready.push(b);
            lock.unlock();
            cv_has_data.notify_one();
        }
    }

    void start() {
        stop_flag.store(false);
        epoch_done.store(false);
        cursor = 0;
        worker = std::thread(&AsyncBatchLoader::worker_fn, this);
    }

    AsyncBatch next() {
        std::unique_lock<std::mutex> lock(mtx);
        cv_has_data.wait(lock, [this]{ return !ready.empty() || epoch_done.load() || stop_flag.load(); });
        if (ready.empty()) return {nullptr, nullptr};
        AsyncBatch b = ready.front();
        ready.pop();
        lock.unlock();
        cv_has_space.notify_one();
        return b;
    }

    bool has_more() {
        std::lock_guard<std::mutex> lock(mtx);
        return !ready.empty() || !epoch_done.load();
    }

    void reset() {
        stop();
        shuffle();
        // Clear leftover batches
        while (!ready.empty()) {
            auto b = ready.front(); ready.pop();
            if (b.data)   { delete[] b.data->data;   delete b.data; }
            if (b.labels) { delete[] b.labels->data;  delete b.labels; }
        }
        start();
    }

    void stop() {
        stop_flag.store(true);
        cv_has_space.notify_all();
        cv_has_data.notify_all();
        if (worker.joinable()) worker.join();
    }

    ~AsyncBatchLoader() { stop(); }
};

static std::vector<AsyncBatchLoader*> g_async_loaders;

// Create async batch iterator: background thread immediately starts prefetching
Tensor* dream_async_iter(Tensor* data, Tensor* labels, Tensor* bs_t) {
    if (!data || !labels || !bs_t) return create_scalar(-1.0);
    data = materialize(data); labels = materialize(labels);
    // Note: AsyncBatchLoader doesn't currently track Tensor refs (only raw
    // pointers). For now we incref but never decref — minor leak, only on
    // exit. Could add ownership tracking later.
    data->refcount++;
    labels->refcount++;

    auto* loader = new AsyncBatchLoader();
    loader->src_data   = data->data;   loader->data_rows = data->d0;   loader->data_cols = data->d1;
    loader->src_labels = labels->data; loader->label_rows = labels->d0; loader->label_cols = labels->d1;
    loader->batch_size = std::max(1, (int)bs_t->data[0]);
    loader->order.resize(loader->data_rows);
    for (int i = 0; i < loader->data_rows; ++i) loader->order[i] = i;
    loader->shuffle();
    loader->start();

    int id = (int)g_async_loaders.size();
    g_async_loaders.push_back(loader);
    return create_scalar((double)id);
}

// Check if more batches are available (1.0 = yes, 0.0 = epoch done)
Tensor* dream_async_has_next(Tensor* iter_t) {
    if (!iter_t) return create_scalar(0.0);
    int id = (int)iter_t->data[0];
    if (id < 0 || id >= (int)g_async_loaders.size()) return create_scalar(0.0);
    return create_scalar(g_async_loaders[id]->has_more() ? 1.0 : 0.0);
}

// Get next pre-fetched batch as (data, labels) tuple. Blocks only if worker is behind.
Tensor* dream_async_next(Tensor* iter_t) {
    if (!iter_t) return create_scalar(0.0);
    int id = (int)iter_t->data[0];
    if (id < 0 || id >= (int)g_async_loaders.size()) return create_scalar(0.0);

    AsyncBatch b = g_async_loaders[id]->next();
    if (!b.data || !b.labels) return create_scalar(0.0);

    // Wrap raw buffers into proper arena-tracked tensors
    Tensor* bd = alloc_tensor(b.data->d0, b.data->d1, 1, 1);
    std::copy(b.data->data, b.data->data + b.data->size, bd->data);
    Tensor* bl = alloc_tensor(b.labels->d0, b.labels->d1, 1, 1);
    std::copy(b.labels->data, b.labels->data + b.labels->size, bl->data);

    // Free worker-allocated raw buffers
    delete[] b.data->data;   delete b.data;
    delete[] b.labels->data; delete b.labels;

    Tensor** arr = new Tensor*[2]{bd, bl};
    Tensor* result = dream_pack(arr, 2);
    delete[] arr;
    // Release alloc's +1 from bd/bl; dream_pack increfed them as parents.
    tensor_decref(bd);
    tensor_decref(bl);
    return result;
}

// Shuffle and restart for next epoch. Joins worker, reshuffles, relaunches.
Tensor* dream_async_reset(Tensor* iter_t) {
    if (!iter_t) return create_scalar(0.0);
    int id = (int)iter_t->data[0];
    if (id < 0 || id >= (int)g_async_loaders.size()) return create_scalar(0.0);
    g_async_loaders[id]->reset();
    return create_scalar(1.0);
}

// Cleanly shut down the background thread
Tensor* dream_async_stop(Tensor* iter_t) {
    if (!iter_t) return create_scalar(0.0);
    int id = (int)iter_t->data[0];
    if (id < 0 || id >= (int)g_async_loaders.size()) return create_scalar(0.0);
    g_async_loaders[id]->stop();
    return create_scalar(1.0);
}

// Disk-streaming async DataLoader: reads BMP files from disk on-the-fly.
// Never loads entire dataset into memory. Background thread reads + decodes + resizes.
// Usage:
//   let loader = disk_loader("file_list.txt", "label_list.txt", 32, 64, 64, 10)
//   while (disk_has_next(loader) > 0.5) {
//       let (bx, by) = disk_next(loader)
//       // train...
//   }
//   disk_reset(loader)

struct DiskBatch { double* data; double* labels; int n; int dcols; int lcols; };

struct DiskStreamLoader {
    std::vector<std::string> files;
    std::vector<int> file_labels;
    int num_classes;
    int H, W;
    int batch_size;
    std::vector<int> order;

    static const int PREFETCH = 3;
    std::queue<DiskBatch> ready;
    std::mutex mtx;
    std::condition_variable cv_space, cv_data;
    std::thread worker;
    std::atomic<bool> stop_flag{false};
    std::atomic<bool> done{false};

    void shuffle() {
        static std::mt19937 rng(std::random_device{}());
        for(int i=(int)order.size()-1;i>0;--i) {
            int j=std::uniform_int_distribution<int>(0,i)(rng);
            std::swap(order[i],order[j]);
        }
    }

    void worker_fn() {
        int pos = 0;
        int dcols = H * W;
        int lcols = num_classes;
        while (!stop_flag.load() && pos < (int)files.size()) {
            int end = std::min(pos + batch_size, (int)files.size());
            int bs = end - pos;
            if (bs <= 0) break;

            double* bd = new double[bs * dcols]();
            double* bl = new double[bs * lcols]();

            for (int i = 0; i < bs; ++i) {
                int idx = order[pos + i];
                // Load and resize BMP
                Tensor* img = dream_load_bmp(files[idx].c_str());
                if (img && img->size > 1) {
                    if (img->d0 != H || img->d1 != W) {
                        Tensor* nh = create_scalar((double)H);
                        Tensor* nw = create_scalar((double)W);
                        img = dream_resize(img, nh, nw);
                    }
                    int copy_n = std::min(img->size, dcols);
                    std::copy(img->data, img->data + copy_n, bd + i * dcols);
                }
                // One-hot label
                int lbl = file_labels[idx];
                if (lbl >= 0 && lbl < lcols)
                    bl[i * lcols + lbl] = 1.0;
            }
            pos += bs;

            std::unique_lock<std::mutex> lock(mtx);
            cv_space.wait(lock, [this]{ return (int)ready.size() < PREFETCH || stop_flag.load(); });
            if (stop_flag.load()) { delete[] bd; delete[] bl; return; }
            ready.push({bd, bl, bs, dcols, lcols});
            lock.unlock();
            cv_data.notify_one();
        }
        done.store(true);
        cv_data.notify_one();
    }

    void start() {
        stop_flag.store(false); done.store(false);
        worker = std::thread(&DiskStreamLoader::worker_fn, this);
    }
    void stop() {
        stop_flag.store(true);
        cv_space.notify_all(); cv_data.notify_all();
        if (worker.joinable()) worker.join();
    }
    void reset() {
        stop();
        while (!ready.empty()) { auto b=ready.front(); ready.pop(); delete[] b.data; delete[] b.labels; }
        shuffle();
        start();
    }
    ~DiskStreamLoader() { stop(); }
};

static std::vector<DiskStreamLoader*> g_disk_loaders;

// Create disk-streaming loader from two text files:
//   file_list: one BMP path per line
//   label_list: one integer label per line (same order)
Tensor* dream_disk_loader(Tensor* flist_t, Tensor* llist_t,
                           Tensor* bs_t, Tensor* h_t, Tensor* w_t, Tensor* nc_t) {
    auto str_from = [](Tensor* t) -> std::string {
        std::string s;
        if (t && t->grad_fn == "string")
            for (int i = 0; i < t->size; ++i) s += (char)(int)t->data[i];
        return s;
    };
    std::string fpath = str_from(flist_t);
    std::string lpath = str_from(llist_t);
    if (fpath.empty() || lpath.empty()) return create_scalar(-1.0);

    auto* ld = new DiskStreamLoader();
    ld->batch_size = std::max(1, (int)bs_t->data[0]);
    ld->H = (int)h_t->data[0]; ld->W = (int)w_t->data[0];
    ld->num_classes = (int)nc_t->data[0];

    // Read file list
    FILE* f = fopen(fpath.c_str(), "r");
    if (!f) { std::cerr << "error: cannot open " << fpath << "\n"; delete ld; return create_scalar(-1.0); }
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        std::string s(line);
        while (!s.empty() && (s.back()=='\n'||s.back()=='\r')) s.pop_back();
        if (!s.empty()) ld->files.push_back(s);
    }
    fclose(f);

    // Read label list
    f = fopen(lpath.c_str(), "r");
    if (!f) { std::cerr << "error: cannot open " << lpath << "\n"; delete ld; return create_scalar(-1.0); }
    while (fgets(line, sizeof(line), f)) {
        ld->file_labels.push_back(std::atoi(line));
    }
    fclose(f);

    // Trim to min of both
    int n = std::min(ld->files.size(), ld->file_labels.size());
    ld->files.resize(n); ld->file_labels.resize(n);
    ld->order.resize(n);
    for (int i = 0; i < n; ++i) ld->order[i] = i;
    ld->shuffle();

    std::cerr << "[dream] disk loader: " << n << " files, batch=" << ld->batch_size
              << ", " << ld->H << "x" << ld->W << ", " << ld->num_classes << " classes\n" << std::flush;

    ld->start();
    int id = (int)g_disk_loaders.size();
    g_disk_loaders.push_back(ld);
    return create_scalar((double)id);
}

Tensor* dream_disk_has_next(Tensor* iter_t) {
    int id = iter_t ? (int)iter_t->data[0] : -1;
    if (id < 0 || id >= (int)g_disk_loaders.size()) return create_scalar(0.0);
    auto* ld = g_disk_loaders[id];
    std::lock_guard<std::mutex> lock(ld->mtx);
    return create_scalar((!ld->ready.empty() || !ld->done.load()) ? 1.0 : 0.0);
}

Tensor* dream_disk_next(Tensor* iter_t) {
    int id = iter_t ? (int)iter_t->data[0] : -1;
    if (id < 0 || id >= (int)g_disk_loaders.size()) return create_scalar(0.0);
    auto* ld = g_disk_loaders[id];

    std::unique_lock<std::mutex> lock(ld->mtx);
    ld->cv_data.wait(lock, [ld]{ return !ld->ready.empty() || ld->done.load() || ld->stop_flag.load(); });
    if (ld->ready.empty()) return create_scalar(0.0);
    DiskBatch b = ld->ready.front(); ld->ready.pop();
    lock.unlock();
    ld->cv_space.notify_one();

    Tensor* bd = alloc_tensor(b.n, b.dcols, 1, 1);
    std::copy(b.data, b.data + b.n * b.dcols, bd->data);
    Tensor* bl = alloc_tensor(b.n, b.lcols, 1, 1);
    std::copy(b.labels, b.labels + b.n * b.lcols, bl->data);
    delete[] b.data; delete[] b.labels;

    Tensor** arr = new Tensor*[2]{bd, bl};
    Tensor* result = dream_pack(arr, 2);
    delete[] arr;
    // Release alloc's +1 from bd/bl; dream_pack increfed them as parents.
    tensor_decref(bd);
    tensor_decref(bl);
    return result;
}

Tensor* dream_disk_reset(Tensor* iter_t) {
    int id = iter_t ? (int)iter_t->data[0] : -1;
    if (id < 0 || id >= (int)g_disk_loaders.size()) return create_scalar(0.0);
    g_disk_loaders[id]->reset();
    return create_scalar(1.0);
}

Tensor* dream_disk_stop(Tensor* iter_t) {
    int id = iter_t ? (int)iter_t->data[0] : -1;
    if (id < 0 || id >= (int)g_disk_loaders.size()) return create_scalar(0.0);
    g_disk_loaders[id]->stop();
    return create_scalar(1.0);
}

// ============================================================
// CSV reader — the #1 most requested thing for real ML work.
// Without this, you can't work with any real-world tabular data.
// ============================================================

// Read a CSV file into a tensor. Auto-detects shape from first 2 lines.
// Optionally skips a header row. Handles common delimiters (, ; tab).
//
// Usage:
//   let data = read_csv("iris.csv", 1);   // skip 1 header row
//
// Return shape: (num_rows, num_cols)
// Non-numeric values become 0.0 with a warning.
Tensor* dream_read_csv(Tensor* path_t, Tensor* skip_t) {
    if (!path_t) { std::cerr << "error: read_csv needs a path\n" << std::flush; return create_scalar(0.0); }
    std::string path;
    if (path_t->grad_fn == "string")
        for (int i = 0; i < path_t->size; ++i) path += (char)(int)path_t->data[i];
    else {
        std::cerr << "error: read_csv path must be a string\n" << std::flush;
        return create_scalar(0.0);
    }

    int skip = skip_t ? (int)skip_t->data[0] : 0;

    FILE* f = fopen(path.c_str(), "r");
    if (!f) {
        std::cerr << "RUNTIME ERROR in read_csv: cannot open '" << path << "'\n"
                  << "  Check the path is correct and the file exists.\n" << std::flush;
        return create_scalar(0.0);
    }

    // First pass: count rows and columns
    char buf[16384];
    int rows = 0, cols = 0;
    int skipped = 0;

    // Determine delimiter from first non-skipped line
    char delim = ',';
    long first_line_pos = ftell(f);
    while (fgets(buf, sizeof(buf), f)) {
        if (skipped < skip) { skipped++; first_line_pos = ftell(f); continue; }
        // Count commas, semicolons, tabs in first data line
        int n_comma = 0, n_semi = 0, n_tab = 0;
        for (char* p = buf; *p && *p != '\n'; ++p) {
            if (*p == ',') n_comma++;
            else if (*p == ';') n_semi++;
            else if (*p == '\t') n_tab++;
        }
        if (n_tab > n_comma && n_tab > n_semi) delim = '\t';
        else if (n_semi > n_comma) delim = ';';
        cols = 1;
        for (char* p = buf; *p && *p != '\n'; ++p)
            if (*p == delim) cols++;
        rows = 1;
        break;
    }
    // Count remaining rows
    while (fgets(buf, sizeof(buf), f)) {
        // Skip empty lines
        bool empty = true;
        for (char* p = buf; *p && *p != '\n'; ++p)
            if (!std::isspace((unsigned char)*p)) { empty = false; break; }
        if (!empty) rows++;
    }

    if (rows == 0 || cols == 0) {
        fclose(f);
        std::cerr << "RUNTIME ERROR in read_csv: empty file or only header in '" << path << "'\n" << std::flush;
        return create_scalar(0.0);
    }

    // Second pass: read data
    rewind(f);
    skipped = 0;
    while (skipped < skip && fgets(buf, sizeof(buf), f)) skipped++;

    Tensor* out = alloc_tensor(rows, cols, 1, 1);
    int row = 0, non_numeric = 0;
    while (fgets(buf, sizeof(buf), f) && row < rows) {
        // Check if line has any non-whitespace content
        bool empty = true;
        for (char* p = buf; *p && *p != '\n'; ++p)
            if (!std::isspace((unsigned char)*p)) { empty = false; break; }
        if (empty) continue;

        char* p = buf;
        for (int c = 0; c < cols; ++c) {
            // Skip leading whitespace
            while (*p && std::isspace((unsigned char)*p) && *p != delim) p++;
            char* end;
            double val = std::strtod(p, &end);
            if (end == p) {
                // Not a number — try to count as non-numeric
                non_numeric++;
                val = 0.0;
                // Advance to next delimiter
                while (*end && *end != delim && *end != '\n') end++;
            }
            out->data[row * cols + c] = val;
            // Advance past delimiter
            p = end;
            while (*p && *p != delim && *p != '\n') p++;
            if (*p == delim) p++;
        }
        row++;
    }
    fclose(f);

    std::cerr << "[read_csv] loaded '" << path << "': " << row << " rows x " << cols << " cols";
    if (non_numeric > 0) std::cerr << " (" << non_numeric << " non-numeric -> 0)";
    std::cerr << "\n" << std::flush;
    return out;
}

// Write a tensor to CSV. For saving predictions, processed data, etc.
Tensor* dream_write_csv(Tensor* t, Tensor* path_t) {
    if (!t || !path_t) return create_scalar(0.0);
    t = materialize(t);
    std::string path;
    if (path_t->grad_fn == "string")
        for (int i = 0; i < path_t->size; ++i) path += (char)(int)path_t->data[i];

    dream_mkdirs_for(path);
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        std::cerr << "RUNTIME ERROR in write_csv: cannot write '" << path << "'\n" << std::flush;
        return create_scalar(0.0);
    }
    for (int i = 0; i < t->d0; ++i) {
        for (int j = 0; j < t->d1; ++j) {
            if (j > 0) fprintf(f, ",");
            fprintf(f, "%.9g", t->data[i * t->d1 + j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    std::cerr << "[write_csv] saved " << t->d0 << "x" << t->d1 << " to '" << path << "'\n" << std::flush;
    return create_scalar(1.0);
}

// Aliases
Tensor* read_csv(Tensor* p, Tensor* s) { return dream_read_csv(p, s); }
Tensor* write_csv(Tensor* t, Tensor* p) { return dream_write_csv(t, p); }
