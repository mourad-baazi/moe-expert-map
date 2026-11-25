// WARNING: on llama.cpp prebuilt b12622 this tool records NaN routing (experts 0-7) for Inkling 2 Mini layer
// passes 18 and later, on both CPU and GPU. Use moe-trace for results; this file is kept to reproduce the issue.
//
// moe-trace-gpu: same output as moe-trace, but uses only the llama.cpp C API so it can run on top of the
// official prebuilt llama.cpp DLLs (CUDA build). Attention and shared layers run on the GPU, all experts stay in RAM.
//
// Usage: llama-moe-trace-gpu <model.gguf> <text.txt> <trace.csv> [n_ctx=4096] [n_batch=512] [n_gpu_layers=99]
// Texts longer than n_ctx are traced in independent windows of n_ctx tokens (n_ctx must be a multiple of n_batch).
// Output CSV: pos,layer,e0..e{k-1},w0..w{k-1}   plus <trace.csv>.tokens.csv with pos,token_id,piece

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct trace_state {
    FILE * out = nullptr;
    int pos0 = 0;
    std::map<int, std::vector<int32_t>> ids;
    std::map<int, std::vector<float>> wts;
    std::map<int, int> k;
    long rows = 0;
};

static int layer_of(const char * name, const char * prefix) {
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0 || name[n] != '-') return -1;
    return atoi(name + n + 1);
}

static void flush_layer(trace_state & st, int il) {
    auto it = st.ids.find(il);
    auto iw = st.wts.find(il);
    if (it == st.ids.end() || iw == st.wts.end()) return;
    const int k = st.k[il];
    const int n_tok = (int) it->second.size() / k;
    for (int t = 0; t < n_tok; ++t) {
        fprintf(st.out, "%d,%d", st.pos0 + t, il);
        for (int j = 0; j < k; ++j) fprintf(st.out, ",%d", it->second[t * k + j]);
        for (int j = 0; j < k; ++j) fprintf(st.out, ",%.4f", iw->second[t * k + j]);
        fputc('\n', st.out);
        st.rows++;
    }
    st.ids.erase(it);
    st.wts.erase(iw);
}

static bool trace_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto & st = *(trace_state *) user_data;
    const int il_ids = layer_of(t->name, "ffn_moe_topk");
    const int il_wts = layer_of(t->name, "ffn_moe_weights");
    if (ask) return il_ids >= 0 || il_wts >= 0;

    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> buf(nbytes);
    if (t->buffer == nullptr || ggml_backend_buffer_is_host(t->buffer)) memcpy(buf.data(), t->data, nbytes);
    else ggml_backend_tensor_get(t, buf.data(), 0, nbytes);

    if (il_ids >= 0 && t->type == GGML_TYPE_I32) {
        const int k = (int) t->ne[0], n_tok = (int) t->ne[1];
        std::vector<int32_t> v(k * n_tok);
        for (int r = 0; r < n_tok; ++r) memcpy(&v[r * k], buf.data() + r * t->nb[1], k * sizeof(int32_t));
        st.ids[il_ids] = std::move(v);
        st.k[il_ids] = k;
        flush_layer(st, il_ids);
    } else if (il_wts >= 0 && t->type == GGML_TYPE_F32) {
        const bool lead1 = t->ne[0] == 1;
        const int k = (int) (lead1 ? t->ne[1] : t->ne[0]);
        const int n_tok = (int) (lead1 ? t->ne[2] : t->ne[1]);
        const size_t rs = lead1 ? t->nb[2] : t->nb[1], cs = lead1 ? t->nb[1] : t->nb[0];
        std::vector<float> v(k * n_tok);
        for (int r = 0; r < n_tok; ++r)
            for (int j = 0; j < k; ++j) memcpy(&v[r * k + j], buf.data() + r * rs + j * cs, sizeof(float));
        st.wts[il_wts] = std::move(v);
        if (!st.k.count(il_wts)) st.k[il_wts] = k;
        flush_layer(st, il_wts);
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.gguf text.txt trace.csv [n_ctx] [n_batch]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const std::string trace_out = argv[3];
    const int n_ctx = argc > 4 ? atoi(argv[4]) : 4096;
    const int n_batch = argc > 5 ? atoi(argv[5]) : 512;
    const int n_gpu_layers = argc > 6 ? atoi(argv[6]) : 99;  // 0 = run everything on the CPU

    std::ifstream in(argv[2], std::ios::binary);
    std::stringstream ss; ss << in.rdbuf();
    const std::string text = ss.str();

    ggml_backend_load_all();
    llama_backend_init();

    // all expert tensors stay in system RAM; everything else goes to the GPU
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    llama_model_tensor_buft_override ov[2] = {{"exps", ggml_backend_dev_buffer_type(cpu)}, {nullptr, nullptr}};
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    mp.tensor_buft_overrides = ov;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    trace_state st;
    st.out = fopen(trace_out.c_str(), "w");
    if (!st.out) { fprintf(stderr, "cannot open %s\n", trace_out.c_str()); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_batch;
    cp.n_ubatch = n_batch;
    cp.n_threads = 6;
    cp.n_threads_batch = 6;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;  // keeps the attention buffer small at long context
    cp.cb_eval = trace_cb;
    cp.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tokens.data(), (int32_t) tokens.size(),
                           llama_vocab_get_add_bos(vocab), true);
    if (n < 0) { tokens.resize(-n); n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tokens.data(),
                                                       (int32_t) tokens.size(), llama_vocab_get_add_bos(vocab), true); }
    tokens.resize(n);
    // long texts are traced in windows of n_ctx tokens; the context memory is cleared between windows

    fprintf(st.out, "pos,layer");
    for (int j = 0; j < 8; ++j) fprintf(st.out, ",e%d", j);
    for (int j = 0; j < 8; ++j) fprintf(st.out, ",w%d", j);
    fputc('\n', st.out);

    if (FILE * tf = fopen((trace_out + ".tokens.csv").c_str(), "w")) {
        fprintf(tf, "pos,token_id,piece\n");
        char piece[256];
        for (size_t i = 0; i < tokens.size(); ++i) {
            int len = llama_token_to_piece(vocab, tokens[i], piece, sizeof(piece) - 1, 0, true);
            std::string p(piece, len > 0 ? len : 0), esc;
            for (char c : p) { if (c == '"') esc += "\"\""; else if (c == '\n') esc += "\\n"; else if (c == '\r') esc += "\\r"; else esc += c; }
            fprintf(tf, "%zu,%d,\"%s\"\n", i, tokens[i], esc.c_str());
        }
        fclose(tf);
    }

    const int64_t t0 = ggml_time_us();
    for (size_t i = 0; i < tokens.size(); i += n_batch) {
        if (i > 0 && i % (size_t) n_ctx == 0) llama_memory_clear(llama_get_memory(ctx), true);
        const size_t window_end = (i / n_ctx + 1) * (size_t) n_ctx;
        const int m = (int) std::min<size_t>(std::min<size_t>(n_batch, tokens.size() - i), window_end - i);
        st.pos0 = (int) i;
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + i, m))) { fprintf(stderr, "decode failed at %zu\n", i); break; }
        fprintf(stderr, "traced %zu / %zu tokens\n", i + m, tokens.size());
    }
    const double secs = (ggml_time_us() - t0) / 1e6;
    fclose(st.out);
    fprintf(stderr, "wrote %ld rows to %s (%.1f s, %.1f tokens/s)\n", st.rows, trace_out.c_str(), secs, tokens.size() / secs);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
