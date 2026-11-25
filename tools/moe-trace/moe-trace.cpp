// moe-trace: records which experts a Mixture-of-Experts model picks for every token and layer.
//
// Usage:
//   llama-moe-trace -m model.gguf -f text.txt --trace-out trace.csv [-c 4096] [-b 512]
//
// Output CSV columns: pos,layer,e0,...,e{k-1},w0,...,w{k-1}
//   pos   = token position in the text
//   layer = logical layer index as named by llama.cpp (looped passes get their own index)
//   e*    = chosen expert ids, w* = their routing weights
// A second file <trace-out>.tokens.csv lists pos,token_id,piece.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct trace_state {
    FILE * out = nullptr;
    int    pos0 = 0;  // position of the first token in the current batch
    // weights arrive in a separate tensor; keep ids until both are seen
    std::map<int, std::vector<int32_t>> ids;   // layer -> ids for the batch
    std::map<int, std::vector<float>>   wts;   // layer -> weights for the batch
    std::map<int, int>                  k;     // layer -> experts per token
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

    const bool host = t->buffer == nullptr || ggml_backend_buffer_is_host(t->buffer);
    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> buf(nbytes);
    if (host) memcpy(buf.data(), t->data, nbytes);
    else      ggml_backend_tensor_get(t, buf.data(), 0, nbytes);

    if (il_ids >= 0 && t->type == GGML_TYPE_I32) {
        const int k = (int) t->ne[0];
        const int n_tok = (int) t->ne[1];
        std::vector<int32_t> v(k * n_tok);
        // rows may be strided (topk is a view); copy row by row
        for (int r = 0; r < n_tok; ++r) memcpy(&v[r * k], buf.data() + r * t->nb[1], k * sizeof(int32_t));
        st.ids[il_ids] = std::move(v);
        st.k[il_ids] = k;
        flush_layer(st, il_ids);
    } else if (il_wts >= 0 && t->type == GGML_TYPE_F32) {
        // weights tensor is [1, k, n_tok] or [k, n_tok]
        const int k = (int) (t->ne[0] == 1 ? t->ne[1] : t->ne[0]);
        const int n_tok = (int) (t->ne[0] == 1 ? t->ne[2] : t->ne[1]);
        const size_t row_stride = t->ne[0] == 1 ? t->nb[2] : t->nb[1];
        const size_t col_stride = t->ne[0] == 1 ? t->nb[1] : t->nb[0];
        std::vector<float> v(k * n_tok);
        for (int r = 0; r < n_tok; ++r)
            for (int j = 0; j < k; ++j) memcpy(&v[r * k + j], buf.data() + r * row_stride + j * col_stride, sizeof(float));
        st.wts[il_wts] = std::move(v);
        if (!st.k.count(il_wts)) st.k[il_wts] = k;
        flush_layer(st, il_wts);
    }
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // pull our own flag out before the common parser sees it
    std::string trace_out = "trace.csv";
    std::vector<char *> args;
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--trace-out") == 0 && i + 1 < argc) { trace_out = argv[++i]; continue; }
        args.push_back(argv[i]);
    }

    common_params params;
    common_init();
    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_COMMON)) return 1;

    trace_state st;
    st.out = fopen(trace_out.c_str(), "w");
    if (!st.out) { LOG_ERR("cannot open %s\n", trace_out.c_str()); return 1; }

    llama_backend_init();
    llama_numa_init(params.numa);
    params.cb_eval = trace_cb;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    auto init = common_init_from_params(params);
    auto * model = init->model();
    auto * ctx = init->context();
    if (!model || !ctx) { LOG_ERR("failed to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    const int n_ctx = llama_n_ctx(ctx);
    if ((int) tokens.size() > n_ctx) {
        LOG_WRN("text has %zu tokens, keeping the first %d (raise -c to keep more)\n", tokens.size(), n_ctx);
        tokens.resize(n_ctx);
    }

    // header
    int k_guess = 8;
    fprintf(st.out, "pos,layer");
    for (int j = 0; j < k_guess; ++j) fprintf(st.out, ",e%d", j);
    for (int j = 0; j < k_guess; ++j) fprintf(st.out, ",w%d", j);
    fputc('\n', st.out);

    // token list
    FILE * tf = fopen((trace_out + ".tokens.csv").c_str(), "w");
    if (tf) {
        fprintf(tf, "pos,token_id,piece\n");
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::string piece = common_token_to_piece(ctx, tokens[i]);
            std::string esc;
            for (char c : piece) { if (c == '"') esc += "\"\""; else if (c == '\n') esc += "\\n"; else if (c == '\r') esc += "\\r"; else esc += c; }
            fprintf(tf, "%zu,%d,\"%s\"\n", i, tokens[i], esc.c_str());
        }
        fclose(tf);
    }

    const int n_batch = params.n_batch > 0 ? params.n_batch : 512;
    const int64_t t0 = ggml_time_us();
    for (size_t i = 0; i < tokens.size(); i += n_batch) {
        const int n = (int) std::min<size_t>(n_batch, tokens.size() - i);
        st.pos0 = (int) i;
        llama_batch batch = llama_batch_get_one(tokens.data() + i, n);
        if (llama_decode(ctx, batch)) { LOG_ERR("decode failed at token %zu\n", i); break; }
        LOG_INF("traced %zu / %zu tokens\n", i + n, tokens.size());
    }
    const double secs = (ggml_time_us() - t0) / 1e6;

    fclose(st.out);
    LOG_INF("wrote %ld rows to %s (%.1f s, %.1f tokens/s)\n", st.rows, trace_out.c_str(), secs, tokens.size() / secs);
    llama_backend_free();
    return 0;
}
