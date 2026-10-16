// Fused CPU MoE expert kernel (AVX-512 BF16) for decode-sized batches.
//
// out[t,:] += sum_e  w_e[t] * Wd_e @ ( silu(Wg_e @ x[t]) * (Wu_e @ x[t]) )
// gu_e = [Wg_e; Wu_e] : [2F, H] bf16 row-major,  d_e : [H, F] bf16 row-major,  x : [T, H] bf16.
// Each expert e has a list of token rows (tok) and routing weights (w).
//
// Decode is weight-bandwidth bound (31.5 MB/expert), so the kernel streams every weight row exactly once
// per call, covering all tokens routed to that expert, using vdpbf16ps (32 bf16 MACs / instruction).
#include <torch/extension.h>
#include <immintrin.h>
#include <omp.h>
#include <vector>
#include <cmath>

static inline float hsum(__m512 v) { return _mm512_reduce_add_ps(v); }

// dot(bf16 a[n], bf16 b[n]), n % 32 == 0
static inline float dot_bf16(const uint16_t* a, const uint16_t* b, int n) {
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        acc0 = _mm512_dpbf16_ps(acc0, (__m512bh)_mm512_loadu_si512(a + i), (__m512bh)_mm512_loadu_si512(b + i));
        acc1 = _mm512_dpbf16_ps(acc1, (__m512bh)_mm512_loadu_si512(a + i + 32), (__m512bh)_mm512_loadu_si512(b + i + 32));
    }
    for (; i < n; i += 32)
        acc0 = _mm512_dpbf16_ps(acc0, (__m512bh)_mm512_loadu_si512(a + i), (__m512bh)_mm512_loadu_si512(b + i));
    return hsum(_mm512_add_ps(acc0, acc1));
}

static inline uint16_t f2bf(float f) {  // round-to-nearest-even
    uint32_t u; memcpy(&u, &f, 4);
    u += 0x7FFF + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

// x: [T,H] bf16; gus/ds: per-expert weights; toks: per-expert int64 token rows; ws: per-expert f32 weights.
// returns out [T,H] f32
torch::Tensor cpu_moe(torch::Tensor x, std::vector<torch::Tensor> gus, std::vector<torch::Tensor> ds,
                      std::vector<torch::Tensor> toks, std::vector<torch::Tensor> ws) {
    const int T = x.size(0), H = x.size(1);
    const int E = gus.size();
    auto out = torch::zeros({T, H}, torch::kFloat32);
    if (E == 0) return out;
    const int F = ds[0].size(1);
    TORCH_CHECK(H % 32 == 0 && F % 32 == 0, "H,F must be multiples of 32");
    const uint16_t* X = (const uint16_t*)x.data_ptr();

    // per (expert, token-slot) activation buffer in bf16: act[e][j][F]
    std::vector<int> off(E + 1, 0);
    for (int e = 0; e < E; e++) off[e + 1] = off[e] + toks[e].size(0);
    const int R = off[E];
    std::vector<uint16_t> act((size_t)R * F);
    std::vector<float> gbuf((size_t)R * 2 * F);

    // phase 1: gate/up rows. work unit = (expert, block of 64 rows of the 2F rows)
    const int RB = 64, nblk = (2 * F) / RB;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int u = 0; u < E * nblk; u++) {
        const int e = u / nblk, b = u % nblk;
        const uint16_t* G = (const uint16_t*)gus[e].data_ptr();
        const int64_t* tk = toks[e].data_ptr<int64_t>();
        const int nt = toks[e].size(0);
        for (int r = b * RB; r < (b + 1) * RB; r++) {
            const uint16_t* row = G + (size_t)r * H;
            for (int j = 0; j < nt; j++)
                gbuf[(size_t)(off[e] + j) * 2 * F + r] = dot_bf16(row, X + (size_t)tk[j] * H, H);
        }
    }
    // silu(g) * u -> bf16
    #pragma omp parallel for schedule(static)
    for (int rj = 0; rj < R; rj++) {
        const float* g = &gbuf[(size_t)rj * 2 * F];
        for (int f = 0; f < F; f++) {
            float gv = g[f], uv = g[F + f];
            act[(size_t)rj * F + f] = f2bf(gv / (1.f + std::exp(-gv)) * uv);
        }
    }
    // phase 2: down rows. each thread owns a block of output rows h -> no atomics; loops over all experts.
    float* O = out.data_ptr<float>();
    const int HB = 32;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int hb = 0; hb < H / HB; hb++) {
        for (int e = 0; e < E; e++) {
            const uint16_t* D = (const uint16_t*)ds[e].data_ptr();
            const int64_t* tk = toks[e].data_ptr<int64_t>();
            const float* w = ws[e].data_ptr<float>();
            const int nt = toks[e].size(0);
            for (int h = hb * HB; h < (hb + 1) * HB; h++) {
                const uint16_t* row = D + (size_t)h * F;
                for (int j = 0; j < nt; j++)
                    O[(size_t)tk[j] * H + h] += w[j] * dot_bf16(row, &act[(size_t)(off[e] + j) * F], F);
            }
        }
    }
    return out;
}

void set_threads(int n) { omp_set_num_threads(n); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // release the GIL while the OpenMP kernel runs so NPU-driver / main Python threads keep going
    m.def("cpu_moe", &cpu_moe, "fused AVX512-BF16 MoE experts (decode)",
          py::call_guard<py::gil_scoped_release>());
    m.def("set_threads", &set_threads);
}
