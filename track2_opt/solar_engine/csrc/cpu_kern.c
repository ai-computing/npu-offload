// Host CPU kernels for the Solar engine (AVX-512 BF16 + OpenMP). Weights are row-major bf16 (raw u16).
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline float dot_bf16(const uint16_t* a, const uint16_t* b, int n) {
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        acc0 = _mm512_dpbf16_ps(acc0, (__m512bh)_mm512_loadu_si512(a + i), (__m512bh)_mm512_loadu_si512(b + i));
        acc1 = _mm512_dpbf16_ps(acc1, (__m512bh)_mm512_loadu_si512(a + i + 32), (__m512bh)_mm512_loadu_si512(b + i + 32));
    }
    for (; i < n; i += 32)
        acc0 = _mm512_dpbf16_ps(acc0, (__m512bh)_mm512_loadu_si512(a + i), (__m512bh)_mm512_loadu_si512(b + i));
    return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
}

static inline float bf2f_c(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

static inline uint16_t f2bf(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    u += 0x7FFF + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

void ck_set_threads(int n) { omp_set_num_threads(n); }

// y[M] (f32) = W[M,K] . x[K]
void ck_gemv(const uint16_t* W, const uint16_t* x, float* y, int M, int K) {
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) y[m] = dot_bf16(W + (size_t)m * K, x, K);
}

// Routed experts on CPU for ONE token (decode): out[H] (f32) += sum_e w[e] * Wd_e @ (silu(Wg_e x) * (Wu_e x))
// g[e], u[e]: [F,H] row-major; d[e]: [H,F] row-major. x: [H] bf16.
void ck_moe1(int E, const uint16_t* const* g, const uint16_t* const* u, const uint16_t* const* d,
             const float* w, const uint16_t* x, float* out, int H, int F) {
    if (E <= 0) return;
    uint16_t* act = (uint16_t*)aligned_alloc(64, (size_t)E * F * 2 + 64);
    float* gb = (float*)aligned_alloc(64, (size_t)E * 2 * F * 4 + 64);
    const int RB = 64, nblk = (2 * F) / RB;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int t = 0; t < E * nblk; t++) {
        int e = t / nblk, b = t % nblk;
        for (int r = b * RB; r < (b + 1) * RB; r++) {
            const uint16_t* row = r < F ? g[e] + (size_t)r * H : u[e] + (size_t)(r - F) * H;
            gb[(size_t)e * 2 * F + r] = dot_bf16(row, x, H);
        }
    }
    for (int e = 0; e < E; e++)
        for (int f = 0; f < F; f++) {
            float gv = gb[(size_t)e * 2 * F + f], uv = gb[(size_t)e * 2 * F + F + f];
            act[(size_t)e * F + f] = f2bf((gv / (1.f + expf(-gv))) * uv);
        }
    const int HB = 32;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int hb = 0; hb < H / HB; hb++)
        for (int e = 0; e < E; e++)
            for (int h = hb * HB; h < (hb + 1) * HB; h++)
                out[h] += w[e] * dot_bf16(d[e] + (size_t)h * F, act + (size_t)e * F, F);
    free(act); free(gb);
}

// two rows at once sharing the x loads (more memory-level parallelism)
static inline void dot2_bf16(const uint16_t* a0, const uint16_t* a1, const uint16_t* b, int n, float* r0, float* r1) {
    __m512 c0 = _mm512_setzero_ps(), c1 = _mm512_setzero_ps(), c2 = _mm512_setzero_ps(), c3 = _mm512_setzero_ps();
    for (int i = 0; i < n; i += 64) {
        __m512bh x0 = (__m512bh)_mm512_loadu_si512(b + i), x1 = (__m512bh)_mm512_loadu_si512(b + i + 32);
        c0 = _mm512_dpbf16_ps(c0, (__m512bh)_mm512_loadu_si512(a0 + i), x0);
        c1 = _mm512_dpbf16_ps(c1, (__m512bh)_mm512_loadu_si512(a0 + i + 32), x1);
        c2 = _mm512_dpbf16_ps(c2, (__m512bh)_mm512_loadu_si512(a1 + i), x0);
        c3 = _mm512_dpbf16_ps(c3, (__m512bh)_mm512_loadu_si512(a1 + i + 32), x1);
    }
    *r0 = _mm512_reduce_add_ps(_mm512_add_ps(c0, c1));
    *r1 = _mm512_reduce_add_ps(_mm512_add_ps(c2, c3));
}

// Same math as ck_moe1, one parallel region (one wake-up) and paired-row dots. Requires F % 64 == 0, H % 64 == 0.
void ck_moe2(int E, const uint16_t* const* g, const uint16_t* const* u, const uint16_t* const* d,
             const float* w, const uint16_t* x, float* out, int H, int F) {
    if (E <= 0) return;
    uint16_t* act = (uint16_t*)aligned_alloc(64, (size_t)E * F * 2 + 64);
    float* gb = (float*)aligned_alloc(64, (size_t)E * 2 * F * 4 + 64);
    const int RB = 64, nblk = (2 * F) / RB, HB = 32;
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 1)
        for (int t = 0; t < E * nblk; t++) {
            int e = t / nblk, b = t % nblk;
            for (int r = b * RB; r < (b + 1) * RB; r += 2) {
                const uint16_t* r0 = r < F ? g[e] + (size_t)r * H : u[e] + (size_t)(r - F) * H;
                const uint16_t* r1 = (r + 1) < F ? g[e] + (size_t)(r + 1) * H : u[e] + (size_t)(r + 1 - F) * H;
                dot2_bf16(r0, r1, x, H, &gb[(size_t)e * 2 * F + r], &gb[(size_t)e * 2 * F + r + 1]);
            }
        }
        #pragma omp for schedule(static)
        for (int t = 0; t < E * F; t++) {
            int e = t / F, f = t % F;
            float gv = gb[(size_t)e * 2 * F + f], uv = gb[(size_t)e * 2 * F + F + f];
            act[t] = f2bf((gv / (1.f + expf(-gv))) * uv);
        }
        #pragma omp for schedule(dynamic, 1)
        for (int hb = 0; hb < H / HB; hb++)
            for (int e = 0; e < E; e++)
                for (int h = hb * HB; h < (hb + 1) * HB; h += 2) {
                    float a, b2;
                    dot2_bf16(d[e] + (size_t)h * F, d[e] + (size_t)(h + 1) * F, act + (size_t)e * F, F, &a, &b2);
                    out[h] += w[e] * a; out[h + 1] += w[e] * b2;
                }
    }
    free(act); free(gb);
}

// ---- NUMA-local variant: thread t owns the same row ranges in placement and compute (static) ----
static inline void range_of(int n, int T, int t, int* lo, int* hi) {
    // rows split in multiples of 2 (paired dots)
    int per = ((n / 2) + T - 1) / T * 2;
    *lo = t * per; *hi = *lo + per; if (*lo > n) *lo = n; if (*hi > n) *hi = n;
}

// Copy one expert into dst ([g F*H][u F*H][d H*F]) so that each page is first touched by the thread that
// will read it in ck_moe3 (same static partition; requires the same OMP thread count + binding).
void ck_place_expert(const uint16_t* g, const uint16_t* u, const uint16_t* d, uint16_t* dst, int H, int F) {
    #pragma omp parallel
    {
        int T = omp_get_num_threads(), t = omp_get_thread_num(), lo, hi;
        range_of(2 * F, T, t, &lo, &hi);
        for (int r = lo; r < hi; r++) {
            const uint16_t* src = r < F ? g + (size_t)r * H : u + (size_t)(r - F) * H;
            memcpy(dst + (size_t)r * H, src, (size_t)H * 2);
        }
        range_of(H, T, t, &lo, &hi);
        memcpy(dst + (size_t)2 * F * H + (size_t)lo * F, d + (size_t)lo * F, (size_t)(hi - lo) * F * 2);
    }
}

// experts given as base pointers of ck_place_expert layouts
void ck_moe3(int E, const uint16_t* const* base, const float* w, const uint16_t* x, float* out, int H, int F) {
    if (E <= 0) return;
    uint16_t* act = (uint16_t*)aligned_alloc(64, (size_t)E * F * 2 + 64);
    float* gb = (float*)aligned_alloc(64, (size_t)E * 2 * F * 4 + 64);
    #pragma omp parallel
    {
        int T = omp_get_num_threads(), t = omp_get_thread_num(), lo, hi;
        range_of(2 * F, T, t, &lo, &hi);
        for (int e = 0; e < E; e++)
            for (int r = lo; r < hi; r += 2)
                dot2_bf16(base[e] + (size_t)r * H, base[e] + (size_t)(r + 1) * H, x, H,
                          &gb[(size_t)e * 2 * F + r], &gb[(size_t)e * 2 * F + r + 1]);
        #pragma omp barrier
        int flo = (int)((long)F * t / T), fhi = (int)((long)F * (t + 1) / T);
        for (int e = 0; e < E; e++)
            for (int f = flo; f < fhi; f++) {
                float gv = gb[(size_t)e * 2 * F + f], uv = gb[(size_t)e * 2 * F + F + f];
                act[(size_t)e * F + f] = f2bf((gv / (1.f + expf(-gv))) * uv);
            }
        #pragma omp barrier
        range_of(H, T, t, &lo, &hi);
        for (int e = 0; e < E; e++) {
            const uint16_t* D = base[e] + (size_t)2 * F * H;
            for (int h = lo; h < hi; h += 2) {
                float a, b2;
                dot2_bf16(D + (size_t)h * F, D + (size_t)(h + 1) * F, act + (size_t)e * F, F, &a, &b2);
                out[h] += w[e] * a; out[h + 1] += w[e] * b2;
            }
        }
    }
    free(act); free(gb);
}

// read-bandwidth probe: each thread streams its own first-touched region (bytes per thread = n)
double ck_read_bw(size_t n_per_thread, int iters) {
    double best = 0;
    #pragma omp parallel
    {
        int T = omp_get_num_threads(), t = omp_get_thread_num();
        uint16_t* buf = (uint16_t*)aligned_alloc(64, n_per_thread);
        memset(buf, 1, n_per_thread);
        for (int it = 0; it < iters; it++) {
            #pragma omp barrier
            double t0 = omp_get_wtime();
            __m512 acc = _mm512_setzero_ps();
            for (size_t i = 0; i < n_per_thread / 2; i += 32) acc = _mm512_add_ps(acc, (__m512)_mm512_loadu_si512(buf + i));
            volatile float sink = _mm512_reduce_add_ps(acc); (void)sink;
            #pragma omp barrier
            double dt = omp_get_wtime() - t0;
            #pragma omp master
            { double bw = (double)n_per_thread * T / dt / 1e9; if (bw > best) best = bw; }
        }
        free(buf);
    }
    return best;
}

// ---------------- int8 cold experts (approximate mode) ----------------
// Placed layout per expert (all row ranges first-touched by the owning thread, same partition as ck_moe3):
//   q   : int8  [(2F + H) rows]: rows 0..2F = gate|up rows (len H), rows 2F.. = down rows (len F)
//   sc  : f32   [2F + H] per-row scale (w = q * sc)
//   rs  : int32 [2F + H] per-row sum of q (for the u8 offset trick)
// Dot with activation: x quantized per call to s8 (sx), fed as u8 = x + 128 to vpdpbusd:
//   sum(q * x) = dpbusd(u8, q) - 128 * rs
typedef struct { int8_t* q; float* sc; int32_t* rs; } q8exp;

static inline int32_t dot_u8s8(const uint8_t* a, const int8_t* b, int n) {
    __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
    int i = 0;
    for (; i + 128 <= n; i += 128) {
        acc0 = _mm512_dpbusd_epi32(acc0, _mm512_loadu_si512(a + i), _mm512_loadu_si512(b + i));
        acc1 = _mm512_dpbusd_epi32(acc1, _mm512_loadu_si512(a + i + 64), _mm512_loadu_si512(b + i + 64));
    }
    for (; i < n; i += 64) acc0 = _mm512_dpbusd_epi32(acc0, _mm512_loadu_si512(a + i), _mm512_loadu_si512(b + i));
    return _mm512_reduce_add_epi32(_mm512_add_epi32(acc0, acc1));
}

static void quant_row(const uint16_t* src, int n, int8_t* dst, float* sc, int32_t* rs) {
    float m = 0;
    for (int i = 0; i < n; i++) { float v = fabsf(bf2f_c(src[i])); if (v > m) m = v; }
    float s = m > 0 ? m / 127.f : 1.f, inv = 1.f / s; int32_t sum = 0;
    for (int i = 0; i < n; i++) { int q = (int)lrintf(bf2f_c(src[i]) * inv); q = q > 127 ? 127 : (q < -127 ? -127 : q); dst[i] = (int8_t)q; sum += q; }
    *sc = s; *rs = sum;
}

// buffer must hold (2F+H)*max(H,F) int8 + (2F+H)*(4+4) bytes; returns offsets via out pointers
void ck_place_expert_q8(const uint16_t* g, const uint16_t* u, const uint16_t* d, int8_t* qbuf, float* sc, int32_t* rs,
                        int H, int F) {
    #pragma omp parallel
    {
        int T = omp_get_num_threads(), t = omp_get_thread_num(), lo, hi;
        range_of(2 * F, T, t, &lo, &hi);
        for (int r = lo; r < hi; r++) {
            const uint16_t* src = r < F ? g + (size_t)r * H : u + (size_t)(r - F) * H;
            quant_row(src, H, qbuf + (size_t)r * H, &sc[r], &rs[r]);
        }
        range_of(H, T, t, &lo, &hi);
        int8_t* qd = qbuf + (size_t)2 * F * H;
        for (int h = lo; h < hi; h++) quant_row(d + (size_t)h * F, F, qd + (size_t)h * F, &sc[2 * F + h], &rs[2 * F + h]);
    }
}

static void quant_act(const float* v, int n, uint8_t* out, float* s_out) {
    float m = 0; for (int i = 0; i < n; i++) { float a = fabsf(v[i]); if (a > m) m = a; }
    float s = m > 0 ? m / 127.f : 1.f, inv = 1.f / s;
    for (int i = 0; i < n; i++) { int q = (int)lrintf(v[i] * inv); q = q > 127 ? 127 : (q < -127 ? -127 : q); out[i] = (uint8_t)(q + 128); }
    *s_out = s;
}

void ck_moe_q8(int E, int8_t* const* q, float* const* sc, int32_t* const* rs, const float* w,
               const uint16_t* x, float* out, int H, int F) {
    if (E <= 0) return;
    float* xf = (float*)malloc((size_t)H * 4);
    for (int i = 0; i < H; i++) xf[i] = bf2f_c(x[i]);
    uint8_t* xq = (uint8_t*)aligned_alloc(64, (size_t)H + 64); float sx;
    quant_act(xf, H, xq, &sx);
    float* gb = (float*)aligned_alloc(64, (size_t)E * 2 * F * 4 + 64);
    float* actf = (float*)aligned_alloc(64, (size_t)E * F * 4 + 64);
    uint8_t* aq = (uint8_t*)aligned_alloc(64, (size_t)E * F + 64);
    float* sa = (float*)malloc((size_t)E * 4);
    #pragma omp parallel
    {
        int T = omp_get_num_threads(), t = omp_get_thread_num(), lo, hi;
        range_of(2 * F, T, t, &lo, &hi);
        for (int e = 0; e < E; e++)
            for (int r = lo; r < hi; r++) {
                int32_t a = dot_u8s8(xq, q[e] + (size_t)r * H, H) - 128 * rs[e][r];
                gb[(size_t)e * 2 * F + r] = (float)a * sx * sc[e][r];
            }
        #pragma omp barrier
        #pragma omp for schedule(static)
        for (int t2 = 0; t2 < E * F; t2++) {
            int e = t2 / F, f = t2 % F;
            float gv = bf2f_c(f2bf(gb[(size_t)e * 2 * F + f])), uv = bf2f_c(f2bf(gb[(size_t)e * 2 * F + F + f]));
            actf[t2] = bf2f_c(f2bf((gv / (1.f + expf(-gv))) * uv));
        }
        #pragma omp single
        for (int e = 0; e < E; e++) quant_act(actf + (size_t)e * F, F, aq + (size_t)e * F, &sa[e]);
        range_of(H, T, t, &lo, &hi);
        for (int e = 0; e < E; e++) {
            const int8_t* qd = q[e] + (size_t)2 * F * H;
            for (int h = lo; h < hi; h++) {
                int32_t a = dot_u8s8(aq + (size_t)e * F, qd + (size_t)h * F, F) - 128 * rs[e][2 * F + h];
                out[h] += w[e] * (float)a * sa[e] * sc[e][2 * F + h];
            }
        }
    }
    free(xf); free(xq); free(gb); free(actf); free(aq); free(sa);
}

// weight-only int8: x stays bf16; int8 weights are widened exactly to bf16 and fed to vdpbf16ps.
static inline __m512bh i8x32_to_bf16(const int8_t* p) {
    __m512i w16 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)p));
    __m512 lo = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(w16)));
    __m512 hi = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(w16, 1)));
    return _mm512_cvtne2ps_pbh(hi, lo);
}
static inline float dot_bf16_i8(const uint16_t* x, const int8_t* w, int n) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    for (int i = 0; i < n; i += 64) {
        a0 = _mm512_dpbf16_ps(a0, (__m512bh)_mm512_loadu_si512(x + i), i8x32_to_bf16(w + i));
        a1 = _mm512_dpbf16_ps(a1, (__m512bh)_mm512_loadu_si512(x + i + 32), i8x32_to_bf16(w + i + 32));
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
}
void ck_moe_q8w(int E, int8_t* const* q, float* const* sc, const float* w, const uint16_t* x, float* out, int H, int F) {
    if (E <= 0) return;
    float* gb = (float*)aligned_alloc(64, (size_t)E * 2 * F * 4 + 64);
    uint16_t* act = (uint16_t*)aligned_alloc(64, (size_t)E * F * 2 + 64);
    #pragma omp parallel
    {
        int T = omp_get_num_threads(), t = omp_get_thread_num(), lo, hi;
        range_of(2 * F, T, t, &lo, &hi);
        for (int e = 0; e < E; e++)
            for (int r = lo; r < hi; r++)
                gb[(size_t)e * 2 * F + r] = dot_bf16_i8(x, q[e] + (size_t)r * H, H) * sc[e][r];
        #pragma omp barrier
        int flo = (int)((long)F * t / T), fhi = (int)((long)F * (t + 1) / T);
        for (int e = 0; e < E; e++)
            for (int f = flo; f < fhi; f++) {
                float gv = gb[(size_t)e * 2 * F + f], uv = gb[(size_t)e * 2 * F + F + f];
                act[(size_t)e * F + f] = f2bf((gv / (1.f + expf(-gv))) * uv);
            }
        #pragma omp barrier
        range_of(H, T, t, &lo, &hi);
        for (int e = 0; e < E; e++) {
            const int8_t* qd = q[e] + (size_t)2 * F * H;
            for (int h = lo; h < hi; h++)
                out[h] += w[e] * dot_bf16_i8(act + (size_t)e * F, qd + (size_t)h * F, F) * sc[e][2 * F + h];
        }
    }
    free(gb); free(act);
}

// int8 weights x 16-bit activations via two VNNI passes: x16 = hi*256 + lo, lo in [0,255] (u8),
// hi signed 8-bit -> fed as u8 (hi+128) with the rowsum correction.
//   sum(w*x16) = 256*(dpbusd(hi+128, w) - 128*rs) + dpbusd(lo, w)
static void quant16(const float* v, int n, uint8_t* hi, uint8_t* lo, float* s_out) {
    float m = 0; for (int i = 0; i < n; i++) { float a = fabsf(v[i]); if (a > m) m = a; }
    float s = m > 0 ? m / 32767.f : 1.f, inv = 1.f / s;
    for (int i = 0; i < n; i++) {
        int q = (int)lrintf(v[i] * inv); q = q > 32767 ? 32767 : (q < -32767 ? -32767 : q);
        int h = q >> 8; int l = q & 255;           // q = h*256 + l, h in [-128,127]
        hi[i] = (uint8_t)(h + 128); lo[i] = (uint8_t)l;
    }
    *s_out = s;
}
static inline int64_t dot16(const uint8_t* hi, const uint8_t* lo, const int8_t* w, int n, int32_t rs) {
    __m512i ah = _mm512_setzero_si512(), al = _mm512_setzero_si512();
    for (int i = 0; i < n; i += 64) {
        __m512i wv = _mm512_loadu_si512(w + i);
        ah = _mm512_dpbusd_epi32(ah, _mm512_loadu_si512(hi + i), wv);
        al = _mm512_dpbusd_epi32(al, _mm512_loadu_si512(lo + i), wv);
    }
    int64_t h = (int64_t)_mm512_reduce_add_epi32(ah) - 128LL * rs;
    return h * 256 + (int64_t)_mm512_reduce_add_epi32(al);
}
void ck_moe_q8x(int E, int8_t* const* q, float* const* sc, int32_t* const* rs, const float* w,
                const uint16_t* x, float* out, int H, int F) {
    if (E <= 0) return;
    float* xf = (float*)malloc((size_t)H * 4);
    for (int i = 0; i < H; i++) xf[i] = bf2f_c(x[i]);
    uint8_t* xh = (uint8_t*)aligned_alloc(64, (size_t)H + 64); uint8_t* xl = (uint8_t*)aligned_alloc(64, (size_t)H + 64);
    float sx; quant16(xf, H, xh, xl, &sx);
    float* gb = (float*)aligned_alloc(64, (size_t)E * 2 * F * 4 + 64);
    float* actf = (float*)aligned_alloc(64, (size_t)E * F * 4 + 64);
    uint8_t* ah = (uint8_t*)aligned_alloc(64, (size_t)E * F + 64); uint8_t* al = (uint8_t*)aligned_alloc(64, (size_t)E * F + 64);
    float* sa = (float*)malloc((size_t)E * 4);
    #pragma omp parallel
    {
        int T = omp_get_num_threads(), t = omp_get_thread_num(), lo, hi;
        range_of(2 * F, T, t, &lo, &hi);
        for (int e = 0; e < E; e++)
            for (int r = lo; r < hi; r++)
                gb[(size_t)e * 2 * F + r] = (float)dot16(xh, xl, q[e] + (size_t)r * H, H, rs[e][r]) * sx * sc[e][r];
        #pragma omp barrier
        #pragma omp for schedule(static)
        for (int t2 = 0; t2 < E * F; t2++) {
            int e = t2 / F, f = t2 % F;
            float gv = gb[(size_t)e * 2 * F + f], uv = gb[(size_t)e * 2 * F + F + f];
            actf[t2] = bf2f_c(f2bf((gv / (1.f + expf(-gv))) * uv));
        }
        #pragma omp for schedule(static)
        for (int e = 0; e < E; e++) quant16(actf + (size_t)e * F, F, ah + (size_t)e * F, al + (size_t)e * F, &sa[e]);
        range_of(H, T, t, &lo, &hi);
        for (int e = 0; e < E; e++) {
            const int8_t* qd = q[e] + (size_t)2 * F * H;
            for (int h = lo; h < hi; h++)
                out[h] += w[e] * (float)dot16(ah + (size_t)e * F, al + (size_t)e * F, qd + (size_t)h * F, F, rs[e][2 * F + h]) * sa[e] * sc[e][2 * F + h];
        }
    }
    free(xf); free(xh); free(xl); free(gb); free(actf); free(ah); free(al); free(sa);
}

// Pull the row ranges each OpenMP thread will read in ck_moe3 into that thread's caches (L2/L3).
// Same static partition as ck_moe3/ck_place_expert, so thread t later reads exactly what it touched.
// Exact: only moves data, no math. Returns a sink to keep the loads alive.
float ck_touch_experts(int E, const uint16_t* const* base, int H, int F) {
    float sink_all = 0.f;
    if (E <= 0) return 0.f;
    #pragma omp parallel reduction(+:sink_all)
    {
        int T = omp_get_num_threads(), t = omp_get_thread_num(), lo, hi;
        __m512 acc = _mm512_setzero_ps();
        range_of(2 * F, T, t, &lo, &hi);
        for (int e = 0; e < E; e++) {
            const uint16_t* p = base[e] + (size_t)lo * H;
            size_t n = (size_t)(hi - lo) * H;
            for (size_t i = 0; i < n; i += 32) acc = _mm512_add_ps(acc, (__m512)_mm512_loadu_si512(p + i));
        }
        range_of(H, T, t, &lo, &hi);
        for (int e = 0; e < E; e++) {
            const uint16_t* p = base[e] + (size_t)2 * F * H + (size_t)lo * F;
            size_t n = (size_t)(hi - lo) * F;
            for (size_t i = 0; i < n; i += 32) acc = _mm512_add_ps(acc, (__m512)_mm512_loadu_si512(p + i));
        }
        sink_all += _mm512_reduce_add_ps(acc);
    }
    return sink_all;
}

// Read device memory through the mapped BAR window: uncached 64-byte loads are serialized per core
// (~0.7 us each), so split the lines over `ck_mmio_threads` threads (they overlap on PCIe).
static int ck_mmio_threads = 8;
void ck_mmio_set_threads(int n) { ck_mmio_threads = n; }
void ck_mmio_read(const uint8_t* src, uint8_t* dst, size_t len) {
    size_t lines = len / 64;
    int nt = ck_mmio_threads;
    if (lines < (size_t)nt * 2) nt = 1;
    if (nt <= 1) {
        for (size_t i = 0; i < lines; i++) _mm512_storeu_si512(dst + i * 64, _mm512_load_si512(src + i * 64));
    } else {
        #pragma omp parallel for num_threads(nt) schedule(static)
        for (size_t i = 0; i < lines; i++) _mm512_storeu_si512(dst + i * 64, _mm512_load_si512(src + i * 64));
    }
    for (size_t i = lines * 64; i < len; i++) dst[i] = ((volatile const uint8_t*)src)[i];
}
