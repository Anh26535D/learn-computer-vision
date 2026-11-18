#define BUILDING_SIFT_DLL
#include "sift_c.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SIFT_DESCR_WIDTH 4
#define SIFT_DESCR_HIST_BINS 8
#define SIFT_INIT_SIGMA 0.5f
#define SIFT_IMG_BORDER 5
#define SIFT_MAX_INTERP_STEPS 5
#define SIFT_ORI_HIST_BINS 36
#define SIFT_ORI_SIG_FCTR 1.5f
#define SIFT_ORI_RADIUS 4.5f
#define SIFT_ORI_PEAK_RATIO 0.8f
#define SIFT_DESCR_SCL_FCTR 3.0f
#define SIFT_DESCR_MAG_THR 0.2f
#define SIFT_INT_DESCR_FCTR 512.0f
#define SIFT_FIXPT_SCALE 1

typedef struct {
    int width;
    int height;
    float* data;
} FloatImage;

static FloatImage create_image(int width, int height) {
    FloatImage img;
    img.width = width;
    img.height = height;
    img.data = (float*)malloc(width * height * sizeof(float));
    return img;
}

static void free_image(FloatImage* img) {
    if (img && img->data) {
        free(img->data);
        img->data = NULL;
    }
}

// BORDER_REFLECT_101 reflection indexing
static inline int reflect101(int p, int len) {
    if (len <= 1) return 0;
    while (p < 0 || p >= len) {
        if (p < 0) {
            p = -p;
        } else {
            p = 2 * len - 2 - p;
        }
    }
    return p;
}

// Bilinear 2x upsampling matching OpenCV INTER_LINEAR
static FloatImage resize_up2(const FloatImage* src) {
    int out_w = src->width * 2;
    int out_h = src->height * 2;
    FloatImage dst = create_image(out_w, out_h);

    for (int y = 0; y < out_h; ++y) {
        float src_y = ((float)y + 0.5f) * 0.5f - 0.5f;
        int y0 = (int)floorf(src_y);
        int y1 = y0 + 1;
        float wy = src_y - (float)y0;
        int y0_c = y0 < 0 ? 0 : (y0 >= src->height ? src->height - 1 : y0);
        int y1_c = y1 < 0 ? 0 : (y1 >= src->height ? src->height - 1 : y1);

        const float* row0 = &src->data[y0_c * src->width];
        const float* row1 = &src->data[y1_c * src->width];
        float* dst_row = &dst.data[y * out_w];

        for (int x = 0; x < out_w; ++x) {
            float src_x = ((float)x + 0.5f) * 0.5f - 0.5f;
            int x0 = (int)floorf(src_x);
            int x1 = x0 + 1;
            float wx = src_x - (float)x0;
            int x0_c = x0 < 0 ? 0 : (x0 >= src->width ? src->width - 1 : x0);
            int x1_c = x1 < 0 ? 0 : (x1 >= src->width ? src->width - 1 : x1);

            float top = (1.0f - wx) * row0[x0_c] + wx * row0[x1_c];
            float bot = (1.0f - wx) * row1[x0_c] + wx * row1[x1_c];
            dst_row[x] = (1.0f - wy) * top + wy * bot;
        }
    }
    return dst;
}

// Separable 1D Gaussian Blur with BORDER_REFLECT_101
static FloatImage gaussian_blur(const FloatImage* src, float sigma) {
    int ksize = ((int)roundf(sigma * 4.0f * 2.0f + 1.0f)) | 1;
    int radius = (ksize - 1) / 2;

    float* kernel = (float*)malloc(ksize * sizeof(float));
    double sum = 0.0;
    for (int i = 0; i < ksize; ++i) {
        float x = (float)(i - radius);
        kernel[i] = expf(-0.5f * (x / sigma) * (x / sigma));
        sum += kernel[i];
    }
    for (int i = 0; i < ksize; ++i) {
        kernel[i] = (float)(kernel[i] / sum);
    }

    int w = src->width;
    int h = src->height;

    // Horizontal pass: src -> tmp
    FloatImage tmp = create_image(w, h);
    for (int y = 0; y < h; ++y) {
        const float* src_row = &src->data[y * w];
        float* tmp_row = &tmp.data[y * w];
        for (int x = 0; x < w; ++x) {
            float val = 0.0f;
            for (int k = 0; k < ksize; ++k) {
                int px = reflect101(x + k - radius, w);
                val += kernel[k] * src_row[px];
            }
            tmp_row[x] = val;
        }
    }

    // Vertical pass: tmp -> dst (row-major streaming for maximum cache locality)
    FloatImage dst = create_image(w, h);
    memset(dst.data, 0, w * h * sizeof(float));
    for (int k = 0; k < ksize; ++k) {
        float kw = kernel[k];
        for (int y = 0; y < h; ++y) {
            int py = reflect101(y + k - radius, h);
            const float* tmp_row = &tmp.data[py * w];
            float* dst_row = &dst.data[y * w];
            for (int x = 0; x < w; ++x) {
                dst_row[x] += kw * tmp_row[x];
            }
        }
    }

    free(kernel);
    free_image(&tmp);
    return dst;
}

// Downsampling by 2 matching OpenCV Size(src.cols/2, src.rows/2)
static FloatImage downsample_half(const FloatImage* src) {
    int dst_w = src->width / 2;
    int dst_h = src->height / 2;
    FloatImage dst = create_image(dst_w, dst_h);
    for (int y = 0; y < dst_h; ++y) {
        const float* src_row = &src->data[(y * 2) * src->width];
        float* dst_row = &dst.data[y * dst_w];
        for (int x = 0; x < dst_w; ++x) {
            dst_row[x] = src_row[x * 2];
        }
    }
    return dst;
}

// Fast Atan2 returning degrees in [0, 360) matching OpenCV cv::fastAtan2
static inline float cv_fast_atan2(float y, float x) {
    float ax = fabsf(x), ay = fabsf(y);
    if (ax == 0.0f && ay == 0.0f) return 0.0f;
    float a, c, c2;
    if (ax >= ay) {
        c = ay / (ax + 1e-12f);
        c2 = c * c;
        a = (((-0.0464964749f * c2 + 0.15931422f) * c2 - 0.327622764f) * c2 + 0.99986663f) * c;
    } else {
        c = ax / (ay + 1e-12f);
        c2 = c * c;
        a = (((-0.0464964749f * c2 + 0.15931422f) * c2 - 0.327622764f) * c2 + 0.99986663f) * c;
        a = 1.5707963267948966f - a;
    }
    if (x < 0.0f) a = 3.141592653589793f - a;
    if (y < 0.0f) a = 6.283185307179586f - a;
    return a * (180.0f / 3.141592653589793f);
}

// Adjust local extrema using 3D quadratic Taylor expansion
typedef struct {
    float pt_x, pt_y;
    float size;
    float response;
    int octave;
    int layer;
    int r1, c1;
} ExtremaCandidate;

static int adjust_local_extrema(
    const FloatImage* dog_oct,
    int octv, int layer, int r, int c,
    int n_octave_layers,
    float contrast_threshold,
    float edge_threshold,
    float sigma,
    ExtremaCandidate* out_cand
) {
    const float img_scale = 1.0f / (255.0f * SIFT_FIXPT_SCALE);
    const float deriv_scale = img_scale * 0.5f;
    const float second_deriv_scale = img_scale;
    const float cross_deriv_scale = img_scale * 0.25f;

    float xc = 0.0f, xr = 0.0f, xi = 0.0f;
    int curr_l = layer;
    int curr_r = r;
    int curr_c = c;
    int converged = 0;

    int W = dog_oct[0].width;
    int H = dog_oct[0].height;

    for (int step = 0; step < SIFT_MAX_INTERP_STEPS; ++step) {
        const float* img = dog_oct[curr_l].data;
        const float* prev = dog_oct[curr_l - 1].data;
        const float* next = dog_oct[curr_l + 1].data;

        // Gradient vector dD
        float dD[3] = {
            (img[curr_r * W + (curr_c + 1)] - img[curr_r * W + (curr_c - 1)]) * deriv_scale,
            (img[(curr_r + 1) * W + curr_c] - img[(curr_r - 1) * W + curr_c]) * deriv_scale,
            (next[curr_r * W + curr_c] - prev[curr_r * W + curr_c]) * deriv_scale
        };

        // Hessian 3x3 matrix
        float v2 = img[curr_r * W + curr_c] * 2.0f;
        float dxx = (img[curr_r * W + (curr_c + 1)] + img[curr_r * W + (curr_c - 1)] - v2) * second_deriv_scale;
        float dyy = (img[(curr_r + 1) * W + curr_c] + img[(curr_r - 1) * W + curr_c] - v2) * second_deriv_scale;
        float dss = (next[curr_r * W + curr_c] + prev[curr_r * W + curr_c] - v2) * second_deriv_scale;
        float dxy = (img[(curr_r + 1) * W + (curr_c + 1)] - img[(curr_r + 1) * W + (curr_c - 1)] -
                     img[(curr_r - 1) * W + (curr_c + 1)] + img[(curr_r - 1) * W + (curr_c - 1)]) * cross_deriv_scale;
        float dxs = (next[curr_r * W + (curr_c + 1)] - next[curr_r * W + (curr_c - 1)] -
                     prev[curr_r * W + (curr_c + 1)] + prev[curr_r * W + (curr_c - 1)]) * cross_deriv_scale;
        float dys = (next[(curr_r + 1) * W + curr_c] - next[(curr_r - 1) * W + curr_c] -
                     prev[(curr_r + 1) * W + curr_c] + prev[(curr_r - 1) * W + curr_c]) * cross_deriv_scale;

        // Invert 3x3 Hessian analytically (Cramer's rule)
        float c00 = dyy * dss - dys * dys;
        float c01 = dys * dxs - dxy * dss;
        float c02 = dxy * dys - dyy * dxs;
        float det = dxx * c00 + dxy * c01 + dxs * c02;

        if (fabsf(det) < 1e-10f) return 0;

        float inv_det = 1.0f / det;
        float c11 = dxx * dss - dxs * dxs;
        float c12 = dxs * dxy - dxx * dys;
        float c22 = dxx * dyy - dxy * dxy;

        float X0 = (c00 * dD[0] + c01 * dD[1] + c02 * dD[2]) * inv_det;
        float X1 = (c01 * dD[0] + c11 * dD[1] + c12 * dD[2]) * inv_det;
        float X2 = (c02 * dD[0] + c12 * dD[1] + c22 * dD[2]) * inv_det;

        xc = -X0;
        xr = -X1;
        xi = -X2;

        if (fabsf(xi) < 0.5f && fabsf(xr) < 0.5f && fabsf(xc) < 0.5f) {
            converged = 1;
            break;
        }

        if (fabsf(xi) > 1e6f || fabsf(xr) > 1e6f || fabsf(xc) > 1e6f) return 0;

        curr_c += (int)roundf(xc);
        curr_r += (int)roundf(xr);
        curr_l += (int)roundf(xi);

        if (curr_l < 1 || curr_l > n_octave_layers ||
            curr_c < SIFT_IMG_BORDER || curr_c >= W - SIFT_IMG_BORDER ||
            curr_r < SIFT_IMG_BORDER || curr_r >= H - SIFT_IMG_BORDER) {
            return 0;
        }
    }

    if (!converged) return 0;

    // Contrast check
    const float* img = dog_oct[curr_l].data;
    const float* prev = dog_oct[curr_l - 1].data;
    const float* next = dog_oct[curr_l + 1].data;

    float dD[3] = {
        (img[curr_r * W + (curr_c + 1)] - img[curr_r * W + (curr_c - 1)]) * deriv_scale,
        (img[(curr_r + 1) * W + curr_c] - img[(curr_r - 1) * W + curr_c]) * deriv_scale,
        (next[curr_r * W + curr_c] - prev[curr_r * W + curr_c]) * deriv_scale
    };

    float t = dD[0] * xc + dD[1] * xr + dD[2] * xi;
    float contr = img[curr_r * W + curr_c] * img_scale + t * 0.5f;

    if (fabsf(contr) * (float)n_octave_layers < contrast_threshold) return 0;

    // Edge rejection check
    float v2 = img[curr_r * W + curr_c] * 2.0f;
    float dxx = (img[curr_r * W + (curr_c + 1)] + img[curr_r * W + (curr_c - 1)] - v2) * second_deriv_scale;
    float dyy = (img[(curr_r + 1) * W + curr_c] + img[(curr_r - 1) * W + curr_c] - v2) * second_deriv_scale;
    float dxy = (img[(curr_r + 1) * W + (curr_c + 1)] - img[(curr_r + 1) * W + (curr_c - 1)] -
                 img[(curr_r - 1) * W + (curr_c + 1)] + img[(curr_r - 1) * W + (curr_c - 1)]) * cross_deriv_scale;

    float tr = dxx + dyy;
    float det2 = dxx * dyy - dxy * dxy;

    if (det2 <= 0.0f || (tr * tr * edge_threshold) >= ((edge_threshold + 1.0f) * (edge_threshold + 1.0f) * det2)) {
        return 0;
    }

    float x_oct = (curr_c + xc) * (float)(1 << octv);
    float y_oct = (curr_r + xr) * (float)(1 << octv);
    int octave_packed = octv + (curr_l << 8) + (((int)roundf((xi + 0.5f) * 255.0f)) << 16);
    float size = sigma * powf(2.0f, (curr_l + xi) / (float)n_octave_layers) * (float)(1 << octv) * 2.0f;

    out_cand->pt_x = x_oct;
    out_cand->pt_y = y_oct;
    out_cand->size = size;
    out_cand->response = fabsf(contr);
    out_cand->octave = octave_packed;
    out_cand->layer = curr_l;
    out_cand->r1 = curr_r;
    out_cand->c1 = curr_c;
    return 1;
}

// Orientation histogram calculation
static void calc_orientation_hist(
    const FloatImage* img,
    int c0, int r0,
    int radius,
    float sigma,
    float* hist
) {
    int W = img->width;
    int H = img->height;
    float expf_scale = -1.0f / (2.0f * sigma * sigma);
    float temphist[SIFT_ORI_HIST_BINS] = {0};

    for (int i = -radius; i <= radius; ++i) {
        int y = r0 + i;
        if (y <= 0 || y >= H - 1) continue;
        for (int j = -radius; j <= radius; ++j) {
            int x = c0 + j;
            if (x <= 0 || x >= W - 1) continue;

            float dx = img->data[y * W + (x + 1)] - img->data[y * W + (x - 1)];
            float dy = img->data[(y - 1) * W + x] - img->data[(y + 1) * W + x];

            float w = expf(((float)(i * i + j * j)) * expf_scale);
            float ori = cv_fast_atan2(dy, dx);
            float mag = sqrtf(dx * dx + dy * dy);

            int bin_idx = (int)roundf(((float)SIFT_ORI_HIST_BINS / 360.0f) * ori);
            if (bin_idx >= SIFT_ORI_HIST_BINS) bin_idx -= SIFT_ORI_HIST_BINS;
            if (bin_idx < 0) bin_idx += SIFT_ORI_HIST_BINS;
            temphist[bin_idx] += w * mag;
        }
    }

    // Circular smoothing [1, 4, 6, 4, 1] / 16
    for (int i = 0; i < SIFT_ORI_HIST_BINS; ++i) {
        int im2 = (i - 2 + SIFT_ORI_HIST_BINS) % SIFT_ORI_HIST_BINS;
        int im1 = (i - 1 + SIFT_ORI_HIST_BINS) % SIFT_ORI_HIST_BINS;
        int ip1 = (i + 1) % SIFT_ORI_HIST_BINS;
        int ip2 = (i + 2) % SIFT_ORI_HIST_BINS;
        hist[i] = (temphist[im2] + temphist[ip2]) * (1.0f / 16.0f) +
                  (temphist[im1] + temphist[ip1]) * (4.0f / 16.0f) +
                  temphist[i] * (6.0f / 16.0f);
    }
}

// Keypoint comparator for qsort matching OpenCV KeyPointsFilter::removeDuplicatedSorted
static int kp_compare(const void* a, const void* b) {
    const SiftKeypoint* k1 = (const SiftKeypoint*)a;
    const SiftKeypoint* k2 = (const SiftKeypoint*)b;
    if (k1->x != k2->x) return (k1->x < k2->x) ? -1 : 1;
    if (k1->y != k2->y) return (k1->y < k2->y) ? -1 : 1;
    if (k1->size != k2->size) return (k1->size > k2->size) ? -1 : 1; // -size
    if (k1->angle != k2->angle) return (k1->angle < k2->angle) ? -1 : 1;
    if (k1->response != k2->response) return (k1->response > k2->response) ? -1 : 1; // -response
    if (k1->octave != k2->octave) return (k1->octave > k2->octave) ? -1 : 1; // -octave
    return 0;
}

// Compute 128D SIFT descriptor matching OpenCV calcSIFTDescriptor
static void calc_sift_descriptor(
    const FloatImage* img,
    float ptf_x, float ptf_y,
    float angle,
    float size_half,
    float* out_desc
) {
    int d = SIFT_DESCR_WIDTH;
    int n = SIFT_DESCR_HIST_BINS;

    int pt_x = (int)roundf(ptf_x);
    int pt_y = (int)roundf(ptf_y);

    float rad = angle * (float)(M_PI / 180.0);
    float cos_t = cosf(rad);
    float sin_t = sinf(rad);
    float bins_per_rad = (float)n / 360.0f;
    float exp_scale = -1.0f / ((float)(d * d) * 0.5f);
    float hist_width = SIFT_DESCR_SCL_FCTR * size_half;
    int radius = (int)roundf(hist_width * 1.4142135623730951f * (float)(d + 1) * 0.5f);
    int max_r = (int)sqrtf((float)(img->width * img->width + img->height * img->height));
    if (radius > max_r) radius = max_r;

    cos_t /= hist_width;
    sin_t /= hist_width;

    float hist[6][6][10];
    memset(hist, 0, sizeof(hist));
    int rows = img->height;
    int cols = img->width;

    for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
            float c_rot = (float)j * cos_t - (float)i * sin_t;
            float r_rot = (float)j * sin_t + (float)i * cos_t;
            float rbin = r_rot + (float)d / 2.0f - 0.5f;
            float cbin = c_rot + (float)d / 2.0f - 0.5f;
            int r = pt_y + i;
            int c = pt_x + j;

            if (rbin > -1.0f && rbin < (float)d && cbin > -1.0f && cbin < (float)d &&
                r > 0 && r < rows - 1 && c > 0 && c < cols - 1) {
                float dx = img->data[r * cols + (c + 1)] - img->data[r * cols + (c - 1)];
                float dy = img->data[(r - 1) * cols + c] - img->data[(r + 1) * cols + c];

                float ori = cv_fast_atan2(dy, dx);
                float mag = sqrtf(dx * dx + dy * dy);
                float w = expf((c_rot * c_rot + r_rot * r_rot) * exp_scale);
                mag *= w;

                float obin = (ori - angle) * bins_per_rad;
                int r0 = (int)floorf(rbin);
                int c0 = (int)floorf(cbin);
                int o0 = (int)floorf(obin);

                float dr = rbin - (float)r0;
                float dc = cbin - (float)c0;
                float do_val = obin - (float)o0;

                if (o0 < 0) o0 += n;
                if (o0 >= n) o0 -= n;

                float v_r1 = mag * dr;
                float v_r0 = mag - v_r1;

                float v_rc11 = v_r1 * dc;
                float v_rc10 = v_r1 - v_rc11;
                float v_rc01 = v_r0 * dc;
                float v_rc00 = v_r0 - v_rc01;

                float v_rco111 = v_rc11 * do_val;
                float v_rco110 = v_rc11 - v_rco111;
                float v_rco101 = v_rc10 * do_val;
                float v_rco100 = v_rc10 - v_rco101;
                float v_rco011 = v_rc01 * do_val;
                float v_rco010 = v_rc01 - v_rco011;
                float v_rco001 = v_rc00 * do_val;
                float v_rco000 = v_rc00 - v_rco001;

                hist[r0 + 1][c0 + 1][o0]     += v_rco000;
                hist[r0 + 1][c0 + 1][o0 + 1] += v_rco001;
                hist[r0 + 1][c0 + 2][o0]     += v_rco010;
                hist[r0 + 1][c0 + 2][o0 + 1] += v_rco011;
                hist[r0 + 2][c0 + 1][o0]     += v_rco100;
                hist[r0 + 2][c0 + 1][o0 + 1] += v_rco101;
                hist[r0 + 2][c0 + 2][o0]     += v_rco110;
                hist[r0 + 2][c0 + 2][o0 + 1] += v_rco111;
            }
        }
    }

    float raw_dst[128];
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            hist[i + 1][j + 1][0] += hist[i + 1][j + 1][n];
            hist[i + 1][j + 1][1] += hist[i + 1][j + 1][n + 1];
            for (int k = 0; k < n; ++k) {
                raw_dst[(i * d + j) * n + k] = hist[i + 1][j + 1][k];
            }
        }
    }

    // L2 normalize and 0.2 clipping
    float nrm2 = 0.0f;
    for (int k = 0; k < 128; ++k) nrm2 += raw_dst[k] * raw_dst[k];
    float thr = sqrtf(nrm2) * SIFT_DESCR_MAG_THR;
    for (int k = 0; k < 128; ++k) {
        if (raw_dst[k] > thr) raw_dst[k] = thr;
    }

    // Second L2 normalize and scale by 512
    nrm2 = 0.0f;
    for (int k = 0; k < 128; ++k) nrm2 += raw_dst[k] * raw_dst[k];
    float norm = sqrtf(nrm2);
    if (norm < 1.1920929e-7f) norm = 1.1920929e-7f;
    float scale = SIFT_INT_DESCR_FCTR / norm;

    for (int k = 0; k < 128; ++k) {
        float val = roundf(raw_dst[k] * scale);
        if (val < 0.0f) val = 0.0f;
        if (val > 255.0f) val = 255.0f;
        out_desc[k] = val;
    }
}

// Main SIFT Pipeline implementation
SIFT_API int sift_detect_and_compute(
    const unsigned char* img_data,
    int width,
    int height,
    int n_octave_layers,
    float contrast_threshold,
    float edge_threshold,
    float sigma,
    SiftKeypoint** out_kps,
    float** out_descs,
    int* out_count
) {
    if (!img_data || width <= 0 || height <= 0 || !out_kps || !out_descs || !out_count) {
        return -1;
    }

    int first_octave = -1;

    // Convert uint8 to float32 image
    FloatImage gray = create_image(width, height);
    for (int i = 0; i < width * height; ++i) {
        gray.data[i] = (float)img_data[i];
    }

    // Create base image
    FloatImage base;
    if (first_octave < 0) {
        float sig_diff = sqrtf(fmaxf(sigma * sigma - (SIFT_INIT_SIGMA * 2.0f) * (SIFT_INIT_SIGMA * 2.0f), 0.01f));
        FloatImage dbl = resize_up2(&gray);
        base = gaussian_blur(&dbl, sig_diff);
        free_image(&dbl);
    } else {
        float sig_diff = sqrtf(fmaxf(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f));
        base = gaussian_blur(&gray, sig_diff);
    }
    free_image(&gray);

    int min_dim = base.width < base.height ? base.width : base.height;
    int n_octaves = (int)roundf(logf((float)min_dim) / logf(2.0f) - 2.0f) - first_octave;

    // Compute sigmas
    int n_layers_total = n_octave_layers + 3;
    float k = powf(2.0f, 1.0f / (float)n_octave_layers);
    float* sig = (float*)malloc(n_layers_total * sizeof(float));
    sig[0] = sigma;
    for (int i = 1; i < n_layers_total; ++i) {
        float sig_prev = powf(k, (float)(i - 1)) * sigma;
        float sig_total = sig_prev * k;
        sig[i] = sqrtf(sig_total * sig_total - sig_prev * sig_prev);
    }

    // Build Gaussian Pyramid
    FloatImage** gpyr = (FloatImage**)malloc(n_octaves * sizeof(FloatImage*));
    for (int o = 0; o < n_octaves; ++o) {
        gpyr[o] = (FloatImage*)malloc(n_layers_total * sizeof(FloatImage));
        for (int i = 0; i < n_layers_total; ++i) {
            if (o == 0 && i == 0) {
                gpyr[o][i] = base;
            } else if (i == 0) {
                gpyr[o][i] = downsample_half(&gpyr[o - 1][n_octave_layers]);
            } else {
                gpyr[o][i] = gaussian_blur(&gpyr[o][i - 1], sig[i]);
            }
        }
    }
    free(sig);

    // Build DoG Pyramid
    int n_dog_layers = n_layers_total - 1;
    FloatImage** dogpyr = (FloatImage**)malloc(n_octaves * sizeof(FloatImage*));
    for (int o = 0; o < n_octaves; ++o) {
        dogpyr[o] = (FloatImage*)malloc(n_dog_layers * sizeof(FloatImage));
        for (int i = 0; i < n_dog_layers; ++i) {
            int w = gpyr[o][i].width;
            int h = gpyr[o][i].height;
            dogpyr[o][i] = create_image(w, h);
            for (int p = 0; p < w * h; ++p) {
                dogpyr[o][i].data[p] = gpyr[o][i + 1].data[p] - gpyr[o][i].data[p];
            }
        }
    }

    int threshold = (int)floorf(0.5f * contrast_threshold / (float)n_octave_layers * 255.0f * (float)SIFT_FIXPT_SCALE);

    // Dynamic array for keypoints
    int kp_cap = 2048;
    int kp_count = 0;
    SiftKeypoint* kps = (SiftKeypoint*)malloc(kp_cap * sizeof(SiftKeypoint));

    for (int oct_idx = 0; oct_idx < n_octaves; ++oct_idx) {
        FloatImage* dog_oct = dogpyr[oct_idx];
        for (int layer_idx = 1; layer_idx <= n_octave_layers; ++layer_idx) {
            const FloatImage* prev_img = &dog_oct[layer_idx - 1];
            const FloatImage* curr_img = &dog_oct[layer_idx];
            const FloatImage* next_img = &dog_oct[layer_idx + 1];

            int H = curr_img->height;
            int W = curr_img->width;
            int b = SIFT_IMG_BORDER;

            for (int r = b; r < H - b; ++r) {
                for (int c = b; c < W - b; ++c) {
                    float val = curr_img->data[r * W + c];
                    if (fabsf(val) <= (float)threshold) continue;

                    int is_max = (val > (float)threshold);
                    int is_min = (val < -(float)threshold);

                    // Strict comparison with 26 neighbors
                    for (int di = -1; di <= 1 && (is_max || is_min); ++di) {
                        const FloatImage* l_img = (di == -1) ? prev_img : ((di == 0) ? curr_img : next_img);
                        for (int dr = -1; dr <= 1 && (is_max || is_min); ++dr) {
                            for (int dc = -1; dc <= 1 && (is_max || is_min); ++dc) {
                                if (di == 0 && dr == 0 && dc == 0) continue;
                                float nb = l_img->data[(r + dr) * W + (c + dc)];
                                if (val <= nb) is_max = 0;
                                if (val >= nb) is_min = 0;
                            }
                        }
                    }

                    if (!is_max && !is_min) continue;

                    ExtremaCandidate cand;
                    if (!adjust_local_extrema(dog_oct, oct_idx, layer_idx, r, c,
                                              n_octave_layers, contrast_threshold, edge_threshold, sigma, &cand)) {
                        continue;
                    }

                    float scl_octv = cand.size * 0.5f / (float)(1 << oct_idx);
                    float hist[SIFT_ORI_HIST_BINS];
                    calc_orientation_hist(
                        &gpyr[oct_idx][cand.layer],
                        cand.c1, cand.r1,
                        (int)roundf(SIFT_ORI_RADIUS * scl_octv),
                        SIFT_ORI_SIG_FCTR * scl_octv,
                        hist
                    );

                    float omax = hist[0];
                    for (int j = 1; j < SIFT_ORI_HIST_BINS; ++j) {
                        if (hist[j] > omax) omax = hist[j];
                    }
                    float mag_thr = omax * SIFT_ORI_PEAK_RATIO;

                    for (int j = 0; j < SIFT_ORI_HIST_BINS; ++j) {
                        int l = (j - 1 + SIFT_ORI_HIST_BINS) % SIFT_ORI_HIST_BINS;
                        int r2 = (j + 1) % SIFT_ORI_HIST_BINS;
                        if (hist[j] > hist[l] && hist[j] > hist[r2] && hist[j] >= mag_thr) {
                            float bin_interp = (float)j + 0.5f * (hist[l] - hist[r2]) / (hist[l] - 2.0f * hist[j] + hist[r2]);
                            if (bin_interp < 0.0f) bin_interp += (float)SIFT_ORI_HIST_BINS;
                            else if (bin_interp >= (float)SIFT_ORI_HIST_BINS) bin_interp -= (float)SIFT_ORI_HIST_BINS;

                            float angle = 360.0f - (360.0f / (float)SIFT_ORI_HIST_BINS) * bin_interp;
                            if (fabsf(angle - 360.0f) < 1e-5f) angle = 0.0f;

                            if (kp_count >= kp_cap) {
                                kp_cap *= 2;
                                kps = (SiftKeypoint*)realloc(kps, kp_cap * sizeof(SiftKeypoint));
                            }
                            kps[kp_count].x = cand.pt_x;
                            kps[kp_count].y = cand.pt_y;
                            kps[kp_count].size = cand.size;
                            kps[kp_count].angle = angle;
                            kps[kp_count].response = cand.response;
                            kps[kp_count].octave = cand.octave;
                            kps[kp_count].class_id = -1;
                            kp_count++;
                        }
                    }
                }
            }
        }
    }

    // Sort keypoints according to OpenCV removeDuplicatedSorted
    qsort(kps, kp_count, sizeof(SiftKeypoint), kp_compare);

    // Filter duplicates
    int unique_count = 0;
    SiftKeypoint* unique_kps = (SiftKeypoint*)malloc(kp_count * sizeof(SiftKeypoint));
    for (int i = 0; i < kp_count; ++i) {
        if (unique_count == 0) {
            unique_kps[unique_count++] = kps[i];
        } else {
            const SiftKeypoint* last = &unique_kps[unique_count - 1];
            if (last->x != kps[i].x || last->y != kps[i].y ||
                last->size != kps[i].size || last->angle != kps[i].angle) {
                unique_kps[unique_count++] = kps[i];
            }
        }
    }
    free(kps);

    // Rescale if first_octave < 0
    if (first_octave < 0) {
        float scale = 1.0f / (float)(1 << -first_octave);
        for (int i = 0; i < unique_count; ++i) {
            unique_kps[i].octave = (unique_kps[i].octave & ~255) | ((unique_kps[i].octave + first_octave) & 255);
            unique_kps[i].x *= scale;
            unique_kps[i].y *= scale;
            unique_kps[i].size *= scale;
        }
    }

    // Compute Descriptors
    float* descriptors = (float*)malloc(unique_count * 128 * sizeof(float));
    for (int i = 0; i < unique_count; ++i) {
        int octave_raw = unique_kps[i].octave & 255;
        int layer_raw = (unique_kps[i].octave >> 8) & 255;
        int octv = (octave_raw < 128) ? octave_raw : (octave_raw - 256);
        float scale = (octv >= 0) ? (1.0f / (float)(1 << octv)) : (float)(1 << -octv);

        float ptf_x = unique_kps[i].x * scale;
        float ptf_y = unique_kps[i].y * scale;
        float size_scaled = unique_kps[i].size * scale;
        const FloatImage* img_layer = &gpyr[octv - first_octave][layer_raw];

        float angle = 360.0f - unique_kps[i].angle;
        if (fabsf(angle - 360.0f) < 1e-5f) angle = 0.0f;

        calc_sift_descriptor(img_layer, ptf_x, ptf_y, angle, size_scaled * 0.5f, &descriptors[i * 128]);
    }

    // Clean up pyramids
    for (int o = 0; o < n_octaves; ++o) {
        for (int i = 0; i < n_layers_total; ++i) {
            free_image(&gpyr[o][i]);
        }
        free(gpyr[o]);
        for (int i = 0; i < n_dog_layers; ++i) {
            free_image(&dogpyr[o][i]);
        }
        free(dogpyr[o]);
    }
    free(gpyr);
    free(dogpyr);

    *out_kps = unique_kps;
    *out_descs = descriptors;
    *out_count = unique_count;
    return 0;
}

SIFT_API void sift_free_results(SiftKeypoint* kps, float* descs) {
    if (kps) free(kps);
    if (descs) free(descs);
}
