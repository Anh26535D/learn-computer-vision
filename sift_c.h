#ifndef SIFT_C_H
#define SIFT_C_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
  #ifdef BUILDING_SIFT_DLL
    #define SIFT_API __declspec(dllexport)
  #else
    #define SIFT_API __declspec(dllimport)
  #endif
#else
  #define SIFT_API
#endif

typedef struct {
    float x;
    float y;
    float size;
    float angle;
    float response;
    int octave;
    int class_id;
} SiftKeypoint;

/**
 * Main SIFT detect and compute function in pure C (scalar, no SIMD).
 * @param img_data Grayscale image data (row-major uint8)
 * @param width Image width
 * @param height Image height
 * @param n_octave_layers Number of layers per octave (default: 3)
 * @param contrast_threshold Contrast threshold (default: 0.04)
 * @param edge_threshold Edge threshold (default: 10.0)
 * @param sigma Initial Gaussian blur sigma (default: 1.6)
 * @param out_kps Output pointer to dynamically allocated array of SiftKeypoint
 * @param out_descs Output pointer to dynamically allocated array of descriptors (count * 128 float32)
 * @param out_count Output pointer to number of detected keypoints
 * @return 0 on success, non-zero on error
 */
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
);

/**
 * Free memory allocated by sift_detect_and_compute.
 */
SIFT_API void sift_free_results(SiftKeypoint* kps, float* descs);

#ifdef __cplusplus
}
#endif

#endif // SIFT_C_H
