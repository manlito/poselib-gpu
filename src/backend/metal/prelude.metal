// Metal prelude of the shared kernels (src/kernels/kernels.inc). The Metal source is this file, the
// kernels and entry.metal, concatenated at build time and compiled at jpo_create.
#include <metal_stdlib>
using namespace metal;

typedef float real;
typedef uint jpo_u32;
#define JPO_FN static inline
#define JPO_P thread
#define JPO_PTS const device float*
#define JPO_C(x) x##f
#define JPO_SQRT(x) sqrt(x)
#define JPO_ABS(x) fabs(x)
#define JPO_CBRT(x) jpo_cbrt(x)
#define JPO_ACOS(x) acos(x)
#define JPO_COS(x) cos(x)
#define JPO_TOL 1.0e-6f
#define JPO_STURM_DEPTH 36

static inline float jpo_cbrt(float x) {
  float r = pow(fabs(x), 1.0f / 3.0f);
  return x < 0.0f ? -r : r;
}

