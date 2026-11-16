// The kernel source (src/kernels/kernels.inc) compiled as C++, twice: jpo::kd in double (the
// reference) and jpo::kf in float (what the GPU backends have to reproduce).
#pragma once

#include <cmath>
#include <cstdint>

#define JPO_FN static inline
#define JPO_P
#define JPO_SQRT(x) std::sqrt(x)
#define JPO_ABS(x) std::fabs(x)
#define JPO_CBRT(x) std::cbrt(x)
#define JPO_ACOS(x) std::acos(x)
#define JPO_COS(x) std::cos(x)

namespace jpo {

namespace kd {
typedef double real;
typedef uint32_t jpo_u32;
#define JPO_PTS const double*
#define JPO_C(x) x
#define JPO_TOL 1.0e-10
#define JPO_STURM_DEPTH 60
#include "kernels/kernels.inc"
#undef JPO_PTS
#undef JPO_C
#undef JPO_TOL
#undef JPO_STURM_DEPTH
#undef JPO_MAX_E
#undef JPO_MAX_P
#undef JPO_REL_STRIDE
#undef JPO_ABS_STRIDE
#undef JPO_BIG
#undef JPO_POLISH_STEPS
}  // namespace kd

namespace kf {
typedef float real;
typedef uint32_t jpo_u32;
#define JPO_PTS const float*
#define JPO_C(x) x##f
#define JPO_TOL 1.0e-6f
#define JPO_STURM_DEPTH 36
#include "kernels/kernels.inc"
#undef JPO_PTS
#undef JPO_C
#undef JPO_TOL
#undef JPO_STURM_DEPTH
#undef JPO_BIG
}  // namespace kf

}  // namespace jpo

#undef JPO_FN
#undef JPO_P
#undef JPO_SQRT
#undef JPO_ABS
#undef JPO_CBRT
#undef JPO_ACOS
#undef JPO_COS
