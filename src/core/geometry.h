// Host-side geometry in double: cameras, rotations, the essential matrix, scores and refinement.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "jpo/jpo.h"

namespace jpo {

// ---- cameras (COLMAP models) -------------------------------------------------------------------------

int camera_param_count(int model);  // 0 for an unknown model
double camera_focal(const jpo_camera& cam);
// Pixel -> normalized image plane (x/z, y/z). False (and NaN) where the ray is not in front of the camera.
bool camera_unproject(const jpo_camera& cam, double px, double py, double* x, double* y);
// Point in the camera frame -> pixel. False if it cannot be projected (behind a pinhole camera).
bool camera_project(const jpo_camera& cam, const double X[3], double* px, double* py);

// ---- rotations ---------------------------------------------------------------------------------------

void quat_to_rotmat(const double q[4], double R[9]);  // q = w x y z; R row major
void rotmat_to_quat(const double R[9], double q[4]);
void rotmat_exp(const double w[3], double R[9]);       // Rodrigues
void mat3_mul(const double A[9], const double B[9], double C[9]);

// ---- relative pose -----------------------------------------------------------------------------------

struct Pose {
  double R[9];
  double t[3];
};
typedef Pose RelPose;
typedef Pose AbsPose;

void essential_from_pose(const RelPose& pose, double E[9]);
// The four (R, t) of an essential matrix (PoseLib motion_from_essential, without its cheirality test).
void poses_from_essential(const double E[9], RelPose out[4]);
// PoseLib's test: both depths of the triangulated point above min_depth. b1, b2 unit bearings.
bool check_cheirality(const RelPose& pose, const double b1[3], const double b2[3], double min_depth = 0.01);

// pts: n * 4 (x1 y1 x2 y2, normalized). MSAC score in units of sq_thr (each point adds min(r2/sq_thr, 1)),
// Sampson error, a point that fails cheirality counts as an outlier. inliers may be null.
double score_relpose(const RelPose& pose, const double* pts, size_t n, double sq_thr, size_t* n_inliers,
                     uint8_t* inliers);
// Truncated least squares on the Sampson error over all points, Levenberg-Marquardt, in place.
void refine_relpose(RelPose* pose, const double* pts, size_t n, double sq_thr, int max_iterations);
// Median over the inliers of the angle between the two rays, in degrees.
double median_triangulation_angle(const RelPose& pose, const double* pts, size_t n, const uint8_t* inliers);

// ---- absolute pose -----------------------------------------------------------------------------------

// pts: n * 5 (x y normalized, X Y Z). Reprojection error on the normalized image plane.
double score_abspose(const AbsPose& pose, const double* pts, size_t n, double sq_thr, size_t* n_inliers,
                     uint8_t* inliers);
void refine_abspose(AbsPose* pose, const double* pts, size_t n, double sq_thr, int max_iterations);
// The same in pixels through the camera model: xy (2 n), xyz (3 n). The score is in units of sq_thr_px.
double score_abspose_px(const AbsPose& pose, const jpo_camera& cam, const double* xy, const double* xyz, size_t n,
                        double sq_thr_px, size_t* n_inliers, uint8_t* inliers);
void refine_abspose_px(AbsPose* pose, const jpo_camera& cam, const double* xy, const double* xyz, size_t n,
                       double sq_thr_px, int max_iterations);

}  // namespace jpo
