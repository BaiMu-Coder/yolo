#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <cmath>
#include <optional>

struct DrogueModel {
    double radius_cls0_mm = 0.0;
    double radius_cls1_mm = 0.0;
    double radius_hole_mm = 0.0;
    double length_L_mm    = 0.0;
};

struct Pose6D {
    double yaw_deg   = 0.0;  // around Y
    double pitch_deg = 0.0;  // around X
    double roll_deg  = 0.0;  // fixed 0
    double tx_mm     = 0.0;
    double ty_mm     = 0.0;
    double tz_mm     = 0.0;
};

class PoseEstimatorLM {
public:
    PoseEstimatorLM() = default;

    PoseEstimatorLM(const cv::Mat& cameraK, const cv::Mat& distCoeffs, const DrogueModel& model) {
        Reset(cameraK, distCoeffs, model);
    }

    void Reset(const cv::Mat& cameraK, const cv::Mat& distCoeffs, const DrogueModel& model) {
        K_ = cameraK.clone();
        D_ = distCoeffs.clone();
        model_ = model;

        pts3d_cls0_ = genCircle3D(model_.radius_cls0_mm, 0.0, 32);
        pts3d_cls1_ = genCircle3D(model_.radius_cls1_mm, 0.0, 32);
        pts3d_hole_ = genCircle3D(model_.radius_hole_mm, -model_.length_L_mm, 16);
    }

    Pose6D Solve(const cv::RotatedRect& targetEllipse,
                 const cv::Point2f& holeCenterPx,
                 bool use_cls1,
                 std::optional<double> known_dist_mm = std::nullopt,
                 int max_iters = 30) const
    {
        const auto& pts3d = use_cls1 ? pts3d_cls1_ : pts3d_cls0_;
        const double real_r = use_cls1 ? model_.radius_cls1_mm : model_.radius_cls0_mm;

        // tz init: tz ≈ f * R / r_img
        const double ew = targetEllipse.size.width;
        const double eh = targetEllipse.size.height;
        const double r_img = (ew + eh) * 0.25; // (w+h)/4
        const double f_mean = 0.5 * (K_.at<double>(0,0) + K_.at<double>(1,1));

        double tz_est = 2000.0;
        if (known_dist_mm.has_value()) tz_est = *known_dist_mm;
        else tz_est = (r_img > 1.0) ? (f_mean * real_r / r_img) : 2000.0;

        // yaw/pitch init from hole offset
        const double cx = targetEllipse.center.x;
        const double cy = targetEllipse.center.y;
        const double dx = holeCenterPx.x - cx;
        const double dy = holeCenterPx.y - cy;
        const double L  = model_.length_L_mm;

        double yaw_guess = 0.0, pit_guess = 0.0;
        if (L > 10.0) {
            yaw_guess = rad2deg(-std::atan(dx * tz_est / (f_mean * L)));
            pit_guess = rad2deg( std::atan(dy * tz_est / (f_mean * L)));
        }

        // tx/ty init by back-projecting ellipse center
        double tx0 = (cx - K_.at<double>(0,2)) * tz_est / K_.at<double>(0,0);
        double ty0 = (cy - K_.at<double>(1,2)) * tz_est / K_.at<double>(1,1);

        if (known_dist_mm.has_value()) {
            std::vector<double> x = {yaw_guess, pit_guess, tx0, ty0};
            lmOptimize(x, targetEllipse, holeCenterPx, pts3d, *known_dist_mm, max_iters);
            return {x[0], x[1], 0.0, x[2], x[3], *known_dist_mm};
        } else {
            std::vector<double> x = {yaw_guess, pit_guess, tx0, ty0, tz_est};
            lmOptimize(x, targetEllipse, holeCenterPx, pts3d, std::nullopt, max_iters);
            return {x[0], x[1], 0.0, x[2], x[3], x[4]};
        }
    }

    void DrawAxis(cv::Mat& img, const Pose6D& pose, bool use_cls1) const
    {
        const double length = use_cls1 ? model_.radius_cls1_mm : model_.radius_cls0_mm;

        cv::Vec3d rvec = eulerYXZ_to_rvec(pose.yaw_deg, pose.pitch_deg, pose.roll_deg);
        cv::Vec3d tvec(pose.tx_mm, pose.ty_mm, pose.tz_mm);

        std::vector<cv::Point3f> pts = {
            {0, 0, 0},
            {(float)length, 0, 0},
            {0, (float)length, 0},
            {0, 0, (float)(length * 3.0)}
        };
        std::vector<cv::Point2f> imgpts;
        cv::projectPoints(pts, rvec, tvec, K_, D_, imgpts);

        auto toI = [](const cv::Point2f& p){ return cv::Point((int)std::lround(p.x), (int)std::lround(p.y)); };
        cv::Point o = toI(imgpts[0]);

        cv::line(img, o, toI(imgpts[1]), cv::Scalar(0,0,255), 3);
        cv::line(img, o, toI(imgpts[2]), cv::Scalar(0,255,0), 3);
        cv::line(img, o, toI(imgpts[3]), cv::Scalar(255,0,0), 3);
    }

private:
    cv::Mat K_, D_;
    DrogueModel model_;
    std::vector<cv::Point3f> pts3d_cls0_, pts3d_cls1_, pts3d_hole_;

    static inline double deg2rad(double d){ return d * CV_PI / 180.0; }
    static inline double rad2deg(double r){ return r * 180.0 / CV_PI; }

    static std::vector<cv::Point3f> genCircle3D(double r, double z, int n)
    {
        std::vector<cv::Point3f> pts;
        pts.reserve(n);
        for (int i = 0; i < n; ++i) {
            double th = 2.0 * CV_PI * (double)i / (double)n;
            pts.emplace_back((float)(r * std::cos(th)), (float)(r * std::sin(th)), (float)z);
        }
        return pts;
    }

    // Python: R.from_euler('yxz', [yaw,pitch,roll]) => R = Rz(roll)*Rx(pitch)*Ry(yaw)
    static cv::Vec3d eulerYXZ_to_rvec(double yaw_deg, double pitch_deg, double roll_deg)
    {
        const double y = deg2rad(yaw_deg);
        const double p = deg2rad(pitch_deg);
        const double r = deg2rad(roll_deg);

        cv::Matx33d Ry( std::cos(y), 0, std::sin(y),
                        0,          1, 0,
                       -std::sin(y), 0, std::cos(y) );

        cv::Matx33d Rx( 1, 0, 0,
                        0, std::cos(p), -std::sin(p),
                        0, std::sin(p),  std::cos(p) );

        cv::Matx33d Rz( std::cos(r), -std::sin(r), 0,
                        std::sin(r),  std::cos(r), 0,
                        0, 0, 1 );

        cv::Matx33d R = Rz * Rx * Ry;

        cv::Vec3d rvec;
        cv::Rodrigues(cv::Mat(R), rvec);
        return rvec;
    }

    void computeResidual(const std::vector<double>& x,
                         const cv::RotatedRect& targetEllipse,
                         const cv::Point2f& holeCenterPx,
                         const std::vector<cv::Point3f>& pts3d,
                         std::optional<double> fixed_tz,
                         std::vector<double>& residual) const
    {
        double yaw, pitch, tx, ty, tz;
        if (fixed_tz.has_value()) {
            yaw = x[0]; pitch = x[1]; tx = x[2]; ty = x[3]; tz = *fixed_tz;
        } else {
            yaw = x[0]; pitch = x[1]; tx = x[2]; ty = x[3]; tz = x[4];
        }

        residual.clear();
        residual.reserve((int)pts3d.size() + 1);

        if (tz < 100.0 || targetEllipse.size.width < 2.0 || targetEllipse.size.height < 2.0) {
            residual.assign((int)pts3d.size() + 1, 100000.0);
            return;
        }

        cv::Vec3d rvec = eulerYXZ_to_rvec(yaw, pitch, 0.0);
        cv::Vec3d tvec(tx, ty, tz);

        std::vector<cv::Point2f> proj_outer;
        cv::projectPoints(pts3d, rvec, tvec, K_, D_, proj_outer);

        std::vector<cv::Point3f> hole3d = { cv::Point3f(0.0f, 0.0f, (float)(-model_.length_L_mm)) };
        std::vector<cv::Point2f> proj_hole;
        cv::projectPoints(hole3d, rvec, tvec, K_, D_, proj_hole);
        const cv::Point2f proj_hole_c = proj_hole[0];

        const double ox = targetEllipse.center.x;
        const double oy = targetEllipse.center.y;
        const double ow = targetEllipse.size.width;
        const double oh = targetEllipse.size.height;
        const double ang = deg2rad(targetEllipse.angle);

        const double cos_a = std::cos(ang);
        const double sin_a = std::sin(ang);
        const double a = ow * 0.5;
        const double b = oh * 0.5;

        for (const auto& p2 : proj_outer) {
            const double dx = p2.x - ox;
            const double dy = p2.y - oy;
            const double lx =  dx * cos_a + dy * sin_a;
            const double ly = -dx * sin_a + dy * cos_a;
            const double err = (lx/a)*(lx/a) + (ly/b)*(ly/b) - 1.0;
            residual.push_back(err * 20.0);
        }

        const double dist = std::hypot(proj_hole_c.x - holeCenterPx.x, proj_hole_c.y - holeCenterPx.y);
        residual.push_back(dist * 50.0);
    }

    void lmOptimize(std::vector<double>& x,
                    const cv::RotatedRect& targetEllipse,
                    const cv::Point2f& holeCenterPx,
                    const std::vector<cv::Point3f>& pts3d,
                    std::optional<double> fixed_tz,
                    int max_iters) const
    {
        const int n = (int)x.size();
        double lambda = 1e-3;

        std::vector<double> r, r_new;
        computeResidual(x, targetEllipse, holeCenterPx, pts3d, fixed_tz, r);

        auto costOf = [](const std::vector<double>& rr){
            double c = 0.0;
            for (double v : rr) c += v*v;
            return 0.5 * c;
        };

        double cost = costOf(r);

        for (int it = 0; it < max_iters; ++it) {
            const int m = (int)r.size();
            cv::Mat J(m, n, CV_64F);
            cv::Mat rMat(m, 1, CV_64F);
            for (int i = 0; i < m; ++i) rMat.at<double>(i,0) = r[i];

            const double eps_base = 1e-4;
            for (int j = 0; j < n; ++j) {
                std::vector<double> x_eps = x;
                double eps = eps_base * (std::abs(x[j]) + 1.0);
                x_eps[j] += eps;

                computeResidual(x_eps, targetEllipse, holeCenterPx, pts3d, fixed_tz, r_new);
                for (int i = 0; i < m; ++i) {
                    J.at<double>(i,j) = (r_new[i] - r[i]) / eps;
                }
            }

            cv::Mat A = J.t() * J;
            cv::Mat g = J.t() * rMat;
            for (int d = 0; d < n; ++d) A.at<double>(d,d) += lambda;

            cv::Mat delta;
            if (!cv::solve(A, -g, delta, cv::DECOMP_SVD)) break;

            double step_norm = 0.0;
            std::vector<double> x_try = x;
            for (int j = 0; j < n; ++j) {
                const double dj = delta.at<double>(j,0);
                x_try[j] += dj;
                step_norm += dj*dj;
            }
            if (step_norm < 1e-12) break;

            computeResidual(x_try, targetEllipse, holeCenterPx, pts3d, fixed_tz, r_new);
            double cost_new = costOf(r_new);

            if (cost_new < cost) {
                x = std::move(x_try);
                r = std::move(r_new);
                cost = cost_new;
                lambda = std::max(lambda * 0.5, 1e-9);
            } else {
                lambda *= 2.0;
                if (lambda > 1e9) break;
            }
        }
    }
};
