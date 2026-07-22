#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

enum class EllipseFitMode
{
    PreferMask,
    PreferMaskNoEdge,
    ForceBox
};

enum class EllipseSource
{
    Box,
    Mask,
    Edge
};

const char *EllipseSourceName(EllipseSource source);

struct EllipseFitConfig
{
    float center_deviation_ratio = 0.30f;
    int ransac_iterations = 160;
    float inlier_threshold_px = 3.0f;
    float minimum_inlier_ratio = 0.42f;
    float maximum_axis_ratio = 6.0f;
    int maximum_points = 180;
    double minimum_contour_area_ratio = 0.005;
    float contour_center_distance_ratio = 0.80f;
    float minimum_candidate_quality = 0.52f;
    bool enable_edge_fallback = true;
    float edge_search_quality_threshold = 0.72f;
    double canny_low_threshold = 45.0;
    double canny_high_threshold = 135.0;
    uint32_t random_seed = 12345;
};

struct EllipseFitResult
{
    cv::RotatedRect ellipse;
    bool valid = false;
    bool from_mask = false; // 兼容旧调用；新代码优先检查 source。
    EllipseSource source = EllipseSource::Box;
    int inliers = 0;
    int sampled_points = 0;
    float inlier_ratio = 0.0f;
    float mean_error_px = std::numeric_limits<float>::quiet_NaN();
    float center_deviation_ratio = 0.0f;
    float quality = 0.0f;
    bool geometry_consistent = true;
    bool temporally_filtered = false;
};

float EllipseSelectionScore(const EllipseFitResult &ellipse, float detection_confidence);

// 统一候选器：支持 Mask、可选灰度边缘和检测框内切圆。
// 外/中圈生产路径使用 PreferMaskNoEdge：Mask 不合格时直接回退内切圆。
class EllipseFitter
{
public:
    explicit EllipseFitter(EllipseFitConfig config = {});

    EllipseFitResult Fit(const cv::Mat &image,
                         const cv::Rect &detection_box,
                         const uint8_t *mask_data,
                         EllipseFitMode mode = EllipseFitMode::PreferMask) const;

    // 兼容没有原图的调用；该重载无法启用灰度边缘候选。
    EllipseFitResult Fit(cv::Size image_size,
                         const cv::Rect &detection_box,
                         const uint8_t *mask_data,
                         EllipseFitMode mode = EllipseFitMode::PreferMask) const;

    static EllipseFitResult BoxInscribedCircle(const cv::Rect &detection_box);

private:
    struct RansacResult
    {
        bool valid = false;
        cv::RotatedRect ellipse;
        int inliers = 0;
        int sampled_points = 0;
        float mean_error_px = std::numeric_limits<float>::infinity();
    };

    static float RadialErrorPx(const cv::RotatedRect &ellipse,
                               const cv::Point2f &point);
    RansacResult FitRansac(const std::vector<cv::Point> &points) const;
    std::vector<cv::Point> CollectMaskPoints(const cv::Mat &binary_roi,
                                             const cv::Point2f &box_center_roi) const;
    std::vector<cv::Point> CollectEdgePoints(const cv::Mat &edge_roi,
                                             const cv::Point2f &box_center_roi) const;
    EllipseFitResult BuildCandidate(const std::vector<cv::Point> &points,
                                    const cv::Rect &roi,
                                    const cv::Rect &detection_box,
                                    EllipseSource source) const;

    EllipseFitConfig config_;
};

struct RingConsistencyConfig
{
    float expected_middle_to_outer_ratio = 980.0f / 1200.0f;
    float diameter_ratio_tolerance = 0.22f;
    float maximum_center_offset_ratio = 0.14f;
    float maximum_axis_ratio_difference = 0.28f;
    float maximum_angle_difference_deg = 24.0f;
    float center_fusion_strength = 0.35f;
};

struct RingConsistencyResult
{
    bool evaluated = false;
    bool consistent = true;
    float score = 1.0f;
};

struct RingPairSelection
{
    int outer_index = -1;
    int middle_index = -1;
    RingConsistencyResult consistency;
};

// 对同一帧外圈/中圈施加同心、尺度、轴比和方向约束。
class RingPairRefiner
{
public:
    explicit RingPairRefiner(RingConsistencyConfig config = {});
    RingConsistencyResult Refine(EllipseFitResult &outer,
                                 EllipseFitResult &middle) const;
    RingPairSelection SelectAndRefine(const std::vector<int> &class_ids,
                                      const std::vector<float> &detection_confidences,
                                      std::vector<EllipseFitResult> &ellipses,
                                      int outer_class_id = 0,
                                      int middle_class_id = 1) const;

private:
    RingConsistencyConfig config_;
};

struct EllipseTemporalConfig
{
    float minimum_alpha = 0.18f;
    float maximum_alpha = 0.72f;
    float hard_jump_ratio = 0.45f;
    float hard_size_change_ratio = 0.55f;
    float rejection_quality = 0.62f;
};

// 面向视频的自适应 EMA：质量越高越信任当前帧；低质量突变保持上一帧。
class EllipseTemporalFilter
{
public:
    explicit EllipseTemporalFilter(EllipseTemporalConfig config = {});
    EllipseFitResult Update(int class_id, const EllipseFitResult &measurement);
    void Reset();

private:
    struct State
    {
        bool initialized = false;
        EllipseFitResult value;
    };

    EllipseTemporalConfig config_;
    std::array<State, 3> states_{};
};
