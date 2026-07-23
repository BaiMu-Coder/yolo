#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

/**
 * @brief 单个检测目标的椭圆来源选择策略。
 *
 * Fit() 无论采用哪种模式，都会先准备一个检测框内切圆作为最终兜底。
 * 外圈/中圈生产路径固定使用 PreferMaskNoEdge，避免低质量图像中的纹理、
 * 反光和阴影边缘被误当成圆环边界。
 */
enum class EllipseFitMode
{
    PreferMask,       ///< Mask 不合格时允许尝试灰度 Edge，仍失败则返回 Box。
    PreferMaskNoEdge, ///< 只尝试 Mask；Mask 不合格时直接返回 Box。
    ForceBox          ///< 跳过所有拟合，强制返回检测框内切圆。
};

/// 最终椭圆实际来自哪条路径，用于可视化、记录和后续不确定度加权。
enum class EllipseSource
{
    Box,  ///< 检测框内切圆：最稳定的兜底，但几何精度和权重最低。
    Mask, ///< 分割 Mask 轮廓经过稳健拟合得到。
    Edge  ///< 灰度 Canny 边缘拟合；参考外/中圈默认不会使用。
};

const char *EllipseSourceName(EllipseSource source);

struct EllipseFitConfig
{
    // ---------- 候选几何门限 ----------
    float center_deviation_ratio = 0.30f; ///< 椭圆中心相对检测框中心的最大偏差/框短边。
    float maximum_axis_ratio = 6.0f;      ///< 最大长短轴比，防止极细退化椭圆。
    double minimum_contour_area_ratio = 0.005; ///< 有效轮廓面积至少占检测框面积的比例。
    float contour_center_distance_ratio = 0.80f; ///< 轮廓质心距离框中心的最大值/框短边。
    float minimum_angular_coverage_deg = 160.0f; ///< 内点至少覆盖的椭圆参数角，防止短弧冒充整圆。
    int minimum_occupied_quadrants = 3;          ///< 内点至少分布在几个椭圆象限中。

    // ---------- 出视场/边界截断专用门限 ----------
    int image_border_margin_px = 3; ///< 距原图边界不超过该距离的轮廓点视为裁剪产生的假边界。
    float partial_minimum_angular_coverage_deg = 90.0f; ///< 截断模式允许的最小可见弧覆盖角。
    int partial_minimum_occupied_quadrants = 2; ///< 截断模式最少覆盖象限数。
    float partial_minimum_candidate_quality = 0.28f; ///< 部分弧候选最低质量；位姿层还会继续降权。
    float partial_maximum_center_std_ratio = 0.25f; ///< 截断模式中心标准差/框短边上限。
    float partial_maximum_axis_std_ratio = 0.45f; ///< 截断模式轴长标准差/框短边上限。
    double partial_maximum_covariance_condition = 1e14; ///< 截断短弧允许更大的条件数。

    // ---------- PROSAC / LO-RANSAC ----------
    int ransac_iterations = 160;           ///< 5点直接椭圆假设的最大采样次数。
    int local_optimization_iterations = 2; ///< 用当前内点重新直接拟合的 LO 次数。
    float inlier_threshold_px = 3.0f;      ///< Sampson 像素误差不超过该值才算内点。
    float minimum_inlier_ratio = 0.42f;    ///< 最少内点数/参与拟合点数。
    int maximum_points = 180;              ///< 单个候选最多参与拟合的轮廓点，限制实时开销。
    uint32_t random_seed = 12345;           ///< 固定种子便于相同输入复现实验。

    // ---------- Sampson LM 精修及不确定度门控 ----------
    int refinement_iterations = 15;       ///< LM 最大迭代次数。
    float robust_delta_px = 2.5f;         ///< Huber 损失从二次段切换到线性段的像素阈值。
    float maximum_center_std_ratio = 0.10f; ///< 中心标准差最大值/检测框短边。
    float maximum_axis_std_ratio = 0.18f;   ///< 轴长标准差最大值/检测框短边。
    double maximum_covariance_condition = 1e12; ///< 法方程条件数上限，过大表示拟合退化。

    // ---------- 候选选择与可选 Edge 路径 ----------
    float minimum_candidate_quality = 0.52f;     ///< 低于该综合质量分时返回 Box。
    bool enable_edge_fallback = true;             ///< 仅对 PreferMask 模式生效。
    float edge_search_quality_threshold = 0.72f; ///< Mask 质量低于该值才计算 Edge 候选。
    double canny_low_threshold = 45.0;            ///< Canny 低阈值。
    double canny_high_threshold = 135.0;          ///< Canny 高阈值。
};

/**
 * @brief 一次椭圆拟合的完整结果。
 *
 * ellipse 使用 OpenCV RotatedRect 表示：center 为原图像素坐标，size 是完整轴长
 * （不是半轴），angle 单位为度。协方差参数顺序固定为
 * [center_x, center_y, log(semi_axis_a), log(semi_axis_b), angle_rad]。
 */
struct EllipseFitResult
{
    cv::RotatedRect ellipse; ///< 最终椭圆；即使回退 Box，也会包含可绘制的内切圆。
    bool valid = false;      ///< 检测框和最终椭圆尺寸是否合法。
    bool from_mask = false; // 兼容旧调用；新代码优先检查 source。
    EllipseSource source = EllipseSource::Box; ///< 实际来源，不能只用 valid 判断是否由 Mask 拟合。
    int inliers = 0;                           ///< Sampson 误差通过门限的采样点数。
    int sampled_points = 0;                    ///< 降采样后实际参加 RANSAC 的点数。
    float inlier_ratio = 0.0f;                 ///< inliers / sampled_points。
    float mean_error_px = std::numeric_limits<float>::quiet_NaN(); ///< 内点平均 Sampson 像素误差。
    float center_deviation_ratio = 0.0f; ///< 拟合中心到检测框中心的距离/框短边。
    float quality = 0.0f;               ///< [0,1] 综合质量分，供候选选择、时序滤波和位姿加权。
    bool geometry_consistent = true;    ///< 与同帧另一参考圆环是否通过物理几何一致性门控。
    bool temporally_filtered = false;   ///< 是否经过视频 EMA 或因低质量突变而保持上一帧。
    bool border_truncated = false; ///< 检测框或 Mask 是否接触原图边界。
    bool partial_visibility = false; ///< 是否使用“开放可见弧”专用门限得到。
    int removed_border_points = 0; ///< 从 Mask 轮廓中删除的图像边界假轮廓点数。
    float visible_arc_ratio = 0.0f; ///< angular_coverage_deg / 360。
    std::vector<cv::Point2f> visible_arc_points; ///< 原图坐标下的有效弧内点，供位姿层直接重投影。

    cv::Matx33d conic = cv::Matx33d::zeros(); ///< 归一化二次曲线矩阵 Q，使 [x y 1]Q[x y 1]^T=0。
    cv::Matx<double, 5, 5> covariance = cv::Matx<double, 5, 5>::zeros(); ///< 上述5参数的近似协方差。
    bool uncertainty_valid = false; ///< 协方差、标准差和条件数是否通过门控。
    float center_std_px = std::numeric_limits<float>::infinity();     ///< x/y 中较大的中心标准差，像素。
    float major_axis_std_px = std::numeric_limits<float>::infinity(); ///< 完整长轴长度的标准差，像素。
    float minor_axis_std_px = std::numeric_limits<float>::infinity(); ///< 完整短轴长度的标准差，像素。
    float angle_std_deg = std::numeric_limits<float>::infinity();     ///< 方向角标准差，度。
    float angular_coverage_deg = 0.0f; ///< 内点在椭圆参数角上的覆盖范围，按5度分箱统计。
    int occupied_quadrants = 0;        ///< 内点覆盖的椭圆象限数，范围0～4。
    double covariance_condition = std::numeric_limits<double>::infinity(); ///< LM 法方程条件数。
};

/// 检测置信度和椭圆质量的联合候选分数；双圆不一致时会附加惩罚。
float EllipseSelectionScore(const EllipseFitResult &ellipse, float detection_confidence);
/// 将拟合误差、协方差、质量分及双圆一致性折算成位姿优化使用的像素标准差。
double EllipseObservationSigmaPx(const EllipseFitResult &ellipse);
/// 将 RotatedRect 转为归一化齐次二次曲线矩阵。
cv::Matx33d EllipseConicMatrix(const cv::RotatedRect &ellipse);

// 统一候选器：支持 Mask、可选灰度边缘和检测框内切圆。
// 外/中圈生产路径使用 PreferMaskNoEdge：Mask 不合格时直接回退内切圆。
class EllipseFitter
{
public:
    explicit EllipseFitter(EllipseFitConfig config = {});

    /**
     * @brief 对一个检测框执行完整椭圆候选流程。
     * @param image 原图；输出椭圆中心始终位于该原图坐标系。
     * @param detection_box YOLO 检测框，允许部分越界，内部会与图像求交。
     * @param mask_data 原图大小的二值实例 Mask；可为空。
     * @param mode Mask/Edge/Box 的选择及回退策略。
     * @param mask_probability 原图大小的 uint8 概率图，0～255 对应概率0～1；
     *        可为空。存在时用于 p=0.5 等值线的亚像素修正及轮廓点加权。
     */
    EllipseFitResult Fit(const cv::Mat &image,
                         const cv::Rect &detection_box,
                         const uint8_t *mask_data,
                         EllipseFitMode mode = EllipseFitMode::PreferMask,
                         const uint8_t *mask_probability = nullptr) const;

    // 兼容没有原图的调用；该重载无法启用灰度边缘候选。
    EllipseFitResult Fit(cv::Size image_size,
                         const cv::Rect &detection_box,
                         const uint8_t *mask_data,
                         EllipseFitMode mode = EllipseFitMode::PreferMask) const;

    /// 生成检测框内切圆；该结果 uncertainty_valid=false，位姿层会自动降低权重。
    static EllipseFitResult BoxInscribedCircle(const cv::Rect &detection_box);

private:
    struct WeightedPoint
    {
        cv::Point2f point;   ///< 检测框 ROI 坐标系内的亚像素轮廓点。
        float weight = 1.0f; ///< 软 Mask 梯度产生的可靠度，清晰边界权重更大。
    };

    struct RansacResult
    {
        bool valid = false;
        cv::RotatedRect ellipse;
        int inliers = 0;
        int sampled_points = 0;
        float mean_error_px = std::numeric_limits<float>::infinity();
    };

    /// 一阶近似的带符号点到椭圆距离，单位近似为像素。
    static float SampsonResidualPx(const cv::RotatedRect &ellipse,
                                   const cv::Point2f &point);
    RansacResult FitRansac(const std::vector<WeightedPoint> &points) const;
    std::vector<WeightedPoint> CollectMaskPoints(const cv::Mat &binary_roi,
                                                 const cv::Mat &probability_roi,
                                                 const cv::Point2f &box_center_roi,
                                                 const cv::Rect &roi_in_image,
                                                 cv::Size image_size,
                                                 int *removed_border_points) const;
    std::vector<WeightedPoint> CollectEdgePoints(const cv::Mat &edge_roi,
                                                 const cv::Point2f &box_center_roi) const;
    EllipseFitResult BuildCandidate(const std::vector<WeightedPoint> &points,
                                    const cv::Rect &roi,
                                    const cv::Rect &detection_box,
                                    EllipseSource source,
                                    bool partial_visibility = false,
                                    int removed_border_points = 0) const;
    bool RefineSampson(const std::vector<WeightedPoint> &points,
                       cv::RotatedRect &ellipse,
                       cv::Matx<double, 5, 5> &covariance,
                       double &condition) const;
    void UpdateGeometryStatistics(const std::vector<WeightedPoint> &points,
                                  const cv::Rect &detection_box,
                                  EllipseFitResult &result,
                                  bool partial_visibility) const;
    static cv::Matx33d EllipseToConic(const cv::RotatedRect &ellipse);

    EllipseFitConfig config_;
};

struct RingConsistencyConfig
{
    float expected_middle_to_outer_ratio = 980.0f / 1200.0f; ///< 中圈/外圈的物理直径比。
    float diameter_ratio_tolerance = 0.22f; ///< 尺寸比的绝对容差。
    float maximum_center_offset_ratio = 0.14f; ///< 两投影中心最大距离/外圈长轴。
    float maximum_axis_ratio_difference = 0.28f; ///< 两椭圆长短轴比最大差值。
    float maximum_angle_difference_deg = 24.0f; ///< 两椭圆长轴方向最大差值，度。
    // 透视投影下同轴圆的图像椭圆中心不必相同，默认禁止图像平面圆心融合。
    float center_fusion_strength = 0.0f;
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

// 对同一帧外圈/中圈施加尺寸比、允许的中心偏移、轴比和方向一致性门控。
// 注意：这是候选筛选，不会强制修改两椭圆中心；真正的共轴约束在位姿投影模型中完成。
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
    float minimum_alpha = 0.18f; ///< 最低 EMA 当前帧权重。
    float maximum_alpha = 0.72f; ///< 最高 EMA 当前帧权重。
    float hard_jump_ratio = 0.45f; ///< 中心跳变量/上一帧长轴的硬门限。
    float hard_size_change_ratio = 0.55f; ///< 长轴相对变化量的硬门限。
    float rejection_quality = 0.62f; ///< 发生硬跳变且质量低于该值时保持上一帧。
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
