#include "ellipse_fitter.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <utility>

namespace
{
// -----------------------------------------------------------------------------
// RotatedRect 基础工具
// OpenCV 的 size.width/height 不保证分别是长轴/短轴，angle 也会随二者交换而变化，
// 所以后续几何比较统一通过这些函数取得“真正的长轴、短轴和长轴方向”。
// -----------------------------------------------------------------------------
float clamp01(float value)
{
    return std::clamp(value, 0.0f, 1.0f);
}

float major_axis(const cv::RotatedRect &ellipse)
{
    return std::max(ellipse.size.width, ellipse.size.height);
}

float minor_axis(const cv::RotatedRect &ellipse)
{
    return std::min(ellipse.size.width, ellipse.size.height);
}

float axis_ratio(const cv::RotatedRect &ellipse)
{
    return major_axis(ellipse) / std::max(1e-3f, minor_axis(ellipse));
}

float major_axis_angle(cv::RotatedRect ellipse)
{
    float angle = ellipse.angle;
    if (ellipse.size.width < ellipse.size.height)
        angle += 90.0f;
    while (angle >= 180.0f) angle -= 180.0f;
    while (angle < 0.0f) angle += 180.0f;
    return angle;
}

float angle_difference(float first, float second)
{
    float difference = std::abs(first - second);
    return std::min(difference, 180.0f - difference);
}

float blend_angle(float previous, float current, float alpha)
{
    // 椭圆方向以180度为周期，而普通角度平均以360度为周期。
    // 先把角度乘2映射到完整圆上做向量平均，再除2，可避免179度和1度被平均成90度。
    const float p = previous * 2.0f * static_cast<float>(CV_PI) / 180.0f;
    const float c = current * 2.0f * static_cast<float>(CV_PI) / 180.0f;
    const float x = (1.0f - alpha) * std::cos(p) + alpha * std::cos(c);
    const float y = (1.0f - alpha) * std::sin(p) + alpha * std::sin(c);
    float result = std::atan2(y, x) * 90.0f / static_cast<float>(CV_PI);
    if (result < 0.0f) result += 180.0f;
    return result;
}
}

const char *EllipseSourceName(EllipseSource source)
{
    switch (source)
    {
    case EllipseSource::Mask: return "mask";
    case EllipseSource::Edge: return "edge";
    default: return "box";
    }
}

float EllipseSelectionScore(const EllipseFitResult &ellipse, float detection_confidence)
{
    // 检测置信度回答“目标是不是该类别”，ellipse.quality 回答“边界几何是否可信”。
    // 两者都保留，避免仅凭高置信检测框选中一个严重退化的 Mask 椭圆。
    const float geometry_penalty = ellipse.geometry_consistent ? 1.0f : 0.72f;
    return 0.55f * clamp01(detection_confidence) +
           0.45f * clamp01(ellipse.quality) * geometry_penalty;
}

double EllipseObservationSigmaPx(const EllipseFitResult &ellipse)
{
    // Box 没有真实轮廓残差和可靠协方差，因此给它与目标尺寸相关的较大 sigma。
    // 位姿 LM 中残差会除以 sigma：sigma 越大，该观测对最终位姿影响越小。
    if (ellipse.source == EllipseSource::Box || !ellipse.uncertainty_valid)
        return std::max(8.0, 0.12 * major_axis(ellipse.ellipse));
    const double residual = std::isfinite(ellipse.mean_error_px) ? ellipse.mean_error_px : 3.0;
    double sigma = std::max(0.5, std::hypot(residual, 0.5 * ellipse.center_std_px));
    // 协方差衡量单个椭圆的局部稳定性；quality/双圈门控补充模型选错风险。
    sigma /= std::max(0.25, static_cast<double>(ellipse.quality));
    if (!ellipse.geometry_consistent) sigma *= 3.0;
    return sigma;
}

cv::Matx33d EllipseConicMatrix(const cv::RotatedRect &ellipse)
{
    // 椭圆局部坐标方程：
    //     u^2/a^2 + v^2/b^2 - 1 = 0
    // 通过旋转 R 和平移 center 展开为齐次形式：
    //     [x y 1] * Q * [x y 1]^T = 0
    // Q 最后做 Frobenius 范数归一化，消除二次曲线矩阵任意比例带来的数值差异。
    const double a = std::max(1e-6, ellipse.size.width * 0.5);
    const double b = std::max(1e-6, ellipse.size.height * 0.5);
    const double angle = ellipse.angle * CV_PI / 180.0;
    const double c = std::cos(angle), s = std::sin(angle);
    const cv::Matx22d rotation(c, -s, s, c);
    const cv::Matx22d diagonal(1.0 / (a * a), 0.0, 0.0, 1.0 / (b * b));
    const cv::Matx22d quadratic = rotation * diagonal * rotation.t();
    const cv::Vec2d center(ellipse.center.x, ellipse.center.y);
    const cv::Vec2d linear = -(quadratic * center);
    cv::Matx33d conic(quadratic(0, 0), quadratic(0, 1), linear[0],
                      quadratic(1, 0), quadratic(1, 1), linear[1],
                      linear[0], linear[1], center.dot(quadratic * center) - 1.0);
    double norm = 0.0;
    for (double value : conic.val) norm += value * value;
    return conic * (1.0 / std::max(1e-15, std::sqrt(norm)));
}

EllipseFitter::EllipseFitter(EllipseFitConfig config) : config_(std::move(config))
{
    // 在构造阶段把外部配置裁剪到合法范围，使核心循环不必反复防御无效参数。
    config_.ransac_iterations = std::max(1, config_.ransac_iterations);
    config_.local_optimization_iterations = std::max(0, config_.local_optimization_iterations);
    config_.refinement_iterations = std::max(1, config_.refinement_iterations);
    config_.maximum_points = std::max(20, config_.maximum_points);
    config_.inlier_threshold_px = std::max(0.1f, config_.inlier_threshold_px);
    config_.minimum_inlier_ratio = clamp01(config_.minimum_inlier_ratio);
    config_.maximum_axis_ratio = std::max(1.0f, config_.maximum_axis_ratio);
    config_.center_deviation_ratio = std::max(0.01f, config_.center_deviation_ratio);
    config_.minimum_candidate_quality = clamp01(config_.minimum_candidate_quality);
    config_.edge_search_quality_threshold = clamp01(config_.edge_search_quality_threshold);
    config_.minimum_angular_coverage_deg = std::clamp(config_.minimum_angular_coverage_deg, 0.0f, 360.0f);
    config_.minimum_occupied_quadrants = std::clamp(config_.minimum_occupied_quadrants, 1, 4);
}

EllipseFitResult EllipseFitter::BoxInscribedCircle(const cv::Rect &detection_box)
{
    // Box 兜底采用检测框短边作为直径，保证圆一定落在框内。
    // 它是一个“始终可用但精度较低”的观测：valid=true 便于后续绘制和解算，
    // uncertainty_valid=false 让位姿层自动使用更大的 sigma 降低其权重。
    EllipseFitResult result;
    const float side = static_cast<float>(std::max(0, std::min(detection_box.width,
                                                               detection_box.height)));
    const cv::Point2f center(detection_box.x + detection_box.width * 0.5f,
                             detection_box.y + detection_box.height * 0.5f);
    result.ellipse = cv::RotatedRect(center, cv::Size2f(side, side), 0.0f);
    result.valid = side > 0.0f;
    result.source = EllipseSource::Box;
    result.quality = result.valid ? 0.30f : 0.0f;
    result.conic = EllipseToConic(result.ellipse);
    const double center_sigma = std::max(2.0, side * 0.12);
    result.covariance(0, 0) = center_sigma * center_sigma;
    result.covariance(1, 1) = center_sigma * center_sigma;
    result.covariance(2, 2) = 0.20 * 0.20;
    result.covariance(3, 3) = 0.20 * 0.20;
    result.covariance(4, 4) = CV_PI * CV_PI / 4.0;
    result.center_std_px = center_sigma;
    result.major_axis_std_px = result.minor_axis_std_px = side * 0.20f;
    result.angle_std_deg = 90.0f;
    result.angular_coverage_deg = 0.0f;
    result.occupied_quadrants = 0;
    result.uncertainty_valid = false;
    return result;
}

float EllipseFitter::SampsonResidualPx(const cv::RotatedRect &ellipse,
                                       const cv::Point2f &point)
{
    // 将点变换到椭圆自身坐标系后，隐式方程为：
    //     F = x^2/a^2 + y^2/b^2 - 1
    // Sampson 距离使用 F / ||∇F||，是一阶点到曲线距离近似。
    // 相比直接比较 F，它已除去局部梯度尺度，结果可近似理解为带符号像素误差。
    const double a = ellipse.size.width * 0.5;
    const double b = ellipse.size.height * 0.5;
    if (a < 1e-3 || b < 1e-3) return std::numeric_limits<float>::infinity();
    const double angle = ellipse.angle * CV_PI / 180.0;
    const double cosine = std::cos(angle), sine = std::sin(angle);
    const double dx = point.x - ellipse.center.x, dy = point.y - ellipse.center.y;
    const double x = cosine * dx + sine * dy;
    const double y = -sine * dx + cosine * dy;
    const double fx = 2.0 * x / (a * a), fy = 2.0 * y / (b * b);
    const double gradient = std::hypot(cosine * fx - sine * fy,
                                       sine * fx + cosine * fy);
    if (gradient < 1e-9) return std::numeric_limits<float>::infinity();
    return static_cast<float>((x * x / (a * a) + y * y / (b * b) - 1.0) / gradient);
}

EllipseFitter::RansacResult EllipseFitter::FitRansac(
    const std::vector<WeightedPoint> &input_points) const
{
    // =========================================================================
    // 阶段1：PROSAC 式质量引导采样 + 5点直接椭圆初值
    // =========================================================================
    RansacResult best;
    const int count = static_cast<int>(input_points.size());
    if (count < 20) return best;
    std::vector<WeightedPoint> points = input_points;
    // 软 Mask 梯度越清晰，点权重越高。先排序后逐步扩大采样池：
    // 早期主要从高质量边界采样，后期再覆盖全部点，兼顾速度和全局搜索能力。
    std::stable_sort(points.begin(), points.end(), [](const WeightedPoint &a, const WeightedPoint &b) {
        return a.weight > b.weight;
    });
    std::mt19937 random(config_.random_seed);
    double best_weighted_inliers = -1.0;

    for (int iteration = 0; iteration < config_.ransac_iterations; ++iteration)
    {
        // pool_size 从20逐渐增长到全部点；这是 PROSAC 思想的轻量实现。
        const int pool_size = std::min(count, std::max(20,
            20 + iteration * std::max(0, count - 20) / std::max(1, config_.ransac_iterations - 1)));
        std::uniform_int_distribution<int> pick(0, pool_size - 1);
        std::vector<cv::Point2f> sample(5);
        std::array<int, 5> indices{};
        for (int sample_index = 0; sample_index < 5;)
        {
            const int point_index = pick(random);
            bool duplicate = false;
            for (int previous = 0; previous < sample_index; ++previous)
                duplicate = duplicate || indices[previous] == point_index;
            if (duplicate) continue;
            indices[sample_index] = point_index;
            sample[sample_index] = points[point_index].point;
            ++sample_index;
        }
        cv::RotatedRect candidate;
        // 一般椭圆有5个自由度，因此5个非退化点即可产生一个最小假设。
        // fitEllipseDirect 使用带椭圆约束的直接最小二乘，比普通 fitEllipse 初值更稳定。
        try { candidate = cv::fitEllipseDirect(sample); }
        catch (const cv::Exception &) { continue; }
        if (minor_axis(candidate) < 2.0f || axis_ratio(candidate) > config_.maximum_axis_ratio)
            continue;

        int inliers = 0;
        double weighted_inliers = 0.0;
        float error_sum = 0.0f;
        for (const WeightedPoint &point : points)
        {
            const float error = std::abs(SampsonResidualPx(candidate, point.point));
            if (error <= config_.inlier_threshold_px)
            {
                ++inliers;
                weighted_inliers += point.weight;
                error_sum += error;
            }
        }
        // 先最大化加权内点数；相同情况下选择平均误差更小的候选。
        // 这样反光或模糊边缘产生的低权重点不容易主导模型选择。
        const float mean_error = inliers > 0 ? error_sum / inliers
                                             : std::numeric_limits<float>::infinity();
        if (weighted_inliers > best_weighted_inliers ||
            (weighted_inliers == best_weighted_inliers && mean_error < best.mean_error_px))
        {
            best_weighted_inliers = weighted_inliers;
            best = {true, candidate, inliers, count, mean_error};
        }
    }

    const int required = std::max(20, static_cast<int>(std::ceil(config_.minimum_inlier_ratio * count)));
    if (!best.valid || best.inliers < required) return {};

    // =========================================================================
    // 阶段2：LO-RANSAC
    // 用当前最佳模型的全部内点重新拟合，再重新划分内点。它能明显减小仅用5点
    // 产生的随机偏差，同时保留 RANSAC 对离群点的鲁棒性。
    // =========================================================================
    for (int local = 0; local < config_.local_optimization_iterations; ++local)
    {
        std::vector<cv::Point2f> inliers;
        for (const WeightedPoint &point : points)
            if (std::abs(SampsonResidualPx(best.ellipse, point.point)) <= config_.inlier_threshold_px)
                inliers.push_back(point.point);
        if (inliers.size() < 5) return {};
        try { best.ellipse = cv::fitEllipseDirect(inliers); }
        catch (const cv::Exception &) { return {}; }
    }
    best.inliers = 0;
    float error_sum = 0.0f;
    for (const WeightedPoint &point : points)
    {
        const float error = std::abs(SampsonResidualPx(best.ellipse, point.point));
        if (error <= config_.inlier_threshold_px) { ++best.inliers; error_sum += error; }
    }
    if (best.inliers < required) return {};
    best.mean_error_px = error_sum / best.inliers;
    return best;
}

std::vector<EllipseFitter::WeightedPoint> EllipseFitter::CollectMaskPoints(
    const cv::Mat &binary_roi, const cv::Mat &probability_roi,
    const cv::Point2f &box_center_roi) const
{
    // =========================================================================
    // Mask 轮廓点生成
    // binary_roi 用于确定拓扑轮廓；probability_roi 用于恢复亚像素边界。
    // 两张图都处在检测框 ROI 坐标系，返回点也暂时保持 ROI 坐标。
    // =========================================================================
    cv::Mat work = binary_roi.clone();
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(work, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    std::vector<WeightedPoint> points;
    const double minimum_area = config_.minimum_contour_area_ratio * binary_roi.cols * binary_roi.rows;
    const float distance_limit = config_.contour_center_distance_ratio *
                                 std::min(binary_roi.cols, binary_roi.rows);
    for (const auto &contour : contours)
    {
        // 先删除面积太小的碎片，再删除质心离检测框中心过远的实例碎片。
        // 这一步主要抑制分割噪点、邻近目标和框边缘残留。
        if (contour.size() < 5 || std::abs(cv::contourArea(contour)) < minimum_area) continue;
        const cv::Moments moments = cv::moments(contour);
        if (std::abs(moments.m00) < 1e-6) continue;
        const cv::Point2f center(static_cast<float>(moments.m10 / moments.m00),
                                 static_cast<float>(moments.m01 / moments.m00));
        if (cv::norm(center - box_center_roi) > distance_limit) continue;
        for (const cv::Point &pixel : contour)
        {
            WeightedPoint point{cv::Point2f(pixel), 1.0f};
            if (!probability_roi.empty() && pixel.x > 0 && pixel.y > 0 &&
                pixel.x + 1 < probability_roi.cols && pixel.y + 1 < probability_roi.rows)
            {
                // 二值轮廓只能落在整数像素栅格上。利用局部一阶近似：
                //     p(x + Δn) ≈ p(x) + |∇p|Δ
                // 解 p=0.5 得 Δ=(0.5-p)/|∇p|，再沿梯度方向移动轮廓点。
                // 位移限制在 ±0.75 像素，防止低梯度区域产生不稳定的大步跳动。
                const float value = probability_roi.at<uint8_t>(pixel) / 255.0f;
                const float gx = (probability_roi.at<uint8_t>(pixel.y, pixel.x + 1) -
                                  probability_roi.at<uint8_t>(pixel.y, pixel.x - 1)) / 510.0f;
                const float gy = (probability_roi.at<uint8_t>(pixel.y + 1, pixel.x) -
                                  probability_roi.at<uint8_t>(pixel.y - 1, pixel.x)) / 510.0f;
                const float gradient = std::hypot(gx, gy);
                if (gradient > 1e-3f)
                {
                    const float shift = std::clamp((0.5f - value) / gradient, -0.75f, 0.75f);
                    point.point.x += shift * gx / gradient;
                    point.point.y += shift * gy / gradient;
                    // 梯度越大，Mask 边界越锐利；该权重同时用于 PROSAC 排序和 LM。
                    point.weight = std::clamp(gradient * 4.0f, 0.15f, 1.0f);
                }
            }
            points.push_back(point);
        }
    }
    return points;
}

std::vector<EllipseFitter::WeightedPoint> EllipseFitter::CollectEdgePoints(
    const cv::Mat &edge_roi, const cv::Point2f &box_center_roi) const
{
    // Edge 是通用实验路径，只在 PreferMask 且 Mask 质量不足时调用。
    // 外圈/中圈生产流程使用 PreferMaskNoEdge，不会进入这里。
    cv::Mat work = edge_roi.clone();
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(work, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    std::vector<WeightedPoint> points;
    const float distance_limit = config_.contour_center_distance_ratio *
                                 std::min(edge_roi.cols, edge_roi.rows);
    for (const auto &contour : contours)
    {
        if (contour.size() < 18) continue;
        const cv::Rect bounds = cv::boundingRect(contour);
        const cv::Point2f center(bounds.x + bounds.width * 0.5f, bounds.y + bounds.height * 0.5f);
        if (cv::norm(center - box_center_roi) <= distance_limit)
            for (const cv::Point &point : contour) points.push_back({cv::Point2f(point), 1.0f});
    }
    return points;
}

bool EllipseFitter::RefineSampson(const std::vector<WeightedPoint> &points,
                                  cv::RotatedRect &ellipse,
                                  cv::Matx<double, 5, 5> &covariance,
                                  double &condition) const
{
    // =========================================================================
    // 阶段3：鲁棒 Sampson-LM 精修
    //
    // 优化变量：[cx, cy, log(a), log(b), angle_rad]
    // 使用 log(a)、log(b) 而不是直接优化半轴，可天然保证轴长始终为正。
    // 每个点的残差为带符号 Sampson 像素距离，外层使用 Huber IRLS 降低残余
    // 离群点影响，内层使用 Levenberg-Marquardt 在高斯牛顿与梯度下降间切换。
    // =========================================================================
    if (points.size() < 6) return false;
    cv::Vec<double, 5> parameters(ellipse.center.x, ellipse.center.y,
                                  std::log(std::max(1.0f, ellipse.size.width * 0.5f)),
                                  std::log(std::max(1.0f, ellipse.size.height * 0.5f)),
                                  ellipse.angle * CV_PI / 180.0);
    auto make_ellipse = [](const cv::Vec<double, 5> &p) {
        return cv::RotatedRect(cv::Point2f(p[0], p[1]),
                               cv::Size2f(2.0f * std::exp(p[2]), 2.0f * std::exp(p[3])),
                               p[4] * 180.0 / CV_PI);
    };
    double lambda = 1e-3;
    cv::Mat final_normal;
    double final_sse = 0.0;
    int final_count = 0;
    for (int iteration = 0; iteration < config_.refinement_iterations; ++iteration)
    {
        const cv::RotatedRect current = make_ellipse(parameters);
        cv::Mat normal = cv::Mat::zeros(5, 5, CV_64F);
        cv::Mat gradient = cv::Mat::zeros(5, 1, CV_64F);
        double cost = 0.0, sse = 0.0;
        int used = 0;
        for (const WeightedPoint &point : points)
        {
            const double residual = SampsonResidualPx(current, point.point);
            if (!std::isfinite(residual)) continue;
            const double absolute = std::abs(residual);
            const double robust_weight = absolute <= config_.robust_delta_px
                                             ? 1.0 : config_.robust_delta_px / absolute;
            // 最终权重 = Mask 边界可靠度 × Huber 鲁棒权重。
            const double weight = std::max(0.01, static_cast<double>(point.weight)) * robust_weight;
            cv::Vec<double, 5> jacobian;
            // 椭圆残差的解析导数较繁琐，这里使用一侧数值差分。
            // 不同参数使用不同 epsilon，避免像素、对数轴长和弧度量纲差异造成精度损失。
            for (int parameter = 0; parameter < 5; ++parameter)
            {
                cv::Vec<double, 5> perturbed = parameters;
                const double epsilon = parameter < 2 ? 0.02 : (parameter < 4 ? 1e-4 : 1e-5);
                perturbed[parameter] += epsilon;
                jacobian[parameter] = (SampsonResidualPx(make_ellipse(perturbed), point.point) - residual) / epsilon;
            }
            for (int row = 0; row < 5; ++row)
            {
                gradient.at<double>(row) += weight * jacobian[row] * residual;
                for (int col = 0; col < 5; ++col)
                    normal.at<double>(row, col) += weight * jacobian[row] * jacobian[col];
            }
            cost += point.weight * (absolute <= config_.robust_delta_px
                        ? 0.5 * residual * residual
                        : config_.robust_delta_px * (absolute - 0.5 * config_.robust_delta_px));
            sse += weight * residual * residual;
            ++used;
        }
        if (used < 6) return false;
        // 即使初值已在极小值附近、后续没有任何一步被接受，
        // 当前法方程仍然是有效的协方差估计，不应把拟合误判为失败。
        final_normal = normal.clone();
        final_sse = sse;
        final_count = used;
        cv::Mat damped = normal.clone();
        // LM 对角阻尼。lambda 较大时步长更保守，较小时接近高斯牛顿。
        for (int diagonal = 0; diagonal < 5; ++diagonal)
            damped.at<double>(diagonal, diagonal) += lambda *
                std::max(1.0, normal.at<double>(diagonal, diagonal));
        cv::Mat delta;
        if (!cv::solve(damped, -gradient, delta, cv::DECOMP_SVD)) return false;
        cv::Vec<double, 5> trial = parameters;
        double step = 0.0;
        for (int i = 0; i < 5; ++i) { trial[i] += delta.at<double>(i); step += delta.at<double>(i) * delta.at<double>(i); }
        if (std::exp(trial[2]) < 1.0 || std::exp(trial[3]) < 1.0) { lambda *= 5.0; continue; }

        double trial_cost = 0.0;
        const cv::RotatedRect trial_ellipse = make_ellipse(trial);
        for (const WeightedPoint &point : points)
        {
            const double r = std::abs(SampsonResidualPx(trial_ellipse, point.point));
            if (!std::isfinite(r)) continue;
            trial_cost += point.weight * (r <= config_.robust_delta_px
                              ? 0.5 * r * r
                              : config_.robust_delta_px * (r - 0.5 * config_.robust_delta_px));
        }
        if (trial_cost < cost)
        {
            // 新模型降低了鲁棒目标：接受更新并减小阻尼。
            parameters = trial;
            lambda = std::max(1e-9, lambda * 0.4);
            if (step < 1e-10) break;
        }
        else
        {
            // 新模型更差：拒绝本次更新并增大阻尼，下轮尝试更小的步长。
            lambda = std::min(1e9, lambda * 5.0);
        }
    }
    ellipse = make_ellipse(parameters);
    if (final_normal.empty()) return false;
    cv::SVD svd(final_normal, cv::SVD::NO_UV);
    // 条件数 = 最大奇异值/最小奇异值。值越大，说明现有轮廓不能稳定约束某些
    // 参数，例如只看到很短的一段弧时，椭圆中心和轴长会存在多组近似解。
    const double largest = svd.w.at<double>(0);
    const double smallest = svd.w.at<double>(svd.w.rows - 1);
    condition = largest / std::max(1e-15, smallest);
    cv::Mat inverse;
    if (!cv::invert(final_normal, inverse, cv::DECOMP_SVD)) return false;
    // 局部线性化下 Cov ≈ (JᵀWJ)^-1 * 残差方差。
    // 该协方差不是绝对真值，但足以用于退化门控和后续位姿观测相对加权。
    inverse *= final_sse / std::max(1, final_count - 5);
    for (int row = 0; row < 5; ++row)
        for (int col = 0; col < 5; ++col)
            covariance(row, col) = inverse.at<double>(row, col);
    return std::isfinite(condition) && cv::checkRange(inverse);
}

cv::Matx33d EllipseFitter::EllipseToConic(const cv::RotatedRect &ellipse)
{
    return EllipseConicMatrix(ellipse);
}

void EllipseFitter::UpdateGeometryStatistics(const std::vector<WeightedPoint> &points,
                                             const cv::Rect &detection_box,
                                             EllipseFitResult &result) const
{
    // =========================================================================
    // 阶段4：覆盖度与不确定度统计
    // 把内点映射到标准单位圆参数角，使用72个5度小格及4个象限统计覆盖范围。
    // 仅有一小段连续弧时，即使残差很小也不能可靠确定完整椭圆，因此必须单独门控。
    // =========================================================================
    std::array<bool, 72> bins{};
    std::array<bool, 4> quadrants{};
    const double angle = result.ellipse.angle * CV_PI / 180.0;
    const double c = std::cos(angle), s = std::sin(angle);
    const double a = result.ellipse.size.width * 0.5, b = result.ellipse.size.height * 0.5;
    for (const WeightedPoint &weighted : points)
    {
        if (std::abs(SampsonResidualPx(result.ellipse, weighted.point)) > config_.inlier_threshold_px) continue;
        const double dx = weighted.point.x - result.ellipse.center.x;
        const double dy = weighted.point.y - result.ellipse.center.y;
        const double x = (c * dx + s * dy) / std::max(1e-6, a);
        const double y = (-s * dx + c * dy) / std::max(1e-6, b);
        double theta = std::atan2(y, x);
        if (theta < 0.0) theta += 2.0 * CV_PI;
        bins[std::min(71, static_cast<int>(theta * 72.0 / (2.0 * CV_PI)))] = true;
        quadrants[std::min(3, static_cast<int>(theta * 4.0 / (2.0 * CV_PI)))] = true;
    }
    result.angular_coverage_deg = 5.0f * std::count(bins.begin(), bins.end(), true);
    result.occupied_quadrants = std::count(quadrants.begin(), quadrants.end(), true);
    result.conic = EllipseToConic(result.ellipse);
    // 将优化参数协方差转换为更直观的像素/角度标准差。
    // 对 log(a) 有 d(2a)/d(log(a))=2a，所以完整轴长标准差约为轴长×log轴标准差。
    const double center_variance = std::max(result.covariance(0, 0), result.covariance(1, 1));
    result.center_std_px = std::sqrt(std::max(0.0, center_variance));
    const double width_std = result.ellipse.size.width * std::sqrt(std::max(0.0, result.covariance(2, 2)));
    const double height_std = result.ellipse.size.height * std::sqrt(std::max(0.0, result.covariance(3, 3)));
    result.major_axis_std_px = std::max(width_std, height_std);
    result.minor_axis_std_px = std::min(width_std, height_std);
    result.angle_std_deg = std::sqrt(std::max(0.0, result.covariance(4, 4))) * 180.0 / CV_PI;
    const float short_side = std::max(1.0f, static_cast<float>(std::min(detection_box.width, detection_box.height)));
    result.uncertainty_valid = std::isfinite(result.center_std_px) &&
        result.center_std_px <= config_.maximum_center_std_ratio * short_side &&
        result.major_axis_std_px <= config_.maximum_axis_std_ratio * short_side &&
        result.covariance_condition <= config_.maximum_covariance_condition;
}

EllipseFitResult EllipseFitter::BuildCandidate(const std::vector<WeightedPoint> &points,
                                                const cv::Rect &roi,
                                                const cv::Rect &detection_box,
                                                EllipseSource source) const
{
    // =========================================================================
    // 阶段5：从轮廓点构造并验收一个完整候选
    // 数据流：
    // 点集降采样 -> PROSAC/LO-RANSAC -> 取内点 -> Sampson LM ->
    // ROI坐标转原图坐标 -> 覆盖度/协方差统计 -> 硬门控 -> 综合质量评分。
    // 任一关键阶段失败都返回 valid=false，由最外层 Fit() 决定继续尝试 Edge 或回退 Box。
    // =========================================================================
    if (points.size() < 20) return {};
    std::vector<WeightedPoint> sampled;
    sampled.reserve(std::min(static_cast<int>(points.size()), config_.maximum_points));
    // 轮廓点通常沿边界有序，等步长抽样可保留空间覆盖，同时限制 RANSAC/LM 开销。
    const int step = std::max(1, static_cast<int>(points.size()) / config_.maximum_points);
    for (int index = 0; index < static_cast<int>(points.size()) &&
                        static_cast<int>(sampled.size()) < config_.maximum_points; index += step)
        sampled.push_back(points[index]);
    const RansacResult fit = FitRansac(sampled);
    if (!fit.valid) return {};

    std::vector<WeightedPoint> inliers;
    // RANSAC 负责从全部点中找到正确吸引域；只把其内点交给更精确但更局部的 LM。
    for (const WeightedPoint &point : sampled)
        if (std::abs(SampsonResidualPx(fit.ellipse, point.point)) <= config_.inlier_threshold_px)
            inliers.push_back(point);
    cv::RotatedRect refined = fit.ellipse;
    cv::Matx<double, 5, 5> covariance = cv::Matx<double, 5, 5>::zeros();
    double condition = std::numeric_limits<double>::infinity();
    if (!RefineSampson(inliers, refined, covariance, condition)) return {};

    EllipseFitResult result;
    // CollectMaskPoints/CollectEdgePoints 返回 ROI 局部坐标；对外结果必须统一回到原图坐标。
    result.ellipse = refined;
    result.ellipse.center += cv::Point2f(roi.x, roi.y);
    result.valid = true;
    result.source = source;
    result.from_mask = source == EllipseSource::Mask;
    result.covariance = covariance;
    result.covariance_condition = condition;
    result.sampled_points = sampled.size();
    std::vector<WeightedPoint> global_points = sampled;
    float error_sum = 0.0f;
    for (WeightedPoint &point : global_points)
    {
        point.point += cv::Point2f(roi.x, roi.y);
        const float error = std::abs(SampsonResidualPx(result.ellipse, point.point));
        if (error <= config_.inlier_threshold_px) { ++result.inliers; error_sum += error; }
    }
    result.inlier_ratio = result.sampled_points > 0
                              ? static_cast<float>(result.inliers) / result.sampled_points : 0.0f;
    result.mean_error_px = result.inliers > 0 ? error_sum / result.inliers
                                               : std::numeric_limits<float>::infinity();
    UpdateGeometryStatistics(global_points, detection_box, result);

    const cv::Point2f box_center(detection_box.x + detection_box.width * 0.5f,
                                 detection_box.y + detection_box.height * 0.5f);
    const float short_side = std::max(1.0f, static_cast<float>(std::min(detection_box.width, detection_box.height)));
    result.center_deviation_ratio = cv::norm(result.ellipse.center - box_center) / short_side;
    // 硬门控用于拒绝“数值上能拟合、物理上不可信”的结果：
    // - 中心不能离检测框太远；
    // - 椭圆不能太小或大幅超出检测框；
    // - 内点必须覆盖足够角度和象限；
    // - 参数协方差必须可观测且没有明显退化。
    if (result.center_deviation_ratio > config_.center_deviation_ratio ||
        minor_axis(result.ellipse) < 0.20f * short_side ||
        major_axis(result.ellipse) > 1.45f * std::max(detection_box.width, detection_box.height) ||
        result.angular_coverage_deg < config_.minimum_angular_coverage_deg ||
        result.occupied_quadrants < config_.minimum_occupied_quadrants || !result.uncertainty_valid)
        return {};

    // 通过硬门控后再计算软质量分。quality 不直接决定几何参数，而用于：
    // 候选/双环选择、视频 EMA 当前帧权重，以及位姿联合优化的观测 sigma。
    const float error_quality = std::exp(-result.mean_error_px / std::max(0.5f, config_.inlier_threshold_px));
    const float center_quality = clamp01(1.0f - result.center_deviation_ratio / config_.center_deviation_ratio);
    const float coverage_quality = clamp01(result.angular_coverage_deg / 300.0f);
    const float uncertainty_quality = clamp01(1.0f - result.center_std_px /
        std::max(1.0f, config_.maximum_center_std_ratio * short_side));
    const float source_bonus = source == EllipseSource::Mask ? 0.04f : 0.0f;
    result.quality = clamp01(0.32f * result.inlier_ratio + 0.22f * error_quality +
                             0.16f * center_quality + 0.16f * coverage_quality +
                             0.14f * uncertainty_quality + source_bonus);
    return result;
}

EllipseFitResult EllipseFitter::Fit(const cv::Mat &image,
                                    const cv::Rect &detection_box,
                                    const uint8_t *mask_data,
                                    EllipseFitMode mode,
                                    const uint8_t *mask_probability) const
{
    // =========================================================================
    // 椭圆拟合总入口
    //
    //   先生成 Box 内切圆兜底
    //          |
    //          +-- ForceBox ------------------------------> 返回 Box
    //          |
    //          +-- 有 Mask -> 亚像素轮廓 -> 稳健拟合/门控 --+
    //          |                                           |
    //          +-- PreferMask 且 Mask 较差 -> 可选 Edge ----+--> 最优候选
    //                                                      |
    //                              候选无效或质量不够 ------+--> 返回 Box
    //
    // PreferMaskNoEdge 不会执行 Canny，这正是外圈/中圈的默认生产策略。
    // =========================================================================
    const cv::Size image_size = image.empty() ? cv::Size() : image.size();
    // 提前准备 fallback，保证后面任何错误分支都有一致、可绘制的返回值。
    EllipseFitResult fallback = BoxInscribedCircle(detection_box);
    if (mode == EllipseFitMode::ForceBox || image_size.width <= 0 || image_size.height <= 0)
        return fallback;
    // 检测框可能因为反投影或取整略微越界，先与原图求交，防止构造 Mat ROI 异常。
    const cv::Rect roi = detection_box & cv::Rect(0, 0, image.cols, image.rows);
    if (roi.width <= 0 || roi.height <= 0) return fallback;
    const cv::Point2f box_center_roi(detection_box.x + detection_box.width * 0.5f - roi.x,
                                     detection_box.y + detection_box.height * 0.5f - roi.y);

    EllipseFitResult best;
    if (mask_data != nullptr)
    {
        // mask_data/mask_probability 都是“原图宽×原图高”的连续单通道数据；
        // Mat 这里只包装外部内存，不复制，再截取与检测框对应的 ROI。
        cv::Mat full_mask(image.size(), CV_8UC1, const_cast<uint8_t *>(mask_data));
        cv::Mat probability_roi;
        if (mask_probability != nullptr)
        {
            cv::Mat full_probability(image.size(), CV_8UC1,
                                     const_cast<uint8_t *>(mask_probability));
            probability_roi = full_probability(roi);
        }
        best = BuildCandidate(CollectMaskPoints(full_mask(roi), probability_roi, box_center_roi),
                              roi, detection_box, EllipseSource::Mask);
    }

    // Mask 已经足够稳定时跳过 Canny，避免在实时路径上重复消耗 CPU。
    if (mode == EllipseFitMode::PreferMask && config_.enable_edge_fallback &&
        (!best.valid || best.quality < config_.edge_search_quality_threshold))
    {
        cv::Mat gray;
        if (image.channels() == 1) gray = image(roi).clone();
        else cv::cvtColor(image(roi), gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.2);
        cv::Mat equalized;
        cv::equalizeHist(gray, equalized);
        cv::Mat edges;
        cv::Canny(equalized, edges, config_.canny_low_threshold,
                  config_.canny_high_threshold, 3, true);
        const int border = std::min(3, std::min(edges.cols, edges.rows) / 4);
        if (border > 0)
        {
            edges.rowRange(0, border).setTo(0);
            edges.rowRange(edges.rows - border, edges.rows).setTo(0);
            edges.colRange(0, border).setTo(0);
            edges.colRange(edges.cols - border, edges.cols).setTo(0);
        }
        EllipseFitResult edge = BuildCandidate(CollectEdgePoints(edges, box_center_roi),
                                               roi, detection_box, EllipseSource::Edge);
        if (edge.valid && (!best.valid || edge.quality > best.quality + 0.03f))
            best = edge;
    }

    // 最终仅接受“通过所有硬门控且综合质量达标”的候选，否则统一使用内切圆。
    return best.valid && best.quality >= config_.minimum_candidate_quality ? best : fallback;
}

EllipseFitResult EllipseFitter::Fit(cv::Size image_size,
                                    const cv::Rect &detection_box,
                                    const uint8_t *mask_data,
                                    EllipseFitMode mode) const
{
    if (image_size.width <= 0 || image_size.height <= 0 || mode == EllipseFitMode::ForceBox)
        return BoxInscribedCircle(detection_box);
    cv::Mat placeholder(image_size, CV_8UC1, cv::Scalar(0));
    EllipseFitConfig no_edge_config = config_;
    no_edge_config.enable_edge_fallback = false;
    return EllipseFitter(no_edge_config).Fit(placeholder, detection_box, mask_data, mode, nullptr);
}

RingPairRefiner::RingPairRefiner(RingConsistencyConfig config) : config_(std::move(config)) {}

RingConsistencyResult RingPairRefiner::Refine(EllipseFitResult &outer,
                                               EllipseFitResult &middle) const
{
    // =========================================================================
    // 同帧双圆环物理一致性门控
    //
    // 外圈和中圈是空间中共轴的两个物理圆，但透视投影后两个图像椭圆的中心
    // 一般不严格重合。因此这里检查“允许范围内的中心偏移”，不修改拟合中心；
    // 真正的共轴/共享法向约束由 PoseEstimatorLM 的联合重投影模型负责。
    // =========================================================================
    RingConsistencyResult result;
    if (!outer.valid || !middle.valid) return result;
    result.evaluated = true;
    const float outer_major = major_axis(outer.ellipse);
    const float middle_major = major_axis(middle.ellipse);
    if (outer_major < 1.0f || middle_major < 1.0f) return result;
    const float size_ratio = middle_major / outer_major;
    // 四项误差都除以各自容差，归一化后 <=1 表示该项通过。
    const float size_error = std::abs(size_ratio - config_.expected_middle_to_outer_ratio) /
                             std::max(0.01f, config_.diameter_ratio_tolerance);
    const float center_error = cv::norm(outer.ellipse.center - middle.ellipse.center) /
                               std::max(1.0f, outer_major * config_.maximum_center_offset_ratio);
    const float axes_error = std::abs(axis_ratio(outer.ellipse) - axis_ratio(middle.ellipse)) /
                             std::max(0.01f, config_.maximum_axis_ratio_difference);
    const float angle_error = angle_difference(major_axis_angle(outer.ellipse),
                                               major_axis_angle(middle.ellipse)) /
                              std::max(1.0f, config_.maximum_angle_difference_deg);
    result.score = clamp01(1.0f - (0.34f * size_error + 0.34f * center_error +
                                  0.18f * axes_error + 0.14f * angle_error));
    result.consistent = size_error <= 1.0f && center_error <= 1.0f &&
                        axes_error <= 1.0f && angle_error <= 1.0f;
    outer.geometry_consistent = result.consistent;
    middle.geometry_consistent = result.consistent;
    if (!result.consistent)
    {
        // 不直接删除不一致候选：保留高质量的一圈作为单圈解算机会，同时显著降低
        // 另一圈 quality。EllipseObservationSigmaPx 会把低质量进一步变成更大的 sigma。
        if (outer.quality >= middle.quality) middle.quality *= 0.35f;
        else outer.quality *= 0.35f;
        return result;
    }

    // 不在图像平面强制同心；真正的共轴约束应在相机投影模型中施加。
    outer.quality = clamp01(outer.quality + 0.12f * result.score);
    middle.quality = clamp01(middle.quality + 0.12f * result.score);
    return result;
}

RingPairSelection RingPairRefiner::SelectAndRefine(
    const std::vector<int> &class_ids,
    const std::vector<float> &detection_confidences,
    std::vector<EllipseFitResult> &ellipses,
    int outer_class_id,
    int middle_class_id) const
{
    // 可能存在同类别多个检测框。先记录每个类别各自最佳项，再遍历所有外圈×中圈
    // 组合，用“两个单体分数 + 双环一致性分数”选择物理上最合理的一对。
    RingPairSelection selection;
    const size_t count = std::min({class_ids.size(), detection_confidences.size(), ellipses.size()});
    float best_outer_score = -1.0f;
    float best_middle_score = -1.0f;
    for (size_t i = 0; i < count; ++i)
    {
        const float score = EllipseSelectionScore(ellipses[i], detection_confidences[i]);
        if (class_ids[i] == outer_class_id && score > best_outer_score)
        {
            selection.outer_index = static_cast<int>(i);
            best_outer_score = score;
        }
        if (class_ids[i] == middle_class_id && score > best_middle_score)
        {
            selection.middle_index = static_cast<int>(i);
            best_middle_score = score;
        }
    }

    float best_pair_score = -std::numeric_limits<float>::infinity();
    int best_outer = -1;
    int best_middle = -1;
    for (size_t outer = 0; outer < count; ++outer)
    {
        if (class_ids[outer] != outer_class_id) continue;
        for (size_t middle = 0; middle < count; ++middle)
        {
            if (class_ids[middle] != middle_class_id) continue;
            EllipseFitResult outer_copy = ellipses[outer];
            EllipseFitResult middle_copy = ellipses[middle];
            // 在副本上评估，防止遍历某个失败组合时提前污染真实候选 quality。
            const RingConsistencyResult consistency = Refine(outer_copy, middle_copy);
            const float geometry_term = consistency.consistent
                                            ? 0.45f * consistency.score : -0.35f;
            const float score = EllipseSelectionScore(outer_copy, detection_confidences[outer]) +
                                EllipseSelectionScore(middle_copy, detection_confidences[middle]) +
                                geometry_term;
            if (score > best_pair_score)
            {
                best_pair_score = score;
                best_outer = static_cast<int>(outer);
                best_middle = static_cast<int>(middle);
            }
        }
    }
    if (best_outer >= 0 && best_middle >= 0)
    {
        // 只对最终胜出的真实候选写入 geometry_consistent 和质量修正。
        selection.outer_index = best_outer;
        selection.middle_index = best_middle;
        selection.consistency = Refine(ellipses[best_outer], ellipses[best_middle]);
    }
    return selection;
}

EllipseTemporalFilter::EllipseTemporalFilter(EllipseTemporalConfig config)
    : config_(std::move(config)) {}

EllipseFitResult EllipseTemporalFilter::Update(int class_id,
                                               const EllipseFitResult &measurement)
{
    // 每个类别维护独立状态，避免外圈、中圈、内孔之间互相串扰。
    if (class_id < 0 || class_id >= static_cast<int>(states_.size()) || !measurement.valid)
        return measurement;
    State &state = states_[class_id];
    if (!state.initialized)
    {
        state.initialized = true;
        state.value = measurement;
        return measurement;
    }

    const float previous_major = std::max(1.0f, major_axis(state.value.ellipse));
    const float jump = cv::norm(measurement.ellipse.center - state.value.ellipse.center) /
                       previous_major;
    const float size_change = std::abs(major_axis(measurement.ellipse) - previous_major) /
                              previous_major;
    if ((jump > config_.hard_jump_ratio || size_change > config_.hard_size_change_ratio) &&
        measurement.quality < config_.rejection_quality)
    {
        // 当前帧同时满足“变化很大”和“质量较低”时，认为更可能是误检。
        // 保持上一帧但略微降低质量；若后续连续出现稳定高质量新位置，仍可重新跟随。
        EllipseFitResult held = state.value;
        held.quality *= 0.92f;
        held.temporally_filtered = true;
        state.value = held;
        return held;
    }

    const float alpha = config_.minimum_alpha +
                        (config_.maximum_alpha - config_.minimum_alpha) * clamp01(measurement.quality);
    // 质量越高 alpha 越大，越相信当前测量；质量越低越依赖历史值。
    EllipseFitResult filtered = measurement;
    filtered.ellipse.center = state.value.ellipse.center * (1.0f - alpha) +
                              measurement.ellipse.center * alpha;
    filtered.ellipse.size.width = state.value.ellipse.size.width * (1.0f - alpha) +
                                  measurement.ellipse.size.width * alpha;
    filtered.ellipse.size.height = state.value.ellipse.size.height * (1.0f - alpha) +
                                   measurement.ellipse.size.height * alpha;
    // 角度不能直接线性插值，例如179度和1度实际只相差2度，需按180度周期融合。
    filtered.ellipse.angle = blend_angle(state.value.ellipse.angle,
                                         measurement.ellipse.angle, alpha);
    filtered.conic = EllipseConicMatrix(filtered.ellipse);
    filtered.temporally_filtered = true;
    state.value = filtered;
    return filtered;
}

void EllipseTemporalFilter::Reset()
{
    states_ = {};
}
