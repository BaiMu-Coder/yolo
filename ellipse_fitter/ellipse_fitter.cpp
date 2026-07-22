#include "ellipse_fitter.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <utility>

namespace
{
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
    const float geometry_penalty = ellipse.geometry_consistent ? 1.0f : 0.72f;
    return 0.55f * clamp01(detection_confidence) +
           0.45f * clamp01(ellipse.quality) * geometry_penalty;
}

EllipseFitter::EllipseFitter(EllipseFitConfig config) : config_(std::move(config))
{
    config_.ransac_iterations = std::max(1, config_.ransac_iterations);
    config_.maximum_points = std::max(20, config_.maximum_points);
    config_.inlier_threshold_px = std::max(0.1f, config_.inlier_threshold_px);
    config_.minimum_inlier_ratio = clamp01(config_.minimum_inlier_ratio);
    config_.maximum_axis_ratio = std::max(1.0f, config_.maximum_axis_ratio);
    config_.center_deviation_ratio = std::max(0.01f, config_.center_deviation_ratio);
    config_.minimum_candidate_quality = clamp01(config_.minimum_candidate_quality);
    config_.edge_search_quality_threshold = clamp01(config_.edge_search_quality_threshold);
}

EllipseFitResult EllipseFitter::BoxInscribedCircle(const cv::Rect &detection_box)
{
    EllipseFitResult result;
    const float side = static_cast<float>(std::max(0, std::min(detection_box.width,
                                                               detection_box.height)));
    const cv::Point2f center(detection_box.x + detection_box.width * 0.5f,
                             detection_box.y + detection_box.height * 0.5f);
    result.ellipse = cv::RotatedRect(center, cv::Size2f(side, side), 0.0f);
    result.valid = side > 0.0f;
    result.source = EllipseSource::Box;
    result.quality = result.valid ? 0.38f : 0.0f;
    return result;
}

float EllipseFitter::RadialErrorPx(const cv::RotatedRect &ellipse,
                                   const cv::Point2f &point)
{
    const float a = ellipse.size.width * 0.5f;
    const float b = ellipse.size.height * 0.5f;
    if (a < 1e-3f || b < 1e-3f)
        return std::numeric_limits<float>::infinity();
    const float x = point.x - ellipse.center.x;
    const float y = point.y - ellipse.center.y;
    const float angle = -ellipse.angle * static_cast<float>(CV_PI) / 180.0f;
    const float cosine = std::cos(angle);
    const float sine = std::sin(angle);
    const float xr = cosine * x - sine * y;
    const float yr = sine * x + cosine * y;
    const float radius = std::sqrt(xr * xr / (a * a) + yr * yr / (b * b));
    return std::abs(radius - 1.0f) * std::min(a, b);
}

EllipseFitter::RansacResult EllipseFitter::FitRansac(
    const std::vector<cv::Point> &points) const
{
    RansacResult best;
    const int count = static_cast<int>(points.size());
    if (count < 20) return best;
    std::mt19937 random(config_.random_seed);
    std::uniform_int_distribution<int> pick(0, count - 1);
    std::vector<cv::Point> sample(5);
    std::array<int, 5> indices{};

    for (int iteration = 0; iteration < config_.ransac_iterations; ++iteration)
    {
        for (int sample_index = 0; sample_index < 5;)
        {
            const int point_index = pick(random);
            bool duplicate = false;
            for (int previous = 0; previous < sample_index; ++previous)
                duplicate = duplicate || indices[previous] == point_index;
            if (duplicate) continue;
            indices[sample_index] = point_index;
            sample[sample_index] = points[point_index];
            ++sample_index;
        }

        cv::RotatedRect candidate;
        try { candidate = cv::fitEllipse(sample); }
        catch (const cv::Exception &) { continue; }
        if (minor_axis(candidate) < 2.0f || axis_ratio(candidate) > config_.maximum_axis_ratio)
            continue;

        int inliers = 0;
        float error_sum = 0.0f;
        for (const cv::Point &point : points)
        {
            const float error = RadialErrorPx(candidate, point);
            if (error <= config_.inlier_threshold_px)
            {
                ++inliers;
                error_sum += error;
            }
        }
        const float mean_error = inliers > 0 ? error_sum / inliers
                                             : std::numeric_limits<float>::infinity();
        if (inliers > best.inliers ||
            (inliers == best.inliers && mean_error < best.mean_error_px))
        {
            best.valid = true;
            best.ellipse = candidate;
            best.inliers = inliers;
            best.sampled_points = count;
            best.mean_error_px = mean_error;
        }
    }

    const int required = std::max(20, static_cast<int>(std::ceil(
                                          config_.minimum_inlier_ratio * count)));
    if (!best.valid || best.inliers < required) return {};

    std::vector<cv::Point> inlier_points;
    inlier_points.reserve(best.inliers);
    for (const cv::Point &point : points)
        if (RadialErrorPx(best.ellipse, point) <= config_.inlier_threshold_px)
            inlier_points.push_back(point);
    if (inlier_points.size() < 5) return {};
    try { best.ellipse = cv::fitEllipse(inlier_points); }
    catch (const cv::Exception &) { return {}; }

    float error_sum = 0.0f;
    for (const cv::Point &point : inlier_points)
        error_sum += RadialErrorPx(best.ellipse, point);
    best.valid = true;
    best.inliers = static_cast<int>(inlier_points.size());
    best.sampled_points = count;
    best.mean_error_px = error_sum / inlier_points.size();
    return best;
}

std::vector<cv::Point> EllipseFitter::CollectMaskPoints(
    const cv::Mat &binary_roi, const cv::Point2f &box_center_roi) const
{
    cv::Mat work = binary_roi.clone();
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(work, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Point> points;
    const double minimum_area = config_.minimum_contour_area_ratio *
                                binary_roi.cols * binary_roi.rows;
    const float distance_limit = config_.contour_center_distance_ratio *
                                 std::min(binary_roi.cols, binary_roi.rows);
    for (const auto &contour : contours)
    {
        if (contour.size() < 5 || std::abs(cv::contourArea(contour)) < minimum_area)
            continue;
        const cv::Moments moments = cv::moments(contour);
        if (std::abs(moments.m00) < 1e-6) continue;
        const cv::Point2f center(static_cast<float>(moments.m10 / moments.m00),
                                 static_cast<float>(moments.m01 / moments.m00));
        if (cv::norm(center - box_center_roi) <= distance_limit)
            points.insert(points.end(), contour.begin(), contour.end());
    }
    return points;
}

std::vector<cv::Point> EllipseFitter::CollectEdgePoints(
    const cv::Mat &edge_roi, const cv::Point2f &box_center_roi) const
{
    cv::Mat work = edge_roi.clone();
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(work, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    std::vector<cv::Point> points;
    const float distance_limit = config_.contour_center_distance_ratio *
                                 std::min(edge_roi.cols, edge_roi.rows);
    for (const auto &contour : contours)
    {
        if (contour.size() < 18) continue;
        const cv::Rect bounds = cv::boundingRect(contour);
        const cv::Point2f center(bounds.x + bounds.width * 0.5f,
                                 bounds.y + bounds.height * 0.5f);
        if (cv::norm(center - box_center_roi) <= distance_limit)
            points.insert(points.end(), contour.begin(), contour.end());
    }
    return points;
}

EllipseFitResult EllipseFitter::BuildCandidate(const std::vector<cv::Point> &points,
                                                const cv::Rect &roi,
                                                const cv::Rect &detection_box,
                                                EllipseSource source) const
{
    if (points.size() < 20) return {};
    std::vector<cv::Point> sampled;
    sampled.reserve(std::min(static_cast<int>(points.size()), config_.maximum_points));
    const int step = std::max(1, static_cast<int>(points.size()) / config_.maximum_points);
    for (int index = 0; index < static_cast<int>(points.size()) &&
                        static_cast<int>(sampled.size()) < config_.maximum_points; index += step)
        sampled.push_back(points[index]);

    const RansacResult fit = FitRansac(sampled);
    if (!fit.valid) return {};
    EllipseFitResult result;
    result.ellipse = fit.ellipse;
    result.ellipse.center.x += roi.x;
    result.ellipse.center.y += roi.y;
    result.valid = true;
    result.source = source;
    result.from_mask = source == EllipseSource::Mask;
    result.inliers = fit.inliers;
    result.sampled_points = fit.sampled_points;
    result.inlier_ratio = fit.sampled_points > 0
                              ? static_cast<float>(fit.inliers) / fit.sampled_points : 0.0f;
    result.mean_error_px = fit.mean_error_px;

    const cv::Point2f box_center(detection_box.x + detection_box.width * 0.5f,
                                 detection_box.y + detection_box.height * 0.5f);
    const float short_side = std::max(1.0f, static_cast<float>(
                                              std::min(detection_box.width, detection_box.height)));
    result.center_deviation_ratio = cv::norm(result.ellipse.center - box_center) / short_side;
    if (result.center_deviation_ratio > config_.center_deviation_ratio ||
        minor_axis(result.ellipse) < 0.20f * short_side ||
        major_axis(result.ellipse) > 1.45f * std::max(detection_box.width, detection_box.height))
        return {};

    const float error_quality = std::exp(-result.mean_error_px /
                                         std::max(0.5f, config_.inlier_threshold_px));
    const float center_quality = clamp01(1.0f - result.center_deviation_ratio /
                                         config_.center_deviation_ratio);
    const float shape_quality = clamp01(1.0f - (axis_ratio(result.ellipse) - 1.0f) /
                                        std::max(1.0f, config_.maximum_axis_ratio - 1.0f));
    const float source_bonus = source == EllipseSource::Mask ? 0.05f : 0.0f;
    result.quality = clamp01(0.42f * result.inlier_ratio + 0.25f * error_quality +
                             0.20f * center_quality + 0.13f * shape_quality + source_bonus);
    return result;
}

EllipseFitResult EllipseFitter::Fit(const cv::Mat &image,
                                    const cv::Rect &detection_box,
                                    const uint8_t *mask_data,
                                    EllipseFitMode mode) const
{
    const cv::Size image_size = image.empty() ? cv::Size() : image.size();
    EllipseFitResult fallback = BoxInscribedCircle(detection_box);
    if (mode == EllipseFitMode::ForceBox || image_size.width <= 0 || image_size.height <= 0)
        return fallback;
    const cv::Rect roi = detection_box & cv::Rect(0, 0, image.cols, image.rows);
    if (roi.width <= 0 || roi.height <= 0) return fallback;
    const cv::Point2f box_center_roi(detection_box.x + detection_box.width * 0.5f - roi.x,
                                     detection_box.y + detection_box.height * 0.5f - roi.y);

    EllipseFitResult best;
    if (mask_data != nullptr)
    {
        cv::Mat full_mask(image.size(), CV_8UC1, const_cast<uint8_t *>(mask_data));
        best = BuildCandidate(CollectMaskPoints(full_mask(roi), box_center_roi),
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
    return EllipseFitter(no_edge_config).Fit(placeholder, detection_box, mask_data, mode);
}

RingPairRefiner::RingPairRefiner(RingConsistencyConfig config) : config_(std::move(config)) {}

RingConsistencyResult RingPairRefiner::Refine(EllipseFitResult &outer,
                                               EllipseFitResult &middle) const
{
    RingConsistencyResult result;
    if (!outer.valid || !middle.valid) return result;
    result.evaluated = true;
    const float outer_major = major_axis(outer.ellipse);
    const float middle_major = major_axis(middle.ellipse);
    if (outer_major < 1.0f || middle_major < 1.0f) return result;
    const float size_ratio = middle_major / outer_major;
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
        if (outer.quality >= middle.quality) middle.quality *= 0.35f;
        else outer.quality *= 0.35f;
        return result;
    }

    const float outer_weight = std::max(0.05f, outer.quality);
    const float middle_weight = std::max(0.05f, middle.quality);
    const cv::Point2f common_center =
        (outer.ellipse.center * outer_weight + middle.ellipse.center * middle_weight) /
        (outer_weight + middle_weight);
    outer.ellipse.center += (common_center - outer.ellipse.center) * config_.center_fusion_strength;
    middle.ellipse.center += (common_center - middle.ellipse.center) * config_.center_fusion_strength;
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
        EllipseFitResult held = state.value;
        held.quality *= 0.92f;
        held.temporally_filtered = true;
        state.value = held;
        return held;
    }

    const float alpha = config_.minimum_alpha +
                        (config_.maximum_alpha - config_.minimum_alpha) * clamp01(measurement.quality);
    EllipseFitResult filtered = measurement;
    filtered.ellipse.center = state.value.ellipse.center * (1.0f - alpha) +
                              measurement.ellipse.center * alpha;
    filtered.ellipse.size.width = state.value.ellipse.size.width * (1.0f - alpha) +
                                  measurement.ellipse.size.width * alpha;
    filtered.ellipse.size.height = state.value.ellipse.size.height * (1.0f - alpha) +
                                   measurement.ellipse.size.height * alpha;
    filtered.ellipse.angle = blend_angle(state.value.ellipse.angle,
                                         measurement.ellipse.angle, alpha);
    filtered.temporally_filtered = true;
    state.value = filtered;
    return filtered;
}

void EllipseTemporalFilter::Reset()
{
    states_ = {};
}
