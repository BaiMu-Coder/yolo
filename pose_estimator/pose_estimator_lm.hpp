#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <cmath>
#include <optional>

// 锥套物理尺寸大小，单位mm
struct DrogueModel
{
    double radius_cls0_mm = 0.0;
    double radius_cls1_mm = 0.0;
    double radius_hole_mm = 0.0;
    double length_L_mm = 0.0;
};

struct Pose6D
{
    double yaw_deg = 0.0;   // around Y
    double pitch_deg = 0.0; // around X
    double roll_deg = 0.0;  // fixed 0
    double tx_mm = 0.0;
    double ty_mm = 0.0;
    double tz_mm = 0.0;
};

inline static double pose_estimator_clamp(double val, double min, double max)
{
    double f = val <= min ? min : (val >= max ? max : val);
    return f;
}

// 手写Levenberg-Marquardt (LM) 非线性优化算法，来解算加油锥套的6D姿态估计（Pose Estimation）  主要就解算xyz和偏航角和俯仰角
/*
通常我们求姿态用OpenCV的solvePnP就行了，这里做的优化：主要是为了处理特殊的误差模型：

普通 PnP：点对点匹配（3D 点 A 对应 2D 点 a）。

锥套口是一个圆，投影到图像上是个椭圆。

圆周上没有明显的特征点（比如角点），很难确定 3D 圆环上的哪个点对应 2D 椭圆上的哪个点（旋转歧义性）。

创新点：它不强求点对点匹配，而是要求 “投影下来的 3D 圆环，必须完美贴合在 2D 检测到的椭圆轮廓上”。
*/
class PoseEstimatorLM
{
public:
    PoseEstimatorLM() = default;

    PoseEstimatorLM(const cv::Mat &cameraK, const cv::Mat &distCoeffs, const DrogueModel &model)
    {
        Reset(cameraK, distCoeffs, model);
    }

    void Reset(const cv::Mat &cameraK, const cv::Mat &distCoeffs, const DrogueModel &model)
    {
        K_ = cameraK.clone();
        D_ = distCoeffs.clone();
        model_ = model;

        // 根据锥套的物理尺寸，造了三个空间上的圆，点云
        pts3d_cls0_ = genCircle3D(model_.radius_cls0_mm, 0.0, 32); // 32个点
        pts3d_cls1_ = genCircle3D(model_.radius_cls1_mm, 0.0, 32);
        pts3d_hole_ = genCircle3D(model_.radius_hole_mm, -model_.length_L_mm, 16);
        // 原理：利用 genCircle3D 在 3D 空间里虚拟地“造”了一个圆环的点云。
        // 作用：优化时，我们把这些 3D 点投影到图像上，看它们是否落在了检测到的椭圆线上。
        // 作用：用一堆可投影的 3D 圆周点，把“外圈=一个真实圆”的几何先验塞进优化里，
    }

    Pose6D Solve(const cv::RotatedRect &targetEllipse,               // 外圈的椭圆 类别0或1
                 const cv::Point2f &holeCenterPx,                    // 中间的圆心 类别2
                 bool use_cls1,                                      // 是否用的类别1
                 std::optional<double> known_dist_mm = std::nullopt, // 看外部是否传递深度信息   （一般实际情况就是 不传递）
                 int max_iters = 30) const                           // LM最大迭代次数
    {
        const auto &pts3d = use_cls1 ? pts3d_cls1_ : pts3d_cls0_;                       // 预先定义好的3D特征点
        const double real_r = use_cls1 ? model_.radius_cls1_mm : model_.radius_cls0_mm; // 真实半径

        // tz init: tz ≈ f * R / r_img   这里就是利用小孔成像原理
        const double ew = targetEllipse.size.width; // RotatedRect 的 size.width/size.height 是拟合得到的椭圆“外接旋转矩形”的宽高
        const double eh = targetEllipse.size.height;
        double r_img ;  //使用长轴进行距离求解
        if(ew>eh)
        r_img=ew/2.0;
        else
        r_img=eh/2.0;
        const double f_mean = 0.5 * (K_.at<double>(0, 0) + K_.at<double>(1, 1)); // f_mean是相机的焦距    (K_.at<double>(0,0)表示矩阵中(0,0)位置的元素   （这里可以写真值，不用fx fy去反算）

        double tz_est = 200.0; // 这里给一个默认值
        if (known_dist_mm.has_value())
            tz_est = *known_dist_mm;
        else
            tz_est = (r_img > 1.0) ? (f_mean * real_r / r_img) : 200.0; // 小孔成像原理计算深度

        // yaw/pitch init from hole offset   给定一个初值，让LM更容易收敛
        const double cx = targetEllipse.center.x; // 同yolo坐标系 左上角原点
        const double cy = targetEllipse.center.y;
        const double dx = holeCenterPx.x - cx;
        const double dy = holeCenterPx.y - cy;
        const double L = model_.length_L_mm;
        double yaw_guess = 0.0, pit_guess = 0.0;
        if (L > 10.0)
        {
            yaw_guess = rad2deg(-std::atan(dx * tz_est / (f_mean * L))); // 用正视 来计算一个初值
            pit_guess = rad2deg(std::atan(dy * tz_est / (f_mean * L)));
            yaw_guess = pose_estimator_clamp(yaw_guess, -30.0, 30.0);
            pit_guess = pose_estimator_clamp(pit_guess, -30.0, 30.0);
        }

        // tx/ty init by back-projecting ellipse center
        double tx0 = (cx - K_.at<double>(0, 2)) * tz_est / K_.at<double>(0, 0); // 椭圆中心在图像哪里，就反推出“物体中心在相机坐标的 X/Y 方向偏了多少(也就是相机坐标系下的坐标，这里并且通过小孔成像原理将像素坐标转换为了：物理距离的坐标单位毫米)”。
        double ty0 = (cy - K_.at<double>(1, 2)) * tz_est / K_.at<double>(1, 1);

        if (known_dist_mm.has_value())
        {
            std::vector<double> x = {yaw_guess, pit_guess, tx0, ty0};
            lmOptimize(x, targetEllipse, holeCenterPx, pts3d, *known_dist_mm, max_iters);
            return {x[0], x[1], 0.0, x[2], x[3], *known_dist_mm};
        }
        else
        {
            std::vector<double> x = {yaw_guess, pit_guess, tx0, ty0, tz_est};
            lmOptimize(x, targetEllipse, holeCenterPx, pts3d, std::nullopt, max_iters);
            return {x[0], x[1], 0.0, x[2], x[3], x[4]};
        }
    }

    void DrawAxis(cv::Mat &img, const Pose6D &pose, bool use_cls1) const
    {
        const double length = use_cls1 ? model_.radius_cls1_mm : model_.radius_cls0_mm;

        cv::Vec3d rvec = eulerYXZ_to_rvec(pose.yaw_deg, pose.pitch_deg, pose.roll_deg);
        cv::Vec3d tvec(pose.tx_mm, pose.ty_mm, pose.tz_mm);

        std::vector<cv::Point3f> pts = {   //定义3D空间里面的骨架
            {0, 0, 0},
            {(float)length, 0, 0},
            {0, (float)length, 0},
            {0, 0, (float)(length * 3.0)}};
        std::vector<cv::Point2f> imgpts;
        cv::projectPoints(pts, rvec, tvec, K_, D_, imgpts);   //3D投影到2D投影

        //坐标转整数
        auto toI = [](const cv::Point2f &p)
        { return cv::Point((int)std::lround(p.x), (int)std::lround(p.y)); };
        cv::Point o = toI(imgpts[0]);

        cv::line(img, o, toI(imgpts[1]), cv::Scalar(0, 0, 255), 3);
        cv::line(img, o, toI(imgpts[2]), cv::Scalar(0, 255, 0), 3);
        cv::line(img, o, toI(imgpts[3]), cv::Scalar(255, 0, 0), 3);
    }

private:
    cv::Mat K_, D_; // 相机内参 和 畸变系数
    DrogueModel model_;
    std::vector<cv::Point3f> pts3d_cls0_, pts3d_cls1_, pts3d_hole_; // 虚拟模拟出来的三维点云

    static inline double deg2rad(double d) { return d * CV_PI / 180.0; }
    static inline double rad2deg(double r) { return r * 180.0 / CV_PI; }

    // “造一个半径为 r，位于高度 z 的圆，并把它离散化成 n 个 3D 坐标点。”
    static std::vector<cv::Point3f> genCircle3D(double r, double z, int n)
    {
        std::vector<cv::Point3f> pts;
        pts.reserve(n);
        for (int i = 0; i < n; ++i)
        {
            double th = 2.0 * CV_PI * (double)i / (double)n;
            pts.emplace_back((float)(r * std::cos(th)), (float)(r * std::sin(th)), (float)z); // 利用圆的参数方程来生成点
        }
        return pts;
    }

    // Python: R.from_euler('yxz', [yaw,pitch,roll]) => R = Rz(roll)*Rx(pitch)*Ry(yaw)
    // 将 欧拉角 (Euler Angles) 转换为 旋转向量 (Rotation Vector)。 因为OpenCV很多核心函数只能处理旋转向量 （三个数表示旋转轴和旋转角度）
    static cv::Vec3d eulerYXZ_to_rvec(double yaw_deg, double pitch_deg, double roll_deg)
    {
        const double y = deg2rad(yaw_deg);
        const double p = deg2rad(pitch_deg);
        const double r = deg2rad(roll_deg);

        cv::Matx33d Ry(std::cos(y), 0, std::sin(y), // 绕Y轴旋转的矩阵
                       0, 1, 0,
                       -std::sin(y), 0, std::cos(y));

        cv::Matx33d Rx(1, 0, 0, // 绕X轴旋转的矩阵
                       0, std::cos(p), -std::sin(p),
                       0, std::sin(p), std::cos(p));

        cv::Matx33d Rz(std::cos(r), -std::sin(r), 0, // 绕Z轴旋转的矩阵
                       std::sin(r), std::cos(r), 0,
                       0, 0, 1);

        cv::Matx33d R = Rz * Rx * Ry; // 矩阵乘法的顺序代表了旋转的顺序

        cv::Vec3d rvec;                  // 是一个3x1的向量
        cv::Rodrigues(cv::Mat(R), rvec); // Rodrigues用于在 旋转矩阵 和 旋转向量 之间进行无损转换   它的数学原理是：向量的方向表示旋转轴，向量的长度（模）表示旋转的角度。
        return rvec;
    }

    // 为下面误差函数提供的工具，用来限制 误差  抗异常点
    // 小误差时 ≈ r（保持精度）
    // 大误差时 ≈ √(2δ|r|)（增长变慢，抗异常）
    static inline double pseudo_huber(double r, double delta)
    {
        // delta: 像素级阈值（建议 2~5）
        const double x = r / delta;
        return std::sqrt(2.0 * delta * delta * (std::sqrt(1.0 + x * x) - 1.0));
    }

    // 误差函数
    // 前 N 项（N = proj_outer 点数）：外圈/参考圈投影点应该落在检测椭圆上 （计算值：点偏离椭圆边界的程度）
    // 最后 1 项：孔中心投影点应该落在对应检测到的孔中心上 （计算值：欧氏距离）
    // LM 做的就是不断改 yaw,pitch,tx,ty,(tz)，让上面这些残差都趋近 0。（优化方向）
    void computeResidual(const std::vector<double> &x,          // 优化变量，最终结果
                         const cv::RotatedRect &targetEllipse,  // 外圈椭圆
                         const cv::Point2f &holeCenterPx,       // 内孔中心坐标
                         const std::vector<cv::Point3f> &pts3d, // 3D模型点，用来和图像中的投影点作比对
                         std::optional<double> fixed_tz,        // 看是否已知深度
                         std::vector<double> &residual) const   // 用来保存计算出来的残差
    // 常成员函数：不修改类里面的任何成员变量
    {
        // 获取优化参数
        double yaw, pitch, tx, ty, tz;
        if (fixed_tz.has_value())
        {
            yaw = x[0];
            pitch = x[1];
            tx = x[2];
            ty = x[3];
            tz = *fixed_tz;
        }
        else
        {
            yaw = x[0];
            pitch = x[1];
            tx = x[2];
            ty = x[3];
            tz = x[4];
        }

        residual.clear();
        residual.reserve((int)pts3d.size() + 1);

        if (tz < 100.0 || targetEllipse.size.width < 2.0 || targetEllipse.size.height < 2.0)
        { // 深度非常小 或者  椭圆尺寸异常小   返回一个很大的残差，表示计算错误
            residual.assign((int)pts3d.size() + 1, 100000.0);
            return;
        }

        // 将 欧拉角 转换为 旋转向量
        cv::Vec3d rvec = eulerYXZ_to_rvec(yaw, pitch, 0.0);
        cv::Vec3d tvec(tx, ty, tz); // 平移向量

        std::vector<cv::Point2f> proj_outer;
        cv::projectPoints(pts3d, rvec, tvec, K_, D_, proj_outer); // 将 3D 点 pts3d 投影到 2D 图像平面，使用了相机内参 K_ 和畸变系数 D_。
                                                                  // 也就是  把这个真实的圆投影到 我拟合出来的椭圆的那个平面上(也就是相机平面)  （因为我的3D点云是个正视平面，所以用上面的旋转矩阵和平移向量，把这个真实的圆，投影到和拟合的圆平面上去，做误差计算）

        std::vector<cv::Point3f> hole3d = {cv::Point3f(0.0f, 0.0f, (float)(-model_.length_L_mm))}; // 把中心孔中心的3D位置 投影到图像上
        std::vector<cv::Point2f> proj_hole;
        cv::projectPoints(hole3d, rvec, tvec, K_, D_, proj_hole);
        const cv::Point2f proj_hole_c = proj_hole[0];

        const double ox = targetEllipse.center.x;
        const double oy = targetEllipse.center.y;
        const double ow = targetEllipse.size.width; // 这两就是外接矩形的 宽高
        const double oh = targetEllipse.size.height;
        const double ang = deg2rad(targetEllipse.angle);

        const double cos_a = std::cos(ang);
        const double sin_a = std::sin(ang);
        // 计算长短半轴
        const double a = ow * 0.5;
        const double b = oh * 0.5;

        for (const auto &p2 : proj_outer)
        { // 对每个投影点计算 椭圆方程误差
            const double dx = p2.x - ox;
            const double dy = p2.y - oy;
            // 先把点移到椭圆中心
            // 再把点逆旋转到椭圆“轴对齐”的坐标系里
            // 目的：让椭圆变成“水平摆正”的椭圆，方便待入标准椭圆方程
            const double lx = dx * cos_a + dy * sin_a;
            const double ly = -dx * sin_a + dy * cos_a;

            // 用椭圆标准方程算误差
            const double err = (lx / a) * (lx / a) + (ly / b) * (ly / b) - 1.0;

            /*             直接返回误差                   */
            // // 把椭圆约束误差整体放大20倍，给一个加权值
            // residual.push_back(err * 20.0);

            /*             限制一下返回误差                  */
            // 给每一个误差加Huber损失，抗异常点
            //  先加权
            double r = err * 20.0;
            // 再做鲁棒（delta 可以调：2~8）
            r = pseudo_huber(r, 5.0);
            residual.push_back(r);
        }

        // 再加一个约束，孔中心对齐误差
        double dist = std::hypot(proj_hole_c.x - holeCenterPx.x, proj_hole_c.y - holeCenterPx.y);
        //// residual.push_back(dist * 50.0);

        // 加权
        dist *= 50.0;
        // 鲁棒
        dist = pseudo_huber(dist, 10.0); // 孔中心允许更大一点，比如 8~15
        residual.push_back(dist);
    }

    // 非线性最小二乘求解器
    // 基本流程：
    // 1、在当前 x 处算残差 r
    // 2、数值差分得到雅可比 J = ∂r/∂x
    // 3、解线性方程得到参数增量 delta
    // 4、试探更新 x_try = x + delta
    // 5、如果代价函数下降就接受，并减小 lambda；否则拒绝并增大 lambda
    // 6、重复迭代直到收敛或达到次数上限
    void lmOptimize(std::vector<double> &x,                // 优化变量，最终结果
                    const cv::RotatedRect &targetEllipse,  // 外圈椭圆
                    const cv::Point2f &holeCenterPx,       // 内孔中心坐标
                    const std::vector<cv::Point3f> &pts3d, // 3D模型点，用来和图像中的投影点作比对
                    std::optional<double> fixed_tz,        // 看是否已知深度
                    int max_iters) const                   // 最大迭代次数
    {
        const int n = (int)x.size();
        double lambda = 1e-3;     //LM的阻尼因子，也就是搜索步长

        // r：    当前 x 下的残差向量
        // r_new：试探参数 x_try 下的残差，用于比较代价
        std::vector<double> r, r_new;
        computeResidual(x, targetEllipse, holeCenterPx, pts3d, fixed_tz, r);


        //Lambda表达式，定义代价函数   0.5 * Σ r²
        auto costOf = [](const std::vector<double> &rr)  
        {
            double c = 0.0;
            for (double v : rr)
                c += v * v;
            return 0.5 * c;
        };


        double cost = costOf(r);

        for (int it = 0; it < max_iters; ++it)   //开始进行迭代
        {
            const int m = (int)r.size();
            cv::Mat J(m, n, CV_64F);    //mxn矩阵   雅可比矩阵   表示每个 residual（残差） 对每个参数的偏导
            cv::Mat rMat(m, 1, CV_64F);    //残差向量
            for (int i = 0; i < m; ++i)
                rMat.at<double>(i, 0) = r[i];
  
            //数值差分，计算雅可比矩阵    
            const double eps_base = 1e-4;
            for (int j = 0; j < n; ++j)
            {
                //对每个参数 加一个小偏差
                std::vector<double> x_eps = x;
                double eps = eps_base * (std::abs(x[j]) + 1.0);
                x_eps[j] += eps;

                computeResidual(x_eps, targetEllipse, holeCenterPx, pts3d, fixed_tz, r_new);
                for (int i = 0; i < m; ++i)
                {
                    J.at<double>(i, j) = (r_new[i] - r[i]) / eps;
                }
            }

            //构造LM正规方程  A delta = -g
            cv::Mat A = J.t() * J;
            cv::Mat g = J.t() * rMat;
            for (int d = 0; d < n; ++d)
                A.at<double>(d, d) += lambda;

            //解线性方程得到delta，用的SVD方程
            cv::Mat delta;
            if (!cv::solve(A, -g, delta, cv::DECOMP_SVD))
                break;

            //试探更新x_try,检查步长是否太小
            double step_norm = 0.0;
            std::vector<double> x_try = x;
            for (int j = 0; j < n; ++j)
            {
                const double dj = delta.at<double>(j, 0);
                x_try[j] += dj;
                step_norm += dj * dj;
            }
            if (step_norm < 1e-12)   //认为收敛了，退出
                break;


            //用x_try计算新的残差，决定是接受还是拒绝
            computeResidual(x_try, targetEllipse, holeCenterPx, pts3d, fixed_tz, r_new);
            double cost_new = costOf(r_new);
            if (cost_new < cost)  //残差变小，接受
            {
                x = std::move(x_try);
                r = std::move(r_new);
                cost = cost_new;
                lambda = std::max(lambda * 0.5, 1e-9);    //阻尼变小一点
            }
            else   //残差变大，拒绝
            {
                lambda *= 2.0;       //阻尼变大，下次更保守
                if (lambda > 1e9)    //过大,退出
                    break;
            }

        }
    }
};
