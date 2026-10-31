#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace kitti_tracking {

class KalmanBoxTracker {
public:
    static inline int count = 0;

    int id;
    int class_id;
    double score;
    int hits;
    int misses;
    int age;

    Eigen::Matrix<double, 7, 1> x; // [cx, cy, s, r, vx, vy, vs]^T
    Eigen::Matrix<double, 7, 7> F; // State transition
    Eigen::Matrix<double, 4, 7> H; // Measurement matrix
    Eigen::Matrix<double, 7, 7> P; // Covariance
    Eigen::Matrix<double, 7, 7> Q; // Process noise
    Eigen::Matrix<double, 4, 4> R; // Measurement noise

    explicit KalmanBoxTracker(const std::vector<double>& bbox, int cls_id = 0, double scr = 1.0)
        : id(count++), class_id(cls_id), score(scr), hits(1), misses(0), age(0) {
        
        // F: Constant-velocity model
        F.setIdentity();
        F(0, 4) = 1.0; // cx += vx
        F(1, 5) = 1.0; // cy += vy
        F(2, 6) = 1.0; // s += vs

        // H: Observation of [cx, cy, s, r]
        H.setZero();
        H.block<4, 4>(0, 0).setIdentity();

        // P: State error covariance
        P.setIdentity();
        P *= 10.0;
        P.block<3, 3>(4, 4) *= 100.0;

        // Q: Process noise
        Q.setIdentity();
        Q.block<3, 3>(4, 4) *= 0.01;

        // R: Measurement noise
        R.setIdentity();
        R.block<2, 2>(2, 2) *= 10.0;

        // State vector initialization
        x.setZero();
        x.block<4, 1>(0, 0) = to_z(bbox);
    }

    static Eigen::Matrix<double, 4, 1> to_z(const std::vector<double>& b) {
        double w = std::max(b[2] - b[0], 1e-4);
        double h = std::max(b[3] - b[1], 1e-4);
        double cx = b[0] + w / 2.0;
        double cy = b[1] + h / 2.0;
        double s = w * h;
        double r = w / h;

        Eigen::Matrix<double, 4, 1> z;
        z << cx, cy, s, r;
        return z;
    }

    std::vector<double> predict() {
        if (x(2, 0) + x(6, 0) <= 0.0) {
            x(6, 0) = 0.0;
        }

        x = F * x;
        P = F * P * F.transpose() + Q;
        age++;
        misses++;
        return bbox();
    }

    void update(const std::vector<double>& det_bbox, int cls_id = -1, double scr = -1.0) {
        Eigen::Matrix<double, 4, 1> z = to_z(det_bbox);
        Eigen::Matrix<double, 4, 1> y = z - H * x; // Innovation
        Eigen::Matrix<double, 4, 4> S = H * P * H.transpose() + R;
        Eigen::Matrix<double, 7, 4> K = P * H.transpose() * S.inverse(); // Kalman Gain

        x = x + K * y;
        Eigen::Matrix<double, 7, 7> I = Eigen::Matrix<double, 7, 7>::Identity();
        P = (I - K * H) * P;

        hits++;
        misses = 0;
        if (cls_id >= 0) class_id = cls_id;
        if (scr >= 0.0) score = scr;
    }

    std::vector<double> bbox() const {
        double cx = x(0, 0);
        double cy = x(1, 0);
        double s = std::max(x(2, 0), 1e-4);
        double r = std::max(x(3, 0), 1e-4);

        double w = std::sqrt(s * r);
        double h = s / w;

        return {cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0};
    }

    std::pair<double, double> get_velocity() const {
        return {x(4, 0), x(5, 0)};
    }
};

} // namespace kitti_tracking
