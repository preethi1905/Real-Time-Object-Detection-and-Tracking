#pragma once

#include "kalman_box_tracker.hpp"
#include <vector>
#include <set>
#include <algorithm>

namespace kitti_tracking {

struct TrackedObject {
    int id;
    std::vector<double> bbox;
    int class_id;
    double score;
    std::pair<double, double> velocity;
};

struct Detection {
    std::vector<double> bbox; // [x1, y1, x2, y2]
    double score;
    int class_id;
};

inline double compute_iou(const std::vector<double>& a, const std::vector<double>& b) {
    double xx1 = std::max(a[0], b[0]);
    double yy1 = std::max(a[1], b[1]);
    double xx2 = std::min(a[2], b[2]);
    double yy2 = std::min(a[3], b[3]);

    double w = std::max(0.0, xx2 - xx1);
    double h = std::max(0.0, yy2 - yy1);
    double inter = w * h;

    double area_a = (a[2] - a[0]) * (a[3] - a[1]);
    double area_b = (b[2] - b[0]) * (b[3] - b[1]);
    double uni = area_a + area_b - inter;

    if (uni <= 0.0) return 0.0;
    return inter / uni;
}

class Tracker {
public:
    int max_age;
    int min_hits;
    double iou_thr;
    bool class_aware;
    std::vector<KalmanBoxTracker> trackers;

    Tracker(int max_age = 5, int min_hits = 2, double iou_thr = 0.3, bool class_aware = false)
        : max_age(max_age), min_hits(min_hits), iou_thr(iou_thr), class_aware(class_aware) {}

    static void reset() {
        KalmanBoxTracker::count = 0;
    }

    std::vector<TrackedObject> step(const std::vector<Detection>& detections) {
        // 1. Predict all existing trackers
        std::vector<std::vector<double>> preds;
        for (auto& t : trackers) {
            preds.push_back(t.predict());
        }

        std::set<size_t> matched_t;
        std::set<size_t> matched_d;

        // 2. Data Association (Greedy / IoU Hungarian assignment)
        if (!preds.empty() && !detections.empty()) {
            std::vector<std::tuple<double, size_t, size_t>> matches;
            for (size_t r = 0; r < preds.size(); ++r) {
                for (size_t c = 0; c < detections.size(); ++c) {
                    if (class_aware && trackers[r].class_id != detections[c].class_id) {
                        continue;
                    }
                    double iou_val = compute_iou(preds[r], detections[c].bbox);
                    if (iou_val >= iou_thr) {
                        matches.emplace_back(iou_val, r, c);
                    }
                }
            }

            // Sort matches descending by IoU
            std::sort(matches.begin(), matches.end(), [](const auto& a, const auto& b) {
                return std::get<0>(a) > std::get<0>(b);
            });

            for (const auto& m : matches) {
                size_t r = std::get<1>(m);
                size_t c = std::get<2>(m);
                if (matched_t.find(r) == matched_t.end() && matched_d.find(c) == matched_d.end()) {
                    trackers[r].update(detections[c].bbox, detections[c].class_id, detections[c].score);
                    matched_t.insert(r);
                    matched_d.insert(c);
                }
            }
        }

        // 3. Initiate new trackers for unmatched detections
        for (size_t c = 0; c < detections.size(); ++c) {
            if (matched_d.find(c) == matched_d.end()) {
                trackers.emplace_back(detections[c].bbox, detections[c].class_id, detections[c].score);
            }
        }

        // 4. Remove dead tracks
        std::vector<KalmanBoxTracker> active;
        for (auto& t : trackers) {
            if (t.misses <= max_age) {
                active.push_back(t);
            }
        }
        trackers = std::move(active);

        // 5. Output confirmed tracks
        std::vector<TrackedObject> results;
        for (const auto& t : trackers) {
            if (t.hits >= min_hits && t.misses == 0) {
                results.push_back({t.id, t.bbox(), t.class_id, t.score, t.get_velocity()});
            }
        }

        return results;
    }
};

} // namespace kitti_tracking
