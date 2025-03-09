#include "sort.h"
#include <iostream>

KalmanBoxTracker::KalmanBoxTracker(const Eigen::Vector4f& bbox) {
    // Initialize Kalman filter
    kf = Eigen::MatrixXf::Identity(7, 7);
    time_since_update = 0;
    id = 0; // Assign unique ID
    hits = 0;
    hit_streak = 0;
    age = 0;
}

void KalmanBoxTracker::update(const Eigen::Vector4f& bbox) {
    time_since_update = 0;
    history.clear();
    hits++;
    hit_streak++;
    // Update Kalman filter state
}

Eigen::Vector4f KalmanBoxTracker::predict() {
    age++;
    if (time_since_update > 0) hit_streak = 0;
    time_since_update++;
    Eigen::Vector4f prediction = kf.block<4, 1>(0, 0);
    history.push_back(prediction);
    return prediction;
}

Eigen::Vector4f KalmanBoxTracker::get_state() {
    return kf.block<4, 1>(0, 0);
}

Sort::Sort(int max_age, int min_hits, float iou_threshold)
    : max_age(max_age), min_hits(min_hits), iou_threshold(iou_threshold), frame_count(0) {}

std::vector<Eigen::VectorXf> Sort::update(const std::vector<Eigen::VectorXf>& detections) {
    frame_count++;
    std::vector<Eigen::Vector4f> trks(trackers.size());
    std::vector<int> to_del;
    std::vector<Eigen::VectorXf> ret;

    for (size_t t = 0; t < trackers.size(); t++) {
        Eigen::Vector4f pos = trackers[t].predict();
        trks[t] = pos;
        if (std::isnan(pos[0])) to_del.push_back(t);
    }

    for (auto it = to_del.rbegin(); it != to_del.rend(); ++it) {
        trackers.erase(trackers.begin() + *it);
    }

    auto matches = associate_detections_to_trackers(detections, trks);

    for (const auto& m : matches) {
        trackers[m.second].update(detections[m.first].head<4>());
    }

    for (size_t i = 0; i < detections.size(); i++) {
        if (std::find_if(matches.begin(), matches.end(), [i](const std::pair<int, int>& match) { return match.first == i; }) == matches.end()) {
            trackers.emplace_back(detections[i].head<4>());
        }
    }

    for (size_t i = 0; i < trackers.size(); i++) {
        auto d = trackers[i].get_state();
        if (trackers[i].time_since_update < 1 && (trackers[i].hit_streak >= min_hits || frame_count <= min_hits)) {
            Eigen::VectorXf ret_row(5);
            ret_row << d[0], d[1], d[2], d[3], trackers[i].id + 1;
            ret.push_back(ret_row);
        }
        if (trackers[i].time_since_update > max_age) {
            trackers.erase(trackers.begin() + i);
            i--;
        }
    }

    return ret;
}

float Sort::iou(const Eigen::Vector4f& bb_test, const Eigen::Vector4f& bb_gt) {
    float xx1 = std::max(bb_test[0], bb_gt[0]);
    float yy1 = std::max(bb_test[1], bb_gt[1]);
    float xx2 = std::min(bb_test[2], bb_gt[2]);
    float yy2 = std::min(bb_test[3], bb_gt[3]);

    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float wh = w * h;
    float o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
                  + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh);
    return o;
}

std::vector<std::pair<int, int>> Sort::associate_detections_to_trackers(
    const std::vector<Eigen::VectorXf>& detections,
    const std::vector<Eigen::Vector4f>& trackers) {
    std::vector<std::pair<int, int>> matches;
    if (trackers.empty()) return matches;

    Eigen::MatrixXf iou_matrix(detections.size(), trackers.size());
    for (size_t i = 0; i < detections.size(); i++) {
        for (size_t j = 0; j < trackers.size(); j++) {
            iou_matrix(i, j) = iou(detections[i].head<4>(), trackers[j]);
        }
    }

    for (size_t i = 0; i < detections.size(); i++) {
        for (size_t j = 0; j < trackers.size(); j++) {
            if (iou_matrix(i, j) > iou_threshold) {
                matches.emplace_back(i, j);
            }
        }
    }

    return matches;
}