#ifndef SORT_H
#define SORT_H

#include <vector>
#include <Eigen/Dense>

class KalmanBoxTracker {
public:
    KalmanBoxTracker(const Eigen::Vector4f& bbox);
    void update(const Eigen::Vector4f& bbox);
    Eigen::Vector4f predict();
    Eigen::Vector4f get_state();

private:
    Eigen::MatrixXf kf;
    int time_since_update;
    int id;
    std::vector<Eigen::Vector4f> history;
    int hits;
    int hit_streak;
    int age;
};

class Sort {
public:
    Sort(int max_age = 1, int min_hits = 3, float iou_threshold = 0.3);
    std::vector<Eigen::VectorXf> update(const std::vector<Eigen::VectorXf>& detections);

private:
    int max_age;
    int min_hits;
    float iou_threshold;
    std::vector<KalmanBoxTracker> trackers;
    int frame_count;

    float iou(const Eigen::Vector4f& bb_test, const Eigen::Vector4f& bb_gt);
    std::vector<std::pair<int, int>> associate_detections_to_trackers(
        const std::vector<Eigen::Vector4f>& detections,
        const std::vector<Eigen::Vector4f>& trackers);
};

#endif // SORT_H