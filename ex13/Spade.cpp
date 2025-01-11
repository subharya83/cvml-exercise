#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <memory>

using json = nlohmann::json;

// Structure to store pixel defect information
struct PixelDefect {
    int x;
    int y;
    double confidence;
    int start_frame;
    int end_frame;
    int total_appearances;

    PixelDefect(int x_, int y_, double conf_, int start_)
        : x(x_), y(y_), confidence(conf_), start_frame(start_),
          end_frame(-1), total_appearances(1) {}

    json to_json() const {
        return {
            {"x", x},
            {"y", y},
            {"confidence", std::round(confidence * 1000.0) / 1000.0},
            {"start_frame", start_frame},
            {"end_frame", end_frame},
            {"total_appearances", total_appearances},
            {"span", end_frame >= 0 ? end_frame - start_frame : nullptr}
        };
    }
};

// Custom hash function for pixel coordinates
struct CoordHash {
    size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

class DeadPixelDetector {
public:
    DeadPixelDetector(double distance_threshold = 3.0,
                      int min_persistence = 5,
                      double noise_threshold = 0.1)
        : distance_threshold_(distance_threshold),
          min_persistence_(min_persistence),
          noise_threshold_(noise_threshold) {}

    void process_video(const std::string& input_path,
                      const std::string& json_output_path,
                      const std::string& video_output_path = "") {
        cv::VideoCapture cap(input_path);
        if (!cap.isOpened()) {
            throw std::runtime_error("Could not open input video: " + input_path);
        }

        std::unique_ptr<cv::VideoWriter> out;
        if (!video_output_path.empty()) {
            int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
            double fps = cap.get(cv::CAP_PROP_FPS);
            cv::Size frame_size(
                static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
                static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))
            );
            out = std::make_unique<cv::VideoWriter>(
                video_output_path, fourcc, fps, frame_size
            );
        }

        int frame_count = 0;
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        cv::Mat frame, gray_frame;

        std::cout << "Processing video: " << input_path << std::endl;

        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
            auto current_defects = detect_defective_pixels(gray_frame);
            update_defects(current_defects, frame_count);

            if (out) {
                visualize_defects(frame);
                *out << frame;
            }

            frame_count++;
            if (frame_count % 100 == 0) {
                std::cout << "Processed frame " << frame_count << "/"
                         << total_frames << std::endl;
            }
        }

        save_results(json_output_path);
    }

private:
    double distance_threshold_;
    int min_persistence_;
    double noise_threshold_;
    std::unordered_map<std::pair<int, int>, PixelDefect, CoordHash> active_defects_;
    std::vector<PixelDefect> completed_defects_;

    std::vector<cv::Mat> compute_difference_matrices(const cv::Mat& frame) {
        static const std::vector<std::pair<int, int>> directions = {
            {1, 1}, {1, 0}, {1, -1}, {0, -1},
            {-1, -1}, {-1, 0}, {-1, 1}, {0, 1}
        };

        std::vector<cv::Mat> diff_matrices;
        diff_matrices.reserve(directions.size());

        for (const auto& [dx, dy] : directions) {
            cv::Mat shifted;
            cv::Mat translation = (cv::Mat_<double>(2, 3) <<
                1, 0, dx,
                0, 1, dy);
            cv::warpAffine(frame, shifted, translation, frame.size(),
                          cv::INTER_LINEAR, cv::BORDER_REPLICATE);
            
            cv::Mat diff;
            cv::absdiff(frame, shifted, diff);
            diff_matrices.push_back(diff);
        }

        return diff_matrices;
    }

    std::pair<cv::Mat, cv::Mat> compute_gradient_matrices(
        const std::vector<cv::Mat>& diff_matrices) {
        
        std::vector<cv::Mat> first_channels(diff_matrices.begin(),
                                          diff_matrices.begin() + 4);
        cv::Mat first_degree;
        cv::merge(first_channels, first_degree);

        cv::Mat second_degree;
        cv::merge(diff_matrices, second_degree);

        // Apply Gaussian smoothing
        for (int i = 0; i < first_degree.channels(); ++i) {
            cv::Mat channel;
            cv::extractChannel(first_degree, channel, i);
            cv::GaussianBlur(channel, channel, cv::Size(3, 3), 0.5);
            cv::insertChannel(channel, first_degree, i);
        }

        for (int i = 0; i < second_degree.channels(); ++i) {
            cv::Mat channel;
            cv::extractChannel(second_degree, channel, i);
            cv::GaussianBlur(channel, channel, cv::Size(3, 3), 0.5);
            cv::insertChannel(channel, second_degree, i);
        }

        return {first_degree, second_degree};
    }

    double compute_mahalanobis_distance(const cv::Mat& vector,
                                      const cv::Mat& mean,
                                      const cv::Mat& icov) {
        cv::Mat diff = vector - mean;
        cv::Mat mult = diff * icov * diff.t();
        return std::sqrt(mult.at<double>(0, 0));
    }

    std::vector<PixelDefect> detect_defective_pixels(const cv::Mat& frame) {
        auto diff_matrices = compute_difference_matrices(frame);
        auto [first_grad, second_grad] = compute_gradient_matrices(diff_matrices);

        std::vector<PixelDefect> candidates;
        const int border = 2;

        for (int y = border; y < frame.rows - border; ++y) {
            for (int x = border; x < frame.cols - border; ++x) {
                cv::Mat first_grad_vector(1, first_grad.channels(), CV_64F);
                cv::Mat second_grad_vector(1, second_grad.channels(), CV_64F);

                // Extract gradient vectors
                for (int c = 0; c < first_grad.channels(); ++c) {
                    first_grad_vector.at<double>(0, c) =
                        first_grad.at<cv::Vec<uchar, 4>>(y, x)[c];
                }
                for (int c = 0; c < second_grad.channels(); ++c) {
                    second_grad_vector.at<double>(0, c) =
                        second_grad.at<cv::Vec<uchar, 8>>(y, x)[c];
                }

                if (cv::mean(first_grad_vector)[0] < noise_threshold_)
                    continue;

                try {
                    cv::Mat covar, mean;
                    cv::calcCovarMatrix(second_grad_vector, covar, mean,
                                      cv::COVAR_NORMAL | cv::COVAR_ROWS);
                    
                    if (cv::determinant(covar) < 1e-10)
                        continue;

                    cv::Mat icov = covar.inv();
                    double distance = compute_mahalanobis_distance(
                        first_grad_vector, mean, icov);

                    if (distance > distance_threshold_) {
                        candidates.emplace_back(x, y, distance, 0);
                    }
                }
                catch (const cv::Exception&) {
                    continue;
                }
            }
        }

        return candidates;
    }

    void update_defects(const std::vector<PixelDefect>& current_defects,
                       int frame_number) {
        std::unordered_set<std::pair<int, int>, CoordHash> current_coords;

        // Update active defects
        for (const auto& current : current_defects) {
            std::pair<int, int> coord{current.x, current.y};
            current_coords.insert(coord);

            auto it = active_defects_.find(coord);
            if (it != active_defects_.end()) {
                it->second.total_appearances++;
                it->second.confidence = std::max(
                    it->second.confidence, current.confidence);
            }
            else {
                PixelDefect defect = current;
                defect.start_frame = frame_number;
                active_defects_[coord] = defect;
            }
        }

        // Check for ended defects
        std::vector<std::pair<int, int>> to_remove;
        for (const auto& [coord, defect] : active_defects_) {
            if (current_coords.find(coord) == current_coords.end()) {
                if (defect.total_appearances >= min_persistence_) {
                    PixelDefect completed = defect;
                    completed.end_frame = frame_number - 1;
                    completed_defects_.push_back(completed);
                }
                to_remove.push_back(coord);
            }
        }

        for (const auto& coord : to_remove) {
            active_defects_.erase(coord);
        }
    }

    void visualize_defects(cv::Mat& frame) {
        for (const auto& [coord, defect] : active_defects_) {
            double confidence_color = std::min(defect.confidence / 10.0, 1.0);
            cv::Scalar color(
                0,
                static_cast<int>(255 * (1 - confidence_color)),
                static_cast<int>(255 * confidence_color)
            );
            cv::circle(frame, cv::Point(defect.x, defect.y), 2, color, -1);
        }
    }

    void save_results(const std::string& output_path) {
        // Finalize remaining active defects
        for (const auto& [coord, defect] : active_defects_) {
            if (defect.total_appearances >= min_persistence_) {
                PixelDefect completed = defect;
                completed.end_frame = completed.start_frame +
                                    completed.total_appearances - 1;
                completed_defects_.push_back(completed);
            }
        }

        // Create JSON output
        json result;
        result["metadata"]["timestamp"] = get_current_timestamp();
        result["metadata"]["settings"] = {
            {"distance_threshold", distance_threshold_},
            {"min_persistence", min_persistence_},
            {"noise_threshold", noise_threshold_}
        };

        result["defects"] = json::array();
        for (const auto& defect : completed_defects_) {
            result["defects"].push_back(defect.to_json());
        }

        // Save to file
        std::ofstream out(output_path);
        out << std::setw(2) << result << std::endl;

        std::cout << "Results saved to " << output_path << std::endl;
        std::cout << "Total dead pixels detected: "
                 << completed_defects_.size() << std::endl;
    }

    std::string get_current_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y-%m-%dT%H:%M:%S");
        return ss.str();
    }
};

int main(int argc, char** argv) {
    try {
        DeadPixelDetector detector(3.0, 5, 0.1);
        detector.process_video(
            "input_video.mp4",
            "dead_pixels_report.json",
            "output_video.mp4"  // Optional
        );
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}