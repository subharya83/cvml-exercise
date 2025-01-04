#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <memory>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>

using json = nlohmann::json;

// Thread-safe queue for frame processing
template<typename T>
class ThreadSafeQueue {
public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(value));
        cond_.notify_one();
    }

    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    bool wait_and_pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this] { return !queue_.empty() || done_; });
        if (queue_.empty()) return false;
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void set_done() {
        std::lock_guard<std::mutex> lock(mutex_);
        done_ = true;
        cond_.notify_all();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable cond_;
    bool done_ = false;
};

// Memory-efficient frame container
struct Frame {
    cv::Mat data;
    int frame_number;
    
    Frame() = default;
    Frame(cv::Mat&& d, int fn) : data(std::move(d)), frame_number(fn) {}
    Frame(const Frame&) = delete;
    Frame& operator=(const Frame&) = delete;
    Frame(Frame&&) = default;
    Frame& operator=(Frame&&) = default;
};

// Enhanced pixel defect structure with memory optimization
struct PixelDefect {
    int16_t x;
    int16_t y;
    float confidence;
    int32_t start_frame;
    int32_t end_frame;
    int32_t total_appearances;

    PixelDefect(int x_, int y_, double conf_, int start_)
        : x(static_cast<int16_t>(x_)),
          y(static_cast<int16_t>(y_)),
          confidence(static_cast<float>(conf_)),
          start_frame(start_),
          end_frame(-1),
          total_appearances(1) {}

    json to_json() const {
        return {
            {"x", x},
            {"y", y},
            {"confidence", std::round(confidence * 1000.0f) / 1000.0f},
            {"start_frame", start_frame},
            {"end_frame", end_frame},
            {"total_appearances", total_appearances},
            {"span", end_frame >= 0 ? end_frame - start_frame : nullptr}
        };
    }
};

class DeadPixelDetector {
public:
    DeadPixelDetector(double distance_threshold = 3.0,
                      int min_persistence = 5,
                      double noise_threshold = 0.1,
                      size_t num_threads = std::thread::hardware_concurrency())
        : distance_threshold_(distance_threshold),
          min_persistence_(min_persistence),
          noise_threshold_(noise_threshold),
          num_threads_(num_threads) {
        active_defects_.reserve(1000);  // Pre-allocate space for defects
        completed_defects_.reserve(1000);
    }

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

        // Initialize thread pool and processing queues
        ThreadSafeQueue<std::unique_ptr<Frame>> frame_queue;
        ThreadSafeQueue<std::pair<int, std::vector<PixelDefect>>> result_queue;
        std::atomic<bool> processing_complete{false};
        std::vector<std::thread> workers;
        std::mutex results_mutex;

        // Start worker threads
        for (size_t i = 0; i < num_threads_; ++i) {
            workers.emplace_back([this, &frame_queue, &result_queue, &processing_complete] {
                process_frames_worker(frame_queue, result_queue, processing_complete);
            });
        }

        // Start result processing thread
        std::thread result_processor([this, &result_queue, &processing_complete, &results_mutex] {
            process_results_worker(result_queue, processing_complete, results_mutex);
        });

        int frame_count = 0;
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        std::cout << "Processing video: " << input_path << std::endl;

        // Main processing loop
        while (true) {
            auto frame = std::make_unique<Frame>();
            cap >> frame->data;
            if (frame->data.empty()) break;

            frame->frame_number = frame_count++;
            frame_queue.push(std::move(frame));

            if (frame_count % 100 == 0) {
                std::cout << "Processed frame " << frame_count << "/"
                         << total_frames << std::endl;
            }
        }

        // Signal completion and wait for workers
        frame_queue.set_done();
        processing_complete = true;
        for (auto& worker : workers) {
            worker.join();
        }
        result_processor.join();

        // Save results
        save_results(json_output_path);
    }

private:
    double distance_threshold_;
    int min_persistence_;
    double noise_threshold_;
    size_t num_threads_;
    std::unordered_map<std::pair<int, int>, PixelDefect, CoordHash> active_defects_;
    std::vector<PixelDefect> completed_defects_;
    std::mutex defects_mutex_;

    // Worker thread function for frame processing
    void process_frames_worker(
        ThreadSafeQueue<std::unique_ptr<Frame>>& frame_queue,
        ThreadSafeQueue<std::pair<int, std::vector<PixelDefect>>>& result_queue,
        std::atomic<bool>& processing_complete) {
        
        while (!processing_complete) {
            std::unique_ptr<Frame> frame;
            if (!frame_queue.wait_and_pop(frame)) continue;

            cv::Mat gray_frame;
            cv::cvtColor(frame->data, gray_frame, cv::COLOR_BGR2GRAY);
            auto defects = detect_defective_pixels(gray_frame);
            
            result_queue.push(std::make_pair(
                frame->frame_number,
                std::move(defects)
            ));
        }
    }

    // Worker thread function for result processing
    void process_results_worker(
        ThreadSafeQueue<std::pair<int, std::vector<PixelDefect>>>& result_queue,
        std::atomic<bool>& processing_complete,
        std::mutex& results_mutex) {
        
        while (!processing_complete || !result_queue.empty()) {
            std::pair<int, std::vector<PixelDefect>> result;
            if (!result_queue.wait_and_pop(result)) continue;

            std::lock_guard<std::mutex> lock(results_mutex);
            update_defects(result.second, result.first);
        }
    }

    // Memory-optimized difference matrices computation
    std::vector<cv::Mat> compute_difference_matrices(const cv::Mat& frame) {
        static const std::array<std::pair<int, int>, 8> directions = {{
            {1, 1}, {1, 0}, {1, -1}, {0, -1},
            {-1, -1}, {-1, 0}, {-1, 1}, {0, 1}
        }};

        std::vector<cv::Mat> diff_matrices;
        diff_matrices.reserve(directions.size());

        cv::Mat shifted;
        shifted.create(frame.size(), frame.type());

        for (const auto& [dx, dy] : directions) {
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

    // Rest of the class implementation remains similar but with thread-safety additions
    // [Previous methods: compute_gradient_matrices, compute_mahalanobis_distance, 
    // detect_defective_pixels, update_defects, visualize_defects, save_results, 
    // get_current_timestamp remain the same but with proper mutex locks where needed]
    
    void update_defects(const std::vector<PixelDefect>& current_defects,
                       int frame_number) {
        std::lock_guard<std::mutex> lock(defects_mutex_);
        // [Previous update_defects implementation]
    }

    void save_results(const std::string& output_path) {
        std::lock_guard<std::mutex> lock(defects_mutex_);
        // [Previous save_results implementation]
    }
};

int main(int argc, char** argv) {
    try {
        // Use 80% of available CPU cores for processing
        size_t num_threads = (std::thread::hardware_concurrency() * 8) / 10;
        if (num_threads < 1) num_threads = 1;

        DeadPixelDetector detector(3.0, 5, 0.1, num_threads);
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