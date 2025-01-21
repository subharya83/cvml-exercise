#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>

class VideoCorruptor {
public:
    VideoCorruptor(const std::string& video_path, double corruption_rate = 0.00005, const std::vector<int>& color = {255, 255, 255})
        : video_path(video_path), corruption_rate(corruption_rate), corruption_color(color) {
        cap.open(video_path);
        if (!cap.isOpened()) {
            throw std::runtime_error("Error: Could not open video file.");
        }

        width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
        total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    }

    ~VideoCorruptor() {
        if (cap.isOpened()) {
            cap.release();
        }
    }

    void process_video(const std::string& output_path) {
        cv::VideoWriter out(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
        if (!out.isOpened()) {
            throw std::runtime_error("Error: Could not open video writer.");
        }

        int frame_idx = 0;
        cv::Mat frame;
        while (cap.read(frame)) {
            cv::Mat corrupted_frame = corrupt_frame(frame, frame_idx);
            out.write(corrupted_frame);
            frame_idx++;
        }

        out.release();
    }

private:
    std::string video_path;
    double corruption_rate;
    cv::VideoCapture cap;
    int width, height, fps, total_frames;
    std::vector<int> corruption_color;

    struct CorruptionInfo {
        std::vector<int> rgb;
        int end_frame;
    };

    std::unordered_map<std::string, CorruptionInfo> corruption_map;

    std::vector<std::pair<int, int>> generate_corruption_locations() {
        int total_pixels = width * height;
        int num_corrupted_pixels = static_cast<int>(total_pixels * corruption_rate);

        std::vector<std::pair<int, int>> locations;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> x_dist(1, width - 2);
        std::uniform_int_distribution<> y_dist(1, height - 2);

        while (locations.size() < num_corrupted_pixels) {
            int x = x_dist(gen);
            int y = y_dist(gen);

            std::vector<std::pair<int, int>> neighborhood = get_valid_neighborhood(x, y);
            int num_contiguous = std::min(5, static_cast<int>(neighborhood.size()));
            std::shuffle(neighborhood.begin(), neighborhood.end(), gen);

            for (int i = 0; i < num_contiguous && locations.size() < num_corrupted_pixels; ++i) {
                locations.push_back(neighborhood[i]);
            }
        }

        return locations;
    }

    std::vector<std::pair<int, int>> get_valid_neighborhood(int x, int y) {
        std::vector<std::pair<int, int>> neighborhood;
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                int new_x = x + dx;
                int new_y = y + dy;
                if (new_x >= 0 && new_x < width && new_y >= 0 && new_y < height) {
                    neighborhood.emplace_back(new_x, new_y);
                }
            }
        }
        return neighborhood;
    }

    cv::Mat corrupt_frame(const cv::Mat& frame, int frame_idx) {
        cv::Mat corrupted_frame = frame.clone();

        // Clean up expired corruptions
        for (auto it = corruption_map.begin(); it != corruption_map.end(); ) {
            if (it->second.end_frame < frame_idx) {
                it = corruption_map.erase(it);
            } else {
                ++it;
            }
        }

        // Generate new corruptions if needed
        if (frame_idx % 5 == 0) {
            std::vector<std::pair<int, int>> new_locations = generate_corruption_locations();
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> duration_dist(5, 10);

            for (const auto& loc : new_locations) {
                std::string key = std::to_string(loc.first) + "," + std::to_string(loc.second);
                if (corruption_map.find(key) == corruption_map.end()) {
                    corruption_map[key] = {corruption_color, frame_idx + duration_dist(gen)};
                }
            }
        }

        // Apply corruptions
        for (const auto& [key, corruption_info] : corruption_map) {
            if (corruption_info.end_frame >= frame_idx) {
                size_t comma_pos = key.find(',');
                int x = std::stoi(key.substr(0, comma_pos));
                int y = std::stoi(key.substr(comma_pos + 1));
                corrupted_frame.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    corruption_info.rgb[0],
                    corruption_info.rgb[1],
                    corruption_info.rgb[2]
                );
            }
        }

        return corrupted_frame;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " -i <input_video> -o <output_video> -c <color_code>" << std::endl;
        return 1;
    }

    std::string input_video;
    std::string output_video;
    std::vector<int> color = {255, 255, 255}; // Default to white

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-i" && i + 1 < argc) {
            input_video = argv[++i];
        } else if (arg == "-o" && i + 1 < argc) {
            output_video = argv[++i];
        } else if (arg == "-c" && i + 1 < argc) {
            std::string color_code = argv[++i];
            if (color_code.length() == 3) {
                color[0] = (color_code[0] == '1') ? 255 : 0;
                color[1] = (color_code[1] == '1') ? 255 : 0;
                color[2] = (color_code[2] == '1') ? 255 : 0;
            } else {
                std::cerr << "Invalid color code. Using default color (white)." << std::endl;
            }
        }
    }

    try {
        VideoCorruptor corruptor(input_video, 0.00005, color);
        corruptor.process_video(output_video);
        std::cout << "Video corruption completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}