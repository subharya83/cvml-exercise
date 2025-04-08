#include "sift.h"
#include <fstream>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./sift input.png output_keypoints.txt\n";
        return 1;
    }
    Image img = load_image(argv[1]);
    auto keypoints = extract_sift_features(img);
    
    // Save keypoints (format: x y size angle descriptor[0..N-1])
    std::ofstream out(argv[2]);
    for (const auto& kp : keypoints) {
        out << kp.x << " " << kp.y << " " << kp.size << " " << kp.angle;
        for (float val : kp.descriptor) out << " " << val;
        out << "\n";
    }
}