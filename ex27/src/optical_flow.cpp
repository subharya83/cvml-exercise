#include "optical_flow.h"

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: ./optical_flow img1.png img2.png window_size\n";
        return 1;
    }
    Image img1 = load_image(argv[1]), img2 = load_image(argv[2]);
    lucas_kanade(img1, img2, std::stoi(argv[3]));
}