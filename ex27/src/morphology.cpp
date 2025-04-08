#include "morphology.h"

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: ./morphology input.png output.png (dilate|erode)\n";
        return 1;
    }
    Image img = load_image(argv[1]);
    Image result;
    if (std::string(argv[3]) == "dilate") result = dilate(img);
    else if (std::string(argv[3]) == "erode") result = erode(img);
    else throw std::runtime_error("Invalid operation");
    save_image(result, argv[2]);
}