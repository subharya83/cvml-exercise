#include "connected_components.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./components input_binary.png output_labels.png\n";
        return 1;
    }
    Image img = load_image(argv[1]);
    Image labels = connected_components(img);
    save_image(labels, argv[2]);
}