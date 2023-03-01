#include "Renderer.h"
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
int main() {
    Renderer app{};

    try {
        app.run();
    }

    catch (const std::exception& e) {

        std::cerr << e.what() << 'n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
