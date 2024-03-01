#include "Renderer.h"
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

// only include the #define in one source file
// otherwise you will get multiple def linker errors

int main() {
    Renderer app{};

    try {
        app.run();
    }

    catch (const std::exception& e) {

        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
