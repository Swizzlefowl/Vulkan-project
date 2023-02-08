#ifndef RENDERER
#define RENDERER
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pCallback);

void DestroyDebugUtilsMessengerEXT(VkInstance instance,
    VkDebugUtilsMessengerEXT callback,
    const VkAllocationCallbacks* pAllocator);

class Renderer {
  private:
#ifdef NDEBUG
    const bool enableValidationLayers{false};
#else
    const bool enableValidationLayers{true};
#endif
    const int maxFramesInFlight{2};
    // member var for window
    const uint32_t WIDTH{800};
    const uint32_t HEIGHT{600};
    GLFWwindow* window{nullptr};

    // validation layers
    std::vector<const char*> validationLayers{"VK_LAYER_KHRONOS_validation"};
    const std::vector<const char*> deviceExtensions{
        VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkDebugUtilsMessengerEXT callback;

    struct QueueFamilyIndices {

        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;

        bool isComplete() {
            return graphicsFamily.has_value() && presentFamily.has_value();
        }
    };

    struct SwapChainSupportDetails {

        vk::SurfaceCapabilitiesKHR capabilities;
        std::vector<vk::SurfaceFormatKHR> formats;
        std::vector<vk::PresentModeKHR> presentModes;
    };

    // member var for vulkan objects
    vk::Instance instance;
    vk::PhysicalDevice physicalDevice;
    vk::Device device{};
    vk::Queue graphicsQueue;
    vk::Queue presentQueue;
    vk::SurfaceKHR surface;
    vk::SwapchainKHR swapChain;
    std::vector<vk::Image> swapChainImages;
    vk::Format swapChainImagesFormat;
    vk::Extent2D swapChainExtent;
    std::vector<vk::ImageView> swapChainImageViews;
    vk::RenderPass renderPass;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline graphicsPipeline;
    std::vector<vk::Framebuffer> swapChainFrameBuffers;
    vk::CommandPool commandPool;
    std::vector<vk::CommandBuffer> commandBuffers;
    std::vector<vk::Semaphore> imageAvailableSemaphores;
    std::vector<vk::Semaphore> finishedRenderingSemaphores;
    std::vector<vk::Fence> inFlightFences;
    bool framebufferResized{false};

  public:
    void run();

  private:
    // initializes the the window and vulkan instance
    void initWindow();
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height); 
    void initVulkan();
    void pickPhysicalDevice();
    bool checkDeviceExtensionSupport(vk::PhysicalDevice device);
    bool isDeviceSuitable(vk::PhysicalDevice device);
    QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device);

    void mainLoop();
    void cleanup();

    // vulkan val layers fucntions
    bool checkValidationLayerSupport();
    std::vector<const char*> getRequiredExtensions();
    void setupDebugCallback();

    // functions to init vulkan objects
    void createInstance();
    void createLogicalDevice();
    void createSurface();

    // functions for swapChains
    SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice device);
    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
        const std::vector<vk::SurfaceFormatKHR> availableFormats);
    vk::PresentModeKHR chooseSwapPresentMode(
        const std::vector<vk::PresentModeKHR> availavlePresentModes);
    vk::Extent2D chooseSwapExtend(const vk::SurfaceCapabilitiesKHR& capabilities);
    void createSwapChain();
    void createImageViews();
    void createFrameBuffers();
    void recreateSwapChain();
    void cleanupSwapChain();

    // graphics pipeline functions
    void createRenderPass();
    void createGraphicsPipeline();
    static std::vector<char> readFile(const std::string& fileName);
    vk::ShaderModule createShadermodule(const std::vector<char>& shaderCode);

    // functions for command buffers
    void createCommandPool();
    void createCommandBuffers();
    void recordCommandBuffer(vk::CommandBuffer& commandBuffer, uint32_t imageIndex);

    // functions for drawing frames
    void drawFrame();
    void createSyncObjects();

    static VKAPI_ATTR VkBool32 VKAPI_CALL
    debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData);
};
#endif
