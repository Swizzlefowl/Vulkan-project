#include "Renderer.h"

void Renderer::run() {

    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
}

void Renderer::initWindow() {

    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window = glfwCreateWindow(WIDTH, HEIGHT, "hello Vulkan", nullptr, nullptr);
}

void Renderer::initVulkan() {

    createInstance();
    setupDebugCallback();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createGraphicsPipeline();
}

void Renderer::pickPhysicalDevice() {

    uint32_t deviceCount{0};
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0)
        throw std::runtime_error("no Physical device found");

    std::vector<vk::PhysicalDevice> devices{deviceCount};

    instance.enumeratePhysicalDevices(&deviceCount, devices.data());

    for (const auto& device : devices) {

        if (isDeviceSuitable(device)) {
            std::cout << device.getProperties().deviceName << '\n';
            physicalDevice = device;
            break;
        }
    }
}

bool Renderer::checkDeviceExtensionSupport(vk::PhysicalDevice device) {

    std::vector<vk::ExtensionProperties> availableExtensions{
        device.enumerateDeviceExtensionProperties()};

    for (auto& ext : availableExtensions)

        std::cout << ext.extensionName << '\n';
    std::set<std::string> requiredExtensions{deviceExtensions.begin(),
        deviceExtensions.end()};

    for (const auto& extension : availableExtensions)

        requiredExtensions.erase(extension.extensionName);
    return requiredExtensions.empty();
}

bool Renderer::isDeviceSuitable(vk::PhysicalDevice device) {

    vk::PhysicalDeviceProperties deviceProperties{device.getProperties()};
    vk::PhysicalDeviceFeatures deviceFeatures{device.getFeatures()};

    QueueFamilyIndices indices = findQueueFamilies(device);

    bool extensionSupported = checkDeviceExtensionSupport(device);
    bool swapChainAdequate{false};
    if (extensionSupported) {

        SwapChainSupportDetails swapChainSupport{querySwapChainSupport(device)};
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    return deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu && extensionSupported && indices.isComplete() && swapChainAdequate;
}

Renderer::QueueFamilyIndices
Renderer::findQueueFamilies(vk::PhysicalDevice device) {

    QueueFamilyIndices indices;

    std::vector<vk::QueueFamilyProperties> queueFamilies(
        device.getQueueFamilyProperties());

    vk::Bool32 presentSupport{false};

    int i{0};
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
            indices.graphicsFamily = i;
        device.getSurfaceSupportKHR(i, surface, &presentSupport);

        if (presentSupport)
            indices.presentFamily = i;
        if (indices.isComplete())
            break;
        i++;
    }
    return indices;
}

void Renderer::mainLoop() {

    while (!glfwWindowShouldClose(window))

        glfwPollEvents();
}

void Renderer::cleanup() {

    device.destroySwapchainKHR(swapChain);
    for (auto& imageView : swapChainImageViews)

        device.destroyImageView(imageView);
    device.destroy();
    instance.destroySurfaceKHR(surface);
    if (enableValidationLayers)
        DestroyDebugUtilsMessengerEXT(instance, callback, nullptr);
    instance.destroy();
    glfwDestroyWindow(window);
    glfwTerminate();
}

bool Renderer::checkValidationLayerSupport() {

    uint32_t layerCount{};

    vk::enumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<vk::LayerProperties> availableLayers(layerCount);
    vk::enumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers) {

        bool layerFound{false};

        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }
        if (!layerFound)
            return false;
    }
    return true;
}

std::vector<const char*> Renderer::getRequiredExtensions() {

    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions,
        glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers)
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    return extensions;
}

void Renderer::setupDebugCallback() {

    if (!enableValidationLayers)
        return;

    auto createInfo = vk::DebugUtilsMessengerCreateInfoEXT(
        vk::DebugUtilsMessengerCreateFlagsEXT(),
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
        debugCallback, nullptr);

    // NOTE: Vulkan-hpp has methods for this, but they trigger linking errors...
    // instance->createDebugUtilsMessengerEXT(createInfo);
    // instance->createDebugUtilsMessengerEXTUnique(createInfo);

    // NOTE: reinterpret_cast is also used by vulkan.hpp internally for all these
    // structs
    if (CreateDebugUtilsMessengerEXT(
            instance,
            reinterpret_cast<const VkDebugUtilsMessengerCreateInfoEXT*>(
                &createInfo),
            nullptr, &callback)
        != VK_SUCCESS)
        throw std::runtime_error("failed to set up debug callback!");
}

void Renderer::createInstance() {

    if (enableValidationLayers && !checkValidationLayerSupport())
        throw std::runtime_error("validation layers requested, but not available!");

    vk::ApplicationInfo appInfo{};
    appInfo.pApplicationName = "hello Triangle";
    appInfo.applicationVersion = 1;
    appInfo.pEngineName = "no Engine";
    appInfo.engineVersion = 1;
    appInfo.apiVersion = VK_API_VERSION_1_0;

    vk::InstanceCreateInfo createInfo{};
    createInfo.pApplicationInfo = &appInfo;

    uint32_t glfwExtensionCount{0};
    const char** glfwExtension{nullptr};

    glfwExtension = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    auto extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }

    if (vk::createInstance(&createInfo, nullptr, &instance) != vk::Result::eSuccess)
        throw std::runtime_error("failed to create instance!");
}

void Renderer::createLogicalDevice() {

    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(),
        indices.presentFamily.value()};
    float queuePriority = 1.0f;

    for (uint32_t queueFamily : uniqueQueueFamilies) {
        vk::DeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.queueFamilyIndex = indices.graphicsFamily.value();
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    vk::PhysicalDeviceFeatures deviceFeatures{};

    vk::DeviceCreateInfo createInfo{};
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }

    if (physicalDevice.createDevice(&createInfo, nullptr, &device) != vk::Result::eSuccess)
        throw std::runtime_error("failed to create logical device!");

    device.getQueue(indices.graphicsFamily.value(), 0, &graphicsQueue);
    device.getQueue(indices.presentFamily.value(), 0, &presentQueue);
}

void Renderer::createSurface() {
    VkSurfaceKHR surf;
    if (glfwCreateWindowSurface(instance, window, nullptr, &surf) != VK_SUCCESS)
        throw std::runtime_error("failed to create window surface!");

    surface = vk::SurfaceKHR{surf};
}

Renderer::SwapChainSupportDetails
Renderer::querySwapChainSupport(vk::PhysicalDevice device) {

    SwapChainSupportDetails details;
    details.capabilities = device.getSurfaceCapabilitiesKHR(surface);
    details.formats = device.getSurfaceFormatsKHR(surface);
    details.presentModes = device.getSurfacePresentModesKHR(surface);

    return details;
}

vk::SurfaceFormatKHR Renderer::chooseSwapSurfaceFormat(
    const std::vector<vk::SurfaceFormatKHR> availableFormats) {

    for (const auto& availableFormat : availableFormats)

        if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
            return availableFormat;
    return availableFormats[0];
}

vk::PresentModeKHR Renderer::chooseSwapPresentMode(
    const std::vector<vk::PresentModeKHR> availavlePresentModes) {

    for (const auto& presentMode : availavlePresentModes)

        if (presentMode == vk::PresentModeKHR::eMailbox)
            return presentMode;
    return vk::PresentModeKHR::eFifo;
}

vk::Extent2D
Renderer::chooseSwapExtend(const vk::SurfaceCapabilitiesKHR& capabilities) {

    if (capabilities.currentExtent != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    } else {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        vk::Extent2D actualExtent{static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)};
        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width,
            capabilities.maxImageExtent.width);

        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height,
            capabilities.maxImageExtent.height);

        return actualExtent;
    }
    return vk::Extent2D();
}

void Renderer::createSwapChain() {

    SwapChainSupportDetails swapChainSupport{
        querySwapChainSupport(physicalDevice)};
    vk::SurfaceFormatKHR surfaceFormat{
        chooseSwapSurfaceFormat(swapChainSupport.formats)};
    vk::PresentModeKHR presentMode{
        chooseSwapPresentMode(swapChainSupport.presentModes)};
    vk::Extent2D extent{chooseSwapExtend(swapChainSupport.capabilities)};

    uint32_t imageCount{swapChainSupport.capabilities.minImageCount + 1};

    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
        imageCount = swapChainSupport.capabilities.maxImageCount;

    vk::SwapchainCreateInfoKHR createInfo{};
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;

    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    uint32_t queueFamilyIndices[]{indices.graphicsFamily.value(),
        indices.presentFamily.value()};

    if (indices.graphicsFamily != indices.presentFamily) {

        createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {

        createInfo.imageSharingMode = vk::SharingMode::eExclusive;
        createInfo.queueFamilyIndexCount = 0;
        createInfo.pQueueFamilyIndices = nullptr;
    }

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if (device.createSwapchainKHR(&createInfo, nullptr, &swapChain) != vk::Result::eSuccess)
        throw std::runtime_error("failed to create swap chain!");

    swapChainImages = device.getSwapchainImagesKHR(swapChain);
    std::cout << swapChainImages.size() << '\n';

    swapChainImagesFormat = surfaceFormat.format;
    swapChainExtent = extent;
}

void Renderer::createImageViews() {

    swapChainImageViews.resize(swapChainImages.size());

    for (size_t i{0}; i < swapChainImages.size(); i++) {

        vk::ImageViewCreateInfo createInfo{};
        createInfo.image = swapChainImages[i];
        createInfo.viewType = vk::ImageViewType::e2D;
        createInfo.format = swapChainImagesFormat;

        vk::ComponentMapping mappings{
            vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity,
            vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity};
        createInfo.components = mappings;

        // base	MipmapLevel = 0, levelcount = 1, baseArrayLayer = 0, layerCount
        // =
        // 1
        vk::ImageSubresourceRange imageSubResource{vk::ImageAspectFlagBits::eColor,
            0, 1, 0, 1};
        createInfo.subresourceRange = imageSubResource;
        if (device.createImageView(&createInfo, nullptr, &swapChainImageViews[i]) != vk::Result::eSuccess)
            throw std::runtime_error("failed to create image views!");
    }
}

void Renderer::createGraphicsPipeline() {

    auto vertShaderCode{readFile("shaders/vert.spv")};
    auto fragShaderCode{readFile("shaders/frag.spv")};

    vk::ShaderModule vertShaderModule{createShadermodule(vertShaderCode)};
    vk::ShaderModule fragShaderModule{createShadermodule(fragShaderCode)};

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{};
    vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
    vertShaderStageInfo.module = fragShaderModule;
    vertShaderStageInfo.pName = "main";

    vk::PipelineShaderStageCreateInfo shaderStagesInfo[]{vertShaderStageInfo,
        fragShaderStageInfo};

    device.destroyShaderModule(vertShaderModule);
    device.destroyShaderModule(fragShaderModule);
}

std::vector<char> Renderer::readFile(const std::string& fileName) {

    std::ifstream file{fileName, std::ios::ate | std::ios::binary};

    if (!file.is_open())

        throw ::std::runtime_error("could not open file!");
    size_t fileSize{static_cast<size_t>(file.tellg())};
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}

vk::ShaderModule
Renderer::createShadermodule(const std::vector<char>& shaderCode) {

    vk::ShaderModuleCreateInfo createInfo{};
    createInfo.codeSize = shaderCode.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data());
    vk::ShaderModule shaderModule;

    if (device.createShaderModule(&createInfo, nullptr, &shaderModule) != vk::Result::eSuccess)

        throw std::runtime_error("failed to create shader module!");
    return shaderModule;
}

VkBool32 VKAPI_CALL Renderer::debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
}

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pCallback) {

    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr)
        return func(instance, pCreateInfo, pAllocator, pCallback);
    else
        return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance,
    VkDebugUtilsMessengerEXT callback,
    const VkAllocationCallbacks* pAllocator) {

    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr)
        func(instance, callback, pAllocator);
}
