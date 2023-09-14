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
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    window = glfwCreateWindow(WIDTH, HEIGHT, "hello Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

void Renderer::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto app = reinterpret_cast<Renderer*>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
}

void Renderer::initVulkan() {
    createInstance();
    setupDebugCallback();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createUniformBuffers();
    createDescriptorSetLayout();
    createRenderPass();
    createGraphicsPipeline();
    createGraphicsPipeline2();
    createCommandPool();
    createDepthResources();
    createFrameBuffers();
    createTextureImage();
    createTextureImageView();
    createTextureSampler();
    createDescriptorPool();
    createDescriptorSets();
    loadModel();
    createVertexBuffer();
    createIndexBuffer();
    createInstancedata();
    createCommandBuffers();
    createSyncObjects();
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
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        drawFrame();
    }
    device.waitIdle();
}

void Renderer::cleanup() {
    cleanupSwapChain();
    device.destroySampler(textureSampler);
    device.destroyImageView(textureImageView);
    device.destroyImage(textureImage);
    device.freeMemory(textureImageMemory);
    device.destroyCommandPool(commandPool);
    device.destroyPipeline(graphicsPipeline);
    device.destroyPipeline(graphicsPipeline2);

    for (size_t i{}; i < maxFramesInFlight; i++) {
        device.destroyBuffer(uniformBuffers[i]);
        device.unmapMemory(uniformBuffersMemory[i]);
        device.freeMemory(uniformBuffersMemory[i]);
    }

    device.destroyDescriptorPool(descriptorPool);
    device.destroyDescriptorSetLayout(descriptorSetLayout);
    device.destroyPipelineLayout(pipelineLayout);
    device.destroyRenderPass(renderPass);
    device.destroyBuffer(vertexBuffer);
    device.freeMemory(vertexBufferMemory);
    device.destroyBuffer(indexBuffer);
    device.freeMemory(indexBufferMemory);
    device.destroyBuffer(instanceBuffer);
    device.freeMemory(instanceBufferMemory);

    for (size_t i{}; i < maxFramesInFlight; i++) {
        device.destroySemaphore(imageAvailableSemaphores[i]);
        device.destroySemaphore(finishedRenderingSemaphores[i]);
        device.destroyFence(inFlightFences[i]);
    }
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

    vk::PhysicalDeviceFeatures checkDeviceFeatures{physicalDevice.getFeatures()};
    vk::PhysicalDeviceFeatures deviceFeatures{};
    if (checkDeviceFeatures.samplerAnisotropy)
        deviceFeatures.samplerAnisotropy = true;

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
    size_t i{0};
    for (auto& imageView : swapChainImageViews) {
        imageView = createImageView(swapChainImages[i], swapChainImagesFormat, vk::ImageAspectFlagBits::eColor);
        i++;
    }
}

void Renderer::createFrameBuffers() {
    swapChainFrameBuffers.resize(swapChainImageViews.size());

    size_t i{0};
    for (auto& attachment : swapChainImageViews) {
        std::array<vk::ImageView, 2> attachments = {
            swapChainImageViews[i],
            depthImageView};

        vk::FramebufferCreateInfo frameBufferInfo{};
        frameBufferInfo.renderPass = renderPass;
        frameBufferInfo.attachmentCount = attachments.size();
        frameBufferInfo.pAttachments = attachments.data();
        frameBufferInfo.width = swapChainExtent.width;
        frameBufferInfo.height = swapChainExtent.height;
        frameBufferInfo.layers = 1;

        if (device.createFramebuffer(&frameBufferInfo, nullptr, &swapChainFrameBuffers[i]) != vk::Result::eSuccess)
            throw std::runtime_error("failed to create a framebuffer!");
        i++;
    }
}

void Renderer::recreateSwapChain() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }

    device.waitIdle();
    cleanupSwapChain();
    createSwapChain();
    createImageViews();
    createDepthResources();
    createFrameBuffers();
}

void Renderer::createTextureImage() {
    int texWidth{};
    int texHeight{};
    int texChannels{};
    stbi_uc* pixels{nullptr};
    pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    vk::DeviceSize imageSize{static_cast<vk::DeviceSize>(texWidth * texHeight * 4)};

    if (!pixels)
        throw std::runtime_error("failed to load image!");

    vk::Buffer stagingBuffer{};
    vk::DeviceMemory stagingBufferMemory{};
    createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

    void* data = device.mapMemory(stagingBufferMemory, 0, imageSize);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    device.unmapMemory(stagingBufferMemory);
    stbi_image_free(pixels);

    createImage(texWidth, texHeight, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage, textureImageMemory);
    transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
    copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
    transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    device.destroyBuffer(stagingBuffer);
    device.freeMemory(stagingBufferMemory);
}

void Renderer::createImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Image& image, vk::DeviceMemory& imageMemory) {
    vk::ImageCreateInfo imageInfo{};
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.extent.width = static_cast<uint32_t>(width);
    imageInfo.extent.height = static_cast<uint32_t>(height);
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.usage = usage;
    imageInfo.initialLayout = vk::ImageLayout::eUndefined;
    imageInfo.tiling = tiling;
    imageInfo.sharingMode = vk::SharingMode::eExclusive;
    imageInfo.samples = vk::SampleCountFlagBits::e1;
    try {
        image = device.createImage(imageInfo);
    } catch (vk::Error& err) {
        std::cout << err.what();
    }
    vk::MemoryRequirements memRequirements;
    memRequirements = device.getImageMemoryRequirements(image);

    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    try {
        imageMemory = device.allocateMemory(allocInfo);
    } catch (vk::Error& err) {
        std::cout << err.what();
    }
    device.bindImageMemory(image, imageMemory, 0);
}

void Renderer::copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height) {
    vk::CommandBuffer commandBuffer{beginSingleTimeCommands()};

    vk::BufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = vk::Offset3D{0, 0, 0};
    region.imageExtent = vk::Extent3D{
        width,
        height,
        1};

    try {
        commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
    } catch (vk::Error& err) {
        std::cout << "failed to copy image";
    }
    endSingleTimeCommands(commandBuffer);
}

void Renderer::transitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
    vk::CommandBuffer commandBuffer{beginSingleTimeCommands()};

    vk::ImageMemoryBarrier memoryBarrier{};
    memoryBarrier.oldLayout = oldLayout;
    memoryBarrier.newLayout = newLayout;
    memoryBarrier.image = image;
    memoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    memoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    memoryBarrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    memoryBarrier.subresourceRange.baseMipLevel = 0;
    memoryBarrier.subresourceRange.levelCount = 1;
    memoryBarrier.subresourceRange.baseArrayLayer = 0;
    memoryBarrier.subresourceRange.layerCount = 1;

    vk::PipelineStageFlags sourceStage;
    vk::PipelineStageFlags destinationStage;

    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
        memoryBarrier.srcAccessMask = vk::AccessFlagBits::eNone;
        memoryBarrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        memoryBarrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        memoryBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        sourceStage = vk::PipelineStageFlagBits::eTransfer;
        destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    } else {
        throw std::invalid_argument("unsupported layout transition!");
    }
    commandBuffer.pipelineBarrier(sourceStage, destinationStage, vk::DependencyFlagBits{}, nullptr, nullptr, memoryBarrier);

    endSingleTimeCommands(commandBuffer);
}

void Renderer::cleanupSwapChain() {
    for (auto& framebuffer : swapChainFrameBuffers)
        device.destroyFramebuffer(framebuffer);
    for (auto& imageView : swapChainImageViews)
        device.destroyImageView(imageView);
    device.destroySwapchainKHR(swapChain);
    device.destroyImageView(depthImageView);
    device.destroyImage(depthImage);
    device.freeMemory(depthImageMemory);
}

void Renderer::loadModel() {

    // randomly gen data pos for instance data
    // it will get added in the vertex shaders
    // to the original pos of the vertex

    std::default_random_engine rndGenerator((unsigned)time(nullptr));
    std::uniform_real_distribution<float> uniformDist(0.0f, 0.0f);
    for (int index{ 0 }; index < 10; index++) {
        glm::vec3 instance{};
        instance.r = uniformDist(rndGenerator);
        instance.g = uniformDist(rndGenerator);
        instance.b = uniformDist(rndGenerator);
        instances.push_back(instance);
    }
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str()))
        throw std::runtime_error(warn + err);

    std::unordered_map<Vertex, uint32_t> uniqueVertices{};
  
    for (const auto& shape : shapes) {
        for (const auto& index : shape.mesh.indices) {
            Vertex vertex{};

            vertex.pos = {
                // the 3 is here because every vertices takes
                // 3 elements in the vector, because its a 1D
                // array
                // you add the 0 or 1 to take account of the
                // x,y,z of the vector
                attrib.vertices[3 * index.vertex_index + 0],
                attrib.vertices[3 * index.vertex_index + 1],
                attrib.vertices[3 * index.vertex_index + 2]};

            vertex.texCoord = {
                // same for texcoords
                attrib.texcoords[2 * index.texcoord_index + 0],
                attrib.texcoords[2 * index.texcoord_index + 1]};

            vertex.color = {1.0f, 0.0f, 0.0f};

            vertex.normal = {
                attrib.normals[3 * index.normal_index + 0],
                attrib.vertices[3 * index.normal_index + 1],
                attrib.vertices[3 * index.normal_index + 2]};
  
            if (uniqueVertices.count(vertex) == 0) {
                uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                vertices.push_back(vertex);
            }
            indices.push_back(uniqueVertices[vertex]);
        }
    }
}

void Renderer::createGraphicsPipeline() {
    auto vertShaderCode{readFile("vertex.spv")};
    auto fragShaderCode{readFile("fragment.spv")};

    // every  vertex buffer needs its own binding
    // the binding is just the index of the buffer inside an array
    // which you will bind during draw time while recording
    // your command buffer
    // stride is just how much bytes the shader will jump
    // after every invocation
    // input rate is when the shader will take the data here its
    // per instance

    vk::VertexInputBindingDescription InstanceBindingDescription{};
    InstanceBindingDescription.binding = 1;
    InstanceBindingDescription.stride = sizeof(glm::vec3);
    InstanceBindingDescription.inputRate = vk::VertexInputRate::eInstance;

    // offset is wehn you have interleaved data inside
    // your buffers, you need to get the offset of the data
    // inside the struct

    vk::VertexInputAttributeDescription InstanceAttributeDescription{};
    InstanceAttributeDescription.binding = 1;
    InstanceAttributeDescription.location = 4;
    InstanceAttributeDescription.format = vk::Format::eR32G32B32Sfloat;
    InstanceAttributeDescription.offset = 0;

    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescription = Vertex::getAttributeDescriptions();

    static std::array<vk::VertexInputBindingDescription, 2> bindingDescriptions = {bindingDescription, InstanceBindingDescription};
    static std::vector<vk::VertexInputAttributeDescription> attributeDescriptions{};
    for (auto binding : attributeDescription) {
        attributeDescriptions.push_back(binding);
    }
    attributeDescriptions.push_back(InstanceAttributeDescription);
    std::cout << "attribute size is: " << attributeDescriptions.size() << "\n";

    vk::ShaderModule vertShaderModule{createShadermodule(vertShaderCode)};
    vk::ShaderModule fragShaderModule{createShadermodule(fragShaderCode)};

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    std::vector<vk::PipelineShaderStageCreateInfo> shaderStagesInfo{vertShaderStageInfo,
        fragShaderStageInfo};

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.vertexBindingDescriptionCount = bindingDescriptions.size();
    vertexInputInfo.pVertexBindingDescriptions = bindingDescriptions.data();
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
    inputAssembly.primitiveRestartEnable = VK_FALSE;
    vk::PipelineViewportStateCreateInfo viewportState{};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    vk::PipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = vk::PolygonMode::eFill;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = vk::CullModeFlagBits::eBack;
    rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f;
    rasterizer.depthBiasClamp = 0.0f;
    rasterizer.depthBiasSlopeFactor = 0.0f;

    vk::PipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
    multisampling.minSampleShading = 1.0f;
    multisampling.pSampleMask = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    colorBlendAttachment.blendEnable = VK_FALSE;

    vk::PipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = vk::LogicOp::eCopy;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    std::vector<vk::DynamicState> dynamicStates{
        vk::DynamicState::eViewport, vk::DynamicState::eScissor};

    vk::PipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    vk::PipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = vk::CompareOp::eLess;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f; // Optional
    depthStencil.maxDepthBounds = 1.0f; // Optional
    depthStencil.stencilTestEnable = VK_FALSE; // Optional

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;

    if (device.createPipelineLayout(&pipelineLayoutInfo, nullptr, &pipelineLayout) != vk::Result::eSuccess)
        throw std::runtime_error("failed to create pipeline layout!");

    vk::GraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.stageCount = shaderStagesInfo.size();
    pipelineInfo.pStages = shaderStagesInfo.data();
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;

    try {
        graphicsPipeline = device.createGraphicsPipeline(nullptr, pipelineInfo).value;
    } catch (vk::SystemError err) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    device.destroyShaderModule(vertShaderModule);
    device.destroyShaderModule(fragShaderModule);

}

void Renderer::createGraphicsPipeline2() {
    auto vertShaderCode{readFile("vertex2.spv")};
    auto fragShaderCode{readFile("fragment2.spv")};
    //auto bindingDescription = Vertex::getBindingDescription();
    //auto attributeDescriptions = Vertex::getAttributeDescriptions();

    vk::ShaderModule vertShaderModule{createShadermodule(vertShaderCode)};
    vk::ShaderModule fragShaderModule{createShadermodule(fragShaderCode)};

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    std::vector<vk::PipelineShaderStageCreateInfo> shaderStagesInfo{vertShaderStageInfo,
        fragShaderStageInfo};

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.vertexBindingDescriptionCount = 0;
    vertexInputInfo.pVertexBindingDescriptions = nullptr; // Optional
    vertexInputInfo.vertexAttributeDescriptionCount = 0;
    vertexInputInfo.pVertexAttributeDescriptions = nullptr; // Optional

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
    inputAssembly.primitiveRestartEnable = VK_FALSE;
    vk::PipelineViewportStateCreateInfo viewportState{};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    vk::PipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = vk::PolygonMode::eFill;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = vk::CullModeFlagBits::eBack;
    rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f;
    rasterizer.depthBiasClamp = 0.0f;
    rasterizer.depthBiasSlopeFactor = 0.0f;

    vk::PipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
    multisampling.minSampleShading = 1.0f;
    multisampling.pSampleMask = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    colorBlendAttachment.blendEnable = VK_FALSE;

    vk::PipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = vk::LogicOp::eCopy;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    std::vector<vk::DynamicState> dynamicStates{
        vk::DynamicState::eViewport, vk::DynamicState::eScissor};

    vk::PipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    vk::PipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = vk::CompareOp::eLess;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f; // Optional
    depthStencil.maxDepthBounds = 1.0f; // Optional
    depthStencil.stencilTestEnable = VK_FALSE; // Optional

    /* vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.setLayoutCount = 0;
    pipelineLayoutInfo.pSetLayouts = nullptr;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;

    if (device.createPipelineLayout(&pipelineLayoutInfo, nullptr, &pipelineLayout2) != vk::Result::eSuccess)
        throw std::runtime_error("failed to create pipeline layout!");*/

    vk::GraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.stageCount = shaderStagesInfo.size();
    pipelineInfo.pStages = shaderStagesInfo.data();
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;

    try {
        graphicsPipeline2 = device.createGraphicsPipeline(nullptr, pipelineInfo).value;
    } catch (vk::SystemError err) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    device.destroyShaderModule(vertShaderModule);
    device.destroyShaderModule(fragShaderModule);
}

void Renderer::createRenderPass() {
    vk::AttachmentDescription colorAttachment{};
    colorAttachment.format = swapChainImagesFormat;
    colorAttachment.samples = vk::SampleCountFlagBits::e1;
    colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
    colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

    vk::AttachmentDescription depthAttachment{};
    depthAttachment.format = findDepthFormat();
    depthAttachment.samples = vk::SampleCountFlagBits::e1;
    depthAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    depthAttachment.storeOp = vk::AttachmentStoreOp::eDontCare;
    depthAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    depthAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    depthAttachment.initialLayout = vk::ImageLayout::eUndefined;
    depthAttachment.finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

    vk::AttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::AttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

    vk::SubpassDescription subpass{};
    subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    vk::SubpassDependency depedancy{};
    depedancy.srcSubpass = VK_SUBPASS_EXTERNAL;
    depedancy.dstSubpass = 0;
    depedancy.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests;
    depedancy.srcAccessMask = vk::AccessFlagBits::eNone;
    depedancy.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests;
    depedancy.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite;

    std::array<vk::AttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};
    vk::RenderPassCreateInfo renderPassInfo{};
    renderPassInfo.attachmentCount = attachments.size();
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &depedancy;

    if (device.createRenderPass(&renderPassInfo, nullptr, &renderPass) != vk::Result::eSuccess)
        throw std::runtime_error("failed to create a render pass object!");
}

void Renderer::createDescriptorSetLayout() {
    vk::DescriptorSetLayoutBinding binding{};
    vk::DescriptorSetLayoutBinding samplerBinding{};
    vk::DescriptorSetLayoutCreateInfo createInfo{};

    binding.binding = 0;
    binding.descriptorCount = 1;
    binding.descriptorType = vk::DescriptorType::eUniformBuffer;
    binding.pImmutableSamplers = nullptr;
    binding.stageFlags = vk::ShaderStageFlagBits::eVertex;

    samplerBinding.binding = 1;
    samplerBinding.descriptorCount = 1;
    samplerBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    binding.pImmutableSamplers = nullptr;
    samplerBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

    std::array<vk::DescriptorSetLayoutBinding, 2> bindings{binding, samplerBinding};
    createInfo.bindingCount = bindings.size();
    createInfo.pBindings = bindings.data();

    if (device.createDescriptorSetLayout(&createInfo, nullptr, &descriptorSetLayout)
        != vk::Result::eSuccess)
        throw std::runtime_error("failed to create descriptorSetLayout");
}

void Renderer::createDescriptorPool() {
    std::array<vk::DescriptorPoolSize, 2> poolSize{};
    poolSize[0].type = vk::DescriptorType::eUniformBuffer;
    poolSize[0].descriptorCount = static_cast<uint32_t>(maxFramesInFlight);
    poolSize[1].type = vk::DescriptorType::eCombinedImageSampler;
    poolSize[1].descriptorCount = static_cast<uint32_t>(maxFramesInFlight);

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.poolSizeCount = poolSize.size();
    poolInfo.pPoolSizes = poolSize.data();
    poolInfo.maxSets = static_cast<uint32_t>(maxFramesInFlight);

    if (device.createDescriptorPool(&poolInfo, nullptr, &descriptorPool) != vk::Result::eSuccess)
        throw std::runtime_error("failed to create descriptor pool!");
}

void Renderer::createDescriptorSets() {
    std::vector<vk::DescriptorSetLayout> layouts(maxFramesInFlight, descriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.descriptorSetCount = static_cast<uint32_t>(maxFramesInFlight);
    allocateInfo.descriptorPool = descriptorPool;
    allocateInfo.pSetLayouts = layouts.data();

    try {
        descriptorSets = device.allocateDescriptorSets(allocateInfo);
    } catch (vk::Error& err) {
        std::cout << err.what();
    }
    for (size_t i = 0; i < maxFramesInFlight; i++) {
        vk::DescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);

        vk::DescriptorImageInfo imageInfo{};
        imageInfo.imageView = textureImageView;
        imageInfo.sampler = textureSampler;
        imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

        std::array<vk::WriteDescriptorSet, 2> descriptorWrite{};
        descriptorWrite[0].dstSet = descriptorSets[i];
        descriptorWrite[0].dstBinding = 0;
        descriptorWrite[0].dstArrayElement = 0;
        descriptorWrite[0].descriptorType = vk::DescriptorType::eUniformBuffer;
        descriptorWrite[0].descriptorCount = 1;
        descriptorWrite[0].pBufferInfo = &bufferInfo;

        descriptorWrite[1].dstSet = descriptorSets[i];
        descriptorWrite[1].dstBinding = 1;
        descriptorWrite[1].dstArrayElement = 0;
        descriptorWrite[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        descriptorWrite[1].descriptorCount = 1;
        descriptorWrite[1].pImageInfo = &imageInfo;

        device.updateDescriptorSets(descriptorWrite, nullptr);
    }
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

void Renderer::createCommandPool() {
    QueueFamilyIndices queueFamilyIndices{findQueueFamilies(physicalDevice)};

    vk::CommandPoolCreateInfo poolInfo{};
    poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

    if (device.createCommandPool(&poolInfo, nullptr, &commandPool) != vk::Result::eSuccess)
        throw std::runtime_error("failed to create command Pool!");
}

void Renderer::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Buffer& buffer, vk::DeviceMemory& bufferMemory) {
    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = vk::SharingMode::eExclusive;

    if (device.createBuffer(&bufferInfo, nullptr, &buffer) != vk::Result::eSuccess)
        throw std::runtime_error("failed to create vertex buffer");

    vk::MemoryRequirements memRequirments{};
    device.getBufferMemoryRequirements(buffer, &memRequirments);

    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.allocationSize = memRequirments.size;
    allocInfo.memoryTypeIndex = findMemoryType(
        memRequirments.memoryTypeBits, properties);

    if (device.allocateMemory(&allocInfo, nullptr, &bufferMemory) != vk::Result::eSuccess)
        throw std::runtime_error("failed to allocate vertex buffer memory");
    device.bindBufferMemory(buffer, bufferMemory, 0);
}

void Renderer::createVertexBuffer() {
    vk::DeviceSize bufferSize{sizeof(vertices[0]) * vertices.size()};
    std::cout << vertices.size() << '\n';
    vk::Buffer stagingBuffer{};
    vk::DeviceMemory stagingBufferMemory{};

    createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuffer, stagingBufferMemory);

    void* data{
        device.mapMemory(stagingBufferMemory, 0, bufferSize)};
    memcpy(data, vertices.data(), static_cast<size_t>(bufferSize));
    device.unmapMemory(stagingBufferMemory);

    createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        vertexBuffer, vertexBufferMemory);

    copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

    device.destroyBuffer(stagingBuffer);
    device.freeMemory(stagingBufferMemory);
}

void Renderer::createInstancedata() {
    // remember to copy the data to the correct buffer

    vk::DeviceSize bufferSize{(sizeof(instances[0]) * instances.size())};
    std::cout << "buffer size is: " << bufferSize << '\n';
    vk::Buffer stagingBuffer{};
    vk::DeviceMemory stagingBufferMemory{};

    createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuffer, stagingBufferMemory);

    void* data{
        device.mapMemory(stagingBufferMemory, 0, bufferSize)};
    memcpy(data, instances.data(), static_cast<size_t>(bufferSize));
    device.unmapMemory(stagingBufferMemory);

    createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        instanceBuffer, instanceBufferMemory);

    copyBuffer(stagingBuffer, instanceBuffer, bufferSize);

    device.destroyBuffer(stagingBuffer);
    device.freeMemory(stagingBufferMemory);
}

void Renderer::createIndexBuffer() {
    vk::DeviceSize bufferSize{sizeof(indices[0]) * indices.size()};
    std::cout << indices.size() << '\n';
    vk::Buffer stagingBuffer{};
    vk::DeviceMemory stagingBufferMemory{};

    createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuffer, stagingBufferMemory);

    void* data{
        device.mapMemory(stagingBufferMemory, 0, bufferSize)};
    memcpy(data, indices.data(), static_cast<size_t>(bufferSize));
    device.unmapMemory(stagingBufferMemory);

    createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        indexBuffer, indexBufferMemory);

    copyBuffer(stagingBuffer, indexBuffer, bufferSize);

    device.destroyBuffer(stagingBuffer);
    device.freeMemory(stagingBufferMemory);
}

void Renderer::createUniformBuffers() {
    vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

    uniformBuffers.resize(maxFramesInFlight);
    uniformBuffersMemory.resize(maxFramesInFlight);
    uniformBuffersMapped.resize(maxFramesInFlight);

    for (size_t i = 0; i < maxFramesInFlight; i++) {
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, uniformBuffers[i], uniformBuffersMemory[i]);
        uniformBuffersMapped[i] = device.mapMemory(uniformBuffersMemory[i], 0, bufferSize);
    }
}

void Renderer::copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) {
    vk::CommandBuffer commandBuffer{beginSingleTimeCommands()};

    vk::BufferCopy copyRegion{};
    copyRegion.size = size;
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;

    commandBuffer.copyBuffer(srcBuffer, dstBuffer, copyRegion);

    endSingleTimeCommands(commandBuffer);
}

uint32_t Renderer::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties memProperties{
        physicalDevice.getMemoryProperties()};

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    throw std::runtime_error("failed to find suitable memory type!");
}

void Renderer::createCommandBuffers() {
    commandBuffers.resize(maxFramesInFlight);
    vk::CommandBufferAllocateInfo allocateInfo{};
    allocateInfo.commandPool = commandPool;
    allocateInfo.level = vk::CommandBufferLevel::ePrimary;
    allocateInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

    if (device.allocateCommandBuffers(&allocateInfo, commandBuffers.data()) != vk::Result::eSuccess)
        throw std::runtime_error("failed to create command Buffer!");
}

vk::CommandBuffer Renderer::beginSingleTimeCommands() {
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = 1;
    allocInfo.commandPool = commandPool;

    vk::CommandBuffer commandBuffer{};
    device.allocateCommandBuffers(&allocInfo, &commandBuffer);

    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    commandBuffer.begin(beginInfo);

    return commandBuffer;
}

void Renderer::endSingleTimeCommands(vk::CommandBuffer commandBuffer) {
    commandBuffer.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    graphicsQueue.submit(submitInfo);
    graphicsQueue.waitIdle();

    device.freeCommandBuffers(commandPool, commandBuffer);
}

void Renderer::recordCommandBuffer(vk::CommandBuffer& commandBuffer, uint32_t imageIndex) {
    vk::CommandBufferBeginInfo beginInfo{};

    if (commandBuffer.begin(&beginInfo) != vk::Result::eSuccess)
        throw std::runtime_error("failed to begin recording command buffer!");

    vk::RenderPassBeginInfo renderPassInfo{};
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = swapChainFrameBuffers[imageIndex];
    renderPassInfo.renderArea.offset = vk::Offset2D{0, 0};
    renderPassInfo.renderArea.extent = swapChainExtent;

    std::array<vk::ClearValue, 2> clearValues{};
    clearValues[0].color = vk::ClearColorValue{0.0f, 0.0f, 1.0f, 0.0f};
    clearValues[1].depthStencil = vk::ClearDepthStencilValue{1.0f, 0};
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    commandBuffer.beginRenderPass(&renderPassInfo, vk::SubpassContents::eInline);

    vk::Viewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(swapChainExtent.width);
    viewport.height = static_cast<float>(swapChainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    commandBuffer.setViewport(0, 1, &viewport);

    vk::Rect2D scissor{};
    scissor.offset = vk::Offset2D{0, 0};
    scissor.extent = swapChainExtent;

    commandBuffer.setScissor(0, 1, &scissor);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline2);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets[currentFrame], nullptr);
    //commandBuffer.draw(6, 1, 0, 0);

    // you need different pipelines for different shaders
    // after binding them every other operation will refer to that pipeline

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
    std::vector<vk::Buffer> buffers{vertexBuffer, instanceBuffer};

    // offsets are for your vertex buffers if you have your data
    // in the same buffer you will need to give the starting offset
    // of your second data
    std::vector<vk::DeviceSize> offsets{0, 0};
    commandBuffer.bindVertexBuffers(0, buffers, offsets);
    commandBuffer.bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint32);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets[currentFrame], nullptr);
    /* vk::Viewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(swapChainExtent.width);
    viewport.height = static_cast<float>(swapChainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    commandBuffer.setViewport(0, 1, &viewport);

    vk::Rect2D scissor{};
    scissor.offset = vk::Offset2D{0, 0};
    scissor.extent = swapChainExtent;

    commandBuffer.setScissor(0, 1, &scissor);*/
    commandBuffer.drawIndexed(indices.size(), 1, 0, 0, 0);
    //commandBuffer.draw(6, 1, 0, 0);
    commandBuffer.endRenderPass();

    try {
        commandBuffer.end();
    } catch (vk::SystemError err) {
        throw std::runtime_error("failed to record command buffer!");
    }
}

vk::ImageView Renderer::createImageView(vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags) {
    vk::ImageViewCreateInfo createInfo{};
    createInfo.image = image;
    createInfo.viewType = vk::ImageViewType::e2D;
    createInfo.format = format;

    vk::ComponentMapping mappings{
        vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity,
        vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity};
    createInfo.components = mappings;

    // base	MipmapLevel = 0, levelcount = 1, baseArrayLayer = 0, layerCount
    // =
    // 1
    vk::ImageSubresourceRange imageSubResource{aspectFlags,
        0, 1, 0, 1};
    createInfo.subresourceRange = imageSubResource;
    vk::ImageView imageView{};
    if (device.createImageView(&createInfo, nullptr, &imageView) != vk::Result::eSuccess)
        throw std::runtime_error("failed to create image views!");

    return imageView;
}

void Renderer::createTextureImageView() {
    textureImageView = createImageView(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits ::eColor);
}

void Renderer::createTextureSampler() {
    vk::PhysicalDeviceProperties deviceProperties = physicalDevice.getProperties();
    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = deviceProperties.limits.maxSamplerAnisotropy;
    samplerInfo.borderColor = vk::BorderColor::eFloatOpaqueBlack;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = vk::CompareOp::eAlways;
    samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;

    try {
        textureSampler = device.createSampler(samplerInfo);
    } catch (vk::Error& err) {
        std::cout << err.what();
    }
}

vk::Format Renderer::findDepthFormat() {
    return findSupportedFormat({vk::Format::eD32Sfloat,
                                   vk::Format::eD32SfloatS8Uint,
                                   vk::Format::eD24UnormS8Uint},
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment);
}

bool Renderer::hasStencilComponent(vk::Format format) {
    return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
}

vk::Format Renderer::findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features) {
    for (const auto& format : candidates) {
        vk::FormatProperties prop = physicalDevice.getFormatProperties(format);
        if (tiling == vk::ImageTiling::eLinear && (prop.linearTilingFeatures & features) == features)
            return format;
        else if (tiling == vk::ImageTiling::eOptimal && (prop.optimalTilingFeatures & features) == features)
            return format;
    }
    throw std::runtime_error("failed to find supported format!");
}

void Renderer::createDepthResources() {
    vk::Format depthFormat = findDepthFormat();
    createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage, depthImageMemory);
    depthImageView = createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth);
}

void Renderer::drawFrame() {
    device.waitForFences(1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
    uint32_t imageIndex{};
    vk::Result result = device.acquireNextImageKHR(swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

    if (result == vk::Result::eErrorOutOfDateKHR) {
        recreateSwapChain();
        return;
    } else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
        throw std::runtime_error("failed to aquire an image");

    device.resetFences(1, &inFlightFences[currentFrame]);

    commandBuffers[currentFrame].reset();
    recordCommandBuffer(commandBuffers[currentFrame], imageIndex);
    updateUniformBuffer(currentFrame);
    vk::SubmitInfo submitInfo{};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &imageAvailableSemaphores[currentFrame];
    vk::PipelineStageFlags waitStages{vk::PipelineStageFlagBits::eColorAttachmentOutput};
    submitInfo.pWaitDstStageMask = &waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &finishedRenderingSemaphores[currentFrame];

    if (graphicsQueue.submit(1, &submitInfo, inFlightFences[currentFrame]) != vk::Result::eSuccess)
        throw std::runtime_error("failed to submit command buffer!");

    vk::PresentInfoKHR presentInfo{};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &finishedRenderingSemaphores[currentFrame];
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapChain;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr;

    result = presentQueue.presentKHR(&presentInfo);

    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || framebufferResized) {
        framebufferResized = false;
        recreateSwapChain();
    } else if (result != vk::Result::eSuccess)
        throw std::runtime_error("failed to present swap chain image!");

    currentFrame = (currentFrame + 1) % maxFramesInFlight;
}

void Renderer::updateUniformBuffer(uint32_t currentImage) {
    static auto startTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    // these static variables are there to just keep info
    // from the previous frame so the image dosent get reset
    
    // look at sets where the camera is looking at and where the center is
    // rotate just rotates the mesh on some axis you set
    // perspective is the fov and some other things i dont understand

    // matrix multiplication is in reverse order
    // so you have  to first rotate it then scale it
    // even tho it will scale it and rotate it after
    // at the actual transformatation
   
   static glm::mat4 oldmodel{glm::mat4(1.0f)};
   static bool once = true;

   //float scale{0.02};
   static glm::mat4 oldview = glm::lookAt(glm::vec3(3.0f, 3.0f, 3.0), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    UniformBufferObject ubo{};

    if (glfwGetKey(window, GLFW_KEY_A)) {
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(180.f), glm::vec3(0.0f, 0.0f, 1.0f));
        //ubo.model = glm::scale(ubo.model, glm::vec3(scale, scale, scale));
        oldmodel = ubo.model;
    }
    if (glfwGetKey(window, GLFW_KEY_D)) {
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(-180.f), glm::vec3(0.0f, 0.0f, 1.0f));
        //ubo.model = glm::scale(ubo.model, glm::vec3(scale, scale, scale));
        oldmodel = ubo.model;
    }
    static float position{0};
    //ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.f), glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.model = oldmodel;
    //ubo.view = glm::lookAt(glm::vec3(3.0f, 3.0f, 0.0), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    if (glfwGetKey(window, GLFW_KEY_W)) {
        position -= 0.001;
        //ubo.model = glm::translate(ubo.model, glm::vec3(0,position,0));
        oldmodel = ubo.model;
        //std::cout << position << '\n';
        ubo.view = glm::lookAt(glm::vec3(position, position, 3.0), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        //oldview = ubo.view;
    }
       
    if (glfwGetKey(window, GLFW_KEY_S)) {
        position += 0.0004;
        ubo.view = glm::lookAt(glm::vec3(position, position, 3.0), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        oldview = ubo.view;
    }
    ubo.view = oldview;

    ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);
    ubo.proj[1][1] *= -1;

    memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
}

void Renderer::createSyncObjects() {
    vk::SemaphoreCreateInfo semaphoreInfo{};
    vk::FenceCreateInfo fenceInfo{};

    fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;

    try {
        for (size_t i{}; i < maxFramesInFlight; i++) {
            imageAvailableSemaphores.push_back(device.createSemaphore(semaphoreInfo));
            finishedRenderingSemaphores.push_back(device.createSemaphore(semaphoreInfo));
            inFlightFences.push_back(device.createFence(fenceInfo));
        }
    } catch (vk::Error& err) {
        throw std::runtime_error("failed to create sync objects");
    }
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
