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
    createRenderPass();
    createGraphicsPipeline();
    createFrameBuffers();
    createCommandPool();
    createCommandBuffer();
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

    device.destroyCommandPool(commandPool);
    device.destroyPipeline(graphicsPipeline);
    device.destroyPipelineLayout(pipelineLayout);
    for (auto& framebuffer : swapChainFrameBuffers)

        device.destroyFramebuffer(framebuffer);

    device.destroyRenderPass(renderPass);
    device.destroySwapchainKHR(swapChain);
    for (auto& imageView : swapChainImageViews)

        device.destroyImageView(imageView);

    device.destroySemaphore(imageAvailableSemaphore);
    device.destroySemaphore(finishedRenderingSemaphore);
    device.destroyFence(inFlightFence);
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

void Renderer::createFrameBuffers() {

    swapChainFrameBuffers.resize(swapChainImageViews.size());

    size_t i{0};
    for (auto& attachment : swapChainImageViews) {
        vk::FramebufferCreateInfo frameBufferInfo{};
        frameBufferInfo.renderPass = renderPass;
        frameBufferInfo.attachmentCount = 1;
        frameBufferInfo.pAttachments = &attachment;
        frameBufferInfo.width = swapChainExtent.width;
        frameBufferInfo.height = swapChainExtent.height;
        frameBufferInfo.layers = 1;

        if (device.createFramebuffer(&frameBufferInfo, nullptr, &swapChainFrameBuffers[i]) != vk::Result::eSuccess)
            throw std::runtime_error("failed to create a framebuffer!");
        i++;
        
    }
}

void Renderer::createGraphicsPipeline() {

    auto vertShaderCode{readFile("vertex.spv")};
    auto fragShaderCode{readFile("fragment.spv")};

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

    vk::PipelineShaderStageCreateInfo shaderStagesInfo[]{vertShaderStageInfo,
        fragShaderStageInfo};

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.vertexBindingDescriptionCount = 0;
    vertexInputInfo.pVertexBindingDescriptions = nullptr;
    vertexInputInfo.vertexAttributeDescriptionCount = 0;
    vertexInputInfo.pVertexAttributeDescriptions = nullptr;

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
    rasterizer.frontFace = vk::FrontFace::eClockwise;
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

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.setLayoutCount = 0;
    pipelineLayoutInfo.pSetLayouts = nullptr;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;

    if (device.createPipelineLayout(&pipelineLayoutInfo, nullptr, &pipelineLayout) != vk::Result::eSuccess)

        throw std::runtime_error("failed to create pipeline layout!");

    vk::GraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStagesInfo;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = nullptr;
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

    vk::AttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::SubpassDescription subpass{};
    subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    vk::SubpassDependency depedancy{};
    depedancy.srcSubpass = VK_SUBPASS_EXTERNAL;
    depedancy.dstSubpass = 0;
    depedancy.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    depedancy.srcAccessMask = vk::AccessFlagBits::eNone;
    depedancy.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    depedancy.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;

    vk::RenderPassCreateInfo renderPassInfo{};
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &depedancy;

    if (device.createRenderPass(&renderPassInfo, nullptr, &renderPass) != vk::Result::eSuccess)
        throw std::runtime_error("failed to create a render pass object!");
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
        std::runtime_error("failed to create command Pool!");
}

void Renderer::createCommandBuffer() {

    vk::CommandBufferAllocateInfo allocateInfo{};
    allocateInfo.commandPool = commandPool;
    allocateInfo.level = vk::CommandBufferLevel::ePrimary;
    allocateInfo.commandBufferCount = 1;

    if (device.allocateCommandBuffers(&allocateInfo, &commandBuffer) != vk::Result::eSuccess)
        std::runtime_error("failed to create command Buffer!");

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

    vk::ClearValue clearColor{{0.0f, 0.0f, 0.0f, 1.0f}};
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;

    commandBuffer.beginRenderPass(&renderPassInfo, vk::SubpassContents::eInline);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

    vk::Viewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(swapChainExtent.width);
    viewport.height = static_cast<float>(swapChainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    commandBuffer.setViewport( 0, 1, &viewport);

    vk::Rect2D scissor{};
    scissor.offset = vk::Offset2D{0, 0};
    scissor.extent = swapChainExtent;

    commandBuffer.setScissor( 0, 1, &scissor);
    commandBuffer.draw(3, 1, 0, 0);
    commandBuffer.endRenderPass();
  
    try {
        commandBuffer.end();
    } catch (vk::SystemError err) {
        throw std::runtime_error("failed to record command buffer!");
    }
}

void Renderer::drawFrame() {

    device.waitForFences(1, &inFlightFence, VK_TRUE, UINT64_MAX);
    device.resetFences(1, &inFlightFence);

    uint32_t imageIndex{};
    device.acquireNextImageKHR(swapChain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

    commandBuffer.reset();
    recordCommandBuffer(commandBuffer, imageIndex);

    vk::SubmitInfo submitInfo{};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &imageAvailableSemaphore;
    vk::PipelineStageFlags waitStages {vk::PipelineStageFlagBits::eColorAttachmentOutput};
    submitInfo.pWaitDstStageMask = &waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &finishedRenderingSemaphore;

    if (graphicsQueue.submit(1, &submitInfo, inFlightFence) != vk::Result::eSuccess)
        std::runtime_error("failed to submit command buffer!");

    vk::PresentInfoKHR presentInfo{};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &finishedRenderingSemaphore;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapChain;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr;

    presentQueue.presentKHR(&presentInfo);
}

void Renderer::createSyncObjects() {

    vk::SemaphoreCreateInfo semaphoreInfo{};
    vk::FenceCreateInfo fenceInfo{};
    fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;

    if (device.createSemaphore(&semaphoreInfo, nullptr, &imageAvailableSemaphore) != vk::Result::eSuccess)
        throw std::runtime_error("failed to create semaphore!");
    if (device.createSemaphore(&semaphoreInfo, nullptr, &finishedRenderingSemaphore) != vk::Result::eSuccess)
        throw std::runtime_error("failed to create semaphore!");
    if (device.createFence(&fenceInfo, nullptr, &inFlightFence) != vk::Result::eSuccess)
        throw std::runtime_error("failed to create semaphore!");
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
