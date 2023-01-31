#include "Renderer.h"

void Renderer::run(){
	
	initWindow();
	initVulkan();
	mainLoop();
	cleanup();
}

void Renderer::initWindow(){
	
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	window = glfwCreateWindow(WIDTH, HEIGHT, "hello Vulkan", nullptr, nullptr);
}

void Renderer::initVulkan(){
	
	createInstance();
	setupDebugCallback();
	createSurface();
    pickPhysicalDevice();
	createLogicalDevice();
}

void Renderer::pickPhysicalDevice(){

	uint32_t deviceCount {0};
	vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

	if (deviceCount == 0) {
		throw std::runtime_error("no Physical device found");
	}

	std::vector <vk::PhysicalDevice> devices{ deviceCount };

	instance.enumeratePhysicalDevices(&deviceCount, devices.data());

	for (const auto& device : devices) {
		
		if (isDeviceSuitable(device)) {
			std::cout << device.getProperties().deviceName;
			physicalDevice = device;
			break;
		}
	}

}

bool Renderer::isDeviceSuitable(vk::PhysicalDevice device){

	vk::PhysicalDeviceProperties deviceProperties{device.getProperties()};
	vk::PhysicalDeviceFeatures deviceFeatures{ device.getFeatures() };
	

	QueueFamilyIndices indices = findQueueFamilies(device);

		return deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu &&
			deviceFeatures.geometryShader && indices.isComplete();
}

Renderer::QueueFamilyIndices Renderer::findQueueFamilies(vk::PhysicalDevice device){

	QueueFamilyIndices indices;
	
	uint32_t queueFamilyCount{0};

	device.getQueueFamilyProperties(&queueFamilyCount, nullptr);

	std::vector<vk::QueueFamilyProperties> queueFamilies(queueFamilyCount);
	device.getQueueFamilyProperties(&queueFamilyCount, queueFamilies.data());

	vk::Bool32 presentSupport{ false };



	int i{0};
	for (const auto& queueFamily : queueFamilies) {
		if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
			indices.graphicsFamily = i;
		}

		device.getSurfaceSupportKHR(i, surface, &presentSupport);

		if (presentSupport) {

			indices.presentFamily = i;
		}

		if (indices.isComplete()) {
			break;
		}

		i++;
	}


	return indices;
}

void Renderer::mainLoop(){
	
	while (!glfwWindowShouldClose(window)) {
		
		glfwPollEvents();
	}
}

void Renderer::cleanup(){

	//instance.destroySurfaceKHR(surface);
	//device.destroy();

	if (enableValidationLayers) {
		DestroyDebugUtilsMessengerEXT(instance, callback, nullptr);
	}

	instance.destroy();
	glfwDestroyWindow(window);
	glfwTerminate();
}

bool Renderer::checkValidationLayerSupport(){

	uint32_t layerCount{};

	vk::enumerateInstanceLayerProperties(&layerCount, nullptr);

	std::vector<vk::LayerProperties> availableLayers(layerCount);
	vk::enumerateInstanceLayerProperties(&layerCount, availableLayers.data());

	for (const char* layerName : validationLayers) {

		bool layerFound{ false };

		for (const auto& layerProperties : availableLayers) {
			if (strcmp(layerName, layerProperties.layerName) == 0) {
				layerFound = true;
				break;
			}
		}
		if (!layerFound) {
			return false;
		}
	}
	return true;
}

std::vector<const char*> Renderer::getRequiredExtensions(){

	uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
	
}

void Renderer::setupDebugCallback(){

	if (!enableValidationLayers) return;

	auto createInfo = vk::DebugUtilsMessengerCreateInfoEXT(
		vk::DebugUtilsMessengerCreateFlagsEXT(),
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
		vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
		debugCallback,
		nullptr
	);

	// NOTE: Vulkan-hpp has methods for this, but they trigger linking errors...
	//instance->createDebugUtilsMessengerEXT(createInfo);
	//instance->createDebugUtilsMessengerEXTUnique(createInfo);

	// NOTE: reinterpret_cast is also used by vulkan.hpp internally for all these structs
	if (CreateDebugUtilsMessengerEXT(instance, reinterpret_cast<const VkDebugUtilsMessengerCreateInfoEXT*>(&createInfo), nullptr, &callback) != VK_SUCCESS) {
		throw std::runtime_error("failed to set up debug callback!");
	}
}

void Renderer::createInstance(){
	
	if (enableValidationLayers && !checkValidationLayerSupport()) {
		throw std::runtime_error("validation layers requested, but not available!");
	}

	vk::ApplicationInfo appInfo{};
	appInfo.sType = vk::StructureType::eApplicationInfo;
	appInfo.pApplicationName = "hello Triangle";
	appInfo.applicationVersion = 1.0;
	appInfo.pEngineName = "no Engine";
	appInfo.engineVersion = 1.0;
	appInfo.apiVersion = VK_API_VERSION_1_0;

	vk::InstanceCreateInfo createInfo{};
	createInfo.sType = vk::StructureType::eInstanceCreateInfo;
	createInfo.pApplicationInfo = &appInfo;

	uint32_t glfwExtensionCount{ 0 };
	const char** glfwExtension{ nullptr };

	glfwExtension = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	auto extensions = getRequiredExtensions();
	createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
	createInfo.ppEnabledExtensionNames = extensions.data();

	if (enableValidationLayers) {
		createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		createInfo.ppEnabledLayerNames = validationLayers.data();
	}
	else {
		createInfo.enabledLayerCount = 0;
	}

	if (vk::createInstance(&createInfo, nullptr, &instance) != vk::Result::eSuccess) {
		throw std::runtime_error("failed to create instance!");
	}

}

void Renderer::createLogicalDevice(){

	QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

	std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
	std::set<uint32_t> uniqueQueueFamilies = { 
		indices.graphicsFamily.value(), indices.presentFamily.value() 
	};
	float queuePriority = 1.0f;

	for (uint32_t queueFamily : uniqueQueueFamilies) {
		vk::DeviceQueueCreateInfo queueCreateInfo{};
		queueCreateInfo.sType = vk::StructureType::eDeviceQueueCreateInfo;
		queueCreateInfo.queueFamilyIndex = indices.graphicsFamily.value();
		queueCreateInfo.queueCount = 1;
		queueCreateInfo.pQueuePriorities = &queuePriority;
		queueCreateInfos.push_back(queueCreateInfo);
	}

	vk::PhysicalDeviceFeatures deviceFeatures{};

	vk::DeviceCreateInfo createInfo{};
	createInfo.sType = vk::StructureType::eDeviceCreateInfo;
	createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
	createInfo.pQueueCreateInfos = queueCreateInfos.data();
	

	createInfo.pEnabledFeatures = &deviceFeatures;

	if (enableValidationLayers) {
		createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		createInfo.ppEnabledLayerNames = validationLayers.data();
	}
	else {
		createInfo.enabledLayerCount = 0;
	}

	if (physicalDevice.createDevice(&createInfo, nullptr,&device) != vk::Result::eSuccess) {
		throw std::runtime_error("failed to create logical device!");
	}

	device.getQueue(indices.graphicsFamily.value(), 0, &graphicsQueue);
	device.getQueue(indices.presentFamily.value(), 0, &presentQueue);
}

void Renderer::createSurface(){
	VkSurfaceKHR surf;
	if (glfwCreateWindowSurface(instance, window, nullptr, &surf) != VK_SUCCESS) {
		throw std::runtime_error("failed to create window surface!");
	}

	surface = vk::SurfaceKHR{ surf };
}

VkResult CreateDebugUtilsMessengerEXT(vk::Instance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pCallback){
	
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pCallback);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
	
}

void DestroyDebugUtilsMessengerEXT(vk::Instance instance, VkDebugUtilsMessengerEXT callback, const VkAllocationCallbacks* pAllocator){

	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, callback, pAllocator);
	}
}

