#ifndef RENDERER
#define RENDERER
#include <optional>
#include <iostream>
#include <set>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <limits>
#include <algorithm>
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>


VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pCallback);

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT callback, const VkAllocationCallbacks* pAllocator);

class Renderer{
private:

#ifdef NDEBUG
	const bool  enableValidationLayers{ false };
#else
	const bool  enableValidationLayers{ true };
#endif

	//member var for window 
	const uint32_t WIDTH{ 800 };
	const uint32_t HEIGHT{ 600 };
	GLFWwindow* window{ nullptr };

	//validation layers
	std::vector<const char*> validationLayers{ "VK_LAYER_KHRONOS_validation" };
	const std::vector <const char*> deviceExtensions{ VK_KHR_SWAPCHAIN_EXTENSION_NAME };
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
		std::vector <vk::SurfaceFormatKHR> formats;
		std::vector <vk::PresentModeKHR> presentModes;
	};

	//member var for vulkan objects
	vk::Instance instance{};
	vk::PhysicalDevice physicalDevice{};
	vk::Device device{};
	vk::Queue graphicsQueue;
	vk::Queue presentQueue;
	vk::SurfaceKHR surface;
	vk::SwapchainKHR swapChain;
	std::vector<vk::Image> swapChainImages;
	vk::SurfaceFormatKHR swapChainImagesFormat;
	vk::Extent2D swapChainExtent;

public:
	void run();

private:
	//initializes the the window and vulkan lib
	void initWindow();
	void initVulkan();

	void pickPhysicalDevice();
	bool checkDeviceExtensionSupport(vk::PhysicalDevice device);
	bool isDeviceSuitable(vk::PhysicalDevice device);
	QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device);

	void mainLoop();
	void cleanup();

	//vulkan val layers fucntions
	bool checkValidationLayerSupport();
	std::vector<const char*> getRequiredExtensions();
	void setupDebugCallback();

	//functions to init vulkan objects
	void createInstance();
	void createLogicalDevice();
	void createSurface();

	//functions for swapChains
	SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice device);
	vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector <vk::SurfaceFormatKHR> availableFormats);
	vk::PresentModeKHR chooseSwapPresentMode(const std::vector <vk::PresentModeKHR> availavlePresentModes);
	vk::Extent2D chooseSwapExtend(const vk::SurfaceCapabilitiesKHR& capabilities);
	void createSwapChain();

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);
};
#endif


