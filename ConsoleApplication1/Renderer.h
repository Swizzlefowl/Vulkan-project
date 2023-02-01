#ifndef RENDERER
#define RENDERER
#include <optional>
#include <iostream>
#include <set>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <cstring>
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
	VkDebugUtilsMessengerEXT callback;

	struct QueueFamilyIndices {
		
		std::optional<uint32_t> graphicsFamily;
		std::optional<uint32_t> presentFamily;

		bool isComplete() {
			return graphicsFamily.has_value() && presentFamily.has_value();
		}
	};

	//member var for vulkan objects
	vk::Instance instance{};
	vk::PhysicalDevice physicalDevice{};
	vk::Device device{};
	vk::Queue graphicsQueue;
	vk::Queue presentQueue;
	vk::SurfaceKHR surface;
	
public:
	void run();

private:
	//initializes the the window and vulkan lib
	void initWindow();
	void initVulkan();

	void pickPhysicalDevice();
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

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);
};
#endif


