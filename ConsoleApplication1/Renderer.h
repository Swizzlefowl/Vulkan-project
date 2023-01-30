#ifndef RENDERER_H
#define RENDERER_H

#include <iostream>
#include <set>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <optional>
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>


class Renderer{
private:
	//member var for window 
	const uint32_t WIDTH{ 800 };
	const uint32_t HEIGHT{ 600 };
	GLFWwindow* window{ nullptr };

	//validation layers
	std::vector<const char*> validationLayers{ "VK_LAYER_KHRONOS_validation" };

#ifdef NDEBUG
	const bool  enableValidationLayers{ false };
#else
	const bool  enableValidationLayers{ false };
#endif // NDEBUG

	struct QueueFamilyIndices {
		
		std::optional<uint32_t> graphicsFamily;

		bool isComplete() {
			return graphicsFamily.has_value();
		}
	};

	//member var for vulkan objects
	vk::Instance instance{};
	vk::PhysicalDevice physicalDevice{};
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

	//functions to init vulkan objects
	void createInstance();
};
#endif

