#include <Windows.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>
#include <glm/gtx/string_cast.hpp>

// for smooth camera movement, lerp, slerp
#include <glm/gtx/compatibility.hpp>
#include <glm/gtx/quaternion.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_EASY_FONT_IMPLEMENTATION
#include <stb_easy_font.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <array>
#include <optional>
#include <set>
#include <unordered_map>

// openxr headers
#define XR_USE_PLATFORM_WIN32
#define XR_USE_GRAPHICS_API_VULKAN
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include <thread>

#define VK_CHECK(result, message) \
    if (result != VK_SUCCESS) { \
        throw std::runtime_error(message); \
    }

const uint32_t WIDTH = 1200;
const uint32_t HEIGHT = 900;

const std::string TEXTURE_PATH = "textures/apollo11_16bit.png";

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

inline void xr_check(XrResult result, const std::string& message)
{
    if (XR_FAILED(result)) {
        throw std::runtime_error(message + " (XR Result: " + std::to_string(result) + ")");
    }
}

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) func(instance, debugMessenger, pAllocator);
}

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    bool isComplete() { return graphicsFamily.has_value() && presentFamily.has_value(); }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec3 normal;
    glm::vec2 texCoord;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }
    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
        attributeDescriptions[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos) };
        attributeDescriptions[1] = { 1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord) };
        return attributeDescriptions;
    }
    bool operator==(const Vertex& other) const { return pos == other.pos && color == other.color && normal == other.normal && texCoord == other.texCoord; }
};

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const { return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec2>()(vertex.texCoord) << 1))); }
    };
}

struct UniformBufferObject {
    alignas(16) glm::mat4 model = glm::mat4(1.0f);
    alignas(16) glm::mat4 view[2];
    alignas(16) glm::mat4 proj[2];
    alignas(16) glm::vec4 cameraPosition[2];
    alignas(16) glm::vec4 minMaxZ = glm::vec4(-196.42f, 196.42f, 0.0f, 0.0f);
};

// openxr structs
struct XrViewData
{
	VkImage image;
	VkImageView imageView;
};

struct XrSwapchainData
{
    XrSwapchain swapchain;
    VkFormat format;
    uint32_t width;
    uint32_t height;
    std::vector<XrViewData> views;
    std::vector<VkFramebuffer> framebuffers;
};

class TerrainRenderer {
public:
    void run()
    {
        initWindow();
        initVulkan();
        mainLoopXR();
        cleanup();
    }

private:
    GLFWwindow* window;

    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;

    VkQueue graphicsQueue;
    VkQueue presentQueue;

    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;

    VkRenderPass renderPass;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    VkCommandPool commandPool;

    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;

    VkImage xrDepthImage;
    VkDeviceMemory xrDepthImageMemory;
    VkImageView xrDepthImageView;

    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    VkImageView textureImageView;
    VkSampler textureSampler;

    std::vector<Vertex> vertices;
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    uint32_t currentFrame = 0;

    bool framebufferResized = false;

	// openxr variables
	XrInstance xrInstance = XR_NULL_HANDLE;
	XrSystemId xrSystemId = XR_NULL_SYSTEM_ID;
	XrSession xrSession = XR_NULL_HANDLE;
	XrSpace xrAppSpace = XR_NULL_HANDLE;
	XrSessionState xrSessionState = XR_SESSION_STATE_IDLE;
	std::vector<XrSwapchainData> xrSwapchains;
    int64_t xrColorFormat;
	std::vector<XrViewConfigurationView> xrViewConfigs; 
	std::vector<XrView> xrViews; // per-eye view information, including pose and fov
	std::vector<XrCompositionLayerProjectionView> xrProjectionViews; // per-eye projection layer info

	// openxr function pointers, loaded at runtime
	PFN_xrGetVulkanInstanceExtensionsKHR xrGetVulkanInstanceExtensionsKHR = nullptr;
	PFN_xrGetVulkanDeviceExtensionsKHR xrGetVulkanDeviceExtensionsKHR = nullptr;
	PFN_xrGetVulkanGraphicsDeviceKHR xrGetVulkanGraphicsDeviceKHR = nullptr;

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan VR Terrain Renderer", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<TerrainRenderer*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void initVulkan()
    {
        initOpenXR();
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createCommandPool();

        // Create OpenXR swapchains and their associated resources first
        createXrSwapchains();

        // Create desktop window swapchain for mirror view
        createSwapChain();
        createImageViews();

        // The render pass must be compatible with both VR and desktop formats
        // (This should be handled by your dynamic format selection from the previous step)
        createRenderPass();

        // --- NEW CALL ORDER ---
        createXrDepthResources();   // Create VR depth buffer AFTER getting VR swapchain sizes
        createXrFramebuffers();     // Create VR framebuffers using the new VR depth buffer

        createDescriptorSetLayout();
        createGraphicsPipeline();

        createDepthResources();     // Create desktop depth buffer for the mirror window
        createFramebuffers();       // Create framebuffers for the mirror window

        createTextureImage();
        createTextureImageView();
        createTextureSampler();
        loadModel();
        createVertexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

    void mainLoopXR() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            // Handle OpenXR events
            XrEventDataBuffer eventData{ XR_TYPE_EVENT_DATA_BUFFER };
            while (xrPollEvent(xrInstance, &eventData) == XR_SUCCESS) {
                if (eventData.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
                    auto sessionStateChanged = *reinterpret_cast<XrEventDataSessionStateChanged*>(&eventData);
                    xrSessionState = sessionStateChanged.state;

                    switch (xrSessionState) {
                    case XR_SESSION_STATE_READY: {
                        XrSessionBeginInfo beginInfo{ XR_TYPE_SESSION_BEGIN_INFO };
                        beginInfo.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                        xr_check(xrBeginSession(xrSession, &beginInfo), "Failed to begin session");
                        break;
                    }
                    case XR_SESSION_STATE_STOPPING: {
                        xr_check(xrEndSession(xrSession), "Failed to end session");
                        break;
                    }
                    case XR_SESSION_STATE_EXITING:
                    case XR_SESSION_STATE_LOSS_PENDING:
                        glfwSetWindowShouldClose(window, true);
                        break;
                    }
                }
                eventData = { XR_TYPE_EVENT_DATA_BUFFER };
            }

            if (xrSessionState != XR_SESSION_STATE_READY && xrSessionState != XR_SESSION_STATE_FOCUSED) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            // OpenXR Frame Loop
            XrFrameState frameState{ XR_TYPE_FRAME_STATE };
            xr_check(xrWaitFrame(xrSession, nullptr, &frameState), "Failed to wait for XR frame");

            xr_check(xrBeginFrame(xrSession, nullptr), "Failed to begin XR frame");

            if (frameState.shouldRender) {
                // Get headset pose and projection matrices
                XrViewLocateInfo viewLocateInfo{ XR_TYPE_VIEW_LOCATE_INFO };
                viewLocateInfo.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                viewLocateInfo.displayTime = frameState.predictedDisplayTime;
                viewLocateInfo.space = xrAppSpace;

                XrViewState viewState{ XR_TYPE_VIEW_STATE };
                uint32_t viewCount = xrViews.size();
                xr_check(xrLocateViews(xrSession, &viewLocateInfo, &viewState, viewCount, &viewCount, xrViews.data()), "Failed to locate views");

                drawXRFrame();
            }

            // Submit frame to OpenXR compositor
            XrCompositionLayerProjection layer{ XR_TYPE_COMPOSITION_LAYER_PROJECTION };
            layer.space = xrAppSpace;
            layer.viewCount = xrProjectionViews.size();
            layer.views = xrProjectionViews.data();
            const XrCompositionLayerBaseHeader* layers[] = { (XrCompositionLayerBaseHeader*)&layer };

            XrFrameEndInfo endInfo{ XR_TYPE_FRAME_END_INFO };
            endInfo.displayTime = frameState.predictedDisplayTime;
            endInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
            endInfo.layerCount = (xrSessionState == XR_SESSION_STATE_FOCUSED) ? 1 : 0; // Only show layer if focused
            endInfo.layers = layers;

            xr_check(xrEndFrame(xrSession, &endInfo), "Failed to end XR frame");

            // Optional: Render mirror view to desktop window
            // drawFrame(); 
        }
        vkDeviceWaitIdle(device);
    }

    void cleanupSwapChain() {
        vkDestroyImageView(device, depthImageView, nullptr);
        vkDestroyImage(device, depthImage, nullptr);
        vkFreeMemory(device, depthImageMemory, nullptr);
        for (auto framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }
        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }
        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }

    void cleanup()
    {
        cleanupSwapChain();

        // cleanup OpenXR
        for (auto& swapchainData : xrSwapchains) {
            for (auto& fb : swapchainData.framebuffers) vkDestroyFramebuffer(device, fb, nullptr);
            for (auto& viewData : swapchainData.views) vkDestroyImageView(device, viewData.imageView, nullptr);
            xrDestroySwapchain(swapchainData.swapchain);
        }
        if (xrAppSpace != XR_NULL_HANDLE) xrDestroySpace(xrAppSpace);
        if (xrSession != XR_NULL_HANDLE) xrDestroySession(xrSession);
        if (xrInstance != XR_NULL_HANDLE) xrDestroyInstance(xrInstance);

        vkDestroyImageView(device, xrDepthImageView, nullptr);
        vkDestroyImage(device, xrDepthImage, nullptr);
        vkFreeMemory(device, xrDepthImageMemory, nullptr);

        // cleanup Vulkan
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
        }
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroySampler(device, textureSampler, nullptr);
        vkDestroyImageView(device, textureImageView, nullptr);
        vkDestroyImage(device, textureImage, nullptr);
        vkFreeMemory(device, textureImageMemory, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyDevice(device, nullptr);
        if (enableValidationLayers) DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device);
        cleanupSwapChain();
        createSwapChain();
        createImageViews();
        createDepthResources();
        createFramebuffers();
    }

    void initOpenXR() {
        // Application Info
        XrApplicationInfo appInfo{};
        strcpy_s(appInfo.applicationName, "Vulkan VR Terrain Renderer");
        appInfo.applicationVersion = 1;
        strcpy_s(appInfo.engineName, "Custom Engine");
        appInfo.engineVersion = 1;
        appInfo.apiVersion = XR_CURRENT_API_VERSION;

        // Request extensions
        const char* const requiredExtensions[] = { XR_KHR_VULKAN_ENABLE_EXTENSION_NAME };
        uint32_t extensionCount = sizeof(requiredExtensions) / sizeof(requiredExtensions[0]);

        XrInstanceCreateInfo createInfo{ XR_TYPE_INSTANCE_CREATE_INFO };
        createInfo.applicationInfo = appInfo;
        createInfo.enabledExtensionCount = extensionCount;
        createInfo.enabledExtensionNames = requiredExtensions;

        xr_check(xrCreateInstance(&createInfo, &xrInstance), "Failed to create OpenXR instance");

        // Get system
        XrSystemGetInfo systemGetInfo{ XR_TYPE_SYSTEM_GET_INFO };
        systemGetInfo.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
        xr_check(xrGetSystem(xrInstance, &systemGetInfo, &xrSystemId), "Failed to get OpenXR system");

        // Load function pointers
        xr_check(xrGetInstanceProcAddr(xrInstance, "xrGetVulkanInstanceExtensionsKHR", (PFN_xrVoidFunction*)(&xrGetVulkanInstanceExtensionsKHR)), "Failed to get xrGetVulkanInstanceExtensionsKHR");
        xr_check(xrGetInstanceProcAddr(xrInstance, "xrGetVulkanGraphicsDeviceKHR", (PFN_xrVoidFunction*)(&xrGetVulkanGraphicsDeviceKHR)), "Failed to get xrGetVulkanGraphicsDeviceKHR");
        xr_check(xrGetInstanceProcAddr(xrInstance, "xrGetVulkanDeviceExtensionsKHR", (PFN_xrVoidFunction*)(&xrGetVulkanDeviceExtensionsKHR)), "Failed to get xrGetVulkanDeviceExtensionsKHR");
    }

    void createInstance()
    {
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Vulkan VR Terrain Renderer";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_3;

        auto glfwExtensions = getRequiredExtensions();

        uint32_t xrExtCount = 0;
        xr_check(xrGetVulkanInstanceExtensionsKHR(xrInstance, xrSystemId, 0, &xrExtCount, nullptr),
            "Failed to get instance ext count");

        std::vector<char> xrExtBuffer(xrExtCount);
        xr_check(xrGetVulkanInstanceExtensionsKHR(xrInstance, xrSystemId,
            xrExtCount, &xrExtCount, xrExtBuffer.data()),
            "Failed to get instance exts");

        // Parse space-separated extension string into stable storage
        std::vector<std::string> xrExtStrings;
        {
            std::string xrExts(xrExtBuffer.data());
            std::stringstream ss(xrExts);
            std::string ext;
            while (ss >> ext) {
                xrExtStrings.push_back(ext);
            }
        }

        // build final list of const char* (safe: they point into xrExtStrings + glfwExtensions storage)
        std::vector<const char*> extensions;
        extensions.insert(extensions.end(), glfwExtensions.begin(), glfwExtensions.end());
        for (auto& s : xrExtStrings) {
            extensions.push_back(s.c_str());
        }

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
        }
        else {
            createInfo.enabledLayerCount = 0;

            createInfo.pNext = nullptr;
        }

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }
    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
    {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }

    void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    void createSurface() {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    /*void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }
        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }*/

    void pickPhysicalDevice() {
        // Ask OpenXR which VkPhysicalDevice it wants us to use.
        VkPhysicalDevice xrProvidedPhysicalDevice = VK_NULL_HANDLE;
        XrResult xrRes = xrGetVulkanGraphicsDeviceKHR(xrInstance, xrSystemId, instance, &xrProvidedPhysicalDevice);

        // If the call fails, give a clear error (this is the call you saw earlier in logs)
        xr_check(xrRes, "Failed to get physical device from OpenXR");

        if (xrProvidedPhysicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("OpenXR did not provide a valid physical device.");
        }

        // Verify the device returned by OpenXR is among Vulkan's enumerated devices.
        uint32_t deviceCount = 0;
        VK_CHECK(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr), "vkEnumeratePhysicalDevices failed to get count");
        if (deviceCount == 0) {
            throw std::runtime_error("No Vulkan physical devices found on this system.");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        VK_CHECK(vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()), "vkEnumeratePhysicalDevices failed to list devices");

        bool found = false;
        for (auto& d : devices) {
            if (d == xrProvidedPhysicalDevice) {
                found = true;
                break;
            }
        }
        if (!found) {
            // This is unexpected — log useful debugging info and abort.
            throw std::runtime_error("The physical device returned by OpenXR is not present in vkEnumeratePhysicalDevices. This indicates a mismatch between the Vulkan instance used and the runtime's expectation.");
        }

        // Accept the XR-provided device as 'physicalDevice' for the rest of the app.
        physicalDevice = xrProvidedPhysicalDevice;

        // Optional: check it's suitable for our needs (present support, features, queues).
        if (!isDeviceSuitable(physicalDevice)) {
            // If your app truly cannot run on the device OpenXR selected, it is better to fail with a clear message:
            throw std::runtime_error("The physical device selected by OpenXR is not suitable for this application's requirements!");
        }

        // Debug info: query basic properties (helps diagnosing -50)
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        std::cout << "Using physical device chosen by OpenXR: " << props.deviceName << " (API version "
            << VK_VERSION_MAJOR(props.apiVersion) << "." << VK_VERSION_MINOR(props.apiVersion)
            << "." << VK_VERSION_PATCH(props.apiVersion) << ")\n";
    }

    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        if (!indices.graphicsFamily.has_value() || !indices.presentFamily.has_value()) {
            throw std::runtime_error("Device does not support required queue families.");
        }

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };
        float queuePriority = 1.0f;
        for (uint32_t queueFamilyIndex : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
            queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.samplerAnisotropy = VK_TRUE;
        deviceFeatures.tessellationShader = VK_TRUE;
        deviceFeatures.fillModeNonSolid = VK_TRUE;

        // --- CRITICAL CHANGE: ENABLE MULTIVIEW FEATURE ---
        // We must explicitly enable the multiview feature for the logical device.
        VkPhysicalDeviceMultiviewFeatures multiviewFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES };
        multiviewFeatures.multiview = VK_TRUE; // Enable the feature

        VkDeviceCreateInfo createInfo{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        createInfo.pNext = &multiviewFeatures; // Chain the multiview feature struct
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.pEnabledFeatures = &deviceFeatures;

        // --- Extensions (OpenXR will automatically request VK_KHR_multiview if needed) ---
        uint32_t xrDevExtCount = 0;
        xrGetVulkanDeviceExtensionsKHR(xrInstance, xrSystemId, 0, &xrDevExtCount, nullptr);
        std::vector<char> xrDevExtBuffer(xrDevExtCount);
        xrGetVulkanDeviceExtensionsKHR(xrInstance, xrSystemId, xrDevExtCount, &xrDevExtCount, xrDevExtBuffer.data());
        std::vector<std::string> xrDevExtStrings;
        std::string xrExts(xrDevExtBuffer.data());
        std::stringstream ss(xrExts);
        std::string ext;
        while (ss >> ext) xrDevExtStrings.push_back(ext);

        std::vector<const char*> enabledDeviceExtensions;
        for (const char* e : deviceExtensions) enabledDeviceExtensions.push_back(e);
        for (auto& s : xrDevExtStrings) enabledDeviceExtensions.push_back(s.c_str());

        createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledDeviceExtensions.size());
        createInfo.ppEnabledExtensionNames = enabledDeviceExtensions.data();

        // --- Create logical device ---
        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);

        // --- Create OpenXR Session ---
        XrGraphicsBindingVulkanKHR graphicsBinding{ XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR };
        graphicsBinding.instance = instance;
        graphicsBinding.physicalDevice = physicalDevice;
        graphicsBinding.device = device;
        graphicsBinding.queueFamilyIndex = indices.graphicsFamily.value();
        graphicsBinding.queueIndex = 0;

        XrSessionCreateInfo sessionCreateInfo{ XR_TYPE_SESSION_CREATE_INFO };
        sessionCreateInfo.next = &graphicsBinding;
        sessionCreateInfo.systemId = xrSystemId;

        // This call will now succeed because the device has the required multiview feature enabled.
        xr_check(xrCreateSession(xrInstance, &sessionCreateInfo, &xrSession), "Failed to create OpenXR session");

        // --- Create reference space ---
        XrReferenceSpaceCreateInfo spaceCreateInfo{ XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
        spaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
        spaceCreateInfo.poseInReferenceSpace = { {0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f} };
        xr_check(xrCreateReferenceSpace(xrSession, &spaceCreateInfo, &xrAppSpace), "Failed to create OpenXR app space");
    }

    void createXrDepthResources() {
        // We need to find a size that is large enough for either eye.
        // Usually they are the same, but we take the max to be safe.
        uint32_t vrWidth = 0;
        uint32_t vrHeight = 0;
        for (const auto& sc : xrSwapchains) {
            if (sc.width > vrWidth) vrWidth = sc.width;
            if (sc.height > vrHeight) vrHeight = sc.height;
        }

        if (vrWidth == 0 || vrHeight == 0) {
            throw std::runtime_error("VR swapchain dimensions are zero, cannot create depth buffer.");
        }

        VkFormat depthFormat = findDepthFormat();
        createImage(vrWidth, vrHeight, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, xrDepthImage, xrDepthImageMemory);
        xrDepthImageView = createImageView(xrDepthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    void createXrSwapchains() {
        // --- Enumerate and select a supported swapchain format ---
        uint32_t formatCount = 0;
        xr_check(xrEnumerateSwapchainFormats(xrSession, 0, &formatCount, nullptr), "Failed to get swapchain format count");
        std::vector<int64_t> formats(formatCount);
        xr_check(xrEnumerateSwapchainFormats(xrSession, formatCount, &formatCount, formats.data()), "Failed to enumerate swapchain formats");

        // Choose a preferred format. R8G8B8A8_SRGB is common.
        xrColorFormat = formats[0]; // Default to the first one
        for (auto& format : formats) {
            if (format == VK_FORMAT_R8G8B8A8_SRGB) {
                xrColorFormat = format;
                break;
            }
        }

        // --- The rest of the function remains mostly the same, but uses xrColorFormat ---
        uint32_t viewCount = 0;
        xr_check(xrEnumerateViewConfigurationViews(xrInstance, xrSystemId, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, &viewCount, nullptr), "enum view config views count");
        xrViewConfigs.resize(viewCount, { XR_TYPE_VIEW_CONFIGURATION_VIEW });
        xr_check(xrEnumerateViewConfigurationViews(xrInstance, xrSystemId, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, viewCount, &viewCount, xrViewConfigs.data()), "enum view config views data");

        xrSwapchains.resize(viewCount);
        for (uint32_t i = 0; i < viewCount; ++i) {
            auto& swapchainData = xrSwapchains[i];
            const auto& viewConfig = xrViewConfigs[i];

            XrSwapchainCreateInfo swapchainCI{ XR_TYPE_SWAPCHAIN_CREATE_INFO };
            swapchainCI.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
            swapchainCI.format = xrColorFormat; // USE THE SELECTED FORMAT
            swapchainCI.sampleCount = 1;
            swapchainCI.width = viewConfig.recommendedImageRectWidth;
            swapchainCI.height = viewConfig.recommendedImageRectHeight;
            swapchainCI.faceCount = 1;
            swapchainCI.arraySize = 1;
            swapchainCI.mipCount = 1;

            xr_check(xrCreateSwapchain(xrSession, &swapchainCI, &swapchainData.swapchain), "Failed to create XR swapchain");

            swapchainData.width = swapchainCI.width;
            swapchainData.height = swapchainCI.height;
            swapchainData.format = (VkFormat)swapchainCI.format;

            uint32_t imageCount;
            xr_check(xrEnumerateSwapchainImages(swapchainData.swapchain, 0, &imageCount, nullptr), "enum swapchain images count");
            std::vector<XrSwapchainImageVulkanKHR> swapchainImages(imageCount, { XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR });
            xr_check(xrEnumerateSwapchainImages(swapchainData.swapchain, imageCount, &imageCount, (XrSwapchainImageBaseHeader*)swapchainImages.data()), "enum swapchain images data");

            swapchainData.views.resize(imageCount);
            for (uint32_t j = 0; j < imageCount; ++j) {
                swapchainData.views[j].image = swapchainImages[j].image;
                swapchainData.views[j].imageView = createImageView(swapchainImages[j].image, swapchainData.format, VK_IMAGE_ASPECT_COLOR_BIT);
            }
        }

        xrViews.resize(viewCount, { XR_TYPE_VIEW });
        xrProjectionViews.resize(viewCount);
        for (uint32_t i = 0; i < viewCount; ++i) {
            xrProjectionViews[i] = { XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW };
            xrProjectionViews[i].subImage.swapchain = xrSwapchains[i].swapchain;
        }
    }

    void createXrFramebuffers() {
        for (auto& swapchain : xrSwapchains) {
            swapchain.framebuffers.resize(swapchain.views.size());
            for (size_t i = 0; i < swapchain.views.size(); i++) {
                // CRITICAL CHANGE: Use the new VR-specific depth image view
                std::array<VkImageView, 2> attachments = {
                    swapchain.views[i].imageView,
                    xrDepthImageView // Use the correctly-sized VR depth buffer
                };

                VkFramebufferCreateInfo framebufferInfo{};
                framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                framebufferInfo.renderPass = renderPass;
                framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
                framebufferInfo.pAttachments = attachments.data();
                framebufferInfo.width = swapchain.width;
                framebufferInfo.height = swapchain.height;
                framebufferInfo.layers = 1;

                if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapchain.framebuffers[i]) != VK_SUCCESS) {
                    throw std::runtime_error("failed to create XR framebuffer!");
                }
            }
        }
    }

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;

        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());

        for (uint32_t i = 0; i < swapChainImages.size(); i++) {
            swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
        }
    }

    void createRenderPass() {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = (VkFormat)xrColorFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; // Correct for mirror window

        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = findDepthFormat();
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        // --- CRITICAL CHANGE: CONFIGURE RENDER PASS FOR MULTIVIEW ---
        // A view mask of 0b00000011 means that this subpass will render to
        // view 0 (left eye) AND view 1 (right eye) simultaneously.
        const uint32_t viewMask = 0b00000011;

        VkRenderPassMultiviewCreateInfo multiviewCI{ VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO };
        multiviewCI.subpassCount = 1;
        multiviewCI.pViewMasks = &viewMask;
        multiviewCI.correlationMaskCount = 0;
        multiviewCI.pCorrelationMasks = nullptr;

        std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };
        VkRenderPassCreateInfo renderPassInfo{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
        renderPassInfo.pNext = &multiviewCI; // Chain the multiview info
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.pImmutableSamplers = nullptr;
        // The UBO is needed in the TCS and TES. 
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;

        VkDescriptorSetLayoutBinding samplerLayoutBinding{};
        samplerLayoutBinding.binding = 1;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerLayoutBinding.pImmutableSamplers = nullptr;
        // CRITICAL CHANGE: The sampler is needed in the TES (for displacement)
        // AND now also in the Fragment Shader (for normal calculation).
        samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        std::array<VkDescriptorSetLayoutBinding, 2> bindings = { uboLayoutBinding, samplerLayoutBinding };
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }

    void createGraphicsPipeline() {
        // Load all four shader stages ---
        auto vertShaderCode = readFile("shaders/multi_tess.vert.spv");
        auto fragShaderCode = readFile("shaders/multi_tess.frag.spv");
        auto tescShaderCode = readFile("shaders/multi_tess.tesc.spv");
        auto teseShaderCode = readFile("shaders/multi_tess.tese.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);
        VkShaderModule tescShaderModule = createShaderModule(tescShaderCode);
        VkShaderModule teseShaderModule = createShaderModule(teseShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo tescShaderStageInfo{};
        tescShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        tescShaderStageInfo.stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
        tescShaderStageInfo.module = tescShaderModule;
        tescShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo teseShaderStageInfo{};
        teseShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        teseShaderStageInfo.stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
        teseShaderStageInfo.module = teseShaderModule;
        teseShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        std::array<VkPipelineShaderStageCreateInfo, 4> shaderStages = { vertShaderStageInfo, tescShaderStageInfo, teseShaderStageInfo, fragShaderStageInfo };

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        // Use a patch list for tessellation
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        // Tessellation state
        VkPipelineTessellationStateCreateInfo tessellationState{};
        tessellationState.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
        tessellationState.patchControlPoints = 4; // We are using quad patches

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_LINE; // Keep wireframe to see tessellation
        //rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        std::vector<VkDynamicState> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        // Define the push constant range (for multipass)
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(int); // We are pushing a single integer (viewIndex)

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1; // Specify we are using one push constant range
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange; // Point to our range

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineInfo.pStages = shaderStages.data();
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        // Point to the new tessellation state
        pipelineInfo.pTessellationState = &tessellationState;
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

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, teseShaderModule, nullptr);
        vkDestroyShaderModule(device, tescShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            std::array<VkImageView, 2> attachments = {
                swapChainImageViews[i],
                depthImageView
            };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    void createCommandPool() {

        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics command pool!");
        }
    }

    void createDepthResources() {
        VkFormat depthFormat = findDepthFormat();

        createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
        depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {

        for (VkFormat format : candidates) {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

            if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
                return format;
            }
            else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }

        throw std::runtime_error("failed to find supported format!");
    }

    VkFormat findDepthFormat() {
        return findSupportedFormat(
            { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
        );
    }

    bool hasStencilComponent(VkFormat format) {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
    }

    void createTextureImage() {
        int texWidth, texHeight, texChannels;

        // Load the 16bit image as single-channel (grayscale)
        stbi_us* pixels = stbi_load_16(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_grey);

        // load the 8-bit image as single-channel (grayscale)
        // stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_grey);

        if (!pixels) {
            throw std::runtime_error("failed to load 16-bit texture image!");
        }

        // Calculate image size for a single channel
        VkDeviceSize imageSize = texWidth * texHeight * sizeof(stbi_us); // For 16-bit single-channel
        // VkDeviceSize imageSize = texWidth * texHeight * sizeof(stbi_uc); // For 8-bit single-channel

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
        memcpy(data, pixels, static_cast<size_t>(imageSize));
        vkUnmapMemory(device, stagingBufferMemory);

        stbi_image_free(pixels);

        // a single-channel 16-bit Vulkan format 
        // VK_FORMAT_R16G16B16A16_UNORM is replaced with VK_FORMAT_R16_UNORM.
        // 'R16' means one Red channel with 16 bits.
        createImage(texWidth, texHeight, VK_FORMAT_R16_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);
        // 8bit single-channel format
        // createImage(texWidth, texHeight, VK_FORMAT_R8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

        transitionImageLayout(textureImage, VK_FORMAT_R16_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        // For 8-bit single-channel, use VK_FORMAT_R8_UNORM
        // transitionImageLayout(textureImage, VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
        // for 8-bit single-channel, use VK_FORMAT_R8_UNORM
        // copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
        transitionImageLayout(textureImage, VK_FORMAT_R16_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        // for 8-bit single-channel, use VK_FORMAT_R8_UNORM
        // transitionImageLayout(textureImage, VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createTextureImageView()
    {
        // 16-bit single-channel format
        textureImageView = createImageView(textureImage, VK_FORMAT_R16_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);

        // 8-bit single-channel format
        // textureImageView = createImageView(textureImage, VK_FORMAT_R8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    void createTextureSampler() {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

        if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture sampler!");
        }
    }

    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags) {

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspectFlags;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        VkImageView imageView;
        if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image view!");
        }

        return imageView;
    }

    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {

        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }

        vkBindImageMemory(device, image, imageMemory, 0);
    }

    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {

        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else {
            throw std::invalid_argument("unsupported layout transition!");
        }

        vkCmdPipelineBarrier(
            commandBuffer,
            sourceStage, destinationStage,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );

        endSingleTimeCommands(commandBuffer);
    }

    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {

        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = { 0, 0, 0 };
        region.imageExtent = {
            width,
            height,
            1
        };

        vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        endSingleTimeCommands(commandBuffer);
    }

    void loadModel()
    {
        const float halfX = 2100.0f;
        const float halfY = 13965.0f;

        const int grid = 128; // Number of patches in each direction
        const int gridY = 128 * 7; // Number of patches in each direction

        std::vector<Vertex> vertexList;

        for (int y = 0; y <= gridY; ++y) {
            for (int x = 0; x <= grid; ++x) {
                float u = static_cast<float>(x) / grid;
                float v = static_cast<float>(y) / gridY;

                float posX = -halfX + u * 2.0f * halfX;
                float posY = -halfY + v * 2.0f * halfY;

                vertexList.push_back({
                    { posX, posY, 0.0f },
                    { 1.0f, 1.0f, 1.0f },
                    { 0.0f, 0.0f, 1.0f },
                    { u, v }
                    });
            }
        }

        vertices.clear();
        for (int y = 0; y < gridY; ++y) {
            for (int x = 0; x < grid; ++x) {
                int i0 = (y) * (grid + 1) + (x);
                int i1 = (y) * (grid + 1) + (x + 1);
                int i2 = (y + 1) * (grid + 1) + (x);
                int i3 = (y + 1) * (grid + 1) + (x + 1);

                vertices.push_back(vertexList[i2]); // top-left
                vertices.push_back(vertexList[i3]); // top-right
                vertices.push_back(vertexList[i0]); // bottom-left
                vertices.push_back(vertexList[i1]); // bottom-right
            }
        }
    }

    void createVertexBuffer() {
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, vertices.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);
        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createUniformBuffers() {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);

            vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
        }
    }

    void createDescriptorPool() {
        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }

    void createDescriptorSets() {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = textureImageView;
            imageInfo.sampler = textureSampler;

            std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &bufferInfo;

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = descriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &imageInfo;

            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    VkCommandBuffer beginSingleTimeCommands() {

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer) {

        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {

        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommands(commandBuffer);
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {

        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createCommandBuffers() {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();
        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    void recordCommandBufferXR(VkCommandBuffer commandBuffer, uint32_t eyeIndex, uint32_t swapchainImageIndex) {
       /* VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }*/

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass; // Assuming a compatible render pass
        renderPassInfo.framebuffer = xrSwapchains[eyeIndex].framebuffers[swapchainImageIndex];
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = { xrSwapchains[eyeIndex].width, xrSwapchains[eyeIndex].height };

        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
        clearValues[1].depthStencil = { 1.0f, 0 };
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
        VkBuffer vertexBuffers[] = { vertexBuffer };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)xrSwapchains[eyeIndex].width;
        viewport.height = (float)xrSwapchains[eyeIndex].height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = { xrSwapchains[eyeIndex].width, xrSwapchains[eyeIndex].height };
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        // Push the view index for the current eye
        int viewIndex = eyeIndex;
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, 0, sizeof(int), &viewIndex);

        vkCmdDraw(commandBuffer, static_cast<uint32_t>(vertices.size()), 1, 0, 0);

        vkCmdEndRenderPass(commandBuffer);

      /*  if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }*/
    }

    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;

        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
        clearValues[1].depthStencil = { 1.0f, 0 };

        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        // Bind pipeline and buffers once for both eyes
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
        VkBuffer vertexBuffers[] = { vertexBuffer };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
        //vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

        // --- RENDER LEFT EYE ---
        {
            VkViewport viewport{};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = (float)swapChainExtent.width / 2.0f; // Half width
            viewport.height = (float)swapChainExtent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

            VkRect2D scissor{};
            scissor.offset = { 0, 0 };
            scissor.extent = { swapChainExtent.width / 2, swapChainExtent.height };
            vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

            // Push the view index for the left eye (0)
            int viewIndex = 0;
            vkCmdPushConstants(
                commandBuffer,
                pipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
                0,
                sizeof(int),
                &viewIndex);

            //vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
            vkCmdDraw(commandBuffer, static_cast<uint32_t>(vertices.size()), 1, 0, 0);
        }

        // --- RENDER RIGHT EYE ---
        {
            VkViewport viewport{};
            viewport.x = (float)swapChainExtent.width / 2.0f; // Start at the middle
            viewport.y = 0.0f;
            viewport.width = (float)swapChainExtent.width / 2.0f; // Half width
            viewport.height = (float)swapChainExtent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

            VkRect2D scissor{};
            scissor.offset = { (int32_t)(swapChainExtent.width / 2) + 50, 0 };
            scissor.extent = { swapChainExtent.width / 2, swapChainExtent.height };
            vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

            // Push the view index for the right eye (1)
            int viewIndex = 1;
            vkCmdPushConstants(
                commandBuffer,
                pipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
                0,
                sizeof(int),
                &viewIndex);

            //vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
            vkCmdDraw(commandBuffer, static_cast<uint32_t>(vertices.size()), 1, 0, 0);
        }

        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    void createSyncObjects() {

        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }
    }

    void updateUniformBuffer(uint32_t currentImage, uint32_t eyeIndex) {
        UniformBufferObject ubo{};

        // VIEW MATRIX from OpenXR
        const auto& pose = xrViews[eyeIndex].pose;
        glm::quat orientation(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
        glm::vec3 position(pose.position.x, pose.position.y, pose.position.z);
        glm::mat4 viewMatrix = glm::inverse(glm::translate(glm::mat4(1.0f), position) * glm::mat4_cast(orientation));

        // PROJECTION MATRIX from OpenXR
        const auto& fov = xrViews[eyeIndex].fov;
        float nearZ = 0.1f;
        float farZ = 10000.0f;
        float tanLeft = tanf(fov.angleLeft);
        float tanRight = tanf(fov.angleRight);
        float tanDown = tanf(fov.angleDown);
        float tanUp = tanf(fov.angleUp);
        float tanSumX = tanRight - tanLeft;
        float tanSumY = tanUp - tanDown;

        glm::mat4 projMatrix(0.0f);
        projMatrix[0][0] = 2.0f / tanSumX;
        projMatrix[1][1] = 2.0f / tanSumY;
        projMatrix[2][0] = (tanRight + tanLeft) / tanSumX;
        projMatrix[2][1] = (tanUp + tanDown) / tanSumY;
        projMatrix[2][2] = -farZ / (farZ - nearZ);
        projMatrix[2][3] = -1.0f;
        projMatrix[3][2] = -(farZ * nearZ) / (farZ - nearZ);
        projMatrix[1][1] *= -1; // Vulkan Y-flip

        // This is simplified, assuming your UBO is structured to accept these matrices
        ubo.view[eyeIndex] = viewMatrix;
        ubo.proj[eyeIndex] = projMatrix;
        ubo.cameraPosition[eyeIndex] = glm::vec4(position, 1.0f);

        // For the other eye, just copy the same data for now to avoid uninitialized data in shader
        int otherEye = (eyeIndex + 1) % 2;
        ubo.view[otherEye] = viewMatrix;
        ubo.proj[otherEye] = projMatrix;
        ubo.cameraPosition[otherEye] = glm::vec4(position, 1.0f);

        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    void drawXRFrame()
    {
        // wait for the previous frame's GPU work to finish, but do it only ONCE per frame
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        VkCommandBuffer commandBuffer = commandBuffers[currentFrame];
        vkResetCommandBuffer(commandBuffer, 0);

        // Begin recording the command buffer once for all eyes.
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        // keep track of the acquired image index for each eye
        std::vector<uint32_t> imageIndices(xrViews.size());

        // acquire images and record all rendering commands
        for (uint32_t i = 0; i < xrViews.size(); ++i) {
            auto& swapchain = xrSwapchains[i];

            // Acquire the swapchain image for the current eye
            XrSwapchainImageAcquireInfo acquireInfo{ XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
            xr_check(xrAcquireSwapchainImage(swapchain.swapchain, &acquireInfo, &imageIndices[i]), "Failed to acquire swapchain image");

            // Wait for that specific image to be ready
            XrSwapchainImageWaitInfo waitInfo{ XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
            waitInfo.timeout = XR_INFINITE_DURATION;
            xr_check(xrWaitSwapchainImage(swapchain.swapchain, &waitInfo), "Failed to wait for swapchain image");

            // Update the Uniform Buffer Object with matrices for the current eye
            updateUniformBuffer(currentFrame, i);

            // Record the drawing commands for this eye into our single, shared command buffer
            recordCommandBufferXR(commandBuffer, i, imageIndices[i]);
        }

        // End recording after all eyes have been recorded.
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }

        // Submit the command buffer to the GPU just ONCE with all the work.
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]),
            "Failed to submit draw command buffer for XR!");

        // Release images and update compositor data.
        // This happens after we've told the GPU what to do.
        for (uint32_t i = 0; i < xrViews.size(); ++i) {
            auto& swapchain = xrSwapchains[i];

            // Release the swapchain image for this eye
            XrSwapchainImageReleaseInfo releaseInfo{ XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
            xr_check(xrReleaseSwapchainImage(swapchain.swapchain, &releaseInfo), "Failed to release swapchain image");

            // Update projection view for the compositor
            xrProjectionViews[i].pose = xrViews[i].pose;
            xrProjectionViews[i].fov = xrViews[i].fov;
            xrProjectionViews[i].subImage.swapchain = swapchain.swapchain; // It's good practice to re-assign the swapchain handle
            xrProjectionViews[i].subImage.imageArrayIndex = 0;
            xrProjectionViews[i].subImage.imageRect.offset = { 0, 0 };
            xrProjectionViews[i].subImage.imageRect.extent = { (int32_t)swapchain.width, (int32_t)swapchain.height };
        }

        // Advance to the next frame in flight
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void drawFrame() {
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return;
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }
        updateUniformBuffer(currentFrame, 0);
        vkResetFences(device, 1, &inFlightFences[currentFrame]);
        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
        VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        VkSwapchainKHR swapChains[] = { swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;
        result = vkQueuePresentKHR(presentQueue, &presentInfo);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        }
        else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {

        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {

        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {

        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {

        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    bool isDeviceSuitable(VkPhysicalDevice device) {
        QueueFamilyIndices indices = findQueueFamilies(device);
        bool extensionsSupported = checkDeviceExtensionSupport(device);
        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        VkPhysicalDeviceMultiviewFeatures multiviewFeatures{};
        multiviewFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES;

        VkPhysicalDeviceFeatures2 supportedFeatures2{};
        supportedFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        supportedFeatures2.pNext = &multiviewFeatures;

        vkGetPhysicalDeviceFeatures2(device, &supportedFeatures2);

        if (multiviewFeatures.multiview == VK_FALSE) {
            return false; 
        }

        VkPhysicalDeviceFeatures supportedFeatures = supportedFeatures2.features;

        return indices.isComplete() &&
            extensionsSupported &&
            swapChainAdequate &&
            supportedFeatures.samplerAnisotropy &&
            supportedFeatures.tessellationShader &&
            supportedFeatures.fillModeNonSolid;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {

        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {

        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

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

    std::vector<const char*> getRequiredExtensions() {

        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    bool checkValidationLayerSupport() {

        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers) {
            bool layerFound = false;

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

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

        return VK_FALSE;
    }
};

int main() {
    TerrainRenderer app;

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}