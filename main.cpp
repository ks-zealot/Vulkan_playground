#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <glm/glm.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>

const int MICROSECONDS_IN_SECOND = 1000000;

const int WIDTH = 800;
const int HEIGHT = 600;

struct Vertex {
  glm::vec2 pos;
  glm::vec3 color;
  static VkVertexInputBindingDescription getBindingDescription() {
    VkVertexInputBindingDescription bindingDescription = {};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return bindingDescription;
  }
  static std::array<VkVertexInputAttributeDescription, 2>
  getAttributeDescriptions() {
    std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions = {};
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, pos);
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, color);
    return attributeDescriptions;
  }
};

struct Tetramino {
  std::vector<Vertex> vertices;
  std::vector<uint16_t> indices;
  int sizeX;
  int sizeY;
  static Tetramino *clone(Tetramino *origin) {
    Tetramino *t = new Tetramino();
    t->vertices = *new std::vector<Vertex>();
    t->indices = *new std::vector<uint16_t>();
    std::copy(origin->vertices.begin(), origin->vertices.end(),
              std::back_inserter(t->vertices));
    std::copy(origin->indices.begin(), origin->indices.end(),
              std::back_inserter(t->indices));
    t->sizeX = origin->sizeX;
    t->sizeY = origin->sizeY;
    return t;
  }
};

float size =
    0.3f; // размер квадратика. так же используется для клонирования квадратика
float edge; // размер стороны квадратика после ресайза
int sizeXInPixel;
int sizeYInPixel;
// tod3 переделать на алгоритме триангуляции
// todo переделать с учетом того что пространство неравномерно
// todo убрать хардкод, сделать генерацию n polymino

Tetramino T = {{
                   {{size * 0, size * 0}, {1.0f, 0.0f, 0.0f}},
                   {{size * 1, size * 0}, {0.0f, 1.0f, 0.0f}},
                   {{size * 1, size * 1}, {0.0f, 0.0f, 1.0f}},
                   {{size * 2, size * 1}, {1.0f, 1.0f, 0.0f}},
                   {{size * 2, size * 2}, {0.0f, 1.0f, 1.0f}},
                   {{size * 1, size * 2}, {1.0f, 0.0f, 1.0f}},
                   {{size * 1, size * 3}, {1.0f, 1.0f, 1.0f}},
                   {{size * 0, size * 3}, {0.0f, 0.0f, 0.0f}},

               },
               {0, 1, 2, 2, 3, 4, 0, 4, 5, 0, 5, 6, 0, 6, 7},
               3,
               2};

Tetramino I = {{
                   {{size * 0, size * 0}, {1.0f, 0.0f, 0.0f}},
                   {{size * 1, size * 0}, {0.0f, 1.0f, 0.0f}},
                   {{size * 1, size * 4}, {0.0f, 0.0f, 1.0f}},
                   {{size * 0, size * 4}, {1.0f, 1.0f, 0.0f}},

               },
               {0, 1, 2, 2, 3, 0},
               1,
               4};

Tetramino O = {{
                   {{size * 0, size * 0}, {1.0f, 0.0f, 0.0f}},
                   {{size * 2, size * 0}, {0.0f, 1.0f, 0.0f}},
                   {{size * 2, size * 2}, {0.0f, 0.0f, 1.0f}},
                   {{size * 0, size * 2}, {1.0f, 1.0f, 0.0f}},

               },
               {0, 1, 2, 2, 3, 0},
               2,
               2};

Tetramino J = {{
                   {{size * 0, size * 0}, {1.0f, 0.0f, 0.0f}},
                   {{size * 2, size * 0}, {0.0f, 1.0f, 0.0f}},
                   {{size * 2, size * -1}, {0.0f, 0.0f, 1.0f}},
                   {{size * 3, size * -1}, {1.0f, 0.0f, 1.0f}},
                   {{size * 3, size * 1}, {0.0f, 1.0f, 1.0f}},
                   {{size * 0, size * 1}, {1.0f, 1.0f, 0.0f}},

               },
               {0, 1, 4, 1, 2, 4, 2, 3, 4, 4, 5, 0},
               3,
               2};

Tetramino L = {{{{size * 0, size * 0}, {1.0f, 0.0f, 0.0f}},
                {{size * 3, size * 0}, {0.0f, 1.0f, 0.0f}},
                {{size * 3, size * 1}, {0.0f, 0.0f, 1.0f}},
                {{size * 1, size * 1}, {0.0f, 1.0f, 1.0f}},
                {{size * 1, size * 2}, {1.0f, 1.0f, 0.0f}},
                {{size * 0, size * 2}, {1.0f, 0.0f, 1.0f}}},
               {0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5},
               3,
               2};

Tetramino S = {{{{size * 0, size * 0}, {1.0f, 0.0f, 0.0f}},
                {{size * 2, size * 0}, {0.0f, 0.0f, 1.0f}},
                {{size * 2, size * 1}, {0.0f, 1.0f, 0.0f}},
                {{size * 3, size * 1}, {1.0f, 1.0f, 0.0f}},
                {{size * 3, size * 2}, {0.0f, 1.0f, 1.0f}},
                {{size * 1, size * 2}, {1.0f, 0.0f, 1.0f}},
                {{size * 1, size * 1}, {1.0f, 1.0f, 1.0f}},
                {{size * 0, size * 1}, {0.0f, 0.0f, 0.0f}}},
               {
                   0, 1, 2, 0, 2, 4, 2, 3, 4, 0, 4, 6, 4, 5, 6, 6, 7, 0,
               },
               3,
               2};

Tetramino Z = {{
                   {{size * 0, size * 0}, {1.0f, 0.0f, 0.0f}},
                   {{size * 2, size * 0}, {0.0f, 1.0f, 0.0f}},
                   {{size * 2, size * -1}, {0.0f, 0.0f, 1.0f}},
                   {{size * 3, size * -1}, {1.0f, 0.0f, 1.0f}},
                   {{size * 3, size * -2}, {1.0f, 1.0f, 0.0f}},
                   {{size * 1, size * -2}, {0.0f, 1.0f, 1.0f}},
                   {{size * 1, size * -1}, {1.0f, 1.0f, 1.0f}},
                   {{size * 0, size * -1}, {0.0f, 0.0f, 0.0f}},

               },
               {0, 7, 6, 6, 5, 4, 0, 6, 4, 4, 3, 2, 4, 2, 0, 2, 1, 0},
               3,
               2

};
std::vector<Tetramino> tetraminos = {O}; // {T, L, S, Z, I, O, J};
std::vector<Tetramino> blob = std::vector<Tetramino>();
float onePixelX = 0.0f;
float onePixelY = 0.0f;

int step = 100; // сколько мы должны пройти по нажатию, в пикселях
double speed = 5.0;
// скорость. измеряется в step в секунду. если два то
// ускоряется в два раза. еслт 0.5 то замедляется

// todo переделать на функции. тут довольно много кода который чтото делает с
// вертексами в функции. напрашиваются лямбды
glm::vec3 UP;
glm::vec3 DOWN;
glm::vec3 LEFT;
glm::vec3 RIGHT;
glm::vec3 INIT = glm::vec3(0.0f, -1.0f, 0.0f);
const std::vector<const char *> validationLayers = {
    "VK_LAYER_LUNARG_standard_validation"};
const std::vector<const char *> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

VkResult CreateDebugReportCallbackEXT(
    VkInstance instance, const VkDebugReportCallbackCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugReportCallbackEXT *pCallback) {
  auto func = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(
      instance, "vkCreateDebugReportCallbackEXT");
  if (func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pCallback);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

void DestroyDebugReportCallbackEXT(VkInstance instance,
                                   VkDebugReportCallbackEXT callback,
                                   const VkAllocationCallbacks *pAllocator) {
  auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(
      instance, "vkDestroyDebugReportCallbackEXT");
  if (func != nullptr) {
    func(instance, callback, pAllocator);
  }
}

static std::vector<char> loadVertex(const std::string &filename) {
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
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

struct QueueFamilyIndices {
  int graphicsFamily = -1;
  int presentFamily = -1;
  bool isComplete() { return graphicsFamily >= 0 && presentFamily >= 0; }
};
struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
};

class HelloTriangleApplication {

public:
  void run() {
    std::cout << "init window" << std::endl;
    initWindow();
    std::cout << "init vulkan" << std::endl;
    initVulkan();
    std::cout << "loop" << std::endl;
    mainLoop();
    std::cout << "cleanup" << std::endl;
    cleanup();
  }

private:
  ::VkInstance vkInstance;
  ::GLFWwindow *window;

  VkDevice device;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDebugReportCallbackEXT callback;
  VkQueue graphicsQueue;
  VkSurfaceKHR surface;
  VkQueue presentQueue;
  VkSwapchainKHR swapChain;
  VkFormat swapChainImageFormat;
  VkExtent2D swapChainExtent;
  VkPipelineLayout pipelineLayout;
  VkRenderPass renderPass;
  VkPipeline graphicsPipeline;
  VkCommandPool commandPool;
  VkSemaphore imageAvailableSemaphore;
  VkSemaphore renderFinishedSemaphore;
  VkBuffer vertexBuffer;
  VkMemoryRequirements memRequirements;
  VkDeviceMemory vertexBufferMemory;
  VkBuffer indexBuffer;
  VkDeviceMemory indexBufferMemory;

  std::vector<VkImage> swapChainImages;
  std::vector<VkImageView> swapChainImageViews;
  std::vector<VkFramebuffer> swapChainFramebuffers;
  std::vector<VkCommandBuffer> commandBuffers;

  unsigned int width = WIDTH;
  unsigned int height = HEIGHT;

  Tetramino *currentTetramino;

  std::chrono::milliseconds mDeltaTime;
  typedef std::chrono::high_resolution_clock Clock;
  Clock::time_point mLastEndTime = Clock::now();
  Clock::time_point mLastMeasureTime = Clock::now();
  typedef std::chrono::duration<long, std::ratio<1, 60>> sixtieths_of_a_sec;
  static constexpr auto kMaxDeltatime = sixtieths_of_a_sec{1};
  unsigned int SPF;

  int lastKeyPressed = -1;

  unsigned int nbFrames = 0;
  unsigned int frames;

  unsigned int stepCount = 0;

  bool block = false;

  static void onWindowResized(GLFWwindow *window, int width, int height) {
    if (width == 0 || height == 0)
      return;

    HelloTriangleApplication *app =
        reinterpret_cast<HelloTriangleApplication *>(
            glfwGetWindowUserPointer(window));
    app->recreateSwapChain();
    app->initOnePixelSize();
    app->saveNewSize(width, height);
  }

  static void keyCallBack(GLFWwindow *w, int key, int scancode, int action,
                          int mods) {
    HelloTriangleApplication *app =
        reinterpret_cast<HelloTriangleApplication *>(
            glfwGetWindowUserPointer(w));

    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
      app->handleKeyPress(key);
    } else if (action == GLFW_RELEASE) {
      app->releaseKey();
    }
  }

  void initWindow() {
    ::glfwInit();
    ::glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = ::glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    initTetramino();
    initOnePixelSize();
    setTetramino();
    setSquareWidthHeigth();
    glfwSetWindowUserPointer(window, this);
    glfwSetWindowSizeCallback(window,
                              HelloTriangleApplication::onWindowResized);
  }

  void initTetramino() {
    int randomIndex = rand() % tetraminos.size();
    currentTetramino = Tetramino::clone(&tetraminos[randomIndex]);
  }

  void setTetramino() {
    float min = 1.0f;
    for (std::vector<Vertex>::iterator it = currentTetramino->vertices.begin();
         it != currentTetramino->vertices.end(); ++it) {
      Vertex &v = *it;
      min = std::min(min, std::abs(-1.0f - v.pos.y));
    }
    for (std::vector<Vertex>::iterator it = currentTetramino->vertices.begin();
         it != currentTetramino->vertices.end(); ++it) {
      Vertex &v = *it;
      v.pos = v.pos - glm::vec2(0.0f, min);
    }
  }

  void initOnePixelSize() {
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    initOnePixelSize(width, height);
  }

  void initOnePixelSize(int width, int height) {
    onePixelX = 1.0f / (width / 2);
    onePixelY = 1.0f / (height / 2);
    UP = glm::vec3(0.0f, -onePixelY, 0.0f);
    DOWN = glm::vec3(0.0f, onePixelY, 0.0f);
    LEFT = glm::vec3(-onePixelX, 0.0f, 0.0f);
    RIGHT = glm::vec3(onePixelX, 0.0f, 0.0f);
  }

  void setSquareWidthHeigth() {
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    setSquareWidthHeigth(width, height);
  }
  // todo повыкидывать лишние функции
  void resize() {
    double ratio = (double)width / (double)height;
    glm::mat4 scaleMatrix;
    double val;
    if (ratio > 1) {
      val = ratio;
      scaleMatrix = glm::scale(glm::mat4(), glm::vec3(1, val, 1));
    } else {
      val = 1 / ratio;
      scaleMatrix = glm::scale(glm::mat4(), glm::vec3(val, 1, 1));
    }
    glm::vec2 centroid = getCentroid();
    for (std::vector<Vertex>::iterator it = currentTetramino->vertices.begin();
         it != currentTetramino->vertices.end(); ++it) {
      Vertex &v = *it;
      v.pos = v.pos - centroid;
      glm::vec4 vector = glm::vec4(v.pos, 0.0f, 1.0f);
      glm::vec4 transformedVector = scaleMatrix * vector;
      v.pos = glm::vec2(transformedVector);
      v.pos = v.pos + centroid;
    }
  }

  void releaseKey() {
    if (!block) {
      lastKeyPressed = -1;
    }
  }

  /*Функция, стирает нижний ряд в тетрамино
   * как это работает - находим точки с максимальным y
   * к y этих точек прибавляем сторону квадратика
   * с максимальным потому что ось y инвертирована
   * после этого все точки спускаем вниз на сторону квадратика
   * сторону квадратика каждый раз высчитывается заново что наверное нездоров
   */
  void flat() {
    float maxY = -1.0f;

    for (std::vector<Vertex>::iterator it = currentTetramino->vertices.begin();
         it != currentTetramino->vertices.end(); ++it) {
      Vertex &v = *it;
      maxY = std::max(maxY, v.pos.y);
    }

    for (std::vector<Vertex>::iterator it = currentTetramino->vertices.begin();
         it != currentTetramino->vertices.end(); ++it) {

      Vertex &v = *it;
      if (v.pos.y == maxY) {
        v.pos.y = v.pos.y - edge;
      }
    }
    for (std::vector<Vertex>::iterator it = currentTetramino->vertices.begin();
         it != currentTetramino->vertices.end(); ++it) {
      Vertex &v = *it;
      v.pos.y = v.pos.y + edge;
    }
    currentTetramino->sizeY--;
    moveVertex();
  }

  void moveOnePixel() {

    glm::mat4 translationMatrix;
    switch (lastKeyPressed) {
    case GLFW_KEY_UP:
    case GLFW_KEY_W:
      translationMatrix = glm::translate(glm::mat4(), UP);
      break;
    case GLFW_KEY_DOWN:
    case GLFW_KEY_S:
      translationMatrix = glm::translate(glm::mat4(), DOWN);
      break;
    case GLFW_KEY_LEFT:
    case GLFW_KEY_A:
      translationMatrix = glm::translate(glm::mat4(), LEFT);
      break;
    case GLFW_KEY_RIGHT:
    case GLFW_KEY_D:
      translationMatrix = glm::translate(glm::mat4(), RIGHT);
      break;
    default:
      return;
    }

    for (std::vector<Vertex>::iterator it = currentTetramino->vertices.begin();
         it != currentTetramino->vertices.end(); ++it) {
      Vertex &v = *it;
      glm::vec4 vector = glm::vec4(v.pos, 0.0f, 1.0f);
      glm::vec4 transformedVector = translationMatrix * vector;
      v.pos = glm::vec2(transformedVector);
    }
    moveVertex();
  }

  void handleKeyPress(int key) {
    if (block) {
      return;
    }
    if (lastKeyPressed == -1) {
      if (key == GLFW_KEY_ENTER || key == GLFW_KEY_KP_ENTER) {
        summon();
      } else if (key == GLFW_KEY_SPACE) {
        rotateTetramino();
        setSquareWidthHeigth();
        resize();
        moveVertex();
      } else if (key == GLFW_KEY_LEFT_CONTROL ||
                 key == GLFW_KEY_RIGHT_CONTROL) {
        summonNextPentamino();
      } else if (key == GLFW_KEY_F) {
        flat();
      } else {
        lastKeyPressed = key;
        moveVertex();
      }
    }
  }

  void summonNextPentamino() {
    lastKeyPressed = GLFW_KEY_S;
    block = true;
  }

  void summon() {
    storeTetramino();
    removeBlocks();
    initTetramino();
    setTetramino();
    setSquareWidthHeigth();
    recreateObject();
    moveVertex();
  }

  void storeTetramino() { blob.push_back(*currentTetramino); }

  glm::vec2 getCentroid() {
    glm::vec2 centroid = glm::vec2(0.0f, 0.0f);
    for (std::vector<Vertex>::iterator it = currentTetramino->vertices.begin();
         it != currentTetramino->vertices.end(); ++it) {
      Vertex &v = *it;
      centroid.x = centroid.x + v.pos.x;
      centroid.y = centroid.y + v.pos.y;
    }
    centroid.x = centroid.x / currentTetramino->vertices.size();
    centroid.y = centroid.y / currentTetramino->vertices.size();
    return centroid;
  }

  // todo сделать через честное нахождение rotation matrix
  void rotateTetramino() {
    glm::vec2 centroid = getCentroid();
    for (std::vector<Vertex>::iterator it = currentTetramino->vertices.begin();
         it != currentTetramino->vertices.end(); ++it) {
      Vertex &v = *it;
      glm::mat4 translate = glm::translate(
          glm::mat4(), glm::vec3(-centroid.x, -centroid.y, 0.0f));
      glm::mat4 rotate =
          glm::rotate(glm::mat4(), 90.0f, glm::vec3(1.0f, 0.0f, 0.0f));
      glm::mat2 rotationMatrix = glm::mat2x2(0, 1, -1, 0);
      glm::vec4 vector = glm::vec4(v.pos, 0.0f, 1.0f);
      glm::vec4 transformedVector = translate * vector;
      v.pos = glm::vec2(transformedVector);
      v.pos = rotationMatrix * v.pos;
      translate =
          glm::translate(glm::mat4(), glm::vec3(centroid.x, centroid.y, 0.0f));
      vector = glm::vec4(v.pos, 0.0f, 1.0f);
      transformedVector = translate * vector;
      v.pos = glm::vec2(transformedVector);
    }
  }

  void saveNewSize(int width, int height) {
    this->width = width;
    this->height = height;
  }

  void setSquareWidthHeigth(double width, double height) {
    // y -> width
    // x -> heigth
    if (height == width) {
      return;
    }
    double ratio = width / height;
    float minX = 1.0f;
    float maxX = 0.0f;
    float minY = 1.0f;
    float maxY = 0.0f;
    for (std::vector<Vertex>::iterator it = currentTetramino->vertices.begin();
         it != currentTetramino->vertices.end(); ++it) {
      Vertex &v = *it;
      double val;
      if (ratio > 1) {
        val = 1 / ratio;
        v.pos = v.pos * glm::vec2(val, 1);
      } else {
        val = ratio;
        v.pos = v.pos * glm::vec2(1, val);
      }
    }
    for (std::vector<Vertex>::iterator it = currentTetramino->vertices.begin();
         it != currentTetramino->vertices.end(); ++it) {
      Vertex v = *it;
      minX = std::min(minX, v.pos.x);
      maxX = std::max(maxX, v.pos.x);
      minY = std::min(minY, std::abs(v.pos.y));
      maxY = std::max(maxY, std::abs(v.pos.y));
    }
    sizeXInPixel = ((maxX - minX) / currentTetramino->sizeX) / onePixelX;
    sizeYInPixel = ((maxY - minY) / currentTetramino->sizeY) / onePixelY;
    edge = (maxY - minY) / (float)currentTetramino->sizeY;
  }

  void initVulkan() {
    //инициация библиотеки
    createInstance();
    setupDebugCallback();
    // готовим поверхность на которой будем рисовать
    createSurface();
    // подбираем видюху
    pickPhysicalDevice();
    // создаем обертку над видюхой
    createLogicalDevice();
    // создаем свопчейн
    createSwapChain();
    // создаем представление что мы будем рисовать
    createImageViews();
    // описываем алгоритм рендеринга
    createRenderPass();
    // графическая пипелина
    createGraphicsPipeline();
    // создаем фреймбаффер
    createFramebuffers();
    // описываем пул команд
    createCommandPool();
    // здесь мы начинаем непосредственно рисовать
    // создаем массив вертексов
    createVertexBuffer();
    // буфер индексов. описывает, как мы будем переисопльзовать вершины
    createIndexBuffer();
    // описываем команды
    createCommandBuffers(commandBuffers, vertexBuffer, 1);
    // семафоры
    createSemaphores();
    initKeyBindings();
  }

  void createIndexBuffer() {
    // todo убрать лишний сайзоф
    VkDeviceSize bufferSize =
        sizeof(currentTetramino->indices[0]) * currentTetramino->indices.size();
    for (std::vector<Tetramino>::reverse_iterator it = blob.rbegin();
         it != blob.rend(); it++) {
      Tetramino t = *it;
      unsigned int size = sizeof(t.vertices[0]) * t.vertices.size();
      bufferSize += size;
    }

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingBufferMemory);
    std::vector<uint16_t> allIndices = std::vector<uint16_t>();
    std::copy(currentTetramino->indices.begin(),
              currentTetramino->indices.end(), std::back_inserter(allIndices));
    uint16_t lastIndex = *std::max_element(currentTetramino->indices.begin(),
                                           currentTetramino->indices.end()) +
                         1;

    for (std::vector<Tetramino>::iterator it = blob.begin(); it != blob.end();
         ++it) {
      for (std::vector<uint16_t>::iterator i = it->indices.begin();
           i != it->indices.end(); ++i) {
        allIndices.push_back(*i + lastIndex);
      }
      lastIndex = *std::max_element(allIndices.begin(), allIndices.end()) + 1;
    }
    void *data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, allIndices.data(), (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer,
                 indexBufferMemory);

    copyBuffer(stagingBuffer, indexBuffer, bufferSize);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
  }

  void initKeyBindings() { ::glfwSetKeyCallback(window, keyCallBack); };

  void moveVertex() {
    moveVertex(currentTetramino->vertices, &vertexBuffer, &vertexBufferMemory);
  }

  void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = 0; // Optional
    copyRegion.dstOffset = 0; // Optional
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
    vkEndCommandBuffer(commandBuffer);
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
  }

  void createVertexBuffer() {
    // todo убрать лишний сайзоф
    VkDeviceSize bufferSize = sizeof(currentTetramino->vertices[0]) *
                              currentTetramino->vertices.size();

    for (std::vector<Tetramino>::reverse_iterator it = blob.rbegin();
         it != blob.rend(); it++) {
      Tetramino t = *it;
      unsigned int size = sizeof(t.vertices[0]) * t.vertices.size();
      bufferSize += size;
    }

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingBufferMemory);
    void *data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    std::vector<Vertex> allVertices = std::vector<Vertex>();
    std::copy(currentTetramino->vertices.begin(),
              currentTetramino->vertices.end(),
              std::back_inserter(allVertices));
    for (std::vector<Tetramino>::iterator it = blob.begin(); it != blob.end();
         ++it) {
      std::copy(it->vertices.begin(), it->vertices.end(),
                std::back_inserter(allVertices));
    }
    memcpy(data, allVertices.data(), (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer,
                 vertexBufferMemory);
    copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
  }

  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags properties, VkBuffer &buffer,
                    VkDeviceMemory &bufferMemory) {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
      throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
  }

  void moveVertex(std::vector<Vertex> vert, VkBuffer *buff,
                  VkDeviceMemory *mem) {
    VkDeviceSize bufferSize = sizeof(currentTetramino->vertices[0]) *
                              currentTetramino->vertices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingBufferMemory);

    void *data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, vert.data(), (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
  }

  void createVertexBuffer(std::vector<Vertex> vert, VkBuffer *buff,
                          VkDeviceMemory *memory) {

    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = sizeof(vert[0]) * vert.size();
    bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, buff) != VK_SUCCESS) {
      throw std::runtime_error("failed to create vertex buffer!");
    }
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, (*buff), &memRequirements);
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(memRequirements.memoryTypeBits,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (vkAllocateMemory(device, &allocInfo, nullptr, memory) != VK_SUCCESS) {
      throw std::runtime_error("failed to allocate vertex buffer memory!");
    }
    vkBindBufferMemory(device, (*buff), (*memory), 0);
    void *data;
    vkMapMemory(device, (*memory), 0, bufferInfo.size, 0, &data);
    memcpy(data, vert.data(), (size_t)bufferInfo.size);
    vkUnmapMemory(device, (*memory));
  };

  uint32_t findMemoryType(uint32_t typeFilter,
                          VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
      if ((typeFilter & (1 << i)) &&
          (memProperties.memoryTypes[i].propertyFlags & properties) ==
              properties) {
        return i;
      }
    }
    throw std::runtime_error("failed to find suitable memory type!");
  }

  void recreateObject() {
    vkDeviceWaitIdle(device);
    cleanupSwapChain();
    vkDestroyBuffer(device, indexBuffer, nullptr);
    vkFreeMemory(device, indexBufferMemory, nullptr);
    vkDestroyBuffer(device, vertexBuffer, nullptr);

    vkFreeMemory(device, vertexBufferMemory, nullptr);

    createSwapChain();
    createImageViews();
    createRenderPass();
    createGraphicsPipeline();
    createFramebuffers();
    // здесь мы начинаем непосредственно рисовать
    // создаем массив вертексов
    createVertexBuffer();
    // буфер индексов. описывает, как мы будем переисопльзовать вершины
    createIndexBuffer();

    createCommandBuffers(commandBuffers, vertexBuffer, 1);
  }

  void recreateSwapChain() {
    vkDeviceWaitIdle(device);
    cleanupSwapChain();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createGraphicsPipeline();
    createFramebuffers();
    createCommandBuffers(commandBuffers, vertexBuffer, 1);
  }

  void cleanupSwapChain() {
    for (size_t i = 0; i < swapChainFramebuffers.size(); i++) {
      vkDestroyFramebuffer(device, swapChainFramebuffers[i], nullptr);
    }

    vkFreeCommandBuffers(device, commandPool,
                         static_cast<uint32_t>(commandBuffers.size()),
                         commandBuffers.data());

    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);

    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
      vkDestroyImageView(device, swapChainImageViews[i], nullptr);
    }

    vkDestroySwapchainKHR(device, swapChain, nullptr);
  }

  void createSemaphores() {

    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    if (vkCreateSemaphore(device, &semaphoreInfo, nullptr,
                          &imageAvailableSemaphore) != VK_SUCCESS ||
        vkCreateSemaphore(device, &semaphoreInfo, nullptr,
                          &renderFinishedSemaphore) != VK_SUCCESS) {

      throw std::runtime_error("failed to create semaphores!");
    }
  }

  void createCommandBuffers(std::vector<VkCommandBuffer> &commandBuffs,
                            VkBuffer vertexBuffs1, uint32_t buffArraySize) {
    commandBuffs.resize(swapChainFramebuffers.size());
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = commandBuffs.size();
    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffs.data()) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to allocate command buffers!");
    }

    for (size_t i = 0; i < commandBuffs.size(); i++) {
      VkCommandBufferBeginInfo beginInfo = {};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
      beginInfo.pInheritanceInfo = nullptr; // Optional
      vkBeginCommandBuffer(commandBuffs[i], &beginInfo);
      VkRenderPassBeginInfo renderPassInfo = {};
      renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
      renderPassInfo.renderPass = renderPass;
      renderPassInfo.framebuffer = swapChainFramebuffers[i];
      renderPassInfo.renderArea.offset = {0, 0};
      renderPassInfo.renderArea.extent = swapChainExtent;
      VkClearValue clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
      renderPassInfo.clearValueCount = 1;
      renderPassInfo.pClearValues = &clearColor;
      vkCmdBeginRenderPass(commandBuffs[i], &renderPassInfo,
                           VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(commandBuffs[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                        graphicsPipeline);
      VkDeviceSize offsets[] = {0, 0};
      VkBuffer buffer1[] = {vertexBuffs1};
      vkCmdBindVertexBuffers(commandBuffs[i], 0, 1, buffer1, offsets);
      vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0,
                           VK_INDEX_TYPE_UINT16);
      uint16_t size = currentTetramino->indices.size();
      for (std::vector<Tetramino>::iterator it = blob.begin(); it != blob.end();
           ++it) {
        size += it->indices.size();
      }

      vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(size), 1, 0, 0,
                       0);
      vkCmdEndRenderPass(commandBuffs[i]);
      if (vkEndCommandBuffer(commandBuffs[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
      }
    }
  }

  void createCommandPool() {
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily;
    poolInfo.flags = 0; // Optional
    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create command pool!");
    }
  }

  void createFramebuffers() {
    swapChainFramebuffers.resize(swapChainImageViews.size());
    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
      VkImageView attachments[] = {swapChainImageViews[i]};

      VkFramebufferCreateInfo framebufferInfo = {};
      framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      framebufferInfo.renderPass = renderPass;
      framebufferInfo.attachmentCount = 1;
      framebufferInfo.pAttachments = attachments;
      framebufferInfo.width = swapChainExtent.width;
      framebufferInfo.height = swapChainExtent.height;
      framebufferInfo.layers = 1;

      if (vkCreateFramebuffer(device, &framebufferInfo, nullptr,
                              &swapChainFramebuffers[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to create framebuffer!");
      }
    }
  }

  void createRenderPass() {
    VkAttachmentDescription colorAttachment = {};
    colorAttachment.format = swapChainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentReference colorAttachmentRef = {};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create render pass!");
    }
    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                               VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;
  }

  void createGraphicsPipeline() {
    auto vertShaderCode = loadVertex("shaders/vert.spv");

    auto fragShaderCode = loadVertex("shaders/frag.spv");

    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
    vertShaderStageInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
    fragShaderStageInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                      fragShaderStageInfo};

    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
    vertexInputInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)swapChainExtent.width;
    viewport.height = (float)swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = swapChainExtent;

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 0;
    pipelineLayoutInfo.pushConstantRangeCount = 0;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                               &pipelineLayout) != VK_SUCCESS) {
      throw std::runtime_error("failed to create pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                  nullptr, &graphicsPipeline) != VK_SUCCESS) {
      throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
  }

  VkShaderModule createShaderModule(const std::vector<char> &code) {
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create shader module!");
    }
    return shaderModule;
  }

  void createImageViews() {
    swapChainImageViews.resize(swapChainImages.size());
    for (size_t i = 0; i < swapChainImages.size(); i++) {
      VkImageViewCreateInfo createInfo = {};
      createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      createInfo.image = swapChainImages[i];
      createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
      createInfo.format = swapChainImageFormat;
      createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      createInfo.subresourceRange.baseMipLevel = 0;
      createInfo.subresourceRange.levelCount = 1;
      createInfo.subresourceRange.baseArrayLayer = 0;
      createInfo.subresourceRange.layerCount = 1;
      if (vkCreateImageView(device, &createInfo, nullptr,
                            &swapChainImageViews[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image views!");
      }
    }
  }
  void createSwapChain() {
    SwapChainSupportDetails swapChainSupport =
        querySwapChainSupport(physicalDevice);

    VkSurfaceFormatKHR surfaceFormat =
        chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode =
        chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);
    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 &&
        imageCount > swapChainSupport.capabilities.maxImageCount) {
      imageCount = swapChainSupport.capabilities.maxImageCount;
    }
    VkSwapchainCreateInfoKHR createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    uint32_t queueFamilyIndices[] = {(uint32_t)indices.graphicsFamily,
                                     (uint32_t)indices.presentFamily};

    if (indices.graphicsFamily != indices.presentFamily) {
      createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      createInfo.queueFamilyIndexCount = 2;
      createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
      createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
      createInfo.queueFamilyIndexCount = 0;     // Optional
      createInfo.pQueueFamilyIndices = nullptr; // Optional
    }
    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;
    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create swap chain!");
    }
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount,
                            swapChainImages.data());

    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;
  }

  VkSurfaceFormatKHR chooseSwapSurfaceFormat(
      const std::vector<VkSurfaceFormatKHR> &availableFormats) {
    if (availableFormats.size() == 1 &&
        availableFormats[0].format == VK_FORMAT_UNDEFINED) {
      return {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    }

    for (const auto &availableFormat : availableFormats) {
      if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM &&
          availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        return availableFormat;
      }
    }

    return availableFormats[0];
  }

  VkPresentModeKHR chooseSwapPresentMode(
      const std::vector<VkPresentModeKHR> availablePresentModes) {
    VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR;

    for (const auto &availablePresentMode : availablePresentModes) {
      if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
        return availablePresentMode;
      } else if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
        bestMode = availablePresentMode;
      }
    }

    return bestMode;
  }

  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities) {
    if (capabilities.currentExtent.width !=
        std::numeric_limits<uint32_t>::max()) {
      return capabilities.currentExtent;
    } else {
      int width, height;
      glfwGetWindowSize(window, &width, &height);

      VkExtent2D actualExtent = {static_cast<uint32_t>(width),
                                 static_cast<uint32_t>(height)};
      actualExtent.width = std::max(
          capabilities.minImageExtent.width,
          std::min(capabilities.maxImageExtent.width, actualExtent.width));
      actualExtent.height = std::max(
          capabilities.minImageExtent.height,
          std::min(capabilities.maxImageExtent.height, actualExtent.height));

      return actualExtent;
    }
  }

  void createSurface() {
    if (glfwCreateWindowSurface(vkInstance, window, nullptr, &surface) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create window surface!");
    }
  }

  void createLogicalDevice() {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<int> uniqueQueueFamilies = {indices.graphicsFamily,
                                         indices.presentFamily};

    float queuePriority = 1.0f;
    for (int queueFamily : uniqueQueueFamilies) {
      VkDeviceQueueCreateInfo queueCreateInfo = {};
      queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queueCreateInfo.queueFamilyIndex = queueFamily;
      queueCreateInfo.queueCount = 1;
      queueCreateInfo.pQueuePriorities = &queuePriority;
      queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures deviceFeatures = {};

    VkDeviceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    createInfo.queueCreateInfoCount =
        static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();

    createInfo.pEnabledFeatures = &deviceFeatures;

    createInfo.enabledExtensionCount =
        static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (enableValidationLayers) {
      createInfo.enabledLayerCount =
          static_cast<uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
      createInfo.enabledLayerCount = 0;
    }

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create logical device!");
    }

    vkGetDeviceQueue(device, indices.graphicsFamily, 0, &graphicsQueue);
    vkGetDeviceQueue(device, indices.presentFamily, 0, &presentQueue);
  }

  void pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(vkInstance, &deviceCount, nullptr);
    if (deviceCount == 0) {
      throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(vkInstance, &deviceCount, devices.data());
    for (const auto &device : devices) {
      if (isDeviceSuitable(device)) {
        physicalDevice = device;
        break;
      }
    }

    if (physicalDevice == VK_NULL_HANDLE) {
      throw std::runtime_error("failed to find a suitable GPU!");
    }
  }

  bool isDeviceSuitable(::VkPhysicalDevice device) {
    QueueFamilyIndices indices = findQueueFamilies(device);
    bool extensionsSupported = checkDeviceExtensionSupport(device);
    bool swapChainAdequate = false;
    if (extensionsSupported) {
      SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
      swapChainAdequate = !swapChainSupport.formats.empty() &&
                          !swapChainSupport.presentModes.empty();
    }
    return indices.isComplete() && extensionsSupported && swapChainAdequate;
  }

  bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(),
                                             deviceExtensions.end());

    for (const auto &extension : availableExtensions) {
      requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
  }

  SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
    SwapChainSupportDetails details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface,
                                              &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount,
                                         nullptr);

    if (formatCount != 0) {
      details.formats.resize(formatCount);
      vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount,
                                           details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface,
                                              &presentModeCount, nullptr);

    if (presentModeCount != 0) {
      details.presentModes.resize(presentModeCount);
      vkGetPhysicalDeviceSurfacePresentModesKHR(
          device, surface, &presentModeCount, details.presentModes.data());
    }
    return details;
  }

  QueueFamilyIndices findQueueFamilies(::VkPhysicalDevice device) {

    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             queueFamilies.data());
    int i = 0;
    for (const auto &queueFamily : queueFamilies) {
      if (queueFamily.queueCount > 0 &&
          queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        indices.graphicsFamily = i;
      }
      VkBool32 presentSupport = false;
      vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
      if (queueFamily.queueCount > 0 && presentSupport) {
        indices.presentFamily = i;
      }
      if (indices.isComplete()) {
        break;
      }
      i++;
    }
    return indices;
  }

  void setupDebugCallback() {
    if (!enableValidationLayers) {
      return;
    }
    VkDebugReportCallbackCreateInfoEXT createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
    createInfo.flags =
        VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
    createInfo.pfnCallback = debugCallback;

    if (CreateDebugReportCallbackEXT(vkInstance, &createInfo, nullptr,
                                     &callback) != VK_SUCCESS) {
      throw std::runtime_error("failed to set up debug callback!");
    }
  }

  void mainLoop() {
    while (!::glfwWindowShouldClose(window)) {
      ::glfwPollEvents();
      calcFrameRate();
      setPosition();
      drawFrame();
      mLastEndTime = Clock::now();
    }
    vkDeviceWaitIdle(device);
  }

  enum Coord { X, Y };
  // todo разбить на подфункции

  bool blockEdges(Coord c, int keypressed,
                  std::pair<glm::vec2, glm::vec2> currentEdge,
                  std::pair<glm::vec2, glm::vec2> storedEdge) {
    bool res = false;
    if (c == Coord::X) {
      if (storedEdge.first.x == storedEdge.second.x) {
        if (std::abs(storedEdge.second.x - currentEdge.first.x) <= onePixelX ||
            std::abs(storedEdge.second.x - currentEdge.second.x) <= onePixelX) {

          float minStored = std::min(storedEdge.first.y, storedEdge.second.y);
          float maxStored = std::max(storedEdge.first.y, storedEdge.second.y);

          float minCurrent =
              std::min(currentEdge.first.y, currentEdge.second.y);
          float maxCurrent =
              std::max(currentEdge.first.y, currentEdge.second.y);
          if (minStored > maxCurrent || maxStored < minCurrent) {
            return false;
          }
          switch (keypressed) {
          case GLFW_KEY_D:
          case GLFW_KEY_RIGHT:
            if (storedEdge.first.x <= currentEdge.first.x) {
              return false;
            }

            if (currentEdge.first.y >= storedEdge.second.y &&
                currentEdge.first.y <= storedEdge.first.y) {
              return true;
            }
            if (storedEdge.first.y <= currentEdge.second.y &&
                storedEdge.first.y >= currentEdge.first.y) {
              return true;
            }

            break;
          case GLFW_KEY_A:
          case GLFW_KEY_LEFT:
            if (storedEdge.first.x >= currentEdge.first.x) {
              return false;
            }

            if (currentEdge.first.y >= storedEdge.second.y &&
                currentEdge.first.y >= storedEdge.first.y) {
              return true;
            }
            if (storedEdge.first.y <= currentEdge.second.y &&
                storedEdge.first.y >= currentEdge.first.y) {
              return true;
            }

            break;
          }
        }
      }
    } else {
      if (storedEdge.first.y == storedEdge.second.y) {
        if (std::abs(storedEdge.second.y - currentEdge.first.y) <= onePixelY ||
            std::abs(storedEdge.second.y - currentEdge.second.y) <= onePixelY) {
          float minStored = std::min(storedEdge.first.x, storedEdge.second.x);
          float maxStored = std::max(storedEdge.first.x, storedEdge.second.x);

          float minCurrent =
              std::min(currentEdge.first.x, currentEdge.second.x);
          float maxCurrent =
              std::max(currentEdge.first.x, currentEdge.second.x);
          if (minStored > maxCurrent || maxStored < minCurrent) {
            return false;
          }
          switch (keypressed) {
          case GLFW_KEY_W:
          case GLFW_KEY_UP:
            if (storedEdge.first.y >= currentEdge.first.y) {
              return false;
            }

            if (currentEdge.first.x >= storedEdge.second.x &&
                currentEdge.first.x <= storedEdge.first.x) {
              return true;
            }
            if (storedEdge.first.x <= currentEdge.second.x &&
                storedEdge.first.x >= currentEdge.first.x) {
              return true;
            }

            break;
          case GLFW_KEY_S:
          case GLFW_KEY_DOWN:
            if (storedEdge.first.y <= currentEdge.first.y) {
              return false;
            }

            if (currentEdge.first.x <= storedEdge.second.x &&
                currentEdge.first.x >= storedEdge.first.x) {
              return true;
            }
            if (storedEdge.first.x >= currentEdge.second.x &&
                storedEdge.first.x <= currentEdge.first.x) {
              return true;
            }
            break;
          }
        }
      }
    }
    return res;
  }

  std::vector<std::pair<glm::vec2, glm::vec2>> getEdges(Tetramino *t) {
    std::vector<std::pair<glm::vec2, glm::vec2>> currentEdges =
        std::vector<std::pair<glm::vec2, glm::vec2>>();

    for (int idx = 0; idx < t->vertices.size() - 1;) {
      Vertex v1 = t->vertices.at(idx);
      idx++;
      Vertex v2 = t->vertices.at(idx);
      currentEdges.push_back(std::make_pair(v1.pos, v2.pos));
    }
    currentEdges.push_back(std::make_pair(
        currentTetramino->vertices.at(currentTetramino->vertices.size() - 1)
            .pos,
        currentTetramino->vertices.at(0).pos));

    return currentEdges;
  }

  bool couldMoveThroughBlocks(Coord coord, int key) {
    if (blob.empty()) {
      return true;
    }
    std::vector<std::pair<glm::vec2, glm::vec2>> edges =
        std::vector<std::pair<glm::vec2, glm::vec2>>();
    std::vector<std::pair<glm::vec2, glm::vec2>> currentEdges =
        getEdges(currentTetramino);

    for (std::vector<Tetramino>::iterator i = blob.begin(); i != blob.end();
         i++) {
      std::vector<std::pair<glm::vec2, glm::vec2>> e = getEdges(&(*i));
    }

    for (std::vector<std::pair<glm::vec2, glm::vec2>>::iterator e =
             currentEdges.begin();
         e != currentEdges.end(); e++) {
      for (std::vector<std::pair<glm::vec2, glm::vec2>>::iterator ee =
               edges.begin();
           ee != edges.end(); ee++) {
        if (blockEdges(coord, key, *e, *ee)) {
          return false;
        }
      }
    }
    return true;
  }

  bool couldMove(Coord coord, int key) {
    for (std::vector<Vertex>::iterator it = currentTetramino->vertices.begin();
         it != currentTetramino->vertices.end(); ++it) {
      Vertex &v = *it;
      if (coord == Coord::X) {
        switch (key) {
        case GLFW_KEY_LEFT:
        case GLFW_KEY_A:
          if (v.pos.x <= -1.0f) {
            return false;
          }
          break;
        case GLFW_KEY_RIGHT:
        case GLFW_KEY_D:
          if (v.pos.x >= 1.0f) {
            return false;
          }
        }
      } else {
        switch (key) {
        case GLFW_KEY_UP:
        case GLFW_KEY_W:
          if (v.pos.y <= -1.0f) {
            return false;
          }
          break;
        case GLFW_KEY_DOWN:
        case GLFW_KEY_S:
          if (v.pos.y >= 1.0f) {
            return false;
          }
        }
      }
    }
    return true;
  }

  bool couldMove() {
    switch (lastKeyPressed) {
    case GLFW_KEY_UP:
    case GLFW_KEY_W:
    case GLFW_KEY_DOWN:
    case GLFW_KEY_S:
      return couldMoveThroughBlocks(Coord::Y, lastKeyPressed) &&
             couldMove(Coord::Y, lastKeyPressed);
    case GLFW_KEY_LEFT:
    case GLFW_KEY_A:
    case GLFW_KEY_RIGHT:
    case GLFW_KEY_D:
      return couldMoveThroughBlocks(Coord::X, lastKeyPressed) &&
             couldMove(Coord::X, lastKeyPressed);
    default:
      return false;
    }
  }

  int getStepCount() {
    switch (lastKeyPressed) {
    case GLFW_KEY_UP:
    case GLFW_KEY_W:
    case GLFW_KEY_DOWN:
    case GLFW_KEY_S:
      return 1; // sizeXInPixel;
    case GLFW_KEY_LEFT:
    case GLFW_KEY_A:
    case GLFW_KEY_RIGHT:
    case GLFW_KEY_D:
      return 1; // sizeYInPixel + 1; // because reasons
    default:
      return 0;
    }
  }
  //поскольку y не равны друг другу, а лежат в некой деьла окружности,
  //мы их кластеризуем. для этого мы вернем примерные центры кластеров, первый
  //из которых
  // 1.0f и до -1.0f с шагом в сторону квадратика
  std::vector<float> calculateCentroids() {
    float step = 1.0f;
    std::vector<float> res = std::vector<float>();
    while (step > -1.0f) {
      res.push_back(step);

      step = step - edge;
    }
    return res;
  }

  std::vector<float> calculateAxis(std::vector<float> *centroids,
                                   std::vector<float> *points, float distance) {
    std::vector<float> res = std::vector<float>();
    for (std::vector<float>::iterator centroid = centroids->begin();
         centroid != centroids->end(); centroid++) {
      float &c = *centroid;
      std::vector<float> *cluster = new std::vector<float>();
      for (std::vector<float>::iterator point = points->begin();
           point != points->end(); point++) {
        float &p = *point;
        if (std::find(cluster->begin(), cluster->end(), p) == cluster->end() &&
            std::abs(p - c) <= distance) {
          cluster->push_back(p);
        }
      }
      if (cluster->empty()) {
        continue;
      }
      res.push_back(*std::max_element(cluster->begin(), cluster->end()));
    }
    return res;
  }

  bool checkAxis(float axis) {
    float length = 0.0f;
    std::vector<std::pair<glm::vec2, glm::vec2>> edges;
    for (std::vector<Tetramino>::iterator t = blob.begin(); t != blob.end();
         t++) {
      Tetramino &tetramino = *t;
      bool tmblr = false;
      glm::vec2 p;
      std::pair<float, float> points;
      for (std::vector<Vertex>::iterator v = tetramino.vertices.begin();
           v != tetramino.vertices.end(); v++) {
        if (std::abs(v->pos.y - axis) <= edge / 2) {
          if (tmblr) {
            points = std::make_pair(p.x, v->pos.x);
          }
          p = v->pos;
          tmblr = true;
        }
      }

      length = length + (std::max(points.first, points.second) -
                         std::min(points.first, points.second));

      std::vector<std::pair<glm::vec2, glm::vec2>> e = getEdges(&(*t));
      edges.insert(edges.end(), e.begin(), e.end());
    }
    bool tmblr = false;
    glm::vec2 p;
    std::pair<float, float> points;
    float edgeSize = edge;
    for (std::vector<std::pair<glm::vec2, glm::vec2>>::iterator i =
             edges.begin();
         i != edges.end(); i++) {
      std::pair<glm::vec2, glm::vec2> &edge = *i;
      if (edge.first.x == edge.second.x) {
        float min = std::min(edge.first.y, edge.second.y);
        float max = std::max(edge.first.y, edge.second.y);
        if (std::abs(axis - min) < (edgeSize / 2.0f) ||
            std::abs(axis - max) < (edgeSize / 2.0f)) {
          continue;
        }
        if (axis > min && axis < max) {
          if (tmblr) {
            points = std::make_pair(p.x, edge.first.x);
            tmblr = !tmblr;
            length = length + (std::max(points.first, points.second) -
                               std::min(points.first, points.second));
          }
          p = edge.first;
          tmblr = !tmblr;
        }
      }
    }
    std::cout << "for axis " << axis << " length " << length << std::endl;
    return 2.0f - length < edge;
  }

  void cutAxis(float axis) { std::cout << "cut axis " << axis << std::endl; }
  /**
   * Итерируемся по всем тетрамино.
   * Собираем все тетрамино по принципу их Y в дельта окружности друг от друга
   * берем один случайный Yt
   * случайный потому что они более менее равны
   * на ходим максимальный Xmax у точек Y котох больше
   * если у точки с Y меньше Yt X > Xmax то уравниваем c Xmax
   * Y увеличиваем на edge
   * тетрамино после всех преобарзований двигаем вниз пока оно не упрется
   */
  void removeBlocks() {
    std::vector<float> centroids = calculateCentroids();
    std::vector<float> points = std::vector<float>();
    for (std::vector<Tetramino>::iterator t = blob.begin(); t != blob.end();
         t++) {
      Tetramino &tetramino = *t;
      for (std::vector<Vertex>::iterator v = tetramino.vertices.begin();
           v != tetramino.vertices.end(); v++) {
        Vertex &vertex = *v;

        points.push_back(vertex.pos.y);
      }
    }
    for (std::vector<float>::iterator y = centroids.begin();
         y != centroids.end(); y++) {
      if (checkAxis(*y)) {
        cutAxis(*y);
      }
    }
  }

  void setPosition() {
    if (lastKeyPressed > 0) {
      frames++;
      int framesBeforeDraw = MICROSECONDS_IN_SECOND / (SPF * step * speed);
      if (frames > framesBeforeDraw) {
        frames = 0;
        stepCount++;
        if (!couldMove()) {
          stepCount = 0;
          lastKeyPressed = -1;
          if (block) {
            summon();
            block = false;
          }
        } else {
          moveOnePixel();
          if (!block && stepCount == getStepCount()) {
            stepCount = 0;
          }
        }
      }
    }
  }

  void calcFrameRate() {
    auto lastTime = std::chrono::duration_cast<std::chrono::microseconds>(
        Clock::now() - mLastMeasureTime);
    nbFrames++;
    if (lastTime.count() >= MICROSECONDS_IN_SECOND) {
      SPF = MICROSECONDS_IN_SECOND / nbFrames;
      nbFrames = 0;
      mLastMeasureTime = Clock::now();
    }
  }

  void drawFrame() {
    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(
        device, swapChain, std::numeric_limits<uint64_t>::max(),
        imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
      recreateSwapChain();
      return;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
      throw std::runtime_error("failed to acquire swap chain image!");
    }

    VkSubmitInfo submitInfo = {};

    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = {imageAvailableSemaphore};
    VkPipelineStageFlags waitStages[] = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[imageIndex];
    VkSemaphore signalSemaphores[] = {renderFinishedSemaphore};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;
    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to submit draw command buffer!");
    }
    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    VkSwapchainKHR swapChains[] = {swapChain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr; // Optional
    result = vkQueuePresentKHR(presentQueue, &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
      recreateSwapChain();
    } else if (result != VK_SUCCESS) {
      throw std::runtime_error("failed to present swap chain image!");
    }
    vkQueueWaitIdle(presentQueue);
  }

  void cleanup() {
    cleanupSwapChain();
    vkDestroyBuffer(device, indexBuffer, nullptr);
    vkFreeMemory(device, indexBufferMemory, nullptr);
    vkDestroyBuffer(device, vertexBuffer, nullptr);

    vkFreeMemory(device, vertexBufferMemory, nullptr);

    vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);
    vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);

    vkDestroyCommandPool(device, commandPool, nullptr);

    vkDestroyDevice(device, nullptr);
    DestroyDebugReportCallbackEXT(vkInstance, callback, nullptr);
    vkDestroySurfaceKHR(vkInstance, surface, nullptr);
    vkDestroyInstance(vkInstance, nullptr);

    glfwDestroyWindow(window);

    glfwTerminate();
  }

  void createInstance() {
    if (enableValidationLayers && !checkValidationLayerSupport()) {
      throw std::runtime_error(
          "validation layers requested, but not available!");
    }

    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    auto extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    if (enableValidationLayers) {
      createInfo.enabledLayerCount =
          static_cast<uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
      createInfo.enabledLayerCount = 0;
    }
    if (vkCreateInstance(&createInfo, nullptr, &vkInstance) != VK_SUCCESS) {

      throw std::runtime_error("failed to create instance!");
    }
  }

  std::vector<const char *> getRequiredExtensions() {
    std::vector<const char *> extensions;

    unsigned int glfwExtensionCount = 0;
    const char **glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    for (unsigned int i = 0; i < glfwExtensionCount; i++) {
      extensions.push_back(glfwExtensions[i]);
    }

    if (enableValidationLayers) {
      extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    }

    return extensions;
  }

  bool checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
    for (const char *layerName : validationLayers) {
      bool layerFound = false;
      for (const auto &layerProperties : availableLayers) {
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

  static VKAPI_ATTR VkBool32 VKAPI_CALL
  debugCallback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objType,
                uint64_t obj, size_t location, int32_t code,
                const char *layerPrefix, const char *msg, void *userData) {

    std::cerr << "validation layer: " << msg << std::endl;

    return VK_FALSE;
  }
};

int main() {
  HelloTriangleApplication app;
  try {
    app.run();
  } catch (const std::runtime_error &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
