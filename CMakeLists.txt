cmake_minimum_required(VERSION 3.7.2)
project(VulkanTest)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 11)
set(VULKAN_SDK_PATH /home/zealot/VulkanSDK/1.0.61.1/x86_64)
set(VK_LAYER_PATH ${VULKAN_SDK_PATH}/etc/explicit_layer.d)
set(SOURCE_FILES  main.cpp /home/zealot/VulkanSDK/1.0.61.1/source)
find_package(glfw3 3.2 REQUIRED)
MESSAGE( STATUS "GLFW3_INCLUDE_DIR:         " ${GLFW_INCLUDE_DIRS} )
MESSAGE( STATUS "VULKAN_SDK_PATH: " ${VULKAN_SDK_PATH} )

LINK_DIRECTORIES(${VULKAN_SDK_PATH}/lib ${VK_LAYER_PATH} ./VulkanTest)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -I${VULKAN_SDK_PATH}/include")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lvulkan")
execute_process (
        COMMAND /bin/bash compile.sh
	WORKING_DIRECTORY shaders
)
add_executable(VulkanTest ${SOURCE_FILES})
target_link_libraries(VulkanTest glfw)

