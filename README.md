# Vulkan Lunar Terrain Renderer

The primary goal of this work is to move beyond the abstractions of high-level game engines to gain fine-grained, direct control over the GPU. The renderer visualises a high-fidelity, large-scale model of the Apollo 11 landing site, derived from NASA's Lunar Reconnaissance Orbiter Camera (LROC) data. The repository showcases an iterative development process, exploring and analysing the performance of several advanced, real-time rendering techniques.

### Related Repository: Data Processing

The terrain mesh and heightmap assets used in this renderer were generated from raw GeoTIFF data using a custom C++ tool. This tool is available in a separate repository:
* **[GeoTIFF-to-Mesh Converter](https://github.com/DilaraAlbayrak/geotiff-to-mesh)**

## Features

This repository contains multiple, distinct implementations, each demonstrating a different rendering technique.

* **Baseline Static Mesh Rendering (TerrainRenderer.cpp):** A foundational renderer for loading and displaying high-resolution `.obj` terrain models.
* **Dynamic LOD with Tessellation (TerrainRenderer_tessellation.cpp):** An efficient LOD system using **hardware tessellation**. It dynamically adjusts terrain complexity based on camera distance, using a low-resolution control grid and a 16-bit heightmap for displacement.
* **Stereo Rendering (VR Workload Simulation):**
    * **Multi-Pass Stereo (TerrainRenderer_multipass.cpp):** A software-driven approach that renders the scene twice per frame (once for each eye) using separate viewports and draw calls.
    * **Single-Pass Stereo (TerrainRenderer_singlepass.cpp):** A highly efficient, hardware-accelerated approach using Vulkan's **multiview** feature to render both eye views in a single draw call.
* **Dynamic LOD with Tessellation for multipass stereo rendering (TerrainRenderer_tessellated_stereo.cpp):** Combining tessellation and stereo implementation.
* **Full VR Integration with OpenXR (TerrainRenderer_VR.cpp):** The renderer is fully integrated with the OpenXR SDK to provide a PC-tethered VR experience on any compliant headset (tested with Meta Quest 3).

## Getting Started

To get a local copy up and running, follow these steps.

### Prerequisites

You will need the following dependencies installed on your system:
* **Vulkan SDK:** [https://vulkan.lunarg.com/](https://vulkan.lunarg.com/)
* **GLFW:** For window and input management.
* **GLM:** For mathematics (vectors, matrices).
* **TinyObjLoader:** For loading `.obj` files (included in the `externals` directory).
* **OpenXR SDK:** Required only for the `TerrainRenderer_VR.cpp` implementation.
* **C++ Compiler:** A modern C++ compiler (e.g., MSVC, GCC, Clang).

### Building

1.  **Clone the repo:**
    ```sh
    git clone https://github.com/DilaraAlbayrak/terrain-rendering-vulkan.git
    cd terrain-rendering-vulkan
    ```

2.  **Compile Shaders:**
    The shaders in the `/shaders` directory are written in GLSL. They must be compiled to SPIR-V bytecode. Use the `glslc` compiler provided with the Vulkan SDK.

    Example:
    ```sh
    glslc shaders/shader.vert -o shaders/shader.vert.spv
    glslc shaders/shader.frag -o shaders/shader.frag.spv
    ```

    Alternatively, you can use the Custom Build Tool in Visual Studio

    <img width="2032" height="535" alt="custom build tool" src="https://github.com/user-attachments/assets/b5457d9a-b6b7-4b8d-91cb-85bcf01835a2" />

4.  **Configure and Build:**
    The project is set up to be built with Visual Studio 2022.
    * Open the `.sln` file in Visual Studio.
    * Ensure that the Include and Library paths for Vulkan, GLFW, and other dependencies are correctly configured in the project properties.
    * Build the desired solution configuration (e.g., Debug or Release).

## Project Structure

This repository is structured as a collection of separate C++ files, each demonstrating one of the core rendering techniques investigated in the dissertation. To switch between implementations, set the desired `.cpp` file as the main entry point in your build system.

* `TerrainRenderer.cpp`: **Baseline `.obj` Mesh Renderer.**
* `TerrainRenderer_tessellation.cpp`: **Dynamic LOD with Hardware Tessellation.**
* `TerrainRenderer_multipass.cpp`: **Multi-Pass Stereo Rendering.**
* `TerrainRenderer_singlepass.cpp`: **Single-Pass (Multiview) Stereo Rendering.**
* `TerrainRenderer_tessellated_stereo.cpp`: **Combines Multi-Pass Stereo and Tessellation.**
* `TerrainRenderer_VR.cpp`: **Full PC-Tethered VR with OpenXR Integration.**

* `/shaders`: Contains GLSL source code (`.vert`, `.frag`, `.tesc`, `.tese`) and pre-compiled SPIR-V bytecode (`.spv`).
* `/models`: Contains the `.obj` model of the Apollo 11 landing site (you can get .obj files in this directory https://drive.google.com/drive/folders/1WXbS4P3G9To2HtbQ1XPC-WEcsk7rFRDM?usp=sharing).
* `/textures`: Contains the 16-bit heightmap (`.png`) used for tessellation.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.
