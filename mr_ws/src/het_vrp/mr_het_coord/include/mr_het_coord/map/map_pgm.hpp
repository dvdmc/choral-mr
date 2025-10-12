#ifndef MR_HET_COORD_MAP_PGM_HPP
#define MR_HET_COORD_MAP_PGM_HPP
#include <cstdint>
#include <string>
#include <vector>
#include <iostream>

// Struct for BITMAPFILEHEADER (14 bytes)
#pragma pack(push, 1)  // Ensure no padding
struct BITMAPFILEHEADER {
  uint16_t bfType;       // File type, must be 'BM' (0x4D42)
  uint32_t bfSize;       // Size of the file in bytes
  uint16_t bfReserved1;  // Reserved, must be 0
  uint16_t bfReserved2;  // Reserved, must be 0
  uint32_t bfOffBits;    // Offset to start of pixel data
};

// Struct for BITMAPINFOHEADER (40 bytes)
struct BITMAPINFOHEADER {
  uint32_t biSize;         // Size of this header (40 bytes)
  int32_t biWidth;         // Width of the image
  int32_t biHeight;        // Height of the image
  uint16_t biPlanes;       // Number of color planes, must be 1
  uint16_t biBitCount;     // Bits per pixel (24 for RGB)
  uint32_t biCompression;  // Compression type (0 for no compression)
  uint32_t biSizeImage;    // Size of the image data (can be 0 for uncompressed)
  int32_t biXPelsPerMeter;  // Horizontal resolution (pixels per meter)
  int32_t biYPelsPerMeter;  // Vertical resolution (pixels per meter)
  uint32_t biClrUsed;       // Number of colors in the color palette
  uint32_t biClrImportant;  // Number of important colors
};
#pragma pack(pop)

/**
 * @brief Creates a 24-bit BMP image from a given grid map.
 * @details The BMP image is created using the BITMAPFILEHEADER and
 * BITMAPINFOHEADER structs. The pixel data is stored in the bottom-up (BMP)
 * format, with each pixel represented by 3 bytes (RGB). The grid map is
 * traversed from top to bottom, and for each cell that is not free, the
 * corresponding pixel in the image is set to white (RGB = 255, 255, 255).
 * @param filename The name of the file to be created.
 * @param width The width of the image in pixels.
 * @param height The height of the image in pixels.
 * @param grid_map The grid map to be converted to a BMP image.
 */
void create_BMP_map(std::string const& filename, int width, int height,
                    std::vector<std::vector<int>> const& grid_map) const {
  // Create the file header
  BITMAPFILEHEADER fileHeader;
  fileHeader.bfType = 0x4D42;  // 'BM'
  fileHeader.bfReserved1 = 0;
  fileHeader.bfReserved2 = 0;
  fileHeader.bfOffBits =
      sizeof(BITMAPFILEHEADER) +
      sizeof(BITMAPINFOHEADER);  // Pixel data starts after headers

  // Create the info header
  BITMAPINFOHEADER infoHeader;
  infoHeader.biSize = sizeof(BITMAPINFOHEADER);
  infoHeader.biWidth = width;
  infoHeader.biHeight = height;
  infoHeader.biPlanes = 1;
  infoHeader.biBitCount = 24;    // 24 bits (RGB)
  infoHeader.biCompression = 0;  // No compression
  int rowSize =
      (width * 3 + 3) & (~3);  // Row size (padded to multiple of 4 bytes)
  infoHeader.biSizeImage = rowSize * height;  // 3 bytes per pixel (RGB)
  infoHeader.biXPelsPerMeter = 0;
  infoHeader.biYPelsPerMeter = 0;
  infoHeader.biClrUsed = 0;
  infoHeader.biClrImportant = 0;

  // Set the size of the file
  fileHeader.bfSize = fileHeader.bfOffBits + infoHeader.biSizeImage;
  // Create pixel data (RGB, bottom-up)
  std::vector<uint8_t> pixels(infoHeader.biSizeImage,
                              0);  // Initialize all pixels to 0 (black)
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (grid_map[y][x]) {
        int index = (height - y - 1) * rowSize + x * 3;
        pixels[index] = 255;      // Blue channel
        pixels[index + 1] = 255;  // Green channel
        pixels[index + 2] = 255;  // Red channel
      }
    }
  }

  // Write headers and pixel data to the file
  std::ofstream file(filename, std::ios::binary);
  if (file) {
    file.write(reinterpret_cast<char const*>(&fileHeader), sizeof(fileHeader));
    file.write(reinterpret_cast<char const*>(&infoHeader), sizeof(infoHeader));
    file.write(reinterpret_cast<char const*>(pixels.data()), pixels.size());
    file.close();
    std::cout << "Bitmap created successfully!" << std::endl;
  } else {
    std::cerr << "Error: Could not create file " << filename << std::endl;
  }
}

#endif