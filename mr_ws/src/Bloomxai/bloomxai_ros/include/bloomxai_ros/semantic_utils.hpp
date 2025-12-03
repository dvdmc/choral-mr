#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <iostream>

static inline std::vector<uint8_t> jetColor(float x)
{
    x = std::clamp(x, 0.0f, 1.0f);

    float r = std::clamp(1.5f - std::abs(4.0f * x - 3.0f), 0.0f, 1.0f);
    float g = std::clamp(1.5f - std::abs(4.0f * x - 2.0f), 0.0f, 1.0f);
    float b = std::clamp(1.5f - std::abs(4.0f * x - 1.0f), 0.0f, 1.0f);

    return {
        static_cast<uint8_t>(r * 255.0f),
        static_cast<uint8_t>(g * 255.0f),
        static_cast<uint8_t>(b * 255.0f)
    };
}

static const std::vector<std::vector<uint8_t>> LABEL_TO_RGB_8 = {
    {0, 0, 0},        // 0=background
    {128, 0, 128},    // 5=bottle
    {192, 0, 0},      // 9=chair
    {192, 128, 0},    // 11=diningtable
    {192, 128, 128},  // 15=person
    {0, 64, 0},       // 16=pottedplant
    {0, 192, 0},      // 18=sofa
    {0, 64, 128}};    // 20=tvmonitor

static const std::vector<std::vector<uint8_t>> LABEL_TO_RGB_21 = {
    {0, 0, 0},        // 0=background
    {0, 64, 0},       // 1=aeroplane
    {0, 128, 0},      // 2=bicycle
    {128, 128, 0},    // 3=bird
    {0, 0, 128},      // 4=boat
    {128, 0, 128},    // 5=bottle
    {0, 128, 128},    // 6=bus
    {128, 128, 128},  // 7=car
    {64, 0, 0},       // 8=cat
    {192, 0, 0},      // 9=chair
    {64, 128, 0},     // 10=cow
    {192, 128, 0},    // 11=diningtable
    {64, 0, 128},     // 12=dog
    {192, 0, 128},    // 13=horse
    {64, 128, 128},   // 14=motorbike
    {192, 128, 128},  // 15=person
    {0, 64, 0},       // 16=potted plant
    {128, 64, 0},     // 17=sheep
    {0, 192, 0},      // 18=sofa
    {128, 192, 0},    // 19=train
    {0, 64, 128},     // 20=tv/monitor
    {128, 64, 128}};  // 21=background

static const std::vector<std::vector<uint8_t>> LABEL_TO_RGB_150 = {
    {120, 120, 120}, {180, 120, 120}, {6, 230, 230}, {80, 50, 50}, {4, 200, 3},
    {120, 120, 80}, {140, 140, 140}, {204, 5, 255}, {230, 230, 230}, {4, 250, 7},
    {224, 5, 255}, {235, 255, 7}, {150, 5, 61}, {120, 120, 70}, {8, 255, 51},
    {255, 6, 82}, {143, 255, 140}, {204, 255, 4}, {255, 51, 7}, {204, 70, 3},
    {0, 102, 200}, {61, 230, 250}, {255, 6, 51}, {11, 102, 255}, {255, 7, 71},
    {255, 9, 224}, {9, 7, 230}, {220, 220, 220}, {255, 9, 92}, {112, 9, 255},
    {8, 255, 214}, {7, 255, 224}, {255, 184, 6}, {10, 255, 71}, {255, 41, 10},
    {7, 255, 255}, {224, 255, 8}, {102, 8, 255}, {255, 61, 6}, {255, 194, 7},
    {255, 122, 8}, {0, 255, 20}, {255, 8, 41}, {255, 5, 153}, {6, 51, 255},
    {235, 12, 255}, {160, 150, 20}, {0, 163, 255}, {140, 140, 140}, {250, 10, 15},
    {20, 255, 0}, {31, 255, 0}, {255, 31, 0}, {255, 224, 0}, {153, 255, 0},
    {0, 0, 255}, {255, 71, 0}, {0, 235, 255}, {0, 173, 255}, {31, 0, 255},
    {11, 200, 200}, {255, 82, 0}, {0, 255, 245}, {0, 61, 255}, {0, 255, 112},
    {0, 255, 133}, {255, 0, 0}, {255, 163, 0}, {255, 102, 0}, {194, 255, 0},
    {0, 143, 255}, {51, 255, 0}, {0, 82, 255}, {0, 255, 41}, {0, 255, 173},
    {10, 0, 255}, {173, 255, 0}, {0, 255, 153}, {255, 92, 0}, {255, 0, 255},
    {255, 0, 245}, {255, 0, 102}, {255, 173, 0}, {255, 0, 20}, {255, 184, 184},
    {0, 31, 255}, {0, 255, 61}, {0, 71, 255}, {255, 0, 204}, {0, 255, 194},
    {0, 255, 82}, {0, 10, 255}, {0, 112, 255}, {51, 0, 255}, {0, 194, 255},
    {0, 122, 255}, {0, 255, 163}, {255, 153, 0}, {0, 255, 10}, {255, 112, 0},
    {143, 255, 0}, {82, 0, 255}, {163, 255, 0}, {255, 235, 0}, {8, 184, 170},
    {133, 0, 255}, {0, 255, 92}, {184, 0, 255}, {255, 0, 31}, {0, 184, 255},
    {0, 214, 255}, {255, 0, 112}, {92, 255, 0}, {0, 224, 255}, {112, 224, 255},
    {70, 184, 160}, {163, 0, 255}, {153, 0, 255}, {71, 255, 0}, {255, 0, 163},
    {255, 204, 0}, {255, 0, 143}, {0, 255, 235}, {133, 255, 0}, {255, 0, 235},
    {245, 0, 255}, {255, 0, 122}, {255, 245, 0}, {10, 190, 212}, {214, 255, 0},
    {0, 204, 255}, {20, 0, 255}, {255, 255, 0}, {0, 153, 255}, {0, 41, 255},
    {0, 255, 204}, {41, 0, 255}, {41, 255, 0}, {173, 0, 255}, {0, 245, 255},
    {71, 0, 255}, {122, 0, 255}, {0, 255, 184}, {0, 92, 255}, {184, 255, 0},
    {0, 133, 255}, {255, 214, 0}, {25, 194, 194}, {102, 255, 0}, {92, 0, 255}
};

static std::vector<uint8_t> hsvToRgb(float h, float s, float v)
{
    float r, g, b;

    int i = static_cast<int>(std::floor(h * 6.0f));
    float f = h * 6.0f - i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - f * s);
    float t = v * (1.0f - (1.0f - f) * s);

    switch (i % 6) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        case 5: r = v; g = p; b = q; break;
        default: r = g = b = 0.0f; break; // Safety
    }

    return {
        static_cast<uint8_t>(std::round(r * 255.0f)),
        static_cast<uint8_t>(std::round(g * 255.0f)),
        static_cast<uint8_t>(std::round(b * 255.0f))
    };
}

static std::vector<std::vector<uint8_t>> generateColorMap(int n)
{
    std::vector<std::vector<uint8_t>> map;

    if (n == 0)
        return map;

    // 0: background → black
    map.push_back({0, 0, 0});

    int num_fg = n - 1;
    float s = 0.65f;
    float v = 0.95f;

    for (int i = 0; i < num_fg; i++) {
        float h = float(i) / float(num_fg);   // MUST match Python exactly
        map.push_back(hsvToRgb(h, s, v));
    }

    return map;
}

static const std::vector<std::vector<uint8_t>> getLabelMap(const int num_classes) {
  switch (num_classes) {
    case 8:
      return LABEL_TO_RGB_8;
    case 21:
      return LABEL_TO_RGB_21;
    case 150:
      return LABEL_TO_RGB_150;
    default:
      return generateColorMap(num_classes);
  }
}

static const std::vector<std::vector<uint8_t>> bgrToRgb(
    const std::vector<std::vector<uint8_t>> &bgrMap) {
  std::vector<std::vector<uint8_t>> rgbMap;
  for (const auto &color : bgrMap) {
    rgbMap.push_back({color[2], color[1], color[0]});
  }
  return rgbMap;
}

static const std::vector<std::vector<uint8_t>> rgbToBgr(
    const std::vector<std::vector<uint8_t>> &rgbMap) {
  std::vector<std::vector<uint8_t>> bgrMap;
  for (const auto &color : rgbMap) {
    bgrMap.push_back({color[2], color[1], color[0]});
  }
  return bgrMap;
}