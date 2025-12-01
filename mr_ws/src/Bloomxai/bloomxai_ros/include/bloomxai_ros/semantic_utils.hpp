#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <iostream>

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

std::vector<uint8_t> hsvToRgb(float H, float S, float V) {
    if (H > 360.0f || H < 0.0f || S > 1.0f || S < 0.0f || V > 1.0f || V < 0.0f) {
        // Handle invalid inputs or return black
        return {0, 0, 0};
    }

    float R, G, B;
    if (S == 0.0f) {
        R = G = B = V; // Achromatic (gray)
    } else {
        int i = (int)(H / 60.0f);
        float f = (H / 60.0f) - i;
        float p = V * (1.0f - S);
        float q = V * (1.0f - S * f);
        float t = V * (1.0f - S * (1.0f - f));

        switch (i % 6) {
            case 0: R = V; G = t; B = p; break;
            case 1: R = q; G = V; B = p; break;
            case 2: R = p; G = V; B = t; break;
            case 3: R = p; G = q; B = V; break;
            case 4: R = t; G = p; B = V; break;
            case 5: R = V; G = p; B = q; break;
            default: R = G = B = 0; break;
        }
    }

    return {
        (uint8_t)(R * 255.0f),
        (uint8_t)(G * 255.0f),
        (uint8_t)(B * 255.0f)
    };
}

static const std::vector<uint8_t> interpolateColor(
    const std::vector<uint8_t> &color1, const std::vector<uint8_t> &color2) {
  std::vector<uint8_t> interpolatedColor;
  for (int i = 0; i < 3; i++) {
    interpolatedColor.push_back((color1[i] + color2[i]) / 2);
  }
  return interpolatedColor;
}

static std::vector<std::vector<uint8_t>> generateColorMap(int numLabels) {
    std::vector<std::vector<uint8_t>> colorMap;

    // --- 1. Define specific, requested colors for the first few classes ---
    const std::vector<std::vector<uint8_t>> requested_start_map = {
        // Label 0: Black (Background)
        {0, 0, 0},
        // Label 1: Light Gray (Neutral, high V, low S)
        {200, 200, 200},
        // Label 2: Desaturated Reddish (Red Hue, low S, high V)
        hsvToRgb(0.0f, 0.25f, 0.9f),
        // Label 3: Desaturated Blueish (Blue Hue, low S, high V)
        hsvToRgb(240.0f, 0.25f, 0.9f)
    };
    
    // Determine how many requested colors we can use (up to numLabels)
    int initial_colors_count = std::min((int)requested_start_map.size(), numLabels);
    
    // Add the initial colors to the map
    for (int i = 0; i < initial_colors_count; ++i) {
        colorMap.push_back(requested_start_map[i]);
    }

    // --- 2. Generate remaining colors using a controlled HSV distribution ---
    
    // We only need to generate colors if the total is greater than the initial set
    int generated_start_index = colorMap.size();
    int remaining_labels = numLabels - generated_start_index;

    if (remaining_labels > 0) {
        // Controlled Saturation (S): Lower value for neutrality
        const float saturation = 0.35f; 
        // Controlled Value (V): Maintain high brightness for contrast
        const float value = 0.95f;   

        // We will start the hue from a non-red/blue position to ensure
        // the new set of colors is distinct from the hardcoded ones.
        // Let's start around 60 (Yellow/Green) and 120 (Green) is a good spot.
        const float HUE_START_OFFSET = 100.0f; 
        const float HUE_RANGE = 360.0f - HUE_START_OFFSET;

        // Calculate the step in Hue to evenly space the *remaining* colors
        float hueStep = HUE_RANGE / (float)remaining_labels;

        for (int i = 0; i < remaining_labels; ++i) {
            // Calculate the current hue, wrapping around 360 degrees
            float currentHue = std::fmod(HUE_START_OFFSET + (float)i * hueStep, 360.0f);

            // Convert HSV to RGB and add to the map
            colorMap.push_back(hsvToRgb(currentHue, saturation, value));
        }
    }

    // Ensure we don't return more colors than requested (safety check)
    if (colorMap.size() > (size_t)numLabels) {
        colorMap.resize(numLabels);
    }
    
    return colorMap;
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