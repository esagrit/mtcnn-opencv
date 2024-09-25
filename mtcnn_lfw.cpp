#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <vector>
#include <string>

#include "mtcnn/detector.h"
#include "draw.hpp"

namespace fs = boost::filesystem;

void process_directory(const fs::path& dirPath, MTCNNDetector& detector, std::vector<std::string>& undetectedFiles, int& totalDetected, int& totalUndetected) {
    for (fs::recursive_directory_iterator fileIter(dirPath), endIter; fileIter != endIter; ++fileIter) {
        if (fs::is_regular_file(fileIter->status())) {
            std::string filePath = fileIter->path().string();
            std::string fileName = fileIter->path().filename().string();
            
            if (filePath.find_last_of(".") != std::string::npos) {
                std::string ext = filePath.substr(filePath.find_last_of(".") + 1);
                if (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp") {
                    std::cout << "Processing file: " << fileName << std::endl;

                    cv::Mat img = cv::imread(filePath);

                    if (img.empty()) {
                        std::cerr << "Failed to open image: " << filePath << "\n";
                        continue;
                    }

                    std::vector<Face> faces = detector.detect(img, 20.f, 0.709f);

                    if (faces.empty()) {
                        std::cout << "No faces detected in: " << fileName << std::endl;
                        undetectedFiles.push_back(fileName);
                        ++totalUndetected;
                    } else {
                        std::cout << "Faces detected in: " << fileName << std::endl;
                        ++totalDetected;
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_models_dir> <path_to_dataset_dir>\n";
        return 1;
    }

    std::string modelPath = argv[1];
    std::string datasetPath = argv[2];

    ProposalNetwork::Config pConfig;
    pConfig.caffeModel = modelPath + "/det1.caffemodel";
    pConfig.protoText = modelPath + "/det1.prototxt";
    pConfig.threshold = 0.6f;

    RefineNetwork::Config rConfig;
    rConfig.caffeModel = modelPath + "/det2.caffemodel";
    rConfig.protoText = modelPath + "/det2.prototxt";
    rConfig.threshold = 0.7f;

    OutputNetwork::Config oConfig;
    oConfig.caffeModel = modelPath + "/det3.caffemodel";
    oConfig.protoText = modelPath + "/det3.prototxt";
    oConfig.threshold = 0.7f;

    MTCNNDetector detector(pConfig, rConfig, oConfig);

    std::vector<std::string> undetectedFiles;
    int totalDetected = 0;
    int totalUndetected = 0;

    fs::path datasetDir(datasetPath);
    if (fs::exists(datasetDir) && fs::is_directory(datasetDir)) {
        process_directory(datasetDir, detector, undetectedFiles, totalDetected, totalUndetected);
    } else {
        std::cerr << "Dataset directory does not exist or is not a directory.\n";
        return 1;
    }

    std::cout << "Total images detected: " << totalDetected << std::endl;
    std::cout << "Total images not detected: " << totalUndetected << std::endl;

    if (!undetectedFiles.empty()) {
        std::cout << "Files with no faces detected:\n";
        for (const auto& fileName : undetectedFiles) {
            std::cout << fileName << std::endl;
        }
    }

    return 0;
}
