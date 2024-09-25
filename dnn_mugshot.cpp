#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <vector>
#include <string>

namespace fs = boost::filesystem;

void process_directory(const fs::path& dirPath, cv::dnn::Net& net, std::vector<std::string>& undetectedFiles, int& totalDetected, int& totalUndetected) {
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

                    // Preprocess the image for DNN
                    cv::Mat blob;
                    cv::dnn::blobFromImage(img, blob, 1.0, cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0), false, false);
                    net.setInput(blob);

                    // Forward pass
                    cv::Mat detections = net.forward();

                    int numDetections = detections.size[2];
                    bool faceDetected = false;

                    for (int i = 0; i < numDetections; i++) {
                        float confidence = detections.at<float>(0, 0, i, 2);
                        if (confidence > 0.5) {  // Threshold for detection
                            int x1 = static_cast<int>(detections.at<float>(0, 0, i, 3) * img.cols);
                            int y1 = static_cast<int>(detections.at<float>(0, 0, i, 4) * img.rows);
                            int x2 = static_cast<int>(detections.at<float>(0, 0, i, 5) * img.cols);
                            int y2 = static_cast<int>(detections.at<float>(0, 0, i, 6) * img.rows);
                            cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
                            faceDetected = true;
                        }
                    }

                    if (!faceDetected) {
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
        std::cerr << "Usage: " << argv[0] << " <path_to_model> <path_to_dataset_dir>\n";
        return 1;
    }

    std::string modelPath = argv[1];
    std::string datasetPath = argv[2];

    // Load the DNN model
    cv::dnn::Net net = cv::dnn::readNetFromCaffe(modelPath + "/deploy.prototxt", modelPath + "/res10_300x300_ssd_iter_140000.caffemodel");

    std::vector<std::string> undetectedFiles;
    int totalDetected = 0;
    int totalUndetected = 0;

    fs::path datasetDir(datasetPath);
    if (fs::exists(datasetDir) && fs::is_directory(datasetDir)) {
        process_directory(datasetDir, net, undetectedFiles, totalDetected, totalUndetected);
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
