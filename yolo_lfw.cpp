#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <vector>
#include <string>

namespace fs = boost::filesystem;

// Fungsi untuk memproses gambar dan mendeteksi wajah menggunakan YOLO
void detect_faces_yolo(const cv::Mat& img, cv::dnn::Net& net, std::vector<cv::Rect>& faces, const float confThreshold = 0.5f) {
    // Convert image to blob for YOLO
    cv::Mat blob;
    cv::dnn::blobFromImage(img, blob, 1.0 / 255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    
    // Forward pass through the network
    std::vector<cv::Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());

    for (const auto& out : outs) {
        const int numDetections = out.rows;
        for (int i = 0; i < numDetections; ++i) {
            const int confidenceIdx = 5;
            const float confidence = out.at<float>(i, confidenceIdx);
            if (confidence > confThreshold) {
                const int centerX = (int)(out.at<float>(i, 0) * img.cols);
                const int centerY = (int)(out.at<float>(i, 1) * img.rows);
                const int width = (int)(out.at<float>(i, 2) * img.cols);
                const int height = (int)(out.at<float>(i, 3) * img.rows);
                const int left = centerX - width / 2;
                const int top = centerY - height / 2;

                faces.emplace_back(left, top, width, height);
            }
        }
    }
}

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

                    std::vector<cv::Rect> faces;
                    detect_faces_yolo(img, net, faces);

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
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <path_to_yolo_cfg> <path_to_yolo_weights> <path_to_dataset_dir>\n";
        return 1;
    }

    std::string yoloCfg = argv[1];
    std::string yoloWeights = argv[2];
    std::string datasetPath = argv[3];

    // Load YOLO
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(yoloCfg, yoloWeights);
    if (net.empty()) {
        std::cerr << "Could not load YOLO network.\n";
        return 1;
    }

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
