#include <iostream>
#include <filesystem>
#include "mtcnn/helpers.h"
#include <boost/filesystem.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "mtcnn/detector.h"

#include <filesystem>
namespace fs = std::filesystem;

cv::Rect expand_bounding_box(const cv::Rect& bounding_box, int pixels, cv::Mat &img) {
        // Perbesar width dan height dengan menambah beberapa pixel
        int new_x = std::max(bounding_box.x - pixels / 2, 0); // kurangi x untuk menjaga pusat
        int new_y = std::max(bounding_box.y - pixels / 2, 0); // kurangi y untuk menjaga pusat
        int new_width = bounding_box.width + pixels;
        int new_height = bounding_box.height + pixels;

        // Pastikan bounding box yang baru tidak keluar dari batas gambar
        if (new_x + new_width > img.cols) {
            new_width = img.cols - new_x;
        }
        if (new_y + new_height > img.rows) {
            new_height = img.rows - new_y;
        }

        // Return bounding box yang baru
        return cv::Rect(new_x, new_y, new_width, new_height);
}

cv::Mat resizeWithPadding(const cv::Mat& image, cv::Size targetSize) {
        int originalWidth = image.cols;
        int originalHeight = image.rows;
        int targetWidth = targetSize.width;
        int targetHeight = targetSize.height;

        // Hitung rasio untuk mempertahankan aspek rasio
        double ratioWidth = static_cast<double>(targetWidth) / originalWidth;
        double ratioHeight = static_cast<double>(targetHeight) / originalHeight;
        double ratio = std::min(ratioWidth, ratioHeight);

        // Ukuran baru dengan rasio yang dipertahankan
        int newWidth = static_cast<int>(originalWidth * ratio);
        int newHeight = static_cast<int>(originalHeight * ratio);

        // Resize gambar
        cv::Mat resizedImage;
        cv::resize(image, resizedImage, cv::Size(newWidth, newHeight));

        // Hitung padding
        int padLeft = (targetWidth - newWidth) / 2;
        int padRight = targetWidth - newWidth - padLeft;
        int padTop = (targetHeight - newHeight) / 2;
        int padBottom = targetHeight - newHeight - padTop;

        // Tambahkan padding dengan warna hitam (atau warna lain jika diinginkan)
        cv::Mat paddedImage;
        cv::copyMakeBorder(resizedImage, paddedImage, padTop, padBottom, padLeft, padRight, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        return paddedImage;
}

bool is_image_file(const fs::path& file_path) {
    const std::string ext = file_path.extension().string();
    return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp");
}


int main(int argc, char **argv) {



    // Mencatat waktu mulai
    auto start = std::chrono::high_resolution_clock::now();

    // Command to run programs >>> "./build/mtcnn_rafi /Users/mohammadrafii/Documents/engine_fr_cpp/engine_facenet/pure_lfw" <<< 
    if (argc != 2) {
        std::cerr << "Usage: <app_binary> <image_folder>\n";
        return 1;
    }

    // ==================================================================================================== //

        // Init InceptionResnetV1 Models for features extraction
        // MTCNNDetector detector = Helpers::initModel();
        std::string modelPath = "./models/";
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


    // ==================================================================================================== //

    fs::path folder_path = std::string(argv[1]);
    // Mengecek apakah folder_path adalah folder
    if (fs::is_directory(folder_path)) {

        int count = 0;
        std::cout << "\n=====================================================================================================================\n" <<std::endl;
        // Iterasi melalui setiap file dalam folder
        for (const auto& entry : fs::recursive_directory_iterator(folder_path)) {
            if (entry.is_regular_file() && is_image_file(entry.path())) {

                // std::cout << "name file yang terbaca: " << entry.path().string() << "\n";

                // Mendapatkan filename gambar di database
                std::string filename =  entry.path().string();

                std::cout << "filename = " << filename << std::endl;
                
                // Mendapatkan file yang ada di dalam database
                std::string imgPath = filename;
                cv::Mat img = cv::imread(imgPath);
                std::vector<Face> faces = detector.detect(img, 10.f, 0.709f);
                std::cout << "faces size() = " << faces.size() << std::endl;
                std::string output_filename = "./lfw_mtcnn_160/"+entry.path().filename().string();
                if(faces.size() > 0){
                    cv::Rect bounding_box_image = expand_bounding_box(faces[0].bbox.getSquare().getRect(), 10, img);
                    cv::Mat cropped_face = img(bounding_box_image);
                    cv::Mat resized_face = resizeWithPadding(cropped_face, cv::Size(160, 160));
                    bool isSuccess = cv::imwrite(output_filename, resized_face);
                    if (isSuccess) {
                        std::cout << "Gambar berhasil disimpan ke " << output_filename << std::endl;
                    } else {
                        std::cerr << "Error: Tidak dapat menyimpan gambar!" << std::endl;
                    }
                }
                else{
                    cv::Mat resized_face = resizeWithPadding(img, cv::Size(160, 160));
                    bool isSuccess = cv::imwrite(output_filename, resized_face);
                    if (isSuccess) {
                        std::cout << "Gambar berhasil disimpan ke " << output_filename << std::endl;
                    } else {
                        std::cerr << "Error: Tidak dapat menyimpan gambar!" << std::endl;
                    }
                }
                
            }
        }

    

        // Mencatat waktu akhir
        auto end = std::chrono::high_resolution_clock::now();
        // Menghitung durasi dalam milidetik
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Menampilkan waktu yang dibutuhkan
        std::cout << "* Time execution : " << duration.count() << " ms\n" << std::endl;
        std::cout << "=====================================================================================================================\n" << std::endl;
    } else {
        std::cerr << folder_path << " is not a directory." << std::endl;
    }

    return 0;
}
