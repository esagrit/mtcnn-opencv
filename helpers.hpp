 cpp
#ifndef HELPERS_HPP
#define HELPERS_HPP

#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "mtcnn/detector.h"
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/matrix.h>
#include <dlib/geometry/vector.h>
#include <dlib/dnn.h>

using rectPoints = std::pair<cv::Rect, std::vector<cv::Point>>;


class Helpers {
public:
    static std::tuple<cv::Mat, dlib::full_object_detection, dlib::matrix<dlib::rgb_pixel>> face_alignment(dlib::shape_predictor &sp, cv::Mat &img, dlib::full_object_detection &shape) {

        // 2D image points. from image
        std::vector<cv::Point2d> image_points;

        // Nose tip
        cv::Point nose_tip(shape.part(30).x(), shape.part(30).y());
        image_points.push_back(nose_tip); 

        // Chin tip
        cv::Point chin_tip(shape.part(8).x(), shape.part(8).y());
        image_points.push_back(chin_tip); 

        // Left eye left corner 
        cv::Point left_eye_tip(shape.part(45).x(), shape.part(45).y());
        image_points.push_back(left_eye_tip); 

        // Right eye right corner
        cv::Point right_eye_tip(shape.part(36).x(), shape.part(36).y());
        image_points.push_back(right_eye_tip);

        // Left Mouth corner
        cv::Point left_mouth_tip(shape.part(54).x(), shape.part(54).y());
        image_points.push_back(left_mouth_tip);  

        // Right mouth corner
        cv::Point right_mouth_tip(shape.part(48).x(), shape.part(48).y());
        image_points.push_back(right_mouth_tip);   

        // 3D model points. from image 1
        std::vector<cv::Point3d> model_points;
        model_points.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));               // Nose tip
        model_points.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));          // Chin
        model_points.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));       // Left eye left corner
        model_points.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));        // Right eye right corner
        model_points.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));      // Left Mouth corner
        model_points.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));       // Right mouth corner
    
        // Camera internals from image 1
        double focal_length1 = img.cols; // Approximate focal length.
        cv::Point2d center = cv::Point2d(img.cols/2,img.rows/2);
        cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << focal_length1, 0, center.x, 0 , focal_length1, center.y, 0, 0, 1);
        cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type); // Assuming no lens distortion

        // Output rotation and translation
        cv::Mat rotation_vector; // Rotation in axis-angle form
        cv::Mat translation_vector;

        // Solve for pose
        cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

        // Convert rotation vector to rotation matrix
        cv::Mat rotation_matrix;
        cv::Rodrigues(rotation_vector, rotation_matrix);

        double yaw;

        yaw = atan2(rotation_matrix.at<double>(1, 0), rotation_matrix.at<double>(0, 0)) * 180.0 / CV_PI;

        // Access the last element in the Rotation Vector in image 1
        double rot = rotation_vector.at<double>(0,2);

        cv::Mat dst;

        // Convert to degrees
        double theta_deg;

        if(yaw < -30 || yaw > 30){
            // Kalau muka terdeteksi dari samping skip alignment
            theta_deg = 0.0;
        }
        else{
            theta_deg = rot/M_PI*180;
        }

        // Rotate around the center
        cv::Point2f pt(img.cols/2., img.rows/2.);
        cv::Mat r = getRotationMatrix2D(pt, theta_deg, 1.0);

        // determine bounding rectangle
        cv::Rect bbox = cv::RotatedRect(pt,img.size(), theta_deg).boundingRect();

        // adjust transformation matrix
        r.at<double>(0,2) += bbox.width/2.0 - center.x;
        r.at<double>(1,2) += bbox.height/2.0 - center.y;



        // Apply affine transform
        cv::warpAffine(img, dst, r, bbox.size());

        cv::Mat image_final = Helpers::face_cropped_after_align(dst,sp);
        // cv::Mat image_final = resizeWithPadding(dst, cv::Size(160, 160)); 

        dlib::matrix<dlib::rgb_pixel> imgRGB;
        dlib::cv_image<dlib::bgr_pixel> dlib_img_after_align(image_final);
        dlib::assign_image(imgRGB, dlib_img_after_align);

        dlib::rectangle dlib_rect_after_align(0, 0, image_final.cols - 1, image_final.rows - 1);

        dlib::full_object_detection shape_after_align = sp(dlib_img_after_align, dlib_rect_after_align);

        return std::make_tuple(image_final, shape_after_align, imgRGB);
    }
    static cv::Mat face_cropped_after_align(cv::Mat &img, dlib::shape_predictor &sp) {
        dlib::matrix<dlib::rgb_pixel> imgRGB;

        dlib::cv_image<dlib::bgr_pixel> dlib_img(img);

        dlib::assign_image(imgRGB, dlib_img);

        dlib::rectangle dlib_rect(0, 0, img.cols - 1, img.rows - 1);

        dlib::full_object_detection shape_after_align = sp(dlib_img, dlib_rect);

        cv::Rect rectangle_face = expand_bounding_box(create_bounding_box(img, shape_after_align),10,img);

        // Get face image
        cv::Mat cropped_face = img(rectangle_face);

        // Resize image into size 1 : 1 
        cv::Mat resized_face = resizeWithPadding(cropped_face, cv::Size(160, 160)); 

        return resized_face;
    }
    static cv::Rect create_bounding_box(cv::Mat &img, const dlib::full_object_detection& shape) {
    // Pastikan shape memiliki setidaknya satu titik
    if (shape.num_parts() == 0) {
        std::cerr << "Error: Shape tidak memiliki titik." << std::endl;
        return cv::Rect();
    }

    // Inisialisasi nilai min dan max dengan koordinat titik pertama
    long min_x = shape.part(0).x();
    long min_y = shape.part(0).y();
    long max_x = shape.part(0).x();
    long max_y = shape.part(0).y();

    // Iterasi melalui semua 68 titik
    for (unsigned long i = 1; i < shape.num_parts(); ++i) {
        long x = shape.part(i).x();
        long y = shape.part(i).y();

        // Update nilai min dan max
        if (x < min_x) min_x = x;
        if (y < min_y) min_y = y;
        if (x > max_x) max_x = x;
        if (y > max_y) max_y = y;
    }

    // Hitung lebar dan tinggi dari bounding box
    int width = max_x - min_x;
    int height = max_y - min_y;

    // Pastikan bounding box berada dalam batas gambar
    min_x = std::max(min_x, 0L); // Batas minimum x adalah 0
    min_y = std::max(min_y, 0L); // Batas minimum y adalah 0
    width = std::min(width, img.cols - static_cast<int>(min_x));  // Batas width tidak melebihi gambar
    height = std::min(height, img.rows - static_cast<int>(min_y));  // Batas height tidak melebihi gambar

    // Pastikan bounding box tidak negatif atau terlalu kecil
    if (width <= 0 || height <= 0) {
        std::cerr << "Error: Bounding box tidak valid setelah penyesuaian." << std::endl;
        return cv::Rect();
    }

    // Buat dan return cv::Rect dari min_x, min_y, width, height
    return cv::Rect(min_x, min_y, width, height);
    }
    static float euclidean_distance_formula(const dlib::matrix<float, 0, 1>& vec1,const dlib::matrix<float, 0, 1>& vec2) {
        return dlib::length(vec1 - vec2);  // Mengembalikan akar kuadrat dari jumlah kuadrat selisih
    }
    static MTCNNDetector initModel(){

        // Calling models for face detection
        std::string modelPath = "./face_detection/models";
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



        // Declare Detect face using MTCNN Models
        MTCNNDetector detector(pConfig, rConfig, oConfig);
        return detector;
    }
    static cv::Mat resizeWithPadding(const cv::Mat& image, cv::Size targetSize) {
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
    static cv::Rect expand_bounding_box(const cv::Rect& bounding_box, int pixels, cv::Mat &img) {
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
};

#endif  // HELPERS_HPP
