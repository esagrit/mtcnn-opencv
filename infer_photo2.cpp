#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "mtcnn/detector.h"
#include "draw.hpp"

int main(int argc, char **argv) {

  if (argc < 3) {
    std::cerr << "Usage " << ": "
              << "<app_binary> "
              << "<path_to_models_dir>"
              << "<path_to_test_image>\n";
    return 1;
  }

  std::string modelPath = argv[1];

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
  cv::Mat img = cv::imread(argv[2]);

  std::vector<Face> faces =  detector.detect(img, 20.f, 0.709f);

  std::cout << "Number of faces found in the supplied image - " << faces.size()
            << std::endl;

    cv::Rect bbox = faces[0].bbox.getRect();

    cv::Mat cropped_face = img(bbox);

    cv::imshow("image 1", cropped_face);

  cv::waitKey(0);

  return 0;
}
