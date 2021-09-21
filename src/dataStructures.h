
#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <map>
#include <opencv2/core.hpp>

struct LidarPoint { // single lidar point in space
    double x,y,z,r; // x,y,z in [m], r is point reflectivity
};

struct BoundingBox { // bounding box around a classified object (contains both 2D and 3D data)

    int boxID; // unique identifier for this bounding box
    int trackID; // unique identifier for the track to which this bounding box belongs

    cv::Rect roi; // 2D region-of-interest in image coordinates
    int classID; // ID based on class file provided to YOLO framework
    double confidence; // classification trust

    std::vector<LidarPoint> lidarPoints; // Lidar 3D points which project into 2D image roi
    std::vector<cv::KeyPoint> keypoints; // keypoints enclosed by 2D roi
    std::vector<cv::DMatch> kptMatches; // keypoint matches enclosed by 2D roi
};

struct DataFrame { // represents the available sensor information at the same time instance

    cv::Mat cameraImg; // camera image

    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
    std::vector<LidarPoint> lidarPoints;

    std::vector<BoundingBox> boundingBoxes; // ROI around detected objects in 2D image coordinates
    std::map<int,int> bbMatches; // bounding box matches between previous and current frame
};

enum struct DETECTOR
{
    SHITOMASI,
    HARRIS,
    FAST,
    BRISK,
    ORB,
    AKAZE,
    SIFT
};

enum struct DESCRIPTOR
{
    BRISK,
    BRIEF,
    ORB,
    FREAK,
    AKAZE,
    SIFT
};

enum struct DESCRIPTORFAMILY
{
    BIN, //Binary
    HOG //Histograms of Oriented Gradients (HOG)
};

enum struct SELECTOR
{
    SEL_NN,
    SEL_KNN
};

enum struct MATCHER
{
    MAT_BF,
    MAT_FLANN
};

static std::string getDescriptorName(DESCRIPTOR descriptorType)
{
    switch (descriptorType)
    {
        case(DESCRIPTOR::BRISK):
        {
            return "BRISK";
        }
        case(DESCRIPTOR::BRIEF):
        {
            return "BRIEF";
        }
        case(DESCRIPTOR::ORB):
        {
            return "ORB";
        }
        case(DESCRIPTOR::FREAK):
        {
            return "FREAK";
        }
        case(DESCRIPTOR::AKAZE):
        {
            return "AKAZE";
        }
        case(DESCRIPTOR::SIFT):
        {
            return "SIFT";
        }
    }
    return "No valid descriptor";
}

static std::string getDetectorName(DETECTOR detectorType)
{
    switch (detectorType)
    {
        case(DETECTOR::AKAZE):
        {
            return "AKAZE";
        }
        case(DETECTOR::BRISK):
        {
            return "BRISK";
        }
        case(DETECTOR::FAST):
        {
            return "FAST";
        }
        case(DETECTOR::HARRIS):
        {
            return "HARRIS";
        }
        case(DETECTOR::ORB):
        {
            return "ORB";
        }
        case(DETECTOR::SHITOMASI):
        {
            return "SHITOMASI";
        }
        case(DETECTOR::SIFT):
        {
            return "SIFT";
        }
    }
    return "No valid detector";
}

#endif /* dataStructures_h */
