
#ifndef matching2D_hpp
#define matching2D_hpp

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"


/**
 * @brief detKeypointsHarris
 * @param keypoints
 * @param img
 * @param bVis
 * detect keypoints with harris algorithm
 */
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis=false);
/**
 * @brief detKeypointsShiTomasi
 * @param keypoints
 * @param img
 * @param bVis
 * detect features with ShiTomasi
 */
void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis=false);

/**
 * @brief detKeypointsModern
 * @param keypoints
 * @param img
 * @param detectorType
 * @param bVis
 * detect keypoints with different feature detectors
 */
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, DETECTOR detectorType, bool bVis=false);

/**
 * @brief descKeypoints
 * @param keypoints
 * @param img
 * @param descriptors
 * @param descriptorType
 * To extract descriptors from a set of keypoints
 */
void descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, DESCRIPTOR descriptorType);

/**
 * @brief matchDescriptors
 * @param kPtsSource
 * @param kPtsRef
 * @param descSource
 * @param descRef
 * @param matches
 * @param descriptorType
 * @param matcherType
 * @param selectorType
 * find best matches between two images
 */
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, DESCRIPTORFAMILY descriptorFamily, MATCHER matcherType, SELECTOR selectorType);

#endif /* matching2D_hpp */
