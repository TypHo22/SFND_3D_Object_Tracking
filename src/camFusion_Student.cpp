
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox,
                              std::vector<cv::KeyPoint> &kptsPrev,
                              std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    //check which keyPoint matches are inside of the area of interest of the bounding box
    //and do filtering accorindg to physical distance
    //https://knowledge.udacity.com/questions/630004

    std::vector<double> physicalDistances;
    for(auto &matches : kptMatches)
    {
        cv::KeyPoint* c = &kptsCurr[matches.trainIdx];
        cv::KeyPoint* p = &kptsPrev[matches.queryIdx];

        physicalDistances.emplace_back(sqrt(std::pow(c->pt.x - p->pt.x,2) + std::pow(c->pt.x - p->pt.x,2)));
    }

    double meanDist;
    double stdDist;
    //calc mean
    for(auto &m : physicalDistances)
        meanDist += m;

    meanDist /= physicalDistances.size();
    //calc standard deviation
    for(auto &s : physicalDistances)
        stdDist +=  (s - meanDist) * (s - meanDist);

    stdDist /= physicalDistances.size();
    stdDist = sqrt(stdDist);

    for(size_t a = 0; a < kptMatches.size(); ++a)
    {
        const double dist = physicalDistances[a];

        if(dist < meanDist + 2 * stdDist &&
           dist > meanDist - 2 * stdDist)
        {
            if(boundingBox.roi.contains(kptsCurr[kptMatches[a].trainIdx].pt))
            {
                boundingBox.kptMatches.push_back(kptMatches[a]);
                boundingBox.keypoints.push_back(kptsCurr[kptMatches[a].trainIdx]);
            }

        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame

    for(auto it0 = kptMatches.begin(); it0 != kptMatches.end() -1; ++it0)
    {
        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint OuterCurr = kptsCurr.at(it0->trainIdx);
        cv::KeyPoint OuterPrev = kptsPrev.at(it0->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance, taken from workspace. I think this is some kind of heuristic value

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint InnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint InnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(OuterCurr.pt - InnerCurr.pt);
            double distPrev = cv::norm(OuterPrev.pt - InnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)  //https://knowledge.udacity.com/questions/668076
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        }
    }

    std::sort(distRatios.begin(), distRatios.end());

    if(!distRatios.empty())
    {
        const double medianDistRatio = distRatios[distRatios.size() / 2];
        // Finally, calculate a TTC estimate based on these 2D camera features
        TTC = (-1.0 / frameRate) / (1 - medianDistRatio);
    }
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{

    // sort the vectors regarding their x-axis, from low to high (will be needed for median calculation)´
    std::sort(lidarPointsCurr.begin(),lidarPointsCurr.end(),[](LidarPoint& a, LidarPoint& b)
    {
        return a.x < b.x;
    });
    std::sort(lidarPointsPrev.begin(),lidarPointsPrev.end(),[](LidarPoint& a, LidarPoint& b)
    {
        return a.x < b.x;
    });

    double meanCurrent = 0.0;
    double meanPrev = 0.0;
    double medianCurrent = 0.0;
    double medianPrev = 0.0;
    double stdCurrent = 0.0;
    double stdPrev = 0.0;

    for(auto& cur : lidarPointsCurr)
         meanCurrent += cur.x;

    for(auto& prev : lidarPointsPrev)
        meanPrev += prev.x;

    meanCurrent /= lidarPointsCurr.size();
    meanPrev    /= lidarPointsPrev.size();
    medianCurrent = (lidarPointsCurr.begin() + lidarPointsCurr.size()/2)->x;
    medianPrev = (lidarPointsPrev.begin() + lidarPointsPrev.size()/2)->x;

    for(auto& cur : lidarPointsCurr)
    {
        const double val = cur.x - meanCurrent;
        stdCurrent += val * val;
    }
    stdCurrent = sqrt(stdCurrent / lidarPointsCurr.size());

    for(auto& prev : lidarPointsPrev)
    {
        const double val = prev.x - meanPrev;
        stdPrev += val * val;
    }
    stdPrev = sqrt(stdPrev / lidarPointsPrev.size());

    //I define outliers the following everything which is greater or smaller than mean +/- standardDeviation
    //remove outliers, on current
    lidarPointsCurr.erase(std::remove_if(lidarPointsCurr.begin(),lidarPointsCurr.end(),
                [meanCurrent,stdCurrent](LidarPoint& p){ return ((p.x > meanCurrent + 2 * stdCurrent) || (p.x < meanCurrent - 2 * stdCurrent));}
    ),lidarPointsCurr.end());
    //remove outliers, on previous
    lidarPointsPrev.erase(std::remove_if(lidarPointsPrev.begin(),lidarPointsPrev.end(),
                [meanPrev,stdPrev](LidarPoint& p){ return ((p.x > meanPrev + 2 * stdPrev) || (p.x < meanPrev - 2 * stdPrev)) ;}),lidarPointsPrev.end());
    //recalculate without outliers
    meanCurrent = 0.0;
    meanPrev = 0.0;

    for(auto& cur : lidarPointsCurr)
         meanCurrent += cur.x;

    for(auto& prev : lidarPointsPrev)
        meanPrev += prev.x;

    meanCurrent /= lidarPointsCurr.size();
    meanPrev    /= lidarPointsPrev.size();

    const double dT = 1 / frameRate;
    TTC = (meanCurrent * dT) / (meanPrev - meanCurrent);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    //https://knowledge.udacity.com/questions/570553#590698
    map<int, int> boxmap{};

    for(auto matchIt = matches.begin(); matchIt != matches.end(); ++matchIt)
    {
        //https://stackoverflow.com/questions/13318853/opencv-drawmatches-queryidx-and-trainidx
        const cv::KeyPoint prevPoints = prevFrame.keypoints[matchIt->queryIdx];//queryIdx comes from the current frame
        const cv::KeyPoint currPoints = currFrame.keypoints[matchIt->trainIdx];//trainIdx comes from the previous frame
        int prevBoxId = -1;//on default
        int currBoxId = -1;//on default

        for (auto &box : prevFrame.boundingBoxes)
            if (box.roi.contains(prevPoints.pt))
                prevBoxId = box.boxID;

        for (auto &box : currFrame.boundingBoxes)
            if (box.roi.contains(currPoints.pt))
                currBoxId = box.boxID;

        // generate currBoxId-prevBoxId map pair
        boxmap.insert({currBoxId, prevBoxId});
    }

    int CurrBoxSize = currFrame.boundingBoxes.size();
    int prevBoxSize = prevFrame.boundingBoxes.size();
    // find the best matched previous boundingbox for each current boudingbox
    for (int i = 0; i < CurrBoxSize; ++i)
    {
        auto boxmapPair = boxmap.equal_range(i);
        vector<int> currBoxCount(prevBoxSize, 0);
        for (auto pr = boxmapPair.first; pr != boxmapPair.second; ++pr)
        {
            if ((*pr).second != -1)
                currBoxCount[(*pr).second] += 1;
        }
        // find the position of best prev box which has highest number of keypoint correspondences.
        const size_t maxPosition = std::distance(currBoxCount.begin(),std::max_element(currBoxCount.begin(), currBoxCount.end()));

        bbBestMatches.insert({maxPosition, i});
    }
}
