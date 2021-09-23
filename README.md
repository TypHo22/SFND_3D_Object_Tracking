# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

**Tasks:** 

**FP.1 Match 3D Objects**

This section matches the bounding boxes with the highest number of keypoint correspondences.
For this check if a points are in a certain area of interest (box) in the two frames and if so
save their ids in a map. 

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

**FP.2 Compute Lidar-based TTC**
sort the vectors of lidarPoints (only needed for median) and remove outliers. I define an outlier everything which 
is greater or smaller mean + 2 * standardDeviation / mean - 2 * standardDeviation. Thats an own statistical definition of an outlier.
If you want to use box-whiskers plot outlier choose 1.5 instead of 2. For the TTC calculation I used the  mean. I only calculated the median
because I wanted to play around with it and get a better feeling for the data.

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

    
   **FP.3 Associate Keypoint Correspondences with Bounding Boxes**
    This was a kinda tricky one because I first misunderstood what is meant  with "distance" 
    so I first did a filtering for kptMatches.distance. This distance would have meant that we filter only
    for the keypoint matches which have the highest similarity (high distance means high difference between the values in the descriptor vectors -> bad match).
    But because we are assuming a rigid transformation between the images we have to choose the physical keypoint distance. I also filter here everything as
    outlier which is greater or smaller mean + 2 * standardDeviation / mean - 2 * standardDeviation.
    
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
    
 **FP.4 Compute Camera-based TTC**
 The TTC gets calculated based on the median. iterate through kptMatches and calculate the relative distances betwen successive frames. 
 based on that we calculate the median which we use for calculating the TTC of the camera
 
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
    
  **FP.5 Performance Evaluation 1**
  On some frames the lidar ttc differs higher than on other frames. 
  As the scenery looks like the car is moving up into a traffic jam. So this error is probably related to the fact that the vehicle is breaking and does not have   a constant velocity anymore
  
  ![image](https://user-images.githubusercontent.com/42981587/134442342-73223295-be19-4350-b21f-7aaa092e87a3.png)

  ![image](https://user-images.githubusercontent.com/42981587/134442363-5b18e2f4-fad2-4b86-9ca0-5f1749c6bc51.png)

  
  **FP.6 Performance Evaluation 2**
  
  I run the example with different detector/descriptor combinations. And I came to the conclusion that it does not make a sense to choose a 
  combinaton based on TTC only (personal opininion here). A combination which always calculates low TTC values would result in a car which raises a warning to       early  which is annoying for the driver. A combination which always calculates high TTC values would result in a car which is more likely to crash because in    some situations the algorithm could think there is enough time left -> road could be wet or the vehicle could be heavily loaded.
  
  So best would be to choose the fastest algorithm to having another calculation frame with a (hopefully) more trustworthy result.
  
  The final TTC should be calculated in a fusion from lidar and camera and also the predecessing ttcs should be taken into account if a value makes sense. 
  For example we calculate the means from lidar and camera and we have a TTC of 13 seconds. In the next frame we have a TTC of 3 seconds (vehicle infront of our car drives out of a tunnel and partially gets covered in sunlight which causes the camera based system to go nuts) 
  so do we have a valid TTC now and our vehicle should do a full break? Probably not so funny on the motorway :) So I think it should also be checked if the TTC     change between two frames makes sense.
  
![image](https://user-images.githubusercontent.com/42981587/134442384-9f64db16-9135-4059-bbaf-8d67d768b048.png)

  
  
  
 
    
    
    
    
