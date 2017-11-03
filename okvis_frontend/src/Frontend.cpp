/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 * 
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Mar 27, 2015
 *      Author: Andreas Forster (an.forster@gmail.com)
 *    Modified: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file Frontend.cpp
 * @brief Source file for the Frontend class.
 * @author Andreas Forster
 * @author Stefan Leutenegger
 */

#include <okvis/Frontend.hpp>

#include <brisk/brisk.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

#include <glog/logging.h>

#include <okvis/ceres/ImuError.hpp>
#include <okvis/VioKeyframeWindowMatchingAlgorithm.hpp>
#include <okvis/IdProvider.hpp>

// cameras and distortions
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>

// Kneip RANSAC
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/FrameAbsolutePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/FrameRelativePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/FrameRotationOnlySacProblem.hpp>

#include <set>

/// \brief okvis Main namespace of this package.
namespace okvis {

// Constructor.
Frontend::Frontend(size_t numCameras)
    : isInitialized_(false),
      numCameras_(numCameras),
      briskDetectionOctaves_(0),
      briskDetectionThreshold_(50.0),
      briskDetectionAbsoluteThreshold_(800.0),
      briskDetectionMaximumKeypoints_(450),
      briskDescriptionRotationInvariance_(true),
      briskDescriptionScaleInvariance_(false),
      briskDescriptionPatternScale_(1.0),
      briskMatchingThreshold_(60.0), // default 60.0
      briskMatchingRatioThreshold_(3.0),
      briskMatching_best_second_min_dist_(10),
      matcher_(
          std::unique_ptr<okvis::DenseMatcher>(new okvis::DenseMatcher(1, 8, true))), // default 4: 4 matcher threads, 4 num of best, dont use distance ratio
      keyframeInsertionOverlapThreshold_(0.8), // default 0.6, larger value make more keyframes, but keyframes sitting too close will impose triangulation problem
      keyframeInsertionMatchingRatioThreshold_(0.4),//default 0.2, larger value make more keyframes, but keyframes sitting too close will impose triangulation problem
      rotation_only_ratio_(0.9), // default is 0.8, make it larger so easier to initialize
      ransacinlinersminnumber_(10), // default is 10
      ransacthreshold_(2), //default is 9, is the reprojection error in pixels?
      ransacdebugoutputlevel_(0), //default is 0, 0: no debug info, 1: short summary, 2: output each trial
      ransac_max_iteration_(1000), //default is 50
      required3d2dmatches_(5), //default is 5
      matchtolastKeyframesnumber_for_3d_(10), //default is 3
      matchtolastKeyframesnumber_for_2d_(10), //default is 2
      IsOriginalFeatureDetector_(false)

{

    // create mutexes for feature detectors and descriptor extractors
  for (size_t i = 0; i < numCameras_; ++i)
  {
    featureDetectorMutexes_.push_back(
        std::unique_ptr<std::mutex>(new std::mutex()));
  }
  initialiseBriskFeatureDetectors();
}

// Detection and descriptor extraction on a per image basis.
bool Frontend::detectAndDescribe(size_t cameraIndex,
                                 std::shared_ptr<okvis::MultiFrame> frameOut,
                                 const okvis::kinematics::Transformation& T_WC,
                                 const std::vector<cv::KeyPoint> * keypoints)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIndex < numCameras_, "Camera index exceeds number of cameras.");
  std::lock_guard<std::mutex> lock(*featureDetectorMutexes_[cameraIndex]);

  // check there are no keypoints here
  OKVIS_ASSERT_TRUE(Exception, keypoints == nullptr, "external keypoints currently not supported")

  frameOut->setDetector(cameraIndex, featureDetectors_[cameraIndex]);
  frameOut->setExtractor(cameraIndex, descriptorExtractors_[cameraIndex]);

  frameOut->detect(cameraIndex);

  // ExtractionDirection == gravity direction in camera frame
  // From the paper: better matching result are obtained by extracting descriptors
  // oriented along the gravity direction that is projected into the image
  Eigen::Vector3d g_in_W(0, 0, -1);

  //T_WC maps vector in camera coord to world coord
  //T_WC.inverse() maps vector in world coord to camera coord
  //T_WC.inverse().C() maps vector in world coord to some camera coord with correct orientation
  //T_WC.inverse().C() * g_in_W maps the gravity vector (g_in_W) in world coord to the camera coord with correct orientation
  Eigen::Vector3d extractionDirection = T_WC.inverse().C() * g_in_W;

  //Recalculate the keypt angle with reference to gravity direction and call extractor_->compute()
  frameOut->describe(cameraIndex, extractionDirection);

  // set detector/extractor to nullpointer? TODO
  return true;
}

// Matching as well as initialization of landmarks and state.
bool Frontend::dataAssociationAndInitialization(
    okvis::Estimator& estimator,
    okvis::kinematics::Transformation& /*T_WS_propagated*/, // TODO sleutenegger: why is this not used here?
    const okvis::VioParameters &params,
    const std::shared_ptr<okvis::MapPointVector> /*map*/, // TODO sleutenegger: why is this not used here?
    std::shared_ptr<okvis::MultiFrame> framesInOut,
    bool *asKeyframe)
{

  // match new keypoints to existing landmarks/keypoints
  // initialise new landmarks (states)
  // outlier rejection by consistency check
  // RANSAC (2D2D / 3D2D)
  // decide keyframe
  // left-right stereo match & init

//    //Print out all keypts to debug repeatibility problem
//    const size_t ksize = framesInOut->numKeypoints(0);
//    std::string longstring;
//    for (size_t k = 0; k < ksize; ++k)
//    {
//          Eigen::Vector2d keypt;
//          framesInOut->getKeypoint(0, k, keypt);
//          longstring += ("(" + std::to_string(keypt[0]) + ", " +
//                  std::to_string(keypt[1])+ ") ");
//    }
//    LOG(INFO) << "All keypts: " << longstring;

  // find distortion type
  okvis::cameras::NCameraSystem::DistortionType distortionType = params.nCameraSystem
      .distortionType(0);
  for (size_t i = 1; i < params.nCameraSystem.numCameras(); ++i) {
    OKVIS_ASSERT_TRUE(Exception,
                      distortionType == params.nCameraSystem.distortionType(i),
                      "mixed frame types are not supported yet");
  }
  int num3dMatches = 0;

  // first frame? (did do addStates before, so 1 frame minimum in estimator)
  if (estimator.numFrames() > 1)
  {

    //int requiredMatches = 5;

    double uncertainMatchFraction = 0;
    bool rotationOnly = false;

    // match to last keyframe
    // matchToKeyframes() is just a special case of matchToLastFrame(), where last frame is a key frame
    TimerSwitchable matchKeyframesTimer("2.4.1 matchToKeyframes");
    switch (distortionType)
    {
      case okvis::cameras::NCameraSystem::RadialTangential:
      {
        num3dMatches = matchToKeyframes<
            VioKeyframeWindowMatchingAlgorithm<
                okvis::cameras::PinholeCamera<
                    okvis::cameras::RadialTangentialDistortion> > >(
            estimator, params, framesInOut->id(), rotationOnly, false,
            &uncertainMatchFraction);
        break;
      }
      case okvis::cameras::NCameraSystem::Equidistant:
      {
        num3dMatches = matchToKeyframes<
            VioKeyframeWindowMatchingAlgorithm<
                okvis::cameras::PinholeCamera<
                    okvis::cameras::EquidistantDistortion> > >(
            estimator, params, framesInOut->id(), rotationOnly, false,
            &uncertainMatchFraction);
        break;
      }
      case okvis::cameras::NCameraSystem::RadialTangential8:
      {
        num3dMatches = matchToKeyframes<
            VioKeyframeWindowMatchingAlgorithm<
                okvis::cameras::PinholeCamera<
                    okvis::cameras::RadialTangentialDistortion8> > >(
            estimator, params, framesInOut->id(), rotationOnly, false,
            &uncertainMatchFraction);
        break;
      }
      default:
        OKVIS_THROW(Exception, "Unsupported distortion type.")
        break;
    }
    matchKeyframesTimer.stop();

    if (!isInitialized_)
    {
      if (!rotationOnly)
      {
        isInitialized_ = true;
        LOG(INFO) << "Initialized!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";

        //okvis::kinematics::Transformation T_WS;
        //estimator.get_T_WS(framesInOut->id(), T_WS);

      }
    }

    //num3dMatches is the total number of match for 3D-2D with past 3 keyframes and 2D-2D with past 2 keyframes
    if (isInitialized_ && num3dMatches <= required3d2dmatches_) {
      LOG(WARNING) << "Tracking failure. Number of 3d2d-matches: " << num3dMatches << " <= " << required3d2dmatches_;
    }

    // keyframe decision, at the moment only landmarks that match with keyframe are initialised
    *asKeyframe = *asKeyframe || doWeNeedANewKeyframe(estimator, framesInOut);

    // match to last frame
    TimerSwitchable matchToLastFrameTimer("2.4.2 matchToLastFrame");
    switch (distortionType)
    {
      case okvis::cameras::NCameraSystem::RadialTangential:
      {
        matchToLastFrame<
            VioKeyframeWindowMatchingAlgorithm<
                okvis::cameras::PinholeCamera<
                    okvis::cameras::RadialTangentialDistortion> > >(
            estimator, params, framesInOut->id(),
            false);
        break;
      }
      case okvis::cameras::NCameraSystem::Equidistant:
      {
        matchToLastFrame<
            VioKeyframeWindowMatchingAlgorithm<
                okvis::cameras::PinholeCamera<
                    okvis::cameras::EquidistantDistortion> > >(
            estimator, params, framesInOut->id(),
            false);
        break;
      }
      case okvis::cameras::NCameraSystem::RadialTangential8:
      {
        matchToLastFrame<
            VioKeyframeWindowMatchingAlgorithm<
                okvis::cameras::PinholeCamera<
                    okvis::cameras::RadialTangentialDistortion8> > >(
            estimator, params, framesInOut->id(),
            false);

        break;
      }
      default:
        OKVIS_THROW(Exception, "Unsupported distortion type.")
        break;
    }
    matchToLastFrameTimer.stop();
  }
  else
  {
    *asKeyframe = true;  // first frame needs to be keyframe
  }

  // do stereo match to get new landmarks
  TimerSwitchable matchStereoTimer("2.4.3 matchStereo");
  switch (distortionType)
  {
    case okvis::cameras::NCameraSystem::RadialTangential:
    {
      matchStereo<
          VioKeyframeWindowMatchingAlgorithm<
              okvis::cameras::PinholeCamera<
                  okvis::cameras::RadialTangentialDistortion> > >(estimator,
                                                                  framesInOut);
      break;
    }
    case okvis::cameras::NCameraSystem::Equidistant:
    {
      matchStereo<
          VioKeyframeWindowMatchingAlgorithm<
              okvis::cameras::PinholeCamera<
                  okvis::cameras::EquidistantDistortion> > >(estimator,
                                                             framesInOut);
      break;
    }
    case okvis::cameras::NCameraSystem::RadialTangential8:
    {
      matchStereo<
          VioKeyframeWindowMatchingAlgorithm<
              okvis::cameras::PinholeCamera<
                  okvis::cameras::RadialTangentialDistortion8> > >(estimator,
                                                                   framesInOut);
      break;
    }
    default:
      OKVIS_THROW(Exception, "Unsupported distortion type.")
      break;
  }
  matchStereoTimer.stop();

  return isInitialized_;
}

// Propagates pose, speeds and biases with given IMU measurements.
bool Frontend::propagation(const okvis::ImuMeasurementDeque & imuMeasurements,
                           const okvis::ImuParameters & imuParams,
                           okvis::kinematics::Transformation& T_WS_propagated,
                           okvis::SpeedAndBias & speedAndBiases,
                           const okvis::Time& t_start, const okvis::Time& t_end,
                           Eigen::Matrix<double, 15, 15>* covariance, // output covariance
                           Eigen::Matrix<double, 15, 15>* jacobian) const // output jacobian
{
  if (imuMeasurements.size() < 2)
  {
    LOG(WARNING)
        << "- Skipping propagation as only one IMU measurement has been given to frontend."
        << " Normal when starting up.";
    return 0;
  }
  int measurements_propagated = okvis::ceres::ImuError::propagation(
      imuMeasurements, imuParams, T_WS_propagated, speedAndBiases, t_start,
      t_end, covariance, jacobian);

  return measurements_propagated > 0;
}

// Decision whether a new frame should be keyframe or not.
bool Frontend::doWeNeedANewKeyframe(
    const okvis::Estimator& estimator,
    std::shared_ptr<okvis::MultiFrame> currentFrame)
{

  if (estimator.numFrames() < 2) {
    // just starting, so yes, we need this as a new keyframe
    return true;
  }

  if (!isInitialized_)
  {
      LOG(INFO) << "Still not initialized, dont add keyframe!" ;
        return false;
  }

  double overlap = 0.0;
  double ratio = 0.0;

  // go through all the multi-frames and try to match the initialized keypoints
  for (size_t im = 0; im < currentFrame->numFrames(); ++im)
  {

    // get the hull of all keypoints in current frame
    std::vector<cv::Point2f> frameBPoints, frameBHull;
    std::vector<cv::Point2f> frameBMatches, frameBMatchesHull;
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > frameBLandmarks;

    const size_t numB = currentFrame->numKeypoints(im);
    frameBPoints.reserve(numB);
    frameBLandmarks.reserve(numB);
    Eigen::Vector2d keypoint;

    for (size_t k = 0; k < numB; ++k)
    {
      currentFrame->getKeypoint(im, k, keypoint);
      // insert it
      frameBPoints.push_back(cv::Point2f(keypoint[0], keypoint[1]));
      // also remember matches
      if (currentFrame->landmarkId(im, k) != 0) {
        frameBMatches.push_back(cv::Point2f(keypoint[0], keypoint[1]));
      }
    }

    if (frameBPoints.size() < 3)
      continue;
    cv::convexHull(frameBPoints, frameBHull);
    if (frameBMatches.size() < 3)
      continue;
    cv::convexHull(frameBMatches, frameBMatchesHull);

    // areas
    double frameBArea = cv::contourArea(frameBHull);
    double frameBMatchesArea = cv::contourArea(frameBMatchesHull);

    // overlap area
    double overlapArea = frameBMatchesArea / frameBArea;
    // matching ratio inside overlap area: count
    int pointsInFrameBMatchesArea = 0;
    if (frameBMatchesHull.size() > 2)
    {
      for (size_t k = 0; k < frameBPoints.size(); ++k)
      {
        if (cv::pointPolygonTest(frameBMatchesHull, frameBPoints[k], false)> 0) { // +ve means inside the contour
          pointsInFrameBMatchesArea++;
        }
      }
    }

    double matchingRatio = double(frameBMatches.size()) / double(pointsInFrameBMatchesArea);

    // calculate overlap score, take the max. among all multi-frames
    overlap = std::max(overlapArea, overlap);
    ratio = std::max(matchingRatio, ratio);
  }

  // take a decision
  // overlap: hull of projected and matched keypoint area v.s. hull of projected keypoint area
  // ratio: matched keypt num / #of keypt inside hull of matched keypt area
  LOG(INFO) << "overlap: " << overlap << ", ratio: " << ratio ;
  if (overlap > keyframeInsertionOverlapThreshold_
      && ratio > keyframeInsertionMatchingRatioThreshold_)
  {
    LOG(INFO) << "not add to keyframe" ;
    return false;
  }
  else
  {
    LOG(INFO) << "add to keyframe" ;
    return true;
  }
}

// Match a new multiframe to existing keyframes
// Perform 3D-2D with past 3 keyframes and 2D-2D with past 2 keyframes
template<class MATCHING_ALGORITHM>
int Frontend::matchToKeyframes(okvis::Estimator& estimator,
                               const okvis::VioParameters & params,
                               const uint64_t currentFrameId,
                               bool& rotationOnly,
                               bool usePoseUncertainty,
                               double* uncertainMatchFraction,
                               bool removeOutliers)
{
  //LOG(INFO) << "matchToKeyframes at frame" << currentFrameId;
  rotationOnly = true;
  if (estimator.numFrames() < 2) {
    // just starting, so yes, we need this as a new keyframe
     LOG(INFO) << "estimator.numFrames() < 2, add as a new keyframe, return 0";
    return 0;
  }

//  //Print out all landmarks to debug repeatibility problem
//  PointMap landmarks;
//  std::string longstring;
//  estimator.getLandmarks(landmarks);
//  for (auto i: landmarks)
//  {
//      longstring += ("(" + std::to_string(i.second.point[0]) + ", " +
//              std::to_string(i.second.point[1])+ ", " +
//              std::to_string(i.second.point[2]) + ") ");
//  }
//  LOG(INFO) << "All landmarks: " << longstring;

  int retCtr = 0;
  int numUncertainMatches = 0;

  // go through all the past 3 keyframes and try to match the initialized keypoints
  size_t kfcounter = 0;
  for (size_t age = 1; age < estimator.numFrames(); ++age)
  {
    uint64_t olderFrameId = estimator.frameIdByAge(age);
    if (!estimator.isKeyframe(olderFrameId))
      continue;

    for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im)
    {
      MATCHING_ALGORITHM matchingAlgorithm(estimator,
                                           MATCHING_ALGORITHM::Match3D2D,
                                           briskMatchingThreshold_,
                                           briskMatchingRatioThreshold_,
                                           briskMatching_best_second_min_dist_,
                                           usePoseUncertainty);
      matchingAlgorithm.setFrames(olderFrameId, currentFrameId, im, im);

      // match 3D-2D
      matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
      retCtr += matchingAlgorithm.numMatches();
      numUncertainMatches += matchingAlgorithm.numUncertainMatches();
      LOG(INFO) << "olderFrameId: " << olderFrameId << ", MatchToKeyFrame(Match3D2D).numMatches(): " << matchingAlgorithm.numMatches();

    }

    kfcounter++;
    if (kfcounter >= matchtolastKeyframesnumber_for_3d_)
      break;
  }

  // Do the same thing again with Match2D2D with last 2 keyframes
  kfcounter = 0;
  bool firstFrame = true;
  for (size_t age = 1; age < estimator.numFrames(); ++age)
  {
    uint64_t olderFrameId = estimator.frameIdByAge(age);
    if (!estimator.isKeyframe(olderFrameId))
      continue;

    for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im)
    {
      MATCHING_ALGORITHM matchingAlgorithm(estimator,
                                           MATCHING_ALGORITHM::Match2D2D,
                                           briskMatchingThreshold_,
                                           briskMatchingRatioThreshold_,
                                           briskMatching_best_second_min_dist_,
                                           usePoseUncertainty);
      matchingAlgorithm.setFrames(olderFrameId, currentFrameId, im, im);

      // match 2D-2D for initialization of new (mono-)correspondences
      matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
      //retCtr += matchingAlgorithm.numMatches();
      //numUncertainMatches += matchingAlgorithm.numUncertainMatches();
      LOG(INFO) << "olderFrameId: " << olderFrameId << ", MatchToKeyFrame(Match2D2D).numMatches(): " << matchingAlgorithm.numMatches();
    }

    bool rotationOnly_tmp = false;
    if (isInitialized_)
    {
        // only do RANSAC 3D2D with most recent KF to remove outlier
        if (kfcounter == 0)
          runRansac3d2d(estimator, params.nCameraSystem,
                        estimator.multiFrame(currentFrameId), removeOutliers);
    }
    else
    {
        // do RANSAC 2D2D for initialization only
        runRansac2d2d(estimator, params, currentFrameId, olderFrameId, true,
                      removeOutliers, rotationOnly_tmp);
    }


    if (firstFrame) {
      rotationOnly = rotationOnly_tmp;
      firstFrame = false;
    }

    kfcounter++;
    if (kfcounter >= matchtolastKeyframesnumber_for_2d_)
    {
      break; // break when already encounter the last keyframe
    }
  }

  // calculate fraction of safe matches
  if (uncertainMatchFraction) {
    *uncertainMatchFraction = double(numUncertainMatches) / double(retCtr);
  }

  LOG(INFO) << "Number of UncertainMatches: " << numUncertainMatches;
  LOG(INFO) << "Number of matches: " << retCtr;
  return retCtr;
}

// Match a new multiframe to the last frame.
// Doing 3D-2D and 2D-2D with the last frame for once (if last frame not keyframe)
template<class MATCHING_ALGORITHM>
int Frontend::matchToLastFrame(okvis::Estimator& estimator,
                               const okvis::VioParameters& params,
                               const uint64_t currentFrameId,
                               bool usePoseUncertainty,
                               bool removeOutliers)
{
  //LOG(INFO) << "matchToLastFrame at frame" << currentFrameId;
  if (estimator.numFrames() < 2) {
    LOG(INFO) << "estimator.numFrames() < 2"  ;
    // just starting, so yes, we need this as a new keyframe
    return 0;
  }

  uint64_t lastFrameId = estimator.frameIdByAge(1);

  if (estimator.isKeyframe(lastFrameId)) {
      LOG(INFO) << "estimator.isKeyframe(lastFrameId), already done in matchToKeyframes()"  ;
    // already done
    return 0;
  }

  int retCtr = 0;

  for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im)
  {
    MATCHING_ALGORITHM matchingAlgorithm(estimator,
                                         MATCHING_ALGORITHM::Match3D2D,
                                         briskMatchingThreshold_,
                                         briskMatchingRatioThreshold_,
                                         briskMatching_best_second_min_dist_,
                                         usePoseUncertainty);
    matchingAlgorithm.setFrames(lastFrameId, currentFrameId, im, im);

    // match 3D-2D
    matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
    retCtr += matchingAlgorithm.numMatches();
    LOG(INFO) << "MatchToLastFrame(Match3D2D).numMatches(): " << matchingAlgorithm.numMatches();
  }

  runRansac3d2d(estimator, params.nCameraSystem,
                estimator.multiFrame(currentFrameId), removeOutliers);

  for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im)
  {
    MATCHING_ALGORITHM matchingAlgorithm(estimator,
                                         MATCHING_ALGORITHM::Match2D2D,
                                         briskMatchingThreshold_,
                                         briskMatchingRatioThreshold_,
                                         briskMatching_best_second_min_dist_,
                                         usePoseUncertainty);
    matchingAlgorithm.setFrames(lastFrameId, currentFrameId, im, im);

    // match 2D-2D for initialization of new (mono-)correspondences
    matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
    retCtr += matchingAlgorithm.numMatches();
    LOG(INFO) << "MatchToLastFrame(Match2D2D).numMatches(): " << matchingAlgorithm.numMatches();
  }

  // remove outliers only, not to initialize pose
  bool rotationOnly = false;
  if (!isInitialized_)
    runRansac2d2d(estimator, params, currentFrameId, lastFrameId, false,
                  removeOutliers, rotationOnly);

  LOG(INFO) << "Number of matches: " << retCtr;
  return retCtr;
}

// Match the frames inside the multiframe to each other to initialise new landmarks.
template<class MATCHING_ALGORITHM>
void Frontend::matchStereo(okvis::Estimator& estimator,
                           std::shared_ptr<okvis::MultiFrame> multiFrame)
{

  const size_t camNumber = multiFrame->numFrames();
  const uint64_t mfId = multiFrame->id();

  //LOG(INFO) << "Do matchStereo at frame " << mfId;

  for (size_t im0 = 0; im0 < camNumber; im0++)
  {
    for (size_t im1 = im0 + 1; im1 < camNumber; im1++)
    {
      // first, check the possibility for overlap
      // FIXME: implement this in the Multiframe...!!

      // check overlap
      if(!multiFrame->hasOverlap(im0, im1)){
        continue;
      }

      MATCHING_ALGORITHM matchingAlgorithm(estimator,
                                           MATCHING_ALGORITHM::Match2D2D,
                                           briskMatchingThreshold_,
                                           briskMatchingRatioThreshold_,
                                           briskMatching_best_second_min_dist_,
                                           false);  // TODO: make sure this is changed when switching back to uncertainty based matching
      matchingAlgorithm.setFrames(mfId, mfId, im0, im1);  // newest frame

      // match 2D-2D
      matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);

      // match 3D-2D
      matchingAlgorithm.setMatchingType(MATCHING_ALGORITHM::Match3D2D);
      matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);

      // match 2D-3D
      matchingAlgorithm.setFrames(mfId, mfId, im1, im0);  // newest frame
      matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
    }
  }

  // TODO: for more than 2 cameras check that there were no duplications!

  // TODO: ensure 1-1 matching.

  // TODO: no RANSAC ?


  for (size_t im = 0; im < camNumber; im++)
  {
    const size_t ksize = multiFrame->numKeypoints(im);
    std::set<uint64_t> myset;
    std::set<uint64_t> repeat_lm_set;

    //First, check if any two keypt correspond to the same landmark, if so, add it to repeat_lm_set
    for (size_t k = 0; k < ksize; ++k)
    {
      uint64_t lmid = multiFrame->landmarkId(im, k);
      if ( lmid != 0)
      {
        if(myset.find(lmid)!=myset.end())
        {
           //LOG(INFO) << "multiFrame->landmarkId(im, k) got repeated!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
           repeat_lm_set.insert(lmid);
        }
        myset.insert(lmid);
      }
    }


    for (size_t k = 0; k < ksize; ++k)
    {
      uint64_t lmid = multiFrame->landmarkId(im, k);
      bool isRepeat = (repeat_lm_set.find(lmid)!=repeat_lm_set.end());
      if (lmid != 0)
      {
//        Eigen::Vector2d keypt;
//        multiFrame->getKeypoint(im, k, keypt);
//        MapPoint landmark;
//        estimator.getLandmark(mult=iFrame->landmarkId(im, k), landmark);
//        LOG(INFO) << "KeyPoint: " << keypt.transpose();
//        LOG(INFO) << "Landmark, id: " << landmark.id << " pos: " << landmark.point.transpose();

//        if(myset.find(multiFrame->landmarkId(im, k))!=myset.end())
//        {
//           LOG(INFO) << "multiFrame->landmarkId(im, k) got repeated!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
//        }
//        myset.insert(multiFrame->landmarkId(im, k));

        if(isRepeat)
        {
            multiFrame->setLandmarkId(im, k, 0);
        }

        continue;  // already identified correspondence
      }
      multiFrame->setLandmarkId(im, k, okvis::IdProvider::instance().newId());
    }




    //Try to do online calibration of the extrinsic between camera and imu
    //1cm difference in T_SC.r() x and y is normal
    //Current camera pose in world: by SolvePnp with current image frame and landmarks with 3d info
    //Current imu pose in world: integration from last imu pose
    if(isInitialized_ && multiFrame->id() > 6000)
    {
        std::vector<cv::Point2f> imagePoints;
        std::vector<cv::Point3f> objectPoints;

        for (size_t k = 0; k < ksize; ++k)
        {
            u_int64_t lmid = multiFrame->landmarkId(im, k);
            if(lmid!=0) // if keypt k has a corrospoending landmarks
            {
                if(estimator.isLandmarkAdded(lmid) && estimator.isLandmarkInitialized(lmid))
                {
                    Eigen::Vector2d keypt;
                    multiFrame->getKeypoint(im, k, keypt);
                    MapPoint landmark;
                    estimator.getLandmark(lmid, landmark);

                    imagePoints.emplace_back(keypt[0], keypt[1]);
                    objectPoints.emplace_back(landmark.point[0]/landmark.point[3],
                            landmark.point[1]/landmark.point[3],landmark.point[2]/landmark.point[3]);
                }

            }

        }

        if(imagePoints.size()>5)
        {
            LOG(INFO) << "Start online calibration with " << imagePoints.size() << " points pair ...";
            //LOG(INFO) << "There are " << imagePoints.size() << " imagePoints and " << objectPoints.size() << " objectPoints.";
            cv::Mat cameraMatrix(3,3,cv::DataType<double>::type);
            cv::setIdentity(cameraMatrix);
            cameraMatrix.at<double>(0,0) = 432.13078374;
            cameraMatrix.at<double>(1,1) = 430.14855392;
            cameraMatrix.at<double>(0,2) = 305.71627069;
            cameraMatrix.at<double>(1,2) = 266.44839265;

            cv::Mat distCoeffs(4,1,cv::DataType<double>::type);
            distCoeffs.at<double>(0) = -0.04955165;
            distCoeffs.at<double>(1) = 0.02744712;
            distCoeffs.at<double>(2) = -0.00189845;
            distCoeffs.at<double>(3) = -0.00160872;

            cv::Mat rvec(3,1,cv::DataType<double>::type);
            cv::Mat tvec(3,1,cv::DataType<double>::type);

            //cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
            cv::solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
            cv::Mat rotation(3,3,cv::DataType<double>::type);
            cv::Rodrigues(rvec, rotation);

            //LOG(INFO) << "rvec: " << (57.29577951*rvec).t();// in degree
            //LOG(INFO) << "tvec: " << tvec.t();

            okvis::kinematics::Transformation T_WS, T_SC, T_CW;
            estimator.get_T_WS(multiFrame->id(), T_WS);
            Eigen::Matrix4d T_AB = Eigen::Matrix4d::Identity(4,4);
            Eigen::Matrix3d rot;
            Eigen::Vector3d t;
            //cv2eigen(tvec, T_AB.topRightCorner<3, 1>());
            cv::cv2eigen(tvec, t);
            cv::cv2eigen(rotation, rot);
            T_AB.topRightCorner<3, 1>() = t;
            T_AB.topLeftCorner<3, 3>() = rot;
            T_CW.set(T_AB);

            //Want to calibrate T_SC = inv(T_WS) * T_WC
            T_SC = T_WS.inverse()*T_CW.inverse();
            //LOG(INFO) << "T_WC=" << T_CW.inverse().T();
            LOG(INFO) << "Calibrated T_SC.r()=" << T_SC.r().transpose();

        }
        else
        {
            LOG(INFO) << "Matched and initialized landmarks are not enough to do online calibration!";
        }



    }
  }


}

// Perform 3D/2D RANSAC.
// Used to ensure min. number of inliers and kick-out outlier (i.e. set the keypt landmarkID to 0)
int Frontend::runRansac3d2d(okvis::Estimator& estimator,
                            const okvis::cameras::NCameraSystem& nCameraSystem,
                            std::shared_ptr<okvis::MultiFrame> currentFrame,
                            bool removeOutliers)
{

  LOG(INFO) << "Perform 3D/2D RANSAC at frame" << currentFrame->id();
  if (estimator.numFrames() < 2) {
    // nothing to match against, we are just starting up.
    LOG(INFO) <<  "nothing to match against, we are just starting up.";
    return 1;
  }



  /////////////////////
  //   KNEIP RANSAC
  /////////////////////
  int numInliers = 0;

  // absolute pose adapter for Kneip toolchain
  opengv::absolute_pose::FrameNoncentralAbsoluteAdapter adapter(estimator,
                                                                nCameraSystem,
                                                                currentFrame);

  //These are keypoints that have a
  //corresponding landmark which is added to the estimator,
  //has more than one observation and not at infinity
  size_t numCorrespondences = adapter.getNumberCorrespondences();
  LOG(INFO) << "numCorrespondences: " << numCorrespondences;

  //Print out all the keypt->3d points correspondence for debug repeatibility problem
//  const size_t numK = currentFrame->numKeypoints(0);
//  opengv::points_t all3dpts = adapter.getAllpoints();
//  std::string longstring;
//  for (auto i: all3dpts)
//  {
//      longstring += ("(" + std::to_string(i[0]) + ", " + std::to_string(i[1]) + ", " + std::to_string(i[2]) + ") ");
//  }
//  LOG(INFO) << "All 3d pts: " << longstring;

  if (numCorrespondences < 5)
  {
    LOG(INFO) << "numCorrespondences: " << numCorrespondences << " < 5, return from 3d2d directly";
    return numCorrespondences;
  }
  // create a RelativePoseSac problem and RANSAC
  opengv::sac::Ransac<
      opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem> ransac;
  std::shared_ptr<
      opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem> absposeproblem_ptr(
      new opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem(
          adapter,
          opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem::Algorithm::GP3P));
  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ = ransacthreshold_; // threshold angle between measure vector and reprojected vector
  ransac.max_iterations_ = ransac_max_iteration_; // default: 50
  // initial guess not needed...
  // run the ransac
  ransac.computeModel(ransacdebugoutputlevel_);

  // assign transformation
  numInliers = ransac.inliers_.size();
  LOG(INFO) <<   "numInliers: " << numInliers;

  if (numInliers >= ransacinlinersminnumber_ ||
          (numInliers >= 6 && (double)numInliers/(double)numCorrespondences > 0.85))
  {

    // kick out outliers:
    std::vector<bool> inliers(numCorrespondences, false);
    for (size_t k = 0; k < ransac.inliers_.size(); ++k) {
      inliers.at(ransac.inliers_.at(k)) = true;
    }

    for (size_t k = 0; k < numCorrespondences; ++k)
    {
      if (!inliers[k])
      {
        // get the landmark id:
        size_t camIdx = adapter.camIndex(k);
        size_t keypointIdx = adapter.keypointIndex(k);
        uint64_t lmId = currentFrame->landmarkId(camIdx, keypointIdx);

        MapPoint landmark;
        estimator.getLandmark(lmId, landmark);

        //LOG(INFO) << "Landmarks rejected as outlier in 3d2d: " <<
        //             (landmark.point/landmark.point[3]).transpose();

        // reset ID:
        currentFrame->setLandmarkId(camIdx, keypointIdx, 0);

        // remove observation
        if (removeOutliers)
        {
          estimator.removeObservation(lmId, currentFrame->id(), camIdx, keypointIdx);
          //LOG(INFO) << "An outlier observation is removed!!!!!!";
        }
      }
    }

  }
  else
  {
      LOG(INFO) << "numInliers < ransacinlinersminnumber= " << ransacinlinersminnumber_ << ", dont kick-out outliers";
  }
  return numInliers;
}

// Perform 2D/2D RANSAC for initialization.
// Perform rotation only sac and relative pose sac to see if relative pose result outweighs rotation only sac
// the model_coefficients_ after ransac is used for pose initialization
int Frontend::runRansac2d2d(okvis::Estimator& estimator,
                            const okvis::VioParameters& params,
                            uint64_t currentFrameId, uint64_t olderFrameId,
                            bool initializePose,
                            bool removeOutliers,
                            bool& rotationOnly)
{

  LOG(INFO) << "Perform 2D/2D RANSAC at frame" << currentFrameId;
  // match 2d2d
  rotationOnly = false;
  const size_t numCameras = params.nCameraSystem.numCameras();

  size_t totalInlierNumber = 0;
  bool rotation_only_success = false;
  bool rel_pose_success = false;

  // run relative RANSAC
  for (size_t im = 0; im < numCameras; ++im)
  {

    // relative pose adapter for Kneip toolchain
    opengv::relative_pose::FrameRelativeAdapter adapter(estimator,
                                                        params.nCameraSystem,
                                                        olderFrameId, im,
                                                        currentFrameId, im);

    size_t numCorrespondences = adapter.getNumberCorrespondences();

    LOG(INFO) << "olderFrameId: " << olderFrameId << ", currentFrameId: " << currentFrameId;
    LOG(INFO) << "numCorrespondences: " << numCorrespondences;
    //Print out all the keypt->3d points correspondence for debug repeatibility problem
    //const size_t numK = currentFrame->numKeypoints(0);
//    okvis::Matches allmatches = adapter.getAllMatches();
//    std::string longstring;
//    for (auto i: allmatches)
//    {
//        longstring += ("(" + std::to_string(i.idxA) + ", " + std::to_string(i.idxB) + ") ");
//    }
//    LOG(INFO) << "All matches: " << longstring;


    if (numCorrespondences < 10)
    {
      LOG(INFO) << "numCorrespondences: " << numCorrespondences << " < 10, exit ransac2d2d";
      continue;  // won't generate meaningful results. let's hope the few correspondences we have are all inliers!!
    }
    // try both the rotation-only RANSAC and the relative one:

    // create a RelativePoseSac problem and RANSAC
    typedef opengv::sac_problems::relative_pose::FrameRotationOnlySacProblem FrameRotationOnlySacProblem;
    opengv::sac::Ransac<FrameRotationOnlySacProblem> rotation_only_ransac;
    std::shared_ptr<FrameRotationOnlySacProblem> rotation_only_problem_ptr(
        new FrameRotationOnlySacProblem(adapter));
    rotation_only_ransac.sac_model_ = rotation_only_problem_ptr;
    rotation_only_ransac.threshold_ = ransacthreshold_;
    rotation_only_ransac.max_iterations_ = ransac_max_iteration_; // default: 50

    // run the ransac
    rotation_only_ransac.computeModel(ransacdebugoutputlevel_);

    // get quality
    int rotation_only_inliers = rotation_only_ransac.inliers_.size();
    float rotation_only_ratio = float(rotation_only_inliers) / float(numCorrespondences);

    LOG(INFO) << "rotation_only_inliers: " << rotation_only_inliers;
    LOG(INFO) << "rotation_only_ratio: " << rotation_only_ratio;

    // now the rel_pose one:
    typedef opengv::sac_problems::relative_pose::FrameRelativePoseSacProblem FrameRelativePoseSacProblem;
    opengv::sac::Ransac<FrameRelativePoseSacProblem> rel_pose_ransac;
    std::shared_ptr<FrameRelativePoseSacProblem> rel_pose_problem_ptr(
        new FrameRelativePoseSacProblem(
            adapter, FrameRelativePoseSacProblem::STEWENIUS));

    rel_pose_ransac.sac_model_ = rel_pose_problem_ptr;
    rel_pose_ransac.threshold_ = ransacthreshold_;     //(1.0 - cos(0.5/600));
    rel_pose_ransac.max_iterations_ = ransac_max_iteration_; // default: 50

    // run the ransac
    rel_pose_ransac.computeModel(ransacdebugoutputlevel_);

    // assess success
    int rel_pose_inliers = rel_pose_ransac.inliers_.size();
    float rel_pose_ratio = float(rel_pose_inliers) / float(numCorrespondences);
    LOG(INFO) << "rel_pose_inliers: " << rel_pose_inliers;
    LOG(INFO) << "rel_pose_ratio: " << rel_pose_ratio;


    // decide on success and fill inliers
    std::vector<bool> inliers(numCorrespondences, false);
    if (rotation_only_ratio > rel_pose_ratio || rotation_only_ratio > rotation_only_ratio_) // default 0.8
    {
      LOG(INFO) << "rotation_only_ratio > rel_pose_ratio || rotation_only_ratio > 0.9, rotationOnly = true";
      if (rotation_only_inliers > ransacinlinersminnumber_) {
        rotation_only_success = true;
        LOG(INFO) << "rotation_only_success = true";
      }
      rotationOnly = true;


      totalInlierNumber += rotation_only_inliers;
      for (size_t k = 0; k < rotation_only_ransac.inliers_.size(); ++k)
      {
        inliers.at(rotation_only_ransac.inliers_.at(k)) = true;
      }
    }
    else
    {
      if (rel_pose_inliers > ransacinlinersminnumber_) {
        rel_pose_success = true;
        LOG(INFO) << "rel_pose_success = true";
      }

      totalInlierNumber += rel_pose_inliers;
      for (size_t k = 0; k < rel_pose_ransac.inliers_.size(); ++k)
      {
        inliers.at(rel_pose_ransac.inliers_.at(k)) = true;
      }
    }

    // failure?
    if (!rotation_only_success && !rel_pose_success) {
      LOG(INFO) << "!rotation_only_success && !rel_pose_success, both not success, !fail! exit ransac2d2d";
      continue;
    }

    // otherwise: kick out outliers!
    std::shared_ptr<okvis::MultiFrame> multiFrame = estimator.multiFrame(currentFrameId);
    for (size_t k = 0; k < numCorrespondences; ++k)
    {
      size_t idxB = adapter.getMatchKeypointIdxB(k);
      if (!inliers[k])
      {
        uint64_t lmId = multiFrame->landmarkId(im, k);
        // reset ID:
        multiFrame->setLandmarkId(im, k, 0);
        // remove observation
        if (removeOutliers)
        {
          if (lmId != 0 && estimator.isLandmarkAdded(lmId))
          {
            estimator.removeObservation(lmId, currentFrameId, im, idxB);
            //LOG(INFO) << "An outlier observation is removed!!!!!!";
          }
        }
      }
    }

    // initialize pose if necessary (only called in MatchToKeyFrame)
    if (initializePose && !isInitialized_)
    {
      if (rel_pose_success)
        LOG(INFO)
            << "Initializing pose from 2D-2D RANSAC: rel_pose_success==true";
      else
        LOG(INFO)
            << "Initializing pose from 2D-2D RANSAC: orientation only";

      Eigen::Matrix4d T_C1C2_mat = Eigen::Matrix4d::Identity();

      okvis::kinematics::Transformation T_SCA, T_WSA, T_SC0, T_WS0;
      uint64_t idA = olderFrameId;
      uint64_t id0 = currentFrameId;
      // im is camera id
      estimator.getCameraSensorStates(idA, im, T_SCA); // Get camera states for a given pose ID
      estimator.get_T_WS(idA, T_WSA);
      estimator.getCameraSensorStates(id0, im, T_SC0);
      estimator.get_T_WS(id0, T_WS0);

      //T_WSA and T_WS0 are only from imu propagation

      if (rel_pose_success)
      {
        // update pose
        // if the IMU is used, this will be quickly optimized to the correct scale. Hopefully.
        T_C1C2_mat.topLeftCorner<3, 4>() = rel_pose_ransac.model_coefficients_;
        LOG(INFO) << "T_C1C2 by ransac: " << rel_pose_ransac.model_coefficients_;

        //initialize with projected length according to motion prior.
        //T_C1C2 is the transformation from oldframe to currentframe
        okvis::kinematics::Transformation T_C1C2 = (T_WSA*T_SCA).inverse() * (T_WS0 * T_SC0);
        LOG(INFO) << "T_C1C2 by imu propagation: " << T_C1C2.T();

        //Use the translation of the imu guess to scale the ransac result
        //Project T_C1C2.r() onto T_C1C2_mat.topRightCorner<3, 1>() and get the vector projection
        T_C1C2_mat.topRightCorner<3, 1>() = T_C1C2_mat.topRightCorner<3, 1>()
            * std::max( 0.0, double(T_C1C2_mat.topRightCorner<3, 1>().transpose() * T_C1C2.r())/  // r(): translation vector
                        double(T_C1C2_mat.topRightCorner<3, 1>().transpose()*T_C1C2_mat.topRightCorner<3, 1>()));

        LOG(INFO) << "Scale corrected T_C1C2: " << T_C1C2_mat ;
        LOG(INFO) << "Before init pose T_WS=" << T_WS0.T();
      }
      else
      {
        // rotation only assigned...
        T_C1C2_mat.topLeftCorner<3, 3>() = rotation_only_ransac.model_coefficients_;
      }


      // set., id0 is currentframeID
      // So it goes from (1) world to sensor of lastframe (2) sensor to camera of lastframe
      // (3) camera of lastframe to camera of current frame (4) camera to sensor of currentframe
      estimator.set_T_WS(id0, T_WSA * T_SCA * okvis::kinematics::Transformation(T_C1C2_mat)
              * T_SC0.inverse());

      if (rel_pose_success)
      {
          estimator.get_T_WS(id0, T_WS0);
          LOG(INFO) << "After init pose T_WS=" << T_WS0.T();
      }
    }
  }

  if (rel_pose_success || rotation_only_success)
  {
    return totalInlierNumber;
  }
  else
  {
    LOG(INFO) << "both rel_pose_success and rotation_only_success are false! HACK!";
    rotationOnly = true;  // hack...
    return -1;
  }

  return 0;
}

// (re)instantiates feature detectors and descriptor extractors. Used after settings changed or at startup.
void Frontend::initialiseBriskFeatureDetectors()
{
  for (auto it = featureDetectorMutexes_.begin();
      it != featureDetectorMutexes_.end(); ++it)
  {
    (*it)->lock();
  }

  featureDetectors_.clear();
  descriptorExtractors_.clear();

  for (size_t i = 0; i < numCameras_; ++i)
  {
    if(IsOriginalFeatureDetector_)
    {
        featureDetectors_.push_back(
            std::shared_ptr<cv::FeatureDetector>(
                new brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator>(
                    briskDetectionThreshold_, briskDetectionOctaves_,
                    briskDetectionAbsoluteThreshold_,
                    briskDetectionMaximumKeypoints_)));
    }
    else
    {
        featureDetectors_.push_back(
            std::shared_ptr<cv::FeatureDetector>(
                new cv::PyramidAdaptedFeatureDetector (new cv::GridAdaptedFeatureDetector(
                new cv::FastFeatureDetector(briskDetectionThreshold_),
                    briskDetectionMaximumKeypoints_, 4, 4 ), briskDetectionOctaves_))); // from config file, except the 7x4...

    }


    descriptorExtractors_.push_back(
        std::shared_ptr<cv::DescriptorExtractor>(
            new brisk::BriskDescriptorExtractor(
                briskDescriptionRotationInvariance_,
                briskDescriptionScaleInvariance_,
                2, // Using version2 of brisk
                briskDescriptionPatternScale_)));
  }

  for (auto it = featureDetectorMutexes_.begin();
      it != featureDetectorMutexes_.end(); ++it)
  {
    (*it)->unlock();
  }
}

}  // namespace okvis
