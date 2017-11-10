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
 *  Created on: Oct 17, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file VioKeyframeWindowMatchingAlgorithm.cpp
 * @brief Source file for the VioKeyframeWindowMatchingAlgorithm class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <okvis/VioKeyframeWindowMatchingAlgorithm.hpp>
#include <okvis/ceres/ReprojectionError.hpp>
#include <okvis/IdProvider.hpp>
#include <okvis/cameras/CameraBase.hpp>
#include <okvis/MultiFrame.hpp>

// cameras and distortions
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>

#include <opencv2/features2d/features2d.hpp> // for cv::KeyPoint

/// \brief okvis Main namespace of this package.
namespace okvis
{

// Constructor.
template<class CAMERA_GEOMETRY_T>
VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::VioKeyframeWindowMatchingAlgorithm(
    okvis::Estimator& estimator, int matchingType, float distanceThreshold,
    float distanceRatioThreshold, float best_second_min_dist,
    bool usePoseUncertainty)
{
  matchingType_ = matchingType;
  distanceThreshold_ = distanceThreshold;
  distanceRatioThreshold_ = distanceRatioThreshold;
  best_second_min_dist_ = best_second_min_dist;
  estimator_ = &estimator;
  usePoseUncertainty_ = usePoseUncertainty;
}

template<class CAMERA_GEOMETRY_T>
VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::~VioKeyframeWindowMatchingAlgorithm() {

}

// Set which frames to match.
// Current frame is B, old frame is A
template<class CAMERA_GEOMETRY_T>
void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::setFrames(
    uint64_t mfIdA, uint64_t mfIdB, size_t camIdA, size_t camIdB)
{

  OKVIS_ASSERT_TRUE(Exception, !(mfIdA == mfIdB && camIdA == camIdB),
                    "trying to match identical frames.");

  // remember indices
  mfIdA_ = mfIdA;
  mfIdB_ = mfIdB;
  camIdA_ = camIdA;
  camIdB_ = camIdB;

  // frames and related information
  frameA_ = estimator_->multiFrame(mfIdA_);
  frameB_ = estimator_->multiFrame(mfIdB_);

  // focal length
  fA_ = frameA_->geometryAs<CAMERA_GEOMETRY_T>(camIdA_)->focalLengthU();
  fB_ = frameB_->geometryAs<CAMERA_GEOMETRY_T>(camIdB_)->focalLengthU();

  // calculate the relative transformations and uncertainties
  // TODO donno, if and what we need here - I'll see
  estimator_->getCameraSensorStates(mfIdA_, camIdA, T_SaCa_);
  estimator_->getCameraSensorStates(mfIdB_, camIdB, T_SbCb_);
  estimator_->get_T_WS(mfIdA_, T_WSa_);
  estimator_->get_T_WS(mfIdB_, T_WSb_);

  T_SaW_ = T_WSa_.inverse();
  T_SbW_ = T_WSb_.inverse();
  T_WCa_ = T_WSa_ * T_SaCa_;
  T_WCb_ = T_WSb_ * T_SbCb_;
  T_CaW_ = T_WCa_.inverse();
  T_CbW_ = T_WCb_.inverse();
  T_CaCb_ = T_WCa_.inverse() * T_WCb_;
  T_CbCa_ = T_CaCb_.inverse();

  validRelativeUncertainty_ = false;
}

// Set the matching type.
template<class CAMERA_GEOMETRY_T>
void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::setMatchingType(
    int matchingType) {
  matchingType_ = matchingType;
}

// This will be called exactly once for each call to DenseMatcher::match().
template<class CAMERA_GEOMETRY_T>
void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::doSetup()
{

  std::string debugstring;
  // setup stereo triangulator
  // first, let's get the relative uncertainty.
  okvis::kinematics::Transformation T_CaCb;
  Eigen::Matrix<double, 6, 6> UOplus = Eigen::Matrix<double, 6, 6>::Zero();

  if (usePoseUncertainty_) {
    OKVIS_THROW(Exception, "No pose uncertainty use currently supported");
  }
  else
  {
    UOplus.setIdentity();
    UOplus.bottomRightCorner<3, 3>() *= 1e-8; // relative uncertainty of rotation
    uint64_t currentId = estimator_->currentFrameId();

    if (estimator_->isInImuWindow(currentId) && (mfIdA_ != mfIdB_))
    {
      okvis::SpeedAndBias speedAndBias;
      estimator_->getSpeedAndBias(currentId, 0, speedAndBias);
      double scale = std::max(1.0, speedAndBias.head<3>().norm());

      // relative translation uncertainty depends on current estimated speed
      UOplus.topLeftCorner<3, 3>() *= (scale * scale) * 1.0e-2;
    }
    else
    {
      UOplus.topLeftCorner<3, 3>() *= 4e-8;
    }
  }

  //LOG(INFO) << "DoSetup: UOplus:" << UOplus;

  // now set the frames and uncertainty
  probabilisticStereoTriangulator_.resetFrames(frameA_, frameB_, camIdA_,
                                               camIdB_, T_CaCb_, UOplus);

  // reset the match counter
  numMatches_ = 0;
  numUncertainMatches_ = 0;

  const size_t numA = frameA_->numKeypoints(camIdA_);

  //Should keypoint[index] in frame A be skipped
  skipA_.clear();
  skipA_.resize(numA, false);
  raySigmasA_.resize(numA);
  int gotprojectedPtACounter=0, gotobservredPtACounter=0, AIsLandMark=0, InitedLandmark=0;

  // calculate projections only once
  if (matchingType_ == Match3D2D)
  {
    // allocate a matrix to store projections
    projectionsIntoB_ = Eigen::Matrix<double, Eigen::Dynamic, 2>::Zero(sizeA(),
                                                                       2);
    projectionsIntoBUncertainties_ =
        Eigen::Matrix<double, Eigen::Dynamic, 2>::Zero(sizeA() * 2, 2);

    // do the projections for each keypoint, if applicable
    // Only initialized landmark will be projected
    for (size_t k = 0; k < numA; ++k)
    {
      uint64_t lm_id = frameA_->landmarkId(camIdA_, k);

      if (lm_id == 0 || !estimator_->isLandmarkAdded(lm_id)) {
        // this can happen, if you called the 2D-2D version just before,
        // without inserting the landmark into the graph
        skipA_[k] = true;
        continue;
      }

      AIsLandMark++;

      okvis::MapPoint landmark;
      estimator_->getLandmark(lm_id, landmark);
      Eigen::Vector4d hp_W = landmark.point;

      //Only use initialized landmark when projecting 3D-2D
      if (!estimator_->isLandmarkInitialized(lm_id)) {
        skipA_[k] = true;
        continue;
      }

      InitedLandmark++;

      // project (distorted)
      Eigen::Vector2d kptB;
      const Eigen::Vector4d hp_Cb = T_CbW_ * hp_W;
      if (frameB_->geometryAs<CAMERA_GEOMETRY_T>(camIdB_)->projectHomogeneous(
          hp_Cb, &kptB)
          != okvis::cameras::CameraBase::ProjectionStatus::Successful)
      {
        skipA_[k] = true;
        continue;
      }

      gotprojectedPtACounter++;

      if (landmark.observations.size() < 2)
      {
        estimator_->setLandmarkInitialized(lm_id, false);
        skipA_[k] = true;
        continue;
      }

       gotobservredPtACounter++;

      // project and get uncertainty
      Eigen::Matrix<double, 2, 4> jacobian;
      Eigen::Matrix4d P_C = Eigen::Matrix4d::Zero();
      P_C.topLeftCorner<3, 3>() = UOplus.topLeftCorner<3, 3>();  // get from before -- velocity scaled
      frameB_->geometryAs<CAMERA_GEOMETRY_T>(camIdB_)->projectHomogeneous(
          hp_Cb, &kptB, &jacobian);
      projectionsIntoBUncertainties_.block<2, 2>(2 * k, 0) = jacobian * P_C
          * jacobian.transpose();
      projectionsIntoB_.row(k) = kptB;

      // precalculate ray uncertainties
      double keypointAStdDev;
      frameA_->getKeypointSize(camIdA_, k, keypointAStdDev);
      keypointAStdDev = 0.8 * keypointAStdDev / 12.0;
      raySigmasA_[k] = sqrt(sqrt(2)) * keypointAStdDev / fA_;  // (sqrt(MeasurementCovariance.norm()) / _fA)
    }

    LOG(INFO) << "landmark got projected: " << gotobservredPtACounter << "/" <<
                 gotprojectedPtACounter << "/" << InitedLandmark << "/" << AIsLandMark;

//    LOG(INFO) << "dosetup (3D2D) projectionsIntoBUncertainties_: " << projectionsIntoBUncertainties_.transpose();
//    LOG(INFO) << "dosetup (3D2D) projectionsIntoB_: " << projectionsIntoB_.transpose();

//    debugstring="";
//    for(auto i: raySigmasA_)
//        debugstring += (std::to_string(i) + ", ");
//    LOG(INFO) << "dosetup (3D2D) raySigmasA_: " << debugstring;

//    debugstring="";
//    for(auto i: skipA_)
//        debugstring += (std::to_string(i) + ", ");
//    LOG(INFO) << "dosetup (3D2D) skipA_: " << debugstring;
  }

  //matchingType_ == Match2D2D
  else
  {
    for (size_t k = 0; k < numA; ++k)
    {
      double keypointAStdDev;
      frameA_->getKeypointSize(camIdA_, k, keypointAStdDev);
      keypointAStdDev = 0.8 * keypointAStdDev / 12.0;
      raySigmasA_[k] = sqrt(sqrt(2)) * keypointAStdDev / fA_;

      if (frameA_->landmarkId(camIdA_, k) == 0) {
        continue;
      }

      if (estimator_->isLandmarkAdded(frameA_->landmarkId(camIdA_, k)))
      {
        if (estimator_->isLandmarkInitialized(
            frameA_->landmarkId(camIdA_, k))) {
          skipA_[k] = true;
        }
      }

    }

//    debugstring="";
//    for(auto i: raySigmasA_)
//        debugstring += (std::to_string(i) + ", ");
//    LOG(INFO) << "dosetup (2D2D) raySigmasA_: " << debugstring;

//    debugstring="";
//    for(auto i: skipA_)
//        debugstring += (std::to_string(i) + ", ");
//    LOG(INFO) << "dosetup (2D2D) skipA_: " << debugstring;
  }
  const size_t numB = frameB_->numKeypoints(camIdB_);

  // Should keypoint[index] in frame B be skipped
  skipB_.clear();
  skipB_.reserve(numB);
  raySigmasB_.resize(numB);

  // do the projections for each keypoint, if applicable
  if (matchingType_ == Match3D2D)
  {
    for (size_t k = 0; k < numB; ++k)
    {
      okvis::MapPoint landmark;
      if (frameB_->landmarkId(camIdB_, k) != 0
          && estimator_->isLandmarkAdded(frameB_->landmarkId(camIdB_, k)))
      {
        estimator_->getLandmark(frameB_->landmarkId(camIdB_, k), landmark);
        skipB_.push_back(
            landmark.observations.find(
                okvis::KeypointIdentifier(mfIdB_, camIdB_, k))
                != landmark.observations.end());
      }
      else
      {
        skipB_.push_back(false);
      }

      double keypointBStdDev;
      frameB_->getKeypointSize(camIdB_, k, keypointBStdDev);
      keypointBStdDev = 0.8 * keypointBStdDev / 12.0;
      raySigmasB_[k] = sqrt(sqrt(2)) * keypointBStdDev / fB_;
    }

//    debugstring="";
//    for(auto i: raySigmasB_)
//        debugstring += (std::to_string(i) + ", ");
//    LOG(INFO) << "dosetup (3D2D) raySigmasB_: " << debugstring;

//    debugstring="";
//    for(auto i: skipB_)
//        debugstring += (std::to_string(i) + ", ");
//    LOG(INFO) << "dosetup (3D2D) skipB_: " << debugstring;
  }

  //matchingType_ == Match2D2D
  else
  {
    for (size_t k = 0; k < numB; ++k)
    {
      double keypointBStdDev;
      frameB_->getKeypointSize(camIdB_, k, keypointBStdDev);
      keypointBStdDev = 0.8 * keypointBStdDev / 12.0;
      raySigmasB_[k] = sqrt(sqrt(2)) * keypointBStdDev / fB_;

      if (frameB_->landmarkId(camIdB_, k) == 0) {
        skipB_.push_back(false);
        continue;
      }

      if (estimator_->isLandmarkAdded(frameB_->landmarkId(camIdB_, k))) {
        skipB_.push_back(
            estimator_->isLandmarkInitialized(frameB_->landmarkId(camIdB_, k)));  // old: isSet - check.
      }
      else
      {
        skipB_.push_back(false);
      }
    }

//    debugstring="";
//    for(auto i: raySigmasB_)
//        debugstring += (std::to_string(i) + ", ");
//    LOG(INFO) << "dosetup (2D2D) raySigmasB_: " << debugstring;

//    debugstring="";
//    for(auto i: skipB_)
//        debugstring += (std::to_string(i) + ", ");
//    LOG(INFO) << "dosetup (2D2D) skipB_: " << debugstring;
  }

}

// What is the size of list A?
template<class CAMERA_GEOMETRY_T>
size_t VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::sizeA() const {
  return frameA_->numKeypoints(camIdA_);
}
// What is the size of list B?
template<class CAMERA_GEOMETRY_T>
size_t VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::sizeB() const {
  return frameB_->numKeypoints(camIdB_);
}

// Set the distance threshold for which matches exceeding it will not be returned as matches.
template<class CAMERA_GEOMETRY_T>
void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::setDistanceThreshold(
    float distanceThreshold) {
  distanceThreshold_ = distanceThreshold;
}

// Get the distance threshold for which matches exceeding it will not be returned as matches.
template<class CAMERA_GEOMETRY_T>
float VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::distanceThreshold() const {
  return distanceThreshold_;
}

// Get the distance ratio threshold for which matches exceeding it will not be returned as matches.
template<class CAMERA_GEOMETRY_T>
float VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::distanceRatioThreshold() const {
  return distanceRatioThreshold_;
}

// Get the distance ratio threshold for which matches exceeding it will not be returned as matches.
template<class CAMERA_GEOMETRY_T>
float VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::best_second_min_dist() const {
  return best_second_min_dist_;
}

// Geometric verification of a match.
template<class CAMERA_GEOMETRY_T>
bool VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::verifyMatch(
    size_t indexA, size_t indexB) const
{

  if (matchingType_ == Match2D2D)
  {

    // potential 2d2d match - verify by triangulation
    Eigen::Vector4d hP;
    bool isParallel;
    bool valid = probabilisticStereoTriangulator_.stereoTriangulate(
        indexA, indexB, hP, isParallel, std::max(raySigmasA_[indexA], raySigmasB_[indexB]));
    if (valid) {
      return true;
    }
  }

  else
  {
    // get projection into B
    Eigen::Vector2d kptB = projectionsIntoB_.row(indexA);

    // uncertainty
    double keypointBStdDev;
    frameB_->getKeypointSize(camIdB_, indexB, keypointBStdDev);
    keypointBStdDev = 0.8 * keypointBStdDev / 12.0;
    Eigen::Matrix2d U = Eigen::Matrix2d::Identity() * keypointBStdDev
        * keypointBStdDev
        + projectionsIntoBUncertainties_.block<2, 2>(2 * indexA, 0);

    Eigen::Vector2d keypointBMeasurement;
    frameB_->getKeypoint(camIdB_, indexB, keypointBMeasurement);
    Eigen::Vector2d err = kptB - keypointBMeasurement;
    const int chi2 = err.transpose() * U.inverse() * err;

    if (chi2 < 4.0) {
      return true;
    }
  }
  return false;
}

// A function that tells you how many times setMatching() will be called.
template<class CAMERA_GEOMETRY_T>
void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::reserveMatches(
    size_t /*numMatches*/) {
  //_triangulatedPoints.clear();
}

// Get the number of matches.
template<class CAMERA_GEOMETRY_T>
size_t VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::numMatches() {
  return numMatches_;
}

// Get the number of uncertain matches.
template<class CAMERA_GEOMETRY_T>
size_t VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::numUncertainMatches() {
  return numUncertainMatches_;
}

// At the end of the matching step, this function is called once
// for each pair of matches discovered.
// B represents current frames
template<class CAMERA_GEOMETRY_T>
void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::setBestMatch(
    size_t indexA, size_t indexB, double /*distance*/)
{

  // assign correspondences
  uint64_t lmIdA = frameA_->landmarkId(camIdA_, indexA);
  uint64_t lmIdB = frameB_->landmarkId(camIdB_, indexB);

  //Use domain knowledge of the ceiling approx height to kill wrong hP
  const double minheight = 1.8;
  const double maxheight = 6.0;

  if (matchingType_ == Match2D2D)
  {

    // check that not both are set (e.g. not both of them are landmark already)
    if (lmIdA != 0 && lmIdB != 0) {
      return;
    }

    // re-triangulate...
    // potential 2d2d match - verify by triangulation
    Eigen::Vector4d hP_Ca; // homogenous point coord of the match in camera frameA, triangulate in triangulateFast()
    bool canBeInitialized;

    //Being "valid" is a minimum requirement
    //2 rays very parallel can still be valid if (e1.cross(e2)).norm() < 6 * sigma
    //Valid if the reprojectio error to both frameA and frameB < 4px
    //If it is valid, it will be added to landmark already
    //After being "valid", the next level is being "canBeInitialized"
    //"canBeInitialized" need the 2 rays not being parallel and
    //the reprojection error of a "out-epipolar" pt must be large enough
    bool valid = probabilisticStereoTriangulator_.stereoTriangulate(
        indexA, indexB, hP_Ca, canBeInitialized,
        std::max(raySigmasA_[indexA], raySigmasB_[indexB]));

    //LOG(INFO) << "hP_Ca: " << hP_Ca.transpose();
    //LOG(INFO) << "T_WCa_ * hP_Ca: " << (T_WCa_ * hP_Ca).transpose();
    //if((T_WCa_ * hP_Ca)[2]<0 || (T_WCa_ * hP_Ca)[2]>1000) LOG(WARNING) << "triangulation wrong!";
    if (!valid) {
      return;
    }

//    Eigen::Vector4d hPP = T_WCa_ * hP_Ca;
//    if(frameB_->id() > 6000 && // since height measurement at the beginning is not so accurate
//            (hPP[2]/hPP[3] < minheight || hPP[2]/hPP[3] > maxheight))
//    {
//        //LOG(INFO) << "hPP violate domain knowledge: " << (T_WCa_ * hP_Ca).transpose();
//        canBeInitialized=false;
//    }

    // get the uncertainty
    if (canBeInitialized)
    {
      // know more exactly, but pointUOplus_A seems to have no usage
      Eigen::Matrix3d pointUOplus_A;
      probabilisticStereoTriangulator_.getUncertainty(indexA, indexB, hP_Ca,
                                                      pointUOplus_A,
                                                      canBeInitialized);
    }

    // check and adapt landmark status
    bool insertA = lmIdA == 0;
    bool insertB = lmIdB == 0;
    bool insertHomogeneousPointParameterBlock = false; // it determines whether a keypoint will be added to landmark
    uint64_t lmId = 0;  // 0 just to avoid warning

    if (insertA && insertB) // both are not landmark
    {
      // ok, we need to assign a new Id...
      lmId = okvis::IdProvider::instance().newId();
      frameA_->setLandmarkId(camIdA_, indexA, lmId); // since they are looking at physically same landmark
      frameB_->setLandmarkId(camIdB_, indexB, lmId); // since they are looking at physically same landmark
      lmIdA = lmId;
      lmIdB = lmId;
      // and add it to the graph
      insertHomogeneousPointParameterBlock = true;
    }
    else // it happens when one is a new keypt while another is a keypt that is triangulated already
    {
      if (!insertA) // If A already is a landmark
      {
        lmId = lmIdA;
        if (!estimator_->isLandmarkAdded(lmId)) {
          // add landmark and observation to the graph
          insertHomogeneousPointParameterBlock = true;
          insertA = true;
        }
      }
      if (!insertB) {
        lmId = lmIdB;
        if (!estimator_->isLandmarkAdded(lmId)) {
          // add landmark and observation to the graph
          insertHomogeneousPointParameterBlock = true;
          insertB = true;
        }
      }
    }

    // add landmark to graph if necessary
    // T_WCa_ = T_WSa_ * T_SaCa_;
    if (insertHomogeneousPointParameterBlock)
    {
      //Add HomogeneousPointParameterBlock
      estimator_->addLandmark(lmId, T_WCa_ * hP_Ca); // mapPtr_->addParameterBlock() which will be used in ceres
      OKVIS_ASSERT_TRUE(Exception, estimator_->isLandmarkAdded(lmId),
                        lmId<<" not added, bug");
      estimator_->setLandmarkInitialized(lmId, canBeInitialized);
    }
    else
    {

      // update initialization status, set better estimate, if possible
      if (canBeInitialized) {
        estimator_->setLandmarkInitialized(lmId, true);
        estimator_->setLandmark(lmId, T_WCa_ * hP_Ca);
      }
    }

    // in image A
    okvis::MapPoint landmark;
    if (insertA && landmark.observations.find(
            okvis::KeypointIdentifier(mfIdA_, camIdA_, indexA))
            == landmark.observations.end())
    {
      // ensure no double observations...
      // TODO hp_Sa NOT USED!
      Eigen::Vector4d hp_Sa(T_SaCa_ * hP_Ca);
      hp_Sa.normalize();
      frameA_->setLandmarkId(camIdA_, indexA, lmId);
      lmIdA = lmId;
      // initialize in graph
      OKVIS_ASSERT_TRUE(Exception, estimator_->isLandmarkAdded(lmId),
                        "landmark id=" << lmId<<" not added");
      estimator_->addObservation<camera_geometry_t>(lmId, mfIdA_, camIdA_,
                                                    indexA);
    }

    // in image B
    if (insertB && landmark.observations.find(
            okvis::KeypointIdentifier(mfIdB_, camIdB_, indexB))
            == landmark.observations.end())
    {
      // ensure no double observations...
      Eigen::Vector4d hp_Sb(T_SbCb_ * T_CbCa_ * hP_Ca);
      hp_Sb.normalize();
      frameB_->setLandmarkId(camIdB_, indexB, lmId);
      lmIdB = lmId;
      // initialize in graph
      OKVIS_ASSERT_TRUE(Exception, estimator_->isLandmarkAdded(lmId),
                        "landmark " << lmId << " not added");
      estimator_->addObservation<camera_geometry_t>(lmId, mfIdB_, camIdB_,
                                                    indexB);
    }

    // let's check for consistency with other observations:
    okvis::ceres::HomogeneousPointParameterBlock point(T_WCa_ * hP_Ca, 0);
    if(canBeInitialized)
      estimator_->setLandmark(lmId, point.estimate());

  }

  //matchingType_ == Match3D2D
  //Only add observation for frameB only, not setLandMark
  else
  {
    okvis::MapPoint landmark;
    estimator_->getLandmark(lmIdA, landmark);

//    //dont add observation for those landmarks
//    if(frameB_->id() > 6000 && // since height measurement at the beginning is not so accurate
//            (landmark.point[2]/landmark.point[3] < minheight ||
//             landmark.point[2]/landmark.point[3] > maxheight))
//    {
//        //LOG(INFO) << "hPP violate domain knowledge: " << (T_WCa_ * hP_Ca).transpose();
//        return;
//    }

    OKVIS_ASSERT_TRUE_DBG(Exception,lmIdB==0,"bug. Id in frame B already set.");

    // get projection into B, the current frame
    Eigen::Vector2d kptB = projectionsIntoB_.row(indexA);
    Eigen::Vector2d keypointBMeasurement;
    frameB_->getKeypoint(camIdB_, indexB, keypointBMeasurement);

    Eigen::Vector2d err = kptB - keypointBMeasurement;
    double keypointBStdDev;
    frameB_->getKeypointSize(camIdB_, indexB, keypointBStdDev);
    keypointBStdDev = 0.8 * keypointBStdDev / 12.0;

    //Calculate the 2x2 covariance matrix using keyptB scale as stdDev
    Eigen::Matrix2d U_tot = Eigen::Matrix2d::Identity() * keypointBStdDev
        * keypointBStdDev
        + projectionsIntoBUncertainties_.block<2, 2>(2 * indexA, 0);

    const double chi2 = err.transpose().eval() * U_tot.inverse() * err;

    if (chi2 > 4.0) { // default is 4.0, set it tighter to reduce green landmark miss-match problem in close area
        LOG(INFO) << "In 3d-2d setBestMatch, chi2=" << chi2 << " >4.0, cannot set landmark in current frame";
      return;
    }

    // saturate allowed image uncertainty
    if (U_tot.norm() > 25.0 / (keypointBStdDev * keypointBStdDev * sqrt(2))) {
      numUncertainMatches_++;
      //return;
    }

    frameB_->setLandmarkId(camIdB_, indexB, lmIdA);
    lmIdB = lmIdA;


    // initialize in graph
    if (landmark.observations.find(
        okvis::KeypointIdentifier(mfIdB_, camIdB_, indexB))
        == landmark.observations.end())
    {
      // ensure no double observations...
      OKVIS_ASSERT_TRUE(Exception, estimator_->isLandmarkAdded(lmIdB),
                        "not added");
      estimator_->addObservation<camera_geometry_t>(lmIdB, mfIdB_, camIdB_,
                                                    indexB);
    }

  }
  numMatches_++;
}

template class VioKeyframeWindowMatchingAlgorithm<
    okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion> > ;

template class VioKeyframeWindowMatchingAlgorithm<
    okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion> > ;

template class VioKeyframeWindowMatchingAlgorithm<
    okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion8> > ;

}
