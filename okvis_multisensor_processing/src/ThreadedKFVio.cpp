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
 *  Created on: Aug 21, 2014
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file ThreadedKFVio.cpp
 * @brief Source file for the ThreadedKFVio class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <map>

#include <glog/logging.h>

#include <okvis/ThreadedKFVio.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ImuError.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

static const int max_camera_input_queue_size = 10;
static const okvis::Duration temporal_imu_data_overlap(0.02);  // overlap of imu data before and after two consecutive frames [seconds]

#ifdef USE_MOCK
// Constructor for gmock.
ThreadedKFVio::ThreadedKFVio(okvis::VioParameters& parameters, okvis::MockVioBackendInterface& estimator,
    okvis::MockVioFrontendInterface& frontend)
    : speedAndBiases_propagated_(okvis::SpeedAndBias::Zero()),
      imu_params_(parameters.imu),
      repropagationNeeded_(false),
      frameSynchronizer_(okvis::FrameSynchronizer(parameters)),
      lastAddedImageTimestamp_(okvis::Time(0, 0)),
      optimizationDone_(true),
      estimator_(estimator),
      frontend_(frontend),
      parameters_(parameters),
      maxImuInputQueueSize_(60) {
  init();
}
#else
// Constructor.
ThreadedKFVio::ThreadedKFVio(okvis::VioParameters& parameters)
    : speedAndBiases_propagated_(okvis::SpeedAndBias::Zero()),
      imu_params_(parameters.imu),
      repropagationNeeded_(false),
      frameSynchronizer_(okvis::FrameSynchronizer(parameters)),
      lastAddedImageTimestamp_(okvis::Time(0, 0)),
      optimizationDone_(true),
      estimator_(),
      frontend_(parameters.nCameraSystem.numCameras()),
      parameters_(parameters),
      IsImageNormalized_(true),
      GammaCorrectionFactor_(4.0f),
      maxImuInputQueueSize_(2 * max_camera_input_queue_size * parameters.imu.rate
              / parameters.sensors_information.cameraRate)
{
  setBlocking(false);
  init();
}
#endif

// Initialises settings and calls startThreads().
void ThreadedKFVio::init()
{
  assert(parameters_.nCameraSystem.numCameras() > 0);
  numCameras_ = parameters_.nCameraSystem.numCameras();
  numCameraPairs_ = 1;

  frontend_.setBriskDetectionOctaves(parameters_.optimization.detectionOctaves);
  frontend_.setBriskDetectionThreshold(parameters_.optimization.detectionThreshold);
  frontend_.setBriskMatchingThreshold(parameters_.optimization.detectionMatchingThreshold);
  frontend_.setBriskMatchingRatioThreshold(parameters_.optimization.detectionMatchingRatioThreshold);
  frontend_.setBriskMatchingbest_second_min_dist(parameters_.optimization.detectionMatching_best_second_min_dist);
  frontend_.setBriskDetectionMaximumKeypoints(parameters_.optimization.maxNoKeypoints);
  frontend_.setIsOriginalFeatureDetector(parameters_.optimization.IsOriginalFeatureDetector);
  frontend_.setRotation_only_ratio_(parameters_.optimization.rotationOnlyThreshold);
  frontend_.setkeyframeInsertionThreshold(parameters_.optimization.keyframeInsertionOverlapThreshold,
                                          parameters_.optimization.keyframeInsertionMatchingRatioThreshold);

  lastOptimizedStateTimestamp_ = okvis::Time(0.0) + temporal_imu_data_overlap;  // s.t. last_timestamp_ - overlap >= 0 (since okvis::time(-0.02) returns big number)
  lastAddedStateTimestamp_ = okvis::Time(0.0) + temporal_imu_data_overlap;  // s.t. last_timestamp_ - overlap >= 0 (since okvis::time(-0.02) returns big number)

  estimator_.addImu(parameters_.imu);
  for (size_t i = 0; i < numCameras_; ++i)
  {
    // parameters_.camera_extrinsics is never set (default 0's)...
    // do they ever change?
    estimator_.addCamera(parameters_.camera_extrinsics);
    cameraMeasurementsReceived_.emplace_back(
          std::shared_ptr<threadsafe::ThreadSafeQueue<std::shared_ptr<okvis::CameraMeasurement> > >
          (new threadsafe::ThreadSafeQueue<std::shared_ptr<okvis::CameraMeasurement> >()));
  }
  
  // set up windows so things don't crash on Mac OS
  if(parameters_.visualization.displayImages)
  {
    for (size_t im = 0; im < parameters_.nCameraSystem.numCameras(); im++)
    {
      std::stringstream windowname;
      windowname << "camera " << im;
  	  cv::namedWindow(windowname.str());
    }
  }
  
  startThreads();
}

// Start all threads.
void ThreadedKFVio::startThreads() {

  // consumer threads
  //for (size_t i = 0; i < numCameras_; ++i) {
  //  frameConsumerThreads_.emplace_back(&ThreadedKFVio::frameConsumerLoop, this, i);
  //}
  for (size_t i = 0; i < numCameraPairs_; ++i) {
    keypointConsumerThreads_.emplace_back(&ThreadedKFVio::matchingLoop, this);
  }
  imuConsumerThread_ = std::thread(&ThreadedKFVio::imuConsumerLoop, this);
  positionConsumerThread_ = std::thread(&ThreadedKFVio::positionConsumerLoop,
                                        this);
  gpsConsumerThread_ = std::thread(&ThreadedKFVio::gpsConsumerLoop, this);
  magnetometerConsumerThread_ = std::thread(
      &ThreadedKFVio::magnetometerConsumerLoop, this);
  differentialConsumerThread_ = std::thread(
      &ThreadedKFVio::differentialConsumerLoop, this);

  // algorithm threads
  visualizationThread_ = std::thread(&ThreadedKFVio::visualizationLoop, this); // print the matching
  //optimizationThread_ = std::thread(&ThreadedKFVio::optimizationLoop, this);
  publisherThread_ = std::thread(&ThreadedKFVio::publisherLoop, this); // invoke those callback functions
}

// Destructor. This calls Shutdown() for all threadsafe queues and joins all threads.
ThreadedKFVio::~ThreadedKFVio()
{
  for (size_t i = 0; i < numCameras_; ++i) {
    cameraMeasurementsReceived_.at(i)->Shutdown();
  }
  keypointMeasurements_.Shutdown();
  matchedFrames_.Shutdown();
  imuMeasurementsReceived_.Shutdown();
  //optimizationResults_.Shutdown();
  visualizationData_.Shutdown();
  imuFrameSynchronizer_.shutdown();
  positionMeasurementsReceived_.Shutdown();

  // consumer threads
  //for (size_t i = 0; i < numCameras_; ++i) {
    //frameConsumerThreads_.at(i).join();
  //}
  for (size_t i = 0; i < numCameraPairs_; ++i) {
    keypointConsumerThreads_.at(i).join();
  }
  imuConsumerThread_.join();
  positionConsumerThread_.join();
  gpsConsumerThread_.join();
  magnetometerConsumerThread_.join();
  differentialConsumerThread_.join();
  visualizationThread_.join();
  optimizationThread_.join();
  publisherThread_.join();

  /*okvis::kinematics::Transformation endPosition;
  estimator_.get_T_WS(estimator_.currentFrameId(), endPosition);
  std::stringstream s;
  s << endPosition.r();
  LOG(INFO) << "Sensor end position:\n" << s.str();
  LOG(INFO) << "Distance to origin: " << endPosition.r().norm();*/
#ifndef DEACTIVATE_TIMERS
  LOG(INFO) << okvis::timing::Timing::print();
#endif
}

void GammaCorrection(const cv::Mat& src, cv::Mat& dst, float fGamma)
{
    unsigned char lut[256];
    for (int i = 0; i < 256; i++)
    {
        lut[i] = cv::saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
    }

    dst = src.clone();
    const int channels = dst.channels();
    switch (channels)
    {
        cv::MatIterator_<uchar> it, end;
        for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
            *it = lut[(*it)];
        break;
    }
}

// Add a new image.
bool ThreadedKFVio::addImage(const okvis::Time & stamp, size_t cameraIndex,
                             const cv::Mat & image,
                             const std::vector<cv::KeyPoint> * keypoints,
                             bool* /*asKeyframe*/)
{
  assert(cameraIndex<numCameras_);

  if (lastAddedImageTimestamp_ > stamp
      && fabs((lastAddedImageTimestamp_ - stamp).toSec())
          > parameters_.sensors_information.frameTimestampTolerance)
  {
    LOG(ERROR)
        << "Received image from the past. Dropping the image.";
    return false;
  }
  lastAddedImageTimestamp_ = stamp;

  std::shared_ptr<okvis::CameraMeasurement> frame = std::make_shared<
      okvis::CameraMeasurement>();

  if(!IsImageNormalized_)
  {
    frame->measurement.image = image;
  }
  else
  {
      cv::Mat normalizedImg = cv::Mat(image.size(), image.type());

      //cv::normalize not much use, since the light always make 255 in the image
      //cv::normalize(image,  normalizedImg, 0, 255, cv::NORM_MINMAX);

      //cv::equalizeHist not work since it invokes dark noise
      //cv::equalizeHist( image, normalizedImg );

      //Gamma values < 1 will shift the image towards the darker end of the spectrum while
      //gamma values > 1 will make the image appear lighter.
      //e.g. gamma=2 equals to out = sqrt(in/255)*255
      GammaCorrection(image, normalizedImg, GammaCorrectionFactor_);

      frame->measurement.image = normalizedImg;
  }


  frame->timeStamp = stamp;
  frame->sensorId = cameraIndex;

  if (keypoints != nullptr)
  {
    frame->measurement.deliversKeypoints = true;
    frame->measurement.keypoints = *keypoints;
  }
  else
  {
    frame->measurement.deliversKeypoints = false;
  }

  if (blocking_)
  {
    cameraMeasurementsReceived_[cameraIndex]->PushBlockingIfFull(frame, 1);
    return true;
  }
  else
  {
    cameraMeasurementsReceived_[cameraIndex]->PushNonBlockingDroppingIfFull(
        frame, max_camera_input_queue_size);
    return cameraMeasurementsReceived_[cameraIndex]->Size() == 1;
  }
}

// Add an abstracted image observation.
bool ThreadedKFVio::addKeypoints(
    const okvis::Time & /*stamp*/, size_t /*cameraIndex*/,
    const std::vector<cv::KeyPoint> & /*keypoints*/,
    const std::vector<uint64_t> & /*landmarkIds*/,
    const cv::Mat & /*descriptors*/,
    bool* /*asKeyframe*/) {
  OKVIS_THROW(
      Exception,
      "ThreadedKFVio::addKeypoints() not implemented anymore since changes to _keypointMeasurements queue.");
  return false;
}

// Add an IMU measurement.
bool ThreadedKFVio::addImuMeasurement(const okvis::Time & stamp,
                                      const Eigen::Vector3d & alpha,
                                      const Eigen::Vector3d & omega) {

  okvis::ImuMeasurement imu_measurement;
  imu_measurement.measurement.accelerometers = alpha;
  imu_measurement.measurement.gyroscopes = omega;
  imu_measurement.timeStamp = stamp;

  if (blocking_) {
    imuMeasurementsReceived_.PushBlockingIfFull(imu_measurement, 1);
    return true;
  } else {
    imuMeasurementsReceived_.PushNonBlockingDroppingIfFull(
        imu_measurement, maxImuInputQueueSize_);
    return imuMeasurementsReceived_.Size() == 1;
  }
}

// Add a position measurement.
void ThreadedKFVio::addPositionMeasurement(const okvis::Time & stamp,
                                           const Eigen::Vector3d & position,
                                           const Eigen::Vector3d & positionOffset,
                                           const Eigen::Matrix3d & positionCovariance) {
  okvis::PositionMeasurement position_measurement;
  position_measurement.measurement.position = position;
  position_measurement.measurement.positionOffset = positionOffset;
  position_measurement.measurement.positionCovariance = positionCovariance;
  position_measurement.timeStamp = stamp;

  if (blocking_) {
    positionMeasurementsReceived_.PushBlockingIfFull(position_measurement, 1);
    return;
  } else {
    positionMeasurementsReceived_.PushNonBlockingDroppingIfFull(
        position_measurement, maxPositionInputQueueSize_);
    return;
  }
}

// Add a GPS measurement.
void ThreadedKFVio::addGpsMeasurement(const okvis::Time &, double, double,
                                      double, const Eigen::Vector3d &,
                                      const Eigen::Matrix3d &) {
  OKVIS_THROW(Exception, "GPS measurements not supported")
}

// Add a magnetometer measurement.
void ThreadedKFVio::addMagnetometerMeasurement(const okvis::Time &,
                                               const Eigen::Vector3d &, double) {
  OKVIS_THROW(Exception, "Magnetometer measurements not supported")
}

// Add a static pressure measurement.
void ThreadedKFVio::addBarometerMeasurement(const okvis::Time &, double, double) {

  OKVIS_THROW(Exception, "Barometer measurements not supported")
}

// Add a differential pressure measurement.
void ThreadedKFVio::addDifferentialPressureMeasurement(const okvis::Time &,
                                                       double, double) {

  OKVIS_THROW(Exception, "Differential pressure measurements not supported")
}

// Set the blocking variable that indicates whether the addMeasurement() functions
// should return immediately (blocking=false), or only when the processing is complete.
void ThreadedKFVio::setBlocking(bool blocking)
{
  blocking_ = blocking;
  // disable time limit for optimization
  if(blocking_)
  {
    std::lock_guard<std::mutex> lock(estimator_mutex_);
    estimator_.setOptimizationTimeLimit(-1.0,parameters_.optimization.max_iterations);
  }
}

// Loop to process frames from camera with index cameraIndex
void ThreadedKFVio::frameConsumerLoop(size_t cameraIndex)
{
  std::shared_ptr<okvis::CameraMeasurement> frame;
  std::shared_ptr<okvis::MultiFrame> multiFrame;
  TimerSwitchable beforeDetectTimer("1.1 frameLoopBeforeDetect"+std::to_string(cameraIndex),true);
  TimerSwitchable waitForFrameSynchronizerMutexTimer("1.1.1 waitForFrameSynchronizerMutex"+std::to_string(cameraIndex),true);
  TimerSwitchable addNewFrameToSynchronizerTimer("1.1.2 addNewFrameToSynchronizer"+std::to_string(cameraIndex),true);
  TimerSwitchable waitForStateVariablesMutexTimer("1.1.3 waitForStateVariablesMutex"+std::to_string(cameraIndex),true);
  TimerSwitchable propagationTimer("1.1.4 propagationTimer"+std::to_string(cameraIndex),true);
  TimerSwitchable detectTimer("1.2 detectAndDescribe"+std::to_string(cameraIndex),true);
  TimerSwitchable afterDetectTimer("1.3 afterDetect"+std::to_string(cameraIndex),true);
  TimerSwitchable waitForFrameSynchronizerMutexTimer2("1.3.1 waitForFrameSynchronizerMutex2"+std::to_string(cameraIndex),true);
  TimerSwitchable waitForMatchingThreadTimer("1.4 waitForMatchingThread"+std::to_string(cameraIndex),true);


  //long long counter=0;
  for (;;)
  {
    //LOG(INFO) << "frameConsumerLoop: " << counter;
    //counter++;
    // get data and check for termination request
    if (cameraMeasurementsReceived_[cameraIndex]->PopBlocking(&frame) == false) {
      return;
    }
    beforeDetectTimer.start();
    {
      // lock the frame synchronizer
      waitForFrameSynchronizerMutexTimer.start();
      std::lock_guard<std::mutex> lock(frameSynchronizer_mutex_);
      waitForFrameSynchronizerMutexTimer.stop();
      // add new frame to frame synchronizer and get the MultiFrame containing it
      addNewFrameToSynchronizerTimer.start();
      multiFrame = frameSynchronizer_.addNewFrame(frame);
      addNewFrameToSynchronizerTimer.stop();
    }
    // unlock frameSynchronizer only now as we can be sure that not two states are added for the same timestamp


    okvis::kinematics::Transformation T_WS;
    okvis::Time lastTimestamp;
    okvis::SpeedAndBias speedAndBiases;
    // copy last state variables
    {
      waitForStateVariablesMutexTimer.start();
      std::lock_guard<std::mutex> lock(lastState_mutex_);
      waitForStateVariablesMutexTimer.stop();
      T_WS = lastOptimized_T_WS_;
      speedAndBiases = lastOptimizedSpeedAndBiases_;
      lastTimestamp = lastOptimizedStateTimestamp_;
    }

    // -- get relevant imu messages for new state
    // so the range is old frame-20ms to new frame+20ms, all imu data within this range is considered
    okvis::Time imuDataEndTime = multiFrame->timestamp()
        + temporal_imu_data_overlap; // temporal_imu_data_overlap = 20ms
    okvis::Time imuDataBeginTime = lastTimestamp - temporal_imu_data_overlap;

    OKVIS_ASSERT_TRUE_DBG(Exception,imuDataBeginTime < imuDataEndTime,"imu data end time is smaller than begin time.");

    // wait until all relevant imu messages have arrived and check for termination request
    if (imuFrameSynchronizer_.waitForUpToDateImuData(okvis::Time(imuDataEndTime)) == false)
    {
      return;
    }
    OKVIS_ASSERT_TRUE_DBG(Exception,
                          imuDataEndTime < imuMeasurements_.back().timeStamp,
                          "Waiting for up to date imu data seems to have failed!");

    okvis::ImuMeasurementDeque imuData = getImuMeasurments(imuDataBeginTime,
                                                           imuDataEndTime);

    // if imu_data is empty, either end_time > begin_time or
    // no measurements in timeframe, should not happen, as we waited for measurements
    if (imuData.size() == 0) {
      beforeDetectTimer.stop();
      continue;
    }

    if (imuData.front().timeStamp > frame->timeStamp) {
      LOG(WARNING) << "Frame is newer than oldest IMU measurement. Dropping it.";
      beforeDetectTimer.stop();
      continue;
    }

    // get T_WC(camIndx) for detectAndDescribe()
    if (estimator_.numFrames() == 0)
    {
      // first frame ever, init orientation by using imu acceleometer to calculate tilting angle
      bool success = okvis::Estimator::initPoseFromImu(imuData, T_WS);
      {
        std::lock_guard<std::mutex> lock(lastState_mutex_);
        lastOptimized_T_WS_ = T_WS;
        lastOptimizedSpeedAndBiases_.setZero();
        lastOptimizedSpeedAndBiases_.segment<3>(6) = imu_params_.a0;
        lastOptimizedStateTimestamp_ = multiFrame->timestamp();
      }
      OKVIS_ASSERT_TRUE_DBG(Exception, success,
          "pose could not be initialized from imu measurements.");
      if (!success) {
          LOG(INFO) << "pose could not be initialized from imu measurements.";
        beforeDetectTimer.stop();
        continue;
      }
    }
    else
    {
      // get old T_WS
      propagationTimer.start();
      okvis::ceres::ImuError::propagation(imuData, parameters_.imu, T_WS,
                                          speedAndBiases, lastTimestamp,
                                          multiFrame->timestamp());
      propagationTimer.stop();
    }

    okvis::kinematics::Transformation T_WC = T_WS
        * (*parameters_.nCameraSystem.T_SC(frame->sensorId));
    beforeDetectTimer.stop();
    detectTimer.start();

    //Do keypt detection and description, use this approximate T_WC to help extracting along gravity direction
    frontend_.detectAndDescribe(frame->sensorId, multiFrame, T_WC, nullptr);
    detectTimer.stop();
    afterDetectTimer.start();

    bool push = false;
    {
      // we now tell frame synchronizer that detectAndDescribe is done for MF with our timestamp
      waitForFrameSynchronizerMutexTimer2.start();
      std::lock_guard<std::mutex> lock(frameSynchronizer_mutex_);
      waitForFrameSynchronizerMutexTimer2.stop();
      frameSynchronizer_.detectionEndedForMultiFrame(multiFrame->id());

      if (frameSynchronizer_.detectionCompletedForAllCameras(multiFrame->id()))
      {
//        LOG(INFO) << "detection completed for multiframe with id "<< multi_frame->id();
        push = true;
      }
    }  // unlocking frame synchronizer
    afterDetectTimer.stop();
    if (push)
    {
      // use queue size 1 to propagate a congestion to the _cameraMeasurementsReceived queue
      // and check for termination request
      waitForMatchingThreadTimer.start();
      if (keypointMeasurements_.PushBlockingIfFull(multiFrame, 1) == false) {
        return;
      }
      waitForMatchingThreadTimer.stop();
    }
  }
}


double maxdiff_x=0, maxdiff_y=0, maxdiff_z=0, maxdiff_norm=0;
double avgdiff_x=0, avgdiff_y=0, avgdiff_z=0, avgdiff_norm=0;
u_int64_t counter=0;

// Loop that matches frames with existing frames.
void ThreadedKFVio::matchingLoop()
{
    int cameraIndex=0;
    std::shared_ptr<okvis::CameraMeasurement> frame;
    std::shared_ptr<okvis::MultiFrame> multiFrame;
    TimerSwitchable beforeDetectTimer("1.1 frameLoopBeforeDetect"+std::to_string(cameraIndex),true);
    TimerSwitchable waitForFrameSynchronizerMutexTimer("1.1.1 waitForFrameSynchronizerMutex"+std::to_string(cameraIndex),true);
    TimerSwitchable addNewFrameToSynchronizerTimer("1.1.2 addNewFrameToSynchronizer"+std::to_string(cameraIndex),true);
    TimerSwitchable waitForStateVariablesMutexTimer("1.1.3 waitForStateVariablesMutex"+std::to_string(cameraIndex),true);
    TimerSwitchable propagationTimer("1.1.4 propagationTimer"+std::to_string(cameraIndex),true);
    TimerSwitchable detectTimer("1.2 detectAndDescribe"+std::to_string(cameraIndex),true);
    TimerSwitchable afterDetectTimer("1.3 afterDetect"+std::to_string(cameraIndex),true);
    TimerSwitchable waitForFrameSynchronizerMutexTimer2("1.3.1 waitForFrameSynchronizerMutex2"+std::to_string(cameraIndex),true);
    TimerSwitchable waitForMatchingThreadTimer("1.4 waitForMatchingThread"+std::to_string(cameraIndex),true);

  TimerSwitchable prepareToAddStateTimer("2.1 prepareToAddState",true);
  TimerSwitchable waitForOptimizationTimer("2.2 waitForOptimization",true);
  TimerSwitchable addStateTimer("2.3 addState",true);
  TimerSwitchable matchingTimer("2.4 matching",true);

  TimerSwitchable optimizationTimer("3.1 optimization",true);
  TimerSwitchable marginalizationTimer("3.2 marginalization",true);
  TimerSwitchable afterOptimizationTimer("3.3 afterOptimization",true);

  okvis::ImuMeasurement data;
  TimerSwitchable processImuTimer("0 processImuMeasurements",true);

  srand(1); // srand() applies globally, so fix rand() seed of all ransac functions in opengv
  clock_t end, begin;
  bool isInit=false;
  double average_time=0, max_running_time=0;
  //long long counter=0;
  for (;;)
  {
      OptimizationResults result;
      LOG(INFO) << "~~new iteration~~";
    //Frame consumer loop content
    //(1) propagate imu measurement (2) Do detectAndDescribe() (3) Push keypt measurements
      //LOG(INFO) << "frameConsumerLoop: " << counter;
      counter++;

      begin = clock();
      // get data and check for termination request
      if (cameraMeasurementsReceived_[cameraIndex]->PopBlocking(&frame) == false) {
        return;
      }
      beforeDetectTimer.start();
      {
        // lock the frame synchronizer
        waitForFrameSynchronizerMutexTimer.start();
        std::lock_guard<std::mutex> lock(frameSynchronizer_mutex_);
        waitForFrameSynchronizerMutexTimer.stop();
        // add new frame to frame synchronizer and get the MultiFrame containing it
        addNewFrameToSynchronizerTimer.start();
        multiFrame = frameSynchronizer_.addNewFrame(frame);
        addNewFrameToSynchronizerTimer.stop();
      }

      // unlock frameSynchronizer only now as we can be sure that not two states are added for the same timestamp
      okvis::kinematics::Transformation T_WS;
      okvis::Time lastTimestamp;
      okvis::SpeedAndBias speedAndBiases;
      // copy last state variables
      {
        waitForStateVariablesMutexTimer.start();
        std::lock_guard<std::mutex> lock(lastState_mutex_);
        waitForStateVariablesMutexTimer.stop();
        T_WS = lastOptimized_T_WS_;
        speedAndBiases = lastOptimizedSpeedAndBiases_;
        lastTimestamp = lastOptimizedStateTimestamp_;
      }

      // -- get relevant imu messages for new state
      okvis::Time imuDataEndTime = multiFrame->timestamp()
          + temporal_imu_data_overlap;
      okvis::Time imuDataBeginTime = lastTimestamp - temporal_imu_data_overlap;

      OKVIS_ASSERT_TRUE_DBG(Exception,imuDataBeginTime < imuDataEndTime,"imu data end time is smaller than begin time.");

      // wait until all relevant imu messages have arrived and check for termination request
      if (imuFrameSynchronizer_.waitForUpToDateImuData(
        okvis::Time(imuDataEndTime)) == false)  {
        return;
      }
      OKVIS_ASSERT_TRUE_DBG(Exception,
                            imuDataEndTime < imuMeasurements_.back().timeStamp,
                            "Waiting for up to date imu data seems to have failed!");

      okvis::ImuMeasurementDeque imuData = getImuMeasurments(imuDataBeginTime,
                                                             imuDataEndTime);

      // if imu_data is empty, either end_time > begin_time or
      // no measurements in timeframe, should not happen, as we waited for measurements
      if (imuData.size() == 0) {
        beforeDetectTimer.stop();
        continue;
      }

      if (imuData.front().timeStamp > frame->timeStamp) {
        LOG(WARNING) << "Frame is newer than oldest IMU measurement. Dropping it.";
        beforeDetectTimer.stop();
        continue;
      }

      // get T_WC(camIndx) for detectAndDescribe()
      if (estimator_.numFrames() == 0)
      {
        // first frame ever
        bool success = okvis::Estimator::initPoseFromImu(imuData, T_WS);
        {
          std::lock_guard<std::mutex> lock(lastState_mutex_);
          lastOptimized_T_WS_ = T_WS;
          lastOptimizedSpeedAndBiases_.setZero();
          lastOptimizedSpeedAndBiases_.segment<3>(6) = imu_params_.a0;
          lastOptimizedSpeedAndBiases_.segment<3>(3) = imu_params_.g0;
          lastOptimizedStateTimestamp_ = multiFrame->timestamp();
        }
        OKVIS_ASSERT_TRUE_DBG(Exception, success,
            "pose could not be initialized from imu measurements.");
        if (!success) {
          LOG(INFO) << "Init pose from imu not success";
          beforeDetectTimer.stop();
          continue;
        }
      }
      else
      {
        // get old T_WS
        propagationTimer.start();
        okvis::ceres::ImuError::propagation(imuData, parameters_.imu, T_WS,
                                            speedAndBiases, lastTimestamp,
                                            multiFrame->timestamp());
        propagationTimer.stop();
      }
      okvis::kinematics::Transformation T_WC = T_WS
          * (*parameters_.nCameraSystem.T_SC(frame->sensorId));
      beforeDetectTimer.stop();
      detectTimer.start();

      //Using this "imu-estimated" T_WC to do descripe keypt along the gravity direction
      frontend_.detectAndDescribe(frame->sensorId, multiFrame, T_WC, nullptr);
      detectTimer.stop();
      afterDetectTimer.start();

      bool push = false;
      {
        // we now tell frame synchronizer that detectAndDescribe is done for MF with our timestamp
        waitForFrameSynchronizerMutexTimer2.start();
        std::lock_guard<std::mutex> lock(frameSynchronizer_mutex_);
        waitForFrameSynchronizerMutexTimer2.stop();
        frameSynchronizer_.detectionEndedForMultiFrame(multiFrame->id());

        if (frameSynchronizer_.detectionCompletedForAllCameras(multiFrame->id()))
        {
  //        LOG(INFO) << "detection completed for multiframe with id "<< multi_frame->id();
          push = true;
        }
      }
      // unlocking frame synchronizer
      afterDetectTimer.stop();

      if (push)
      {
        // use queue size 1 to propagate a congestion to the _cameraMeasurementsReceived queue
        // and check for termination request
        waitForMatchingThreadTimer.start();
        if (keypointMeasurements_.PushBlockingIfFull(multiFrame, 1) == false) {
          return;
        }
        waitForMatchingThreadTimer.stop();
      }


    end = clock();
    //LOG(INFO) << "Frame consumer loop run in " << (double(end - begin)) / CLOCKS_PER_SEC << "seconds" ;
    //Frame consumer loop content end


    //Matching Loop
    //(1) Add States to parameter and residuals block of mapPtr_
    //(2) matching (match KeyFrames and lastframe), new landmarks will be established in ransac2d2d
    //(3) ransac3d2d and 2d2d (remove outliers and init pose in 2d2d)
    //(4) Push matching result (i.e. which keypt match to which landmarks)

    //LOG(INFO) << "matchingLoop: " << counter;
    //counter++;

    begin = clock();
    // get new frame
    std::shared_ptr<okvis::MultiFrame> frame;

    // get data and check for termination request
    if (keypointMeasurements_.PopBlocking(&frame) == false)
      return;

    prepareToAddStateTimer.start();
    // -- get relevant imu messages for new state
    imuDataEndTime = frame->timestamp() + temporal_imu_data_overlap;
    imuDataBeginTime = lastAddedStateTimestamp_
        - temporal_imu_data_overlap;

    OKVIS_ASSERT_TRUE_DBG(Exception,imuDataBeginTime < imuDataEndTime,
        "imu data end time is smaller than begin time." <<
        "current frametimestamp " << frame->timestamp() << " (id: " << frame->id() <<
        "last timestamp         " << lastAddedStateTimestamp_ << " (id: " << estimator_.currentFrameId());

    // wait until all relevant imu messages have arrived and check for termination request
    if (imuFrameSynchronizer_.waitForUpToDateImuData(
        okvis::Time(imuDataEndTime)) == false)
      return; OKVIS_ASSERT_TRUE_DBG(Exception,
        imuDataEndTime < imuMeasurements_.back().timeStamp,
        "Waiting for up to date imu data seems to have failed!");

    imuData = getImuMeasurments(imuDataBeginTime, imuDataEndTime);

    prepareToAddStateTimer.stop();
    // if imu_data is empty, either end_time > begin_time or
    // no measurements in timeframe, should not happen, as we waited for measurements
    if (imuData.size() == 0)
    {
      LOG(INFO) << "imuData.size() == 0, continue" ;
      continue;
    }

    // make sure that optimization of last frame is over.
    // TODO If we didn't actually 'pop' the _matchedFrames queue until after optimization this would not be necessary
    {
      waitForOptimizationTimer.start();
      std::unique_lock<std::mutex> l(estimator_mutex_);
      while (!optimizationDone_)
        optimizationNotification_.wait(l);
      waitForOptimizationTimer.stop();
      addStateTimer.start();
      okvis::Time t0Matching = okvis::Time::now();
      bool asKeyframe = false;

      okvis::kinematics::Transformation T_WS;
//      if(frame->id()>1)
//      {
//          estimator_.get_T_WS(frame->id()-1, T_WS);
//          LOG(INFO) << "Before estimator_.addStates() ...";
//          LOG(INFO) << "T_WS.r() is: " << std::fixed << std::setprecision(16) << T_WS.r()[0] << ", " << T_WS.r()[1] << ", " <<T_WS.r()[2] ;

//          Eigen::Vector3d ea = T_WS.C().eulerAngles(0, 1, 2);
//          LOG(INFO) << "T_WS.C() is: " << std::fixed << std::setprecision(16) << ea[0] << ", " << ea[1] << ", " << ea[2] ;
//      }


      // fill in statesMap_ and mapPtr_, which is used by the solver
      // frame contains keypt of current frame, which contains the corresponding landmark ID, so
      // can calculate reprojection error
      // imuData let the imu error part can be propagate
      // During matching, setBestMatch() will addLandMark and setLandMark, which will amend the
      // corresponding parameter blocks
      // Immediately after addStates(), estimator_.get_T_WS() will just return an initial guess by imu integration
      if (estimator_.addStates(frame, imuData, asKeyframe))
      {
        lastAddedStateTimestamp_ = frame->timestamp();
        addStateTimer.stop();
      }
      else
      {
        LOG(ERROR) << "Failed to add state! will drop multiframe.";
        addStateTimer.stop();
        continue;
      }

      // -- matching keypoints, initialising landmarks etc.
      estimator_.get_T_WS(frame->id(), T_WS);

      matchingTimer.start();

      //T_WS is not used ... (why?)
      isInit = frontend_.dataAssociationAndInitialization(estimator_, T_WS, parameters_, // Matching as well as initialization of landmarks and state.
                                                 map_, frame, &asKeyframe);
      matchingTimer.stop();

      //Get landmarks here since matching will remove outliers
      estimator_.getLandmarks(result.landmarksVector);

      if (asKeyframe)
        estimator_.setKeyframe(frame->id(), asKeyframe);
      if (!blocking_)
      {
        double timeLimit = parameters_.optimization.timeLimitForMatchingAndOptimization
                           -(okvis::Time::now()-t0Matching).toSec();
        estimator_.setOptimizationTimeLimit(std::max<double>(0.0, timeLimit),
                                            parameters_.optimization.min_iterations);
      }
      optimizationDone_ = false;
    }  // unlock estimator_mutex_

    // use queue size 1 to propagate a congestion to the _matchedFrames queue
    if (matchedFrames_.PushBlockingIfFull(frame, 1) == false)
      return;


    end = clock();
    //LOG(INFO) << "Matching loop run in " << (double(end - begin)) / CLOCKS_PER_SEC << "seconds" ;

    begin = clock();
    //Optimization content
    //LOG(INFO) << "optimizationLoop: " << counter;
    //counter++;
    std::shared_ptr<okvis::MultiFrame> frame_pairs;
    VioVisualizer::VisualizationData::Ptr visualizationDataPtr;
    okvis::Time deleteImuMeasurementsUntil(0, 0);
    if (matchedFrames_.PopBlocking(&frame_pairs) == false)
      return;


    okvis::kinematics::Transformation T_WSt, T_WSf;
    okvis::SpeedAndBias SAB;
    estimator_.get_T_WS(frame_pairs->id(), T_WSt);
    estimator_.getSpeedAndBias(frame_pairs->id(), 0, SAB);
    LOG(INFO) << "Just before Optimization (Just imu integration as initial guess)...";
    LOG(INFO) << "T_WS.r() is: " << std::fixed << std::setprecision(16) << T_WSt.r()[0] << ", " << T_WSt.r()[1] << ", " <<T_WSt.r()[2] ;

    Eigen::Vector3d eaf, ea = T_WSt.C().eulerAngles(0, 1, 2);
    LOG(INFO) << "T_WS.C() is: " << std::fixed << std::setprecision(16) << ea[0] << ", " << ea[1] << ", " << ea[2] ;
    LOG(INFO) << "SpeedAndBias is: " << SAB.transpose();


    {
      std::lock_guard<std::mutex> l(estimator_mutex_);
      optimizationTimer.start();
      //if(frontend_.isInitialized()){
        estimator_.optimize(parameters_.optimization.max_iterations, 2,
                            parameters_.optimization.IsVerbose, 1);// max_iter, num_thread, Isverbose
      //}
      /*if (estimator_.numFrames() > 0 && !frontend_.isInitialized()){
        // undo translation
        for(size_t n=0; n<estimator_.numFrames(); ++n){
          okvis::kinematics::Transformation T_WS_0;
          estimator_.get_T_WS(estimator_.frameIdByAge(n),T_WS_0);
          Eigen::Matrix4d T_WS_0_mat = T_WS_0.T();
          T_WS_0_mat.topRightCorner<3,1>().setZero();
          estimator_.set_T_WS(estimator_.frameIdByAge(n),okvis::kinematics::Transformation(T_WS_0_mat));
          okvis::SpeedAndBias sb_0 = okvis::SpeedAndBias::Zero();
          if(estimator_.getSpeedAndBias(estimator_.frameIdByAge(n), 0, sb_0)){
            sb_0.head<3>().setZero();
            estimator_.setSpeedAndBias(estimator_.frameIdByAge(n), 0, sb_0);
          }
        }
      }*/

      optimizationTimer.stop();

      // get timestamp of last frame in IMU window. Need to do this before marginalization as it will be removed there (if not keyframe)
      if (estimator_.numFrames() > size_t(parameters_.optimization.numImuFrames))
      {
        deleteImuMeasurementsUntil = estimator_.multiFrame(
            estimator_.frameIdByAge(parameters_.optimization.numImuFrames))
            ->timestamp() - temporal_imu_data_overlap;
      }

      marginalizationTimer.start();
      estimator_.applyMarginalizationStrategy(
          parameters_.optimization.numKeyframes,
          parameters_.optimization.numImuFrames, result.transferredLandmarks); // transferredLandmarks save the removed landmarks
      marginalizationTimer.stop();
      afterOptimizationTimer.start();

      // now actually remove measurements
      deleteImuMeasurements(deleteImuMeasurementsUntil);

      // saving optimized state and saving it in OptimizationResults struct
      {
        std::lock_guard<std::mutex> lock(lastState_mutex_);
        estimator_.get_T_WS(frame_pairs->id(), lastOptimized_T_WS_);
        estimator_.getSpeedAndBias(frame_pairs->id(), 0,
                                   lastOptimizedSpeedAndBiases_);

        if(true /*isInit*/)
        {
            LOG(INFO) << "Just after Optimization (after reprojection error is involved)...";
            LOG(INFO) << "T_WS.r() is: " << std::fixed << std::setprecision(16) << lastOptimized_T_WS_.r()[0] << ", " << lastOptimized_T_WS_.r()[1] << ", " <<lastOptimized_T_WS_.r()[2] ;

            eaf = lastOptimized_T_WS_.C().eulerAngles(0, 1, 2);
            LOG(INFO) << "T_WS.C() is: " << std::fixed << std::setprecision(16) << eaf[0] << ", " << eaf[1] << ", " << eaf[2] ;
            LOG(INFO) << "SpeedAndBias is: " << lastOptimizedSpeedAndBiases_.transpose();

            Eigen::Vector3d T_diff = T_WSt.r()-lastOptimized_T_WS_.r();
            Eigen::Vector3d C_diff = ea-eaf;
            LOG(INFO) << "Difference with Optimization...";
            LOG(INFO) << "Current Speed is: " << lastOptimizedSpeedAndBiases_[0] << ", " << lastOptimizedSpeedAndBiases_[1] << ", " << lastOptimizedSpeedAndBiases_[2];
            LOG(INFO) << "T_WS.r() is: " << std::fixed << std::setprecision(16) << fabs(T_diff[0]) << ", " << fabs(T_diff[1]) << ", " <<fabs(T_diff[2]) << "   norm= " << T_diff.norm() ;
            LOG(INFO) << "T_WS.C() is: " << std::fixed << std::setprecision(16) << fabs(C_diff[0]) << ", " << fabs(C_diff[1]) << ", " << fabs(C_diff[2]) << "  norm= " << C_diff.norm() ;

            if(fabs(T_diff[0]) > maxdiff_x) maxdiff_x = fabs(T_diff[0]);
            if(fabs(T_diff[1]) > maxdiff_y) maxdiff_y = fabs(T_diff[1]);
            if(fabs(T_diff[1]) > maxdiff_z) maxdiff_z = fabs(T_diff[2]);
            if(T_diff.norm() > maxdiff_norm) maxdiff_norm = T_diff.norm();

            LOG(INFO) << "T_WS.r() max diff so far: " << std::fixed << std::setprecision(16) << maxdiff_x << ", " << maxdiff_y << ", " <<maxdiff_z << "   norm= " << maxdiff_norm ;

            avgdiff_x = (avgdiff_x*counter+fabs(T_diff[0]))/(counter+1);
            avgdiff_y = (avgdiff_y*counter+fabs(T_diff[1]))/(counter+1);
            avgdiff_z = (avgdiff_z*counter+fabs(T_diff[2]))/(counter+1);
            avgdiff_norm = (avgdiff_norm*counter+fabs(T_diff.norm()))/(counter+1);

            LOG(INFO) << "T_WS.r() average diff so far: " << std::fixed << std::setprecision(16) << avgdiff_x << ", " << avgdiff_y << ", " <<avgdiff_z << "   norm= " << avgdiff_norm ;

            okvis::kinematics::Transformation T_SC;
            estimator_.getCameraSensorStates(frame_pairs->id(), 0, T_SC);
            LOG(INFO) << "Optimized T_SC:" << T_SC.T();

            counter++;
        }


        lastOptimizedStateTimestamp_ = frame_pairs->timestamp();

        // if we publish the state after each IMU propagation we do not need to publish it here.
        if (!parameters_.publishing.publishImuPropagatedState)
        {
          result.T_WS = lastOptimized_T_WS_;
          result.speedAndBiases = lastOptimizedSpeedAndBiases_;
          result.stamp = lastOptimizedStateTimestamp_;
          result.onlyPublishLandmarks = false;
        }
        else
        {
          result.onlyPublishLandmarks = true;
        }
        //estimator_.getLandmarks(result.landmarksVector);
        lastIsInitialized_ = isInit;
        lastresultvector_of_T_SCi_.clear();
        for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i)
        {
          okvis::kinematics::Transformation T_SCii;
          estimator_.getCameraSensorStates(frame_pairs->id(), i, T_SCii);
          lastresultvector_of_T_SCi_.push_back(T_SCii);
        }

        repropagationNeeded_ = true;
      }

      if (parameters_.visualization.displayImages)
      {
        // fill in information that requires access to estimator.
        visualizationDataPtr = VioVisualizer::VisualizationData::Ptr(
            new VioVisualizer::VisualizationData());
        visualizationDataPtr->observations.resize(frame_pairs->numKeypoints());
        okvis::MapPoint landmark;
        okvis::ObservationVector::iterator it = visualizationDataPtr
            ->observations.begin();

        for (size_t camIndex = 0; camIndex < frame_pairs->numFrames(); ++camIndex)
        {
          for (size_t k = 0; k < frame_pairs->numKeypoints(camIndex); ++k)
          {
            OKVIS_ASSERT_TRUE_DBG(Exception,it != visualizationDataPtr->observations.end(),"Observation-vector not big enough");
            it->keypointIdx = k;
            frame_pairs->getKeypoint(camIndex, k, it->keypointMeasurement);
            frame_pairs->getKeypointSize(camIndex, k, it->keypointSize);
            it->cameraIdx = camIndex;
            it->frameId = frame_pairs->id();
            it->landmarkId = frame_pairs->landmarkId(camIndex, k);

            //Checks whether the landmark is added to the estimator.
            if (estimator_.isLandmarkAdded(it->landmarkId))
            {
              estimator_.getLandmark(it->landmarkId, landmark);
              it->landmark_W = landmark.point;

              if (estimator_.isLandmarkInitialized(it->landmarkId))
                it->isInitialized = true;
              else
                it->isInitialized = false;
            }
            else
            {
              it->landmark_W = Eigen::Vector4d(0, 0, 0, 0);  // set to infinity to tell visualizer that landmark is not added
            }
            ++it;
          }
        }
        visualizationDataPtr->keyFrames = estimator_.multiFrame(
            estimator_.currentKeyframeId());
        estimator_.get_T_WS(estimator_.currentKeyframeId(),
                            visualizationDataPtr->T_WS_keyFrame);
      }

      optimizationDone_ = true;
    }  // unlock mutex
    optimizationNotification_.notify_all();

    if (!parameters_.publishing.publishImuPropagatedState)
    {
      // adding further elements to result that do not access estimator.
      for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i)
      {
        result.vector_of_T_SCi.push_back(okvis::kinematics::Transformation(
                *parameters_.nCameraSystem.T_SC(i)));
      }
    }
    optimizationResults_.Push(result);

    // adding further elements to visualization data that do not access estimator
    if (parameters_.visualization.displayImages)
    {
      visualizationDataPtr->currentFrames = frame_pairs;
      visualizationDataPtr->lastFrames = estimator_.multiFrame(estimator_.frameIdByAge(std::min(5, (int)(estimator_.numFrames()-1))));
      visualizationData_.PushNonBlockingDroppingIfFull(visualizationDataPtr, 1);
    }
    afterOptimizationTimer.stop();


    end = clock();
    average_time = (average_time*(counter-1)+(double(end - begin)))/(counter);
    if(double(end-begin)>max_running_time) max_running_time = double(end-begin);
    LOG(INFO) << "Matching(single thread) loop run in " << (double(end - begin)) / CLOCKS_PER_SEC << "seconds";
    LOG(INFO) << "average matching loop run in " << average_time / CLOCKS_PER_SEC << "seconds";
    LOG(INFO) << "maximum matching loop run in " << max_running_time / CLOCKS_PER_SEC << "seconds";
    //Optimization content end

  }
}

// Loop to process IMU measurements.
void ThreadedKFVio::imuConsumerLoop()
{
  okvis::ImuMeasurement data;
  TimerSwitchable processImuTimer("0 processImuMeasurements",true);
  //long long counter=0;
  for (;;)
  {
    //LOG(INFO) << "imuConsumerLoop: " << counter;
    //counter++;
    // get data and check for termination request
    if (imuMeasurementsReceived_.PopBlocking(&data) == false)
      return;
    processImuTimer.start();
    okvis::Time start;
    const okvis::Time* end;  // do not need to copy end timestamp
    {
      std::lock_guard<std::mutex> imuLock(imuMeasurements_mutex_);
      OKVIS_ASSERT_TRUE(Exception,
                        imuMeasurements_.empty()
                        || imuMeasurements_.back().timeStamp < data.timeStamp,
                        "IMU measurement from the past received");

      if (parameters_.publishing.publishImuPropagatedState)
      {
        // repropagationNeeded_ is false when init
        // Will set to true after each optimization_loop
        if (!repropagationNeeded_ && imuMeasurements_.size() > 0)
        {
          start = imuMeasurements_.back().timeStamp;
        }
        else if (repropagationNeeded_)
        {
          //It is an "update" phase to the "filter"
          //Which gives measurements to T_WS_propagated_ and speedAndBiases_propagated_
          std::lock_guard<std::mutex> lastStateLock(lastState_mutex_);
          start = lastOptimizedStateTimestamp_;
          T_WS_propagated_ = lastOptimized_T_WS_;
          speedAndBiases_propagated_ = lastOptimizedSpeedAndBiases_;
          repropagationNeeded_ = false;
        }
        else
        {
          start = okvis::Time(0, 0);
        }
        end = &data.timeStamp;
      }
      imuMeasurements_.push_back(data);
    }  // unlock _imuMeasurements_mutex

    // notify other threads that imu data with timeStamp is here.
    imuFrameSynchronizer_.gotImuData(data.timeStamp);

    //get an intermediate result by just having imu integration before the next camera frame comes
    if (parameters_.publishing.publishImuPropagatedState)
    {
      Eigen::Matrix<double, 15, 15> covariance;
      Eigen::Matrix<double, 15, 15> jacobian;

      frontend_.propagation(imuMeasurements_, imu_params_, T_WS_propagated_,
                            speedAndBiases_propagated_, start, *end, &covariance,
                            &jacobian);
      OptimizationResults result;
      result.stamp = *end;
      result.T_WS = T_WS_propagated_;
      result.speedAndBiases = speedAndBiases_propagated_;
      result.omega_S = imuMeasurements_.back().measurement.gyroscopes
          - speedAndBiases_propagated_.segment<3>(3);
      result.IsInitialized = lastIsInitialized_;
      result.vector_of_T_SCi = lastresultvector_of_T_SCi_;

//      for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i)
//      {
//        result.vector_of_T_SCi.push_back(
//            okvis::kinematics::Transformation(
//                *parameters_.nCameraSystem.T_SC(i)));
//      }
      result.onlyPublishLandmarks = false;
      optimizationResults_.PushNonBlockingDroppingIfFull(result,1); // Push to the queue. If full, drop the oldest entry., max queue size is 1
    }
    processImuTimer.stop();
  }
}

// Loop to process position measurements.
void ThreadedKFVio::positionConsumerLoop()
{
  okvis::PositionMeasurement data;
  for (;;)
  {
    // get data and check for termination request
    if (positionMeasurementsReceived_.PopBlocking(&data) == false)
      return;
    // collect
    {
      std::lock_guard<std::mutex> positionLock(positionMeasurements_mutex_);
      positionMeasurements_.push_back(data);
    }
  }
}

// Loop to process GPS measurements.
void ThreadedKFVio::gpsConsumerLoop() {
}

// Loop to process magnetometer measurements.
void ThreadedKFVio::magnetometerConsumerLoop() {
}

// Loop to process differential pressure measurements.
void ThreadedKFVio::differentialConsumerLoop() {
}

// Loop that visualizes completed frames.
void ThreadedKFVio::visualizationLoop()
{
  okvis::VioVisualizer visualizer_(parameters_);
  for (;;)
  {
    VioVisualizer::VisualizationData::Ptr new_data;
    if (visualizationData_.PopBlocking(&new_data) == false)
      return;

    //visualizer_.showDebugImages(new_data);
    std::vector<cv::Mat> out_images(parameters_.nCameraSystem.numCameras());
    for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i)
    {
      out_images[i] = visualizer_.drawMatches(new_data, i, parameters_.publishing.drawingmode);
    }
	displayImages_.PushNonBlockingDroppingIfFull(out_images,1);
  }
}

// trigger display (needed because OSX won't allow threaded display)
void ThreadedKFVio::display()
{
  std::vector<cv::Mat> out_images;
  if (displayImages_.Size() == 0)
	return;

  if (displayImages_.PopBlocking(&out_images) == false)
    return;

  // draw
  for (size_t im = 0; im < parameters_.nCameraSystem.numCameras(); im++)
  {
    std::stringstream windowname;
    windowname << "camera " << im;
    cv::imshow(windowname.str(), out_images[im]);
  }

  cv::waitKey(1);
}

// Get a subset of the recorded IMU measurements.
okvis::ImuMeasurementDeque ThreadedKFVio::getImuMeasurments(
    okvis::Time& imuDataBeginTime, okvis::Time& imuDataEndTime) {
  // sanity checks:
  // if end time is smaller than begin time, return empty queue.
  // if begin time is larger than newest imu time, return empty queue.
  if (imuDataEndTime < imuDataBeginTime
      || imuDataBeginTime > imuMeasurements_.back().timeStamp)
    return okvis::ImuMeasurementDeque();

  std::lock_guard<std::mutex> lock(imuMeasurements_mutex_);
  // get iterator to imu data before previous frame
  okvis::ImuMeasurementDeque::iterator first_imu_package = imuMeasurements_
      .begin();
  okvis::ImuMeasurementDeque::iterator last_imu_package =
      imuMeasurements_.end();
  // TODO go backwards through queue. Is probably faster.
  for (auto iter = imuMeasurements_.begin(); iter != imuMeasurements_.end();
      ++iter) {
    // move first_imu_package iterator back until iter->timeStamp is higher than requested begintime
    if (iter->timeStamp <= imuDataBeginTime)
      first_imu_package = iter;

    // set last_imu_package iterator as soon as we hit first timeStamp higher than requested endtime & break
    if (iter->timeStamp >= imuDataEndTime) {
      last_imu_package = iter;
      // since we want to include this last imu measurement in returned Deque we
      // increase last_imu_package iterator once.
      ++last_imu_package;
      break;
    }
  }

  // create copy of imu buffer
  return okvis::ImuMeasurementDeque(first_imu_package, last_imu_package);
}

// Remove IMU measurements from the internal buffer.
int ThreadedKFVio::deleteImuMeasurements(const okvis::Time& eraseUntil)
{
  std::lock_guard<std::mutex> lock(imuMeasurements_mutex_);
  if (imuMeasurements_.front().timeStamp > eraseUntil)
    return 0;

  okvis::ImuMeasurementDeque::iterator eraseEnd;
  int removed = 0;
  for (auto it = imuMeasurements_.begin(); it != imuMeasurements_.end(); ++it)
  {
    eraseEnd = it;
    if (it->timeStamp >= eraseUntil)
      break;
    ++removed;
  }

  imuMeasurements_.erase(imuMeasurements_.begin(), eraseEnd);

  return removed;
}

// Loop that performs the optimization and marginalisation.
void ThreadedKFVio::optimizationLoop() {
  TimerSwitchable optimizationTimer("3.1 optimization",true);
  TimerSwitchable marginalizationTimer("3.2 marginalization",true);
  TimerSwitchable afterOptimizationTimer("3.3 afterOptimization",true);

  //long long counter=0;
  for (;;)
  {
    //LOG(INFO) << "optimizationLoop: " << counter;
    //counter++;
    std::shared_ptr<okvis::MultiFrame> frame_pairs;
    VioVisualizer::VisualizationData::Ptr visualizationDataPtr;
    okvis::Time deleteImuMeasurementsUntil(0, 0);
    if (matchedFrames_.PopBlocking(&frame_pairs) == false)
      return;

    OptimizationResults result;
    {
      std::lock_guard<std::mutex> l(estimator_mutex_);
      optimizationTimer.start();
      //if(frontend_.isInitialized()){
        estimator_.optimize(parameters_.optimization.max_iterations, 2,
                            parameters_.optimization.IsVerbose, 1); // max_iter, num_thread, Isverbose
      //}
      /*if (estimator_.numFrames() > 0 && !frontend_.isInitialized()){
        // undo translation
        for(size_t n=0; n<estimator_.numFrames(); ++n){
          okvis::kinematics::Transformation T_WS_0;
          estimator_.get_T_WS(estimator_.frameIdByAge(n),T_WS_0);
          Eigen::Matrix4d T_WS_0_mat = T_WS_0.T();
          T_WS_0_mat.topRightCorner<3,1>().setZero();
          estimator_.set_T_WS(estimator_.frameIdByAge(n),okvis::kinematics::Transformation(T_WS_0_mat));
          okvis::SpeedAndBias sb_0 = okvis::SpeedAndBias::Zero();
          if(estimator_.getSpeedAndBias(estimator_.frameIdByAge(n), 0, sb_0)){
            sb_0.head<3>().setZero();
            estimator_.setSpeedAndBias(estimator_.frameIdByAge(n), 0, sb_0);
          }
        }
      }*/

      optimizationTimer.stop();

      // get timestamp of last frame in IMU window. Need to do this before marginalization as it will be removed there (if not keyframe)
      if (estimator_.numFrames() > size_t(parameters_.optimization.numImuFrames))
      {
        deleteImuMeasurementsUntil = estimator_.multiFrame(
            estimator_.frameIdByAge(parameters_.optimization.numImuFrames))
            ->timestamp() - temporal_imu_data_overlap;
      }

      marginalizationTimer.start();
      estimator_.applyMarginalizationStrategy(
          parameters_.optimization.numKeyframes,
          parameters_.optimization.numImuFrames, result.transferredLandmarks);
      marginalizationTimer.stop();
      afterOptimizationTimer.start();

      // now actually remove measurements
      deleteImuMeasurements(deleteImuMeasurementsUntil);

      // saving optimized state and saving it in OptimizationResults struct
      {
        std::lock_guard<std::mutex> lock(lastState_mutex_);
        estimator_.get_T_WS(frame_pairs->id(), lastOptimized_T_WS_);
        estimator_.getSpeedAndBias(frame_pairs->id(), 0,
                                   lastOptimizedSpeedAndBiases_);
        lastOptimizedStateTimestamp_ = frame_pairs->timestamp();

        // if we publish the state after each IMU propagation we do not need to publish it here.
        if (!parameters_.publishing.publishImuPropagatedState)
        {
          result.T_WS = lastOptimized_T_WS_;
          result.speedAndBiases = lastOptimizedSpeedAndBiases_;
          result.stamp = lastOptimizedStateTimestamp_;
          result.onlyPublishLandmarks = false;
        }
        else
          result.onlyPublishLandmarks = true;
        estimator_.getLandmarks(result.landmarksVector);

        repropagationNeeded_ = true;
      }

      if (parameters_.visualization.displayImages)
      {
        // fill in information that requires access to estimator.
        visualizationDataPtr = VioVisualizer::VisualizationData::Ptr(
            new VioVisualizer::VisualizationData());
        visualizationDataPtr->observations.resize(frame_pairs->numKeypoints());
        okvis::MapPoint landmark;
        okvis::ObservationVector::iterator it = visualizationDataPtr->observations.begin();

        for (size_t camIndex = 0; camIndex < frame_pairs->numFrames();++camIndex)
        {
          for (size_t k = 0; k < frame_pairs->numKeypoints(camIndex); ++k)
          {
            OKVIS_ASSERT_TRUE_DBG(Exception,it != visualizationDataPtr->observations.end(),"Observation-vector not big enough");
            it->keypointIdx = k;
            frame_pairs->getKeypoint(camIndex, k, it->keypointMeasurement);
            frame_pairs->getKeypointSize(camIndex, k, it->keypointSize);
            it->cameraIdx = camIndex;
            it->frameId = frame_pairs->id();
            it->landmarkId = frame_pairs->landmarkId(camIndex, k);

            //Checks whether the landmark is added to the estimator.
            if (estimator_.isLandmarkAdded(it->landmarkId))
            {
              estimator_.getLandmark(it->landmarkId, landmark);
              it->landmark_W = landmark.point;
              if (estimator_.isLandmarkInitialized(it->landmarkId))
                it->isInitialized = true;
              else
                it->isInitialized = false;
            }
            else
            {
              it->landmark_W = Eigen::Vector4d(0, 0, 0, 0);  // set to infinity to tell visualizer that landmark is not added
            }
            ++it;
          }
        }
        visualizationDataPtr->keyFrames = estimator_.multiFrame(
            estimator_.currentKeyframeId());
        estimator_.get_T_WS(estimator_.currentKeyframeId(),
                            visualizationDataPtr->T_WS_keyFrame);
      }

      optimizationDone_ = true;
    }  // unlock mutex
    optimizationNotification_.notify_all();

    if (!parameters_.publishing.publishImuPropagatedState)
    {
      // adding further elements to result that do not access estimator.
      for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i)
      {
        result.vector_of_T_SCi.push_back(
            okvis::kinematics::Transformation(
                *parameters_.nCameraSystem.T_SC(i)));
      }
    }
    optimizationResults_.Push(result);

    // adding further elements to visualization data that do not access estimator
    if (parameters_.visualization.displayImages) {
      visualizationDataPtr->currentFrames = frame_pairs;
      visualizationData_.PushNonBlockingDroppingIfFull(visualizationDataPtr, 1);
    }
    afterOptimizationTimer.stop();
  }
}

// Loop that publishes the newest state and landmarks.
void ThreadedKFVio::publisherLoop()
{
  for (;;)
  {
    //cv::waitKey(50); // broadcast at a rate of 20Hz
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // get the result data
    OptimizationResults result;
    if (optimizationResults_.PopBlocking(&result) == false)
      return;

    // call all user callbacks
    if (stateCallback_ && !result.onlyPublishLandmarks)
      stateCallback_(result.stamp, result.T_WS);
    if (fullStateCallback_ && !result.onlyPublishLandmarks)
      fullStateCallback_(result.stamp, result.T_WS, result.speedAndBiases,
                         result.omega_S);

    //The extrinsic is the online calibration betwen camera and imu
    if (fullStateCallbackWithExtrinsics_ && !result.onlyPublishLandmarks)
      fullStateCallbackWithExtrinsics_(result.stamp, result.T_WS,
                                       result.speedAndBiases, result.omega_S,
                                       result.vector_of_T_SCi, result.IsInitialized);
    if (landmarksCallback_ && !result.landmarksVector.empty())
      landmarksCallback_(result.stamp, result.landmarksVector,
                         result.transferredLandmarks);  //TODO(gohlp): why two maps?
  }
}

}  // namespace okvis
