#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <memory>
#include <functional>
#include <atomic>
#include <Eigen/Core>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/opencv.hpp>

#pragma GCC diagnostic pop
#include <okvis/VioParametersReader.hpp>
#include <okvis/ThreadedKFVio.hpp>
#include <boost/filesystem.hpp>
#include <QtSerialPort/QSerialPort>
#include <QTextStream>
#include <QCoreApplication>
#include <QStringList>
#include <sys/time.h>
#include "stdafx.h"
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <vector>
#include "FlyCapture2.h"
#include <sstream>
#include <thread>
#include <mutex>
#include <iomanip>

#define PLAY_DELAY_IN_MS (50) // 0 is stop

using namespace FlyCapture2;
using namespace std;
using namespace cv;

#define GRAVITY (9.80665)
#define _USE_MATH_DEFINES
#define ACCELE_LSB_DIVIDER (1200.0/GRAVITY)
#define ACCELE_MAX_DECIMAL (21600)
#define GYRO_LSB_DIVIDER ((25.0*180)/M_PI)
#define GYRO_MAX_DECIMAL (25000)

#define CAM_IMAGE_WIDTH (640)
#define CAM_IMAGE_HEIGHT (512)

// [AA][acc_x][acc_y][acc_z][gyro_x][gyro_y][gyro_z][whether cam sync trigger][total count][55]
#define IMU_FRAME_LENGTH (17)

#define IsProgramDebug (false)

#define LOG_FOR_CALIB   (0)
#define LOG_FOR_RUN     (1)
#define LOG_MODE        (LOG_FOR_CALIB)

#define SERIAL_READ_TIMEOUT_TIME (5000) // in miliseconds
#define IMU_FRAME_TO_CAPTURE (10000)

#define RUN_MODE_OFFLINE    (0)
#define RUN_MODE_ONLINE     (1)
#define RUN_MODE            (RUN_MODE_OFFLINE)


QT_USE_NAMESPACE

fstream fp;
double total_length_of_travel=0;

void PrintBuildInfo2()
{
    FC2Version fc2Version;
    Utilities::GetLibraryVersion(&fc2Version);

    ostringstream version;
    version << "FlyCapture2 library version: " << fc2Version.major << "."
            << fc2Version.minor << "." << fc2Version.type << "."
            << fc2Version.build;
    cout << version.str() << endl;

    ostringstream timeStamp;
    timeStamp << "Application build date: " << __DATE__ << " " << __TIME__;
    cout << timeStamp.str() << endl << endl;
}

void PrintCameraInfo2(CameraInfo *pCamInfo)
{
    cout << endl;
    cout << "*** CAMERA INFORMATION ***" << endl;
    cout << "Serial number - " << pCamInfo->serialNumber << endl;
    cout << "Camera model - " << pCamInfo->modelName << endl;
    cout << "Camera vendor - " << pCamInfo->vendorName << endl;
    cout << "Sensor - " << pCamInfo->sensorInfo << endl;
    cout << "Resolution - " << pCamInfo->sensorResolution << endl;
    cout << "Firmware version - " << pCamInfo->firmwareVersion << endl;
    cout << "Firmware build time - " << pCamInfo->firmwareBuildTime << endl
         << endl;
}

void PrintError2(Error error) { error.PrintErrorTrace(); }

bool PollForTriggerReady2(Camera *pCam)
{
    const unsigned int k_softwareTrigger = 0x62C;
    Error error;
    unsigned int regVal = 0;

    do
    {
        error = pCam->ReadRegister(k_softwareTrigger, &regVal);
        if (error != PGRERROR_OK)
        {
            PrintError2(error);
            return false;
        }

    } while ((regVal >> 31) != 0);

    return true;
}

int AddImage(Image image, cv::Mat img)
{
    Error error;

    // Create a converted image
    Image convertedImage;

    // Convert the raw image
    error = image.Convert(PIXEL_FORMAT_MONO8, &convertedImage);
    if (error != PGRERROR_OK)
    {
        cout << "Error in image.Convert" << endl;
        PrintError2(error);
        return -1;
    }

    //
    img = cv::Mat(image.GetRows(), image.GetCols(), CV_8UC3, image.GetData());
}

class PoseViewer
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  constexpr static const double imageSize = 1000.0;

  cv::Mat _image;
  std::vector<cv::Point2d> _path;
  std::vector<double> _heights;
  double _scale = 1.0;
  double _min_x = -0.5;
  double _min_y = -0.5;
  double _min_z = -0.5;
  double _max_x = 0.5;
  double _max_y = 0.5;
  double _max_z = 0.5;
  const double _frameScale = 0.2;  // [m]
  std::atomic_bool drawing_;
  std::atomic_bool showing_;

  PoseViewer()
  {
    cv::namedWindow("Top View");
    _image.create(imageSize, imageSize, CV_8UC3);
    drawing_ = false;
    showing_ = false;
  }
  // this we can register as a callback, so will run whether a new state is estimated
  void publishFullStateAsCallback(
      const okvis::Time & /*t*/, const okvis::kinematics::Transformation & T_WS, // T_WS is pose
      const Eigen::Matrix<double, 9, 1> & speedAndBiases,
      const Eigen::Matrix<double, 3, 1> & /*omega_S*/)
  {

    // just append the path
    Eigen::Vector3d r = T_WS.r(); // position
    Eigen::Matrix3d C = T_WS.C(); // Rotation

    if(_path.size()>0)
    {
        double dist = cv::norm(cv::Point2d(r[0], r[1])-_path.back());
        total_length_of_travel += dist;
    }
    _path.push_back(cv::Point2d(r[0], r[1]));



    _heights.push_back(r[2]);
    // maintain scaling
    if (r[0] - _frameScale < _min_x)
      _min_x = r[0] - _frameScale;
    if (r[1] - _frameScale < _min_y)
      _min_y = r[1] - _frameScale;
    if (r[2] < _min_z)
      _min_z = r[2];
    if (r[0] + _frameScale > _max_x)
      _max_x = r[0] + _frameScale;
    if (r[1] + _frameScale > _max_y)
      _max_y = r[1] + _frameScale;
    if (r[2] > _max_z)
      _max_z = r[2];
    _scale = std::min(imageSize / (_max_x - _min_x), imageSize / (_max_y - _min_y));

    // draw it
    while (showing_) {
    }
    drawing_ = true;
    // erase
    _image.setTo(cv::Scalar(10, 10, 10));
    drawPath();
    // draw axes
    Eigen::Vector3d e_x = C.col(0);
    Eigen::Vector3d e_y = C.col(1);
    Eigen::Vector3d e_z = C.col(2);
    cv::line(
        _image,
        convertToImageCoordinates(_path.back()),
        convertToImageCoordinates(
            _path.back() + cv::Point2d(e_x[0], e_x[1]) * _frameScale),
        cv::Scalar(0, 0, 255), 1, CV_AA);
    cv::line(
        _image,
        convertToImageCoordinates(_path.back()),
        convertToImageCoordinates(
            _path.back() + cv::Point2d(e_y[0], e_y[1]) * _frameScale),
        cv::Scalar(0, 255, 0), 1, CV_AA);
    cv::line(
        _image,
        convertToImageCoordinates(_path.back()),
        convertToImageCoordinates(
            _path.back() + cv::Point2d(e_z[0], e_z[1]) * _frameScale),
        cv::Scalar(255, 0, 0), 1, CV_AA);

    // some text:
    std::stringstream postext;
    postext << "position = [" << r[0] << ", " << r[1] << ", " << r[2] << "]";
    cv::putText(_image, postext.str(), cv::Point(15,15),
                cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1);
    std::stringstream veltext;
    veltext << "velocity = [" << speedAndBiases[0] << ", " << speedAndBiases[1] << ", " << speedAndBiases[2] << "]";
    cv::putText(_image, veltext.str(), cv::Point(15,35),
                    cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1);

    //Also output the position, angle and velocity to result.txt
    fp << r[0] << ", " << r[1] << ", " << r[2] << ", ";

    Eigen::Vector3d ea = C.eulerAngles(0, 1, 2);
    fp << ea[0] << ", " << ea[1] << ", " << ea[2] << ", ";
    fp << speedAndBiases[0] << ", " << speedAndBiases[1] << ", " << speedAndBiases[2] << endl;

    drawing_ = false; // notify
  }
  void display()
  {
    while (drawing_) {
    }
    showing_ = true;
    cv::imshow("Top View", _image);
    showing_ = false;
    cv::waitKey(1);
  }

 private:
  cv::Point2d convertToImageCoordinates(const cv::Point2d & pointInMeters) const
  {
    cv::Point2d pt = (pointInMeters - cv::Point2d(_min_x, _min_y)) * _scale;
    return cv::Point2d(pt.x, imageSize - pt.y); // reverse y for more intuitive top-down plot
  }

  void drawPath()
  {
    for (size_t i = 0; i + 1 < _path.size(); )
    {
      cv::Point2d p0 = convertToImageCoordinates(_path[i]);
      cv::Point2d p1 = convertToImageCoordinates(_path[i + 1]);
      cv::Point2d diff = p1-p0;

      if(diff.dot(diff)<2.0)
      {
        _path.erase(_path.begin() + i + 1);  // clean short segment
        _heights.erase(_heights.begin() + i + 1);
        continue;
      }

      //Purple: 0 level, Red: higher height, Blue: lower height
      double rel_height = (_heights[i] - _min_z + _heights[i + 1] - _min_z)
                      * 0.5 / (_max_z - _min_z);
      cv::line(
          _image,
          p0,
          p1,
          rel_height * cv::Scalar(255, 0, 0)
              + (1.0 - rel_height) * cv::Scalar(0, 0, 255),
          1, CV_AA);


      //For the start, draw a cross to indicate
      if(i==0)
      {
          cv::line(_image, cv::Point2d(p0.x-30, p0.y), cv::Point2d(p0.x+30, p0.y), cv::Scalar(0, 255, 0), 3);
          cv::line(_image, cv::Point2d(p0.x, p0.y-30), cv::Point2d(p0.x, p0.y+30), cv::Scalar(0, 255, 0), 3);
          cv::putText(_image, "Start", cv::Point(p0.x+20,p0.y+20),
                      cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0,255,0), 3);
      }

      i++;

    }
  }

};

//used to tune focal length of camera
int main3(int argc, char **argv)
{
    VideoCapture cap;
    // open the default camera, use something different from 0 otherwise;
    // Check VideoCapture documentation.
    if(!cap.open(0))
        return 0;
    for(;;)
    {
          Mat frame;
          cap >> frame;
          if( frame.empty() ) break; // end of video stream
          imshow("Livestream", frame);
          if( waitKey(10) == 27 ) break; // stop capturing by pressing ESC
    }
    // the camera will be closed automatically upon exit
    // cap.close();

    std::cin.get();
    return 0;
}

// this is just a workbench. most of the stuff here will go into the Frontend class.
int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = 0;  // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
  FLAGS_colorlogtostderr = 1;

  if (RUN_MODE == RUN_MODE_OFFLINE && argc < 3 ||
       RUN_MODE == RUN_MODE_ONLINE && argc < 2 ) {
      // COMPACT_GOOGLE_LOG_ERROR.stream()
    LOG(ERROR)<<
    "Usage: ./" << argv[0] << " configuration-yaml-file dataset-folder [skip-first-seconds]";
    return -1;
  }

  okvis::Duration deltaT(0.0);
  if (argc == 4) {
    deltaT = okvis::Duration(atof(argv[3]));
  }

  // read configuration file
  std::string configFilename(argv[1]);

  okvis::VioParametersReader vio_parameters_reader(configFilename);
  okvis::VioParameters parameters;
  vio_parameters_reader.getParameters(parameters);

  okvis::ThreadedKFVio okvis_estimator(parameters);

  PoseViewer poseViewer;

  //set a function to be called every time a new state is estimated
  //std::bind(member_function, member_instance, ...)


  //Also output the position, angle and velocity to result.txt
  timespec starttime;
  clock_gettime(CLOCK_REALTIME, &starttime);
  std::stringstream filename;
  filename << starttime.tv_sec << "result.txt";
  fp.open(filename.str(), ios::out);
  if(!fp){
      cout<<"Fail to open file: "<<endl;
      std::cin.get();
  }

  fp << "pos_x, pos_y, pos_z, theta_x, theta_y, theta_z, vel_x, vel_y, vel_z" << endl;



  okvis_estimator.setFullStateCallback(
      std::bind(&PoseViewer::publishFullStateAsCallback, &poseViewer,
                std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3, std::placeholders::_4));
  //So the function returned still have 4 variables

  //indicates whether the addMeasurement() functions
  //should return immediately (blocking=false), or only when the processing is complete.
  okvis_estimator.setBlocking(true);

  //Setup folder path
  if (RUN_MODE == RUN_MODE_OFFLINE)
  {
      // the folder path
      std::string path(argv[2]);

      const unsigned int numCameras = parameters.nCameraSystem.numCameras();
      LOG(INFO)<< "numCameras: " << numCameras;

      // open the IMU file
      std::string line;
      std::ifstream imu_file(path + "/imu0/data.csv");
      if (!imu_file.good()) {
        LOG(ERROR)<< "no imu file found at " << path+"/imu0/data.csv";
        return -1;
      }
      int number_of_lines = 0;
      while (std::getline(imu_file, line))
        ++number_of_lines;
      LOG(INFO)<< "No. IMU measurements: " << number_of_lines-1;

      if (number_of_lines - 1 <= 0)
      {
        LOG(ERROR)<< "no imu messages present in " << path+"/imu0/data.csv";
        return -1;
      }

      // set reading position to second line
      imu_file.clear();
      imu_file.seekg(0, std::ios::beg); // beg:: beginning of the stream
      std::getline(imu_file, line); // skip a line

      std::vector<okvis::Time> times;
      okvis::Time latest(0);
      int num_camera_images = 0;
      std::vector < std::vector < std::string >> image_names(numCameras);
      for (size_t i = 0; i < numCameras; ++i)
      {
        num_camera_images = 0; // total counter of image this camera has
        std::string folder(path + "/cam" + std::to_string(i) + "/data");

        for (auto it = boost::filesystem::directory_iterator(folder);
            it != boost::filesystem::directory_iterator(); it++)
        {
          if (!boost::filesystem::is_directory(it->path()))
          {
            //we eliminate directories
            num_camera_images++;
            image_names.at(i).push_back(it->path().filename().string());
          }
          else
          {
            continue;
          }
        }

        if (num_camera_images == 0) {
          LOG(ERROR)<< "no images at " << folder;
          return 1;
        }

        LOG(INFO)<< "No. cam " << i << " images: " << num_camera_images;
        // the filenames are not going to be sorted. So do this here
        std::sort(image_names.at(i).begin(), image_names.at(i).end());
      }

      std::vector < std::vector < std::string > ::iterator
          > cam_iterators(numCameras);
      for (size_t i = 0; i < numCameras; ++i) {
        cam_iterators.at(i) = image_names.at(i).begin();
      }

      int counter = 0;
      okvis::Time start(0.0);
      okvis::Duration ImageDelayInSec(parameters.sensors_information.imageDelay);

      //main loop
      while (true)
      {
        okvis_estimator.display(); // show all OKVIS's camera
        cv::waitKey(PLAY_DELAY_IN_MS);
        poseViewer.display();

        // check if at the end
        for (size_t i = 0; i < numCameras; ++i) {
          if (cam_iterators[i] == image_names[i].end()) {
            fp.close();
            std::cout << "total_length_of_travel: " << total_length_of_travel << endl;
            std::cout << std::endl << "Finished. Press any key to exit." << std::endl << std::flush;
            cv::waitKey(0);
            return 0;
          }
        }

        /// add images
        okvis::Time t;

        for (size_t i = 0; i < numCameras; ++i)
        {
          cv::Mat filtered = cv::imread(
              path + "/cam" + std::to_string(i) + "/data/" + *cam_iterators.at(i),
              cv::IMREAD_GRAYSCALE);

          //The image name contains the second and nanoseconds of the capture time + ".png"
          std::string nanoseconds = cam_iterators.at(i)->substr(
              cam_iterators.at(i)->size() - 13, 9);
          std::string seconds = cam_iterators.at(i)->substr(
              0, cam_iterators.at(i)->size() - 13);

          t = okvis::Time(std::stoi(seconds), std::stoi(nanoseconds)); // stoi: string to signed int
          t += ImageDelayInSec;
          if (start == okvis::Time(0.0)) {
            start = t;
          }

          // get all IMU measurements till then (for the same image)
          okvis::Time t_imu = start;
          do {
            if (!std::getline(imu_file, line))
            {
              fp.close();
              std::cout << "total_length_of_travel: " << total_length_of_travel << endl;
              std::cout << std::endl << "Finished. Press any key to exit." << std::endl << std::flush;
              cv::waitKey(0);
              return 0;
            }

            //Data [timestamp, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z)

            std::stringstream stream(line);
            std::string s;
            std::getline(stream, s, ',');

            //First 10 digits are seconds, remaining 9 digits are nanoseconds
            std::string nanoseconds = s.substr(s.size() - 9, 9);
            std::string seconds = s.substr(0, s.size() - 9);

            Eigen::Vector3d gyr;
            for (int j = 0; j < 3; ++j) {
              std::getline(stream, s, ',');
              gyr[j] = std::stod(s);
            }

            Eigen::Vector3d acc;
            for (int j = 0; j < 3; ++j) {
              std::getline(stream, s, ',');
              acc[j] = std::stod(s);
            }

            t_imu = okvis::Time(std::stoi(seconds), std::stoi(nanoseconds));

            // add the IMU measurement for (blocking) processing if imu timestamp + 1 sec > start
            // deltaT is user input argument [skip-first-second], init is 0
            if (t_imu - start + okvis::Duration(1.0) > deltaT) {
              okvis_estimator.addImuMeasurement(t_imu, acc, gyr);
            }

          } while (t_imu <= t); // continue if image timestamp >= imu timestamp

          // add the image to the frontend for (blocking) processing if image timestamp > start
          if (t - start > deltaT) {
            okvis_estimator.addImage(t, i, filtered);
          }

          cam_iterators[i]++;
        }
        ++counter;

        // display progress
        if (counter % 20 == 0)
        {
          std::cout << "\rProgress: "
              << int(double(counter) / double(num_camera_images) * 100) << "%  "
              << std::flush;
        }

      }
  }
  else
  {
      QTextStream standardOutput(stdout);
      QSerialPort serialPort;
      QString serialPortName = "ttyUSB0";
      serialPort.setPortName(serialPortName);

      int serialPortBaudRate = QSerialPort::Baud115200;
      serialPort.setBaudRate(serialPortBaudRate);

      if (!serialPort.open(QIODevice::ReadOnly)) {
          standardOutput << QObject::tr("Failed to open port %1, error: %2").arg(serialPortName).arg(serialPort.error()) << endl;
          std::cin.get();
          return 1;
      }

      standardOutput << QObject::tr("Opened the serial port %1, at baudrate: %2").arg(serialPortName).arg(serialPortBaudRate) << endl;

      //Setup the pointgrey camera
      PrintBuildInfo2();
      Error error;
      BusManager busMgr;
      unsigned int numCameras;
      error = busMgr.GetNumOfCameras(&numCameras);
      if (error != PGRERROR_OK)
      {
          PrintError2(error);
          std::cin.get();
          return -1;
      }

      cout << "Number of cameras detected: " << numCameras << endl;

      if (numCameras < 1)
      {
          cout << "Insufficient number of cameras... exiting" << endl;
          std::cin.get();
          return -1;
      }

      PGRGuid guid;
      error = busMgr.GetCameraFromIndex(0, &guid);
      if (error != PGRERROR_OK)
      {
          PrintError2(error);
          std::cin.get();
          return -1;
      }

      Camera cam;

      // Connect to a camera
      error = cam.Connect(&guid);
      if (error != PGRERROR_OK)
      {
          PrintError2(error);
          std::cin.get();
          return -1;
      }

      // Power on the camera
      const unsigned int k_cameraPower = 0x610;
      const unsigned int k_powerVal = 0x80000000;
      error = cam.WriteRegister(k_cameraPower, k_powerVal);
      if (error != PGRERROR_OK)
      {
          PrintError2(error);
          std::cin.get();
          return -1;
      }

      const unsigned int millisecondsToSleep = 100;
      unsigned int regVal = 0;
      unsigned int retries = 10;

      // Wait for camera to complete power-up
      do
      {
          struct timespec nsDelay;
          nsDelay.tv_sec = 0;
          nsDelay.tv_nsec = (long)millisecondsToSleep * 1000000L;
          nanosleep(&nsDelay, NULL);

          error = cam.ReadRegister(k_cameraPower, &regVal);
          if (error == PGRERROR_TIMEOUT)
          {
              // ignore timeout errors, camera may not be responding to
              // register reads during power-up
          }
          else if (error != PGRERROR_OK)
          {
              PrintError2(error);
              std::cin.get();
              return -1;
          }

          retries--;
      } while ((regVal & k_powerVal) == 0 && retries > 0);

      // Check for timeout errors after retrying
      if (error == PGRERROR_TIMEOUT)
      {
          PrintError2(error);
          std::cin.get();
          return -1;
      }

      // Get the camera information
      CameraInfo camInfo;
      error = cam.GetCameraInfo(&camInfo);
      if (error != PGRERROR_OK)
      {
          PrintError2(error);
          std::cin.get();
          return -1;
      }

      PrintCameraInfo2(&camInfo);

      auto imageSettings = new Format7ImageSettings();
      imageSettings->mode = MODE_1;
      imageSettings->width = CAM_IMAGE_WIDTH;
      imageSettings->height = CAM_IMAGE_HEIGHT;
      imageSettings->pixelFormat = PIXEL_FORMAT_MONO8;

      bool settingsValid = false;
      Format7PacketInfo packetInfo;
      error = cam.ValidateFormat7Settings(imageSettings, &settingsValid, &packetInfo);
      if (!settingsValid)
      {
          cout << "Settings are not valid" << endl;
          std::cin.get();
          return -1;
      }
      error = cam.SetFormat7Configuration(imageSettings, packetInfo.recommendedBytesPerPacket);
      if (error != PGRERROR_OK)
      {
          PrintError2(error);
          std::cin.get();
          return -1;
      }

      // Get current trigger settings
      TriggerMode triggerMode;
      error = cam.GetTriggerMode(&triggerMode);
      if (error != PGRERROR_OK)
      {
          PrintError2(error);
          std::cin.get();
          return -1;
      }

      // Set camera to trigger mode 0
      triggerMode.onOff = true;
      triggerMode.mode = 0;
      triggerMode.parameter = 0;

      // Triggering the camera externally using source 0.
      triggerMode.source = 0;

      error = cam.SetTriggerMode(&triggerMode);
      if (error != PGRERROR_OK)
      {
          PrintError2(error);
          std::cin.get();
          return -1;
      }

      // Poll to ensure camera is ready
      bool retVal = PollForTriggerReady2(&cam);
      if (!retVal)
      {
          cout << endl;
          cout << "Error polling for trigger ready!" << endl;
          std::cin.get();
          return -1;
      }

      // Get the camera configuration
      FC2Config config;
      error = cam.GetConfiguration(&config);
      if (error != PGRERROR_OK)
      {
          PrintError2(error);
          std::cin.get();
          return -1;
      }

      // Set the grab timeout to 5 seconds
      config.grabTimeout = 5000;

      // Set the camera configuration
      error = cam.SetConfiguration(&config);
      if (error != PGRERROR_OK)
      {
          PrintError2(error);
          std::cin.get();
          return -1;
      }

      // Camera is ready, start capturing images
      error = cam.StartCapture();
      if (error != PGRERROR_OK)
      {
          PrintError2(error);
          std::cin.get();
          return -1;
      }

      cout << "Trigger the camera by sending a trigger pulse to GPIO"
           << triggerMode.source << endl;

      Image image;
      timespec imutime;
      int currentoffset=-1; // -1 means not yet found 0xAA, >=0 means the position away from 0xAA
      std::vector<double> imuvalue;

      int i=0;
      QByteArray readData = serialPort.readAll();
      clock_gettime(CLOCK_REALTIME, &imutime);

      while (serialPort.waitForReadyRead(SERIAL_READ_TIMEOUT_TIME))
      {
          okvis_estimator.display(); // show all OKVIS's camera
          poseViewer.display();

          unsigned int b_prev=0;

          int sum;
          double value;
          //QByteArray tmp = serialPort.readAll();
          QByteArray tmp = serialPort.read(IMU_FRAME_LENGTH);

          if(IsProgramDebug) readData.append(tmp);
          if(IsProgramDebug) standardOutput << tmp.toHex() << endl;


          const char* c=tmp.data();

          if(IsProgramDebug) standardOutput << QObject::tr("tmp.length()=%1").arg(tmp.length()) << endl;
          for (int j=0; j<tmp.length(); j++)
          {
              //Detect "0xAA", get a new frame of imu data
              unsigned int b1=*c;
              auto data_in_hex = b1 & 0xff;
              if(IsProgramDebug) standardOutput << QObject::tr("data_in_hex=%1, j=%2, b1=%3").arg(data_in_hex).arg(j).arg(b1) << endl;
              if(currentoffset<0 && data_in_hex == 0xAA)
              {
                  currentoffset=0;
              }
              else if(currentoffset>=0)
              {
                  if(currentoffset == 16 && data_in_hex == 0x55) currentoffset=-1;
                  else
                  {
                      switch(currentoffset)
                      {
                          //waiting for the next byte to complete
                          case 1: case 3: case 5: case 7: case 9: case 11: case 14:
                              b_prev = data_in_hex;
                              break;

                          case 2: case 4: case 6:// accelerometer
                              sum = (b_prev<<8)+data_in_hex;
                              value = (sum>ACCELE_MAX_DECIMAL?sum-65536:sum)/ACCELE_LSB_DIVIDER;
                              if(IsProgramDebug) standardOutput << QObject::tr("===Acc, sum=%1, value=%2").arg(sum).arg(value) << endl;
                              imuvalue.push_back(value);
                              break;

                          case 8: case 10: case 12: // gyroscope
                              sum = (b_prev<<8)+data_in_hex;
                              value = (sum>GYRO_MAX_DECIMAL?sum-65536:sum)/GYRO_LSB_DIVIDER;
                              if(IsProgramDebug) standardOutput << QObject::tr("===Gyro, sum=%1, value=%2").arg(sum).arg(value) << endl;
                              imuvalue.push_back(value);
                              break;

                          case 13: // whether a syn signal is sent to camera
                              clock_gettime(CLOCK_REALTIME, &imutime);
                              if(data_in_hex == 0x01)
                              {
                                  // Grab image
                                  error = cam.RetrieveBuffer(&image);
                                  if (error != PGRERROR_OK)
                                  {
                                      cout << "Error in RetrieveBuffer, skip this image frame" << endl;
                                      //PrintError2(error);
                                      //std::cin.get();
                                      //return -1;
                                  }
                                  else
                                  {
                                      cv::Mat img;
                                      std::thread (AddImage, image, std::ref(img)).detach();
                                      auto t = okvis::Time(image.GetTimeStamp().seconds, image.GetTimeStamp().microSeconds*1e3);
                                      okvis_estimator.addImage(t, 0, img);

                                      cout << "Finish adding the image to okvis.estimator" << endl;
                                  }

                              }
                              break;

                          case 15: // a counter from 0 to 65535
                              sum = (b_prev<<8)+data_in_hex;
                              if(IsProgramDebug) standardOutput << QObject::tr("Counter=%1").arg(sum) << endl;
                              break;

                          default:
                              if(IsProgramDebug) standardOutput << QObject::tr("Parsing imu data error, j=%1, currentoffset=%2, data_in_hex=%3, re-search 0xAA !!").arg(j).arg(currentoffset).arg(data_in_hex) << endl;
                              currentoffset=-1;
                      }
                  }
              }
              c++;
              if(currentoffset>=0) currentoffset++;

              //Flush a new imu frame if imuvalue has more than 6 entry
              if(imuvalue.size()>=6)
              {
                  Eigen::Vector3d gyr;
                  for (int j = 0; j < 3; ++j) {
                    gyr[j] = imuvalue[3+j];
                  }

                  Eigen::Vector3d acc;
                  for (int j = 0; j < 3; ++j) {
                    acc[j] = imuvalue[j];
                  }

                  auto t_imu = okvis::Time(imutime.tv_sec, imutime.tv_nsec);

                  // add the IMU measurement for (blocking) processing if imu timestamp + 1 sec > start
                  // deltaT is user input argument [skip-first-second], init is 0
                  //if (t_imu - start + okvis::Duration(1.0) > deltaT) {
                    okvis_estimator.addImuMeasurement(t_imu, acc, gyr);
                  //}


                  imuvalue.erase (imuvalue.begin(),imuvalue.begin()+6);
                  cout << "i=" << i << ", An imu frame is captured!" << endl;
              }
          }

      }
      cout << "Main loop ends" << endl;

      // Turn trigger mode off.
      triggerMode.onOff = false;
      error = cam.SetTriggerMode(&triggerMode);
      if (error != PGRERROR_OK)
      {
          PrintError2(error);
          std::cin.get();
          return -1;
      }
      cout << endl;
      cout << "Finished grabbing images" << endl;

      // Stop capturing images
      error = cam.StopCapture();
      if (error != PGRERROR_OK)
      {
          PrintError2(error);
          std::cin.get();
          return -1;
      }

      // Disconnect the camera
      error = cam.Disconnect();
      if (error != PGRERROR_OK)
      {
          PrintError2(error);
          std::cin.get();
          return -1;
      }
  }




  std::cout << std::endl << std::flush;
  return 0;
}
