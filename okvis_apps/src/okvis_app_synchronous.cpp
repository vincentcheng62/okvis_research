#include <iostream>
#include <fstream>
#include <algorithm>
#include <stdlib.h>
#include <memory>
#include <functional>
#include <atomic>
#include <Eigen/Core>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/opencv.hpp>
#include <GL/gl.h>

#pragma GCC diagnostic pop
#include <okvis/VioParametersReader.hpp>
#include <okvis/ThreadedKFVio.hpp>
#include <boost/filesystem.hpp>

#include <QtSerialPort/QSerialPort>
#include <QTextStream>
#include <QCoreApplication>
#include <QStringList>
//#include <sys/time.h>
#include "stdafx.h"
#include <time.h>
//#include <unistd.h>
#include <math.h>
#include <vector>
#include "FlyCapture2.h"
#include <sstream>
#include <thread>
#include <mutex>
#include <iomanip>
#include <ctime>
#include <cstdlib>
#include <experimental/filesystem>
#include <chrono>
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#if defined(_WIN32) || defined(_WIN64)
    #define USBPORTNAME QString("COM")
#else
    #define USBPORTNAME QString("ttyUSB")
#endif

#define PLAY_DELAY_IN_MS (10) // 0 is stop
#define DEFAULT_CONFIG_FILE ("../config/config_me2_inv_mancalib.yaml")

namespace fs = std::experimental::filesystem;
using namespace FlyCapture2;
using namespace std;
using namespace cv;

#define SECOND_TO_SLEEP_AT_START (5)

#define GRAVITY (9.81007) // according to okvis (offical: 9.80665)
#define _USE_MATH_DEFINES
#define ACCELE_LSB_DIVIDER (1200.0/GRAVITY)
#define ACCELE_MAX_DECIMAL (21600)
#define GYRO_LSB_DIVIDER ((25.0*180)/M_PI)
#define GYRO_MAX_DECIMAL (25000)

#define CAM_IMAGE_WIDTH (640)
#define CAM_IMAGE_HEIGHT (512)
#define CAMERA_NUMBER_OF_BUFFER (20)

// Camera hardware setting: NA: 4.0, focusing range: infinity
#define ISCAMERA_SETTING_DEFAULT    (false)
#define CAMERA_SHUTTER_TIME_IN_MS (4.0) // tune the shutter time to avoid motion blur
#define CAMERA_SHUTTER_LOG_FOR_RUN_MULTIPLER    (2.5) // default 2.5 for looking at ceiling
#define CAMERA_EXPOSURE_IN_EV (1.0) // orig: 0.939
#define CAMERA_GAIN_IN_dB (35.0) // orig: 0.939
#define CAMERA_GAIN_LOG_FOR_RUN_MULTIPLER    (0.25) // default 0.25 for looking at ceiling

// [AA][acc_x][acc_y][acc_z][gyro_x][gyro_y][gyro_z][whether cam sync trigger][total count][55]


#define IsProgramDebug (false)
#define MAX_TTY_TRY_INDEX       (10)

#define LOG_FOR_CALIB_CAMERA_IMU        (0)
#define LOG_FOR_RUN                     (1)
#define LOG_FOR_CALIB_IMU_INTRINSIC     (2)
#define LOG_MODE                        (LOG_FOR_RUN)

#define SERIAL_READ_TIMEOUT_TIME (500) // in miliseconds
#define CAMERA_GRAB_TIMEOUT (5000) // in miliseconds
#define CAMERA_POWER_UP_RETRY (1000)

// The following 3 para need to be tuned so that imu data can be captured at ~500Hz in software level
// It is becoz waitForReadyRead() takes 3.5~4ms randomly for each call
#define BYTE_TO_READ_PER_SERIALPORT  (32)
#define BYTE_TO_READ_PER_SERIALPORT2  (33)

// every nth use port2, so to speed up a little bit
// (30) for auto shutter in static
// (26) for 2ms shutter in static
// (35) for 2ms shutter in dynamic
#define BYTE_TO_READ_RATIO  (35)

#define IMU_FRAME_LENGTH (17)
#define IMU_Hz             (500)
#define IMU_TO_CAM_RATIO       (20)
#define CAM_Hz              (IMU_Hz/IMU_TO_CAM_RATIO)

//Keep it 10000 if want to achieve good repeatibility in okvis
#define IMU_FRAME_TO_CAPTURE (50000) // 57000 for luyujie loop in fuyong

//#define IsNormalizeImage (true)
#define SYNC_MARGIN (4000) // margin in ms, after a new camera frame arrive, wait SYNC_MARGIN ms and get the latest imu frame with sync=0x01
#define FIRST_FRAME_MATCH_MARGIN    (SYNC_MARGIN*4)

bool isFirstCamCapture=false;
//okvis::Time latestcamtime;
cv::Mat latestImage;

struct imudata
{
    long index;
    bool IsSync;
    double gyro_x;
    double gyro_y;
    double gyro_z;
    double acc_x;
    double acc_y;
    double acc_z;
};

bool BothStart=false;
bool ManualStop=false;
Image imagecopy;
std::vector<imudata> gIMUframes;
std::vector<TimeStamp> gImageframes;
int awayfromlastsynccounter = -1000; // used in ReadIMUData(), but also helps to determine when to turn BothStart to true
fstream fp ,fp2, fp3, fp4;
int progress=0;
double total_length_of_travel=0;

QT_USE_NAMESPACE

static inline uint32_t __iter_div_u64_rem(uint64_t dividend, uint32_t divisor, uint64_t *remainder)
{
  uint32_t ret = 0;

  while (dividend >= divisor) {
    /* The following asm() prevents the compiler from
       optimising this loop into a modulo operation.  */
    //asm("" : "+rm"(dividend));

    dividend -= divisor;
    ret++;
  }

  *remainder = dividend;

  return ret;
}

#define NSEC_PER_SEC  1000000000L
static inline void timespec_add_ns(struct timespec *a, uint64_t ns)
{
  a->tv_sec += __iter_div_u64_rem(a->tv_nsec + ns, NSEC_PER_SEC, &ns);
  a->tv_nsec = ns;
}



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

void PrintFormat7Capabilities(Format7Info fmt7Info)
{
    cout << "*** Format7 INFORMATION ***" << endl;
    cout << "Max image pixels: (" << fmt7Info.maxWidth << ", "
         << fmt7Info.maxHeight << ")" << endl;
    cout << "Image Unit size: (" << fmt7Info.imageHStepSize << ", "
         << fmt7Info.imageVStepSize << ")" << endl;
    cout << "Offset Unit size: (" << fmt7Info.offsetHStepSize << ", "
         << fmt7Info.offsetVStepSize << ")" << endl;
    cout << "Pixel format bitfield: 0x" << fmt7Info.pixelFormatBitField << endl <<endl;
}

int ConvertToOpenCVImage(Image image, cv::Mat img)
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
    //memcpy( img.data, image.GetData(), image.GetRows()*image.GetCols());
    return 0;
}

void CamRetrieveBuffer(Camera &cam, okvis::ThreadedKFVio &okvis_estimator)
{
    Error error;
    Image image;
    int i=0;
    timespec camtime, tend;
    int last_i_index=0;
    bool IsMatched;
    bool IsFirstMatchedFinished=false;
    unsigned long long imu_period_in_nsec = 1000000000/IMU_Hz;
    okvis::Time lastcamtime;

    while(!ManualStop)
    {
        error = cam.RetrieveBuffer(&image);
        //clock_gettime(CLOCK_REALTIME, &tend);
        //cout << "End timestamp= " << (long long)tend.tv_sec << std::setw(9) <<
                //std::setfill('0') << (long)tend.tv_nsec<< endl;
        //double retrievetime = double((long long)tend.tv_sec - (long long)tstart.tv_sec) + double((long)tend.tv_nsec-(long)tstart.tv_nsec)*1e-9;
        //cout << "RetrieveBuffertime= " << retrievetime << endl;
        if (error != PGRERROR_OK)
        {
            cout << "Error in RetrieveBuffer, skip this image frame" << endl;
            cout << error.GetDescription() << endl;
            cout << error.CollectSupportInformation() << endl;
            cout << error.GetCause().GetDescription() << endl;
            cout << error.GetType() << endl;
            error.PrintErrorTrace();
            //PrintError2(error);
            //std::cin.get();
            //return -1;
        }
        else
        {
            //clock_gettime(CLOCK_REALTIME, &tend);

            //cout << "img Timestamp in micsecond: " << tend.tv_nsec << endl;
            //cout << "img Timestamp in API: " << image.GetTimeStamp().microSeconds*1000L << endl;

            //cout << "success" << endl;
            if(BothStart) gImageframes.push_back(image.GetTimeStamp());



            //imagecopy.DeepCopy((&image));
            if(BothStart)
            {
                //std::thread (SaveImage, imagecopy, dest).detach();
                //std::thread (AddImage, imagecopy, std::ref(img)).detach();
                //ConvertToOpenCVImage(image, &img);

                // Create a converted image
                Image convertedImage;
                image.Convert(PIXEL_FORMAT_MONO8, &convertedImage);

                //if(!IsNormalizeImage)
                //{
                    //cv::Mat img = cv::Mat(image.GetRows(), image.GetCols(), CV_8UC1, convertedImage.GetData());
                    latestImage = cv::Mat(image.GetRows(), image.GetCols(), CV_8UC1);
                    memcpy( latestImage.data, convertedImage.GetData(), image.GetRows()*image.GetCols());
//                }
//                else
//                {
//                    cv::Mat latestImage = cv::Mat(image.GetRows(), image.GetCols(), CV_8UC1);
//                    cv::Mat beforenormImg = cv::Mat(image.GetRows(), image.GetCols(), CV_8UC1);
//                    memcpy( beforenormImg.data, convertedImage.GetData(), image.GetRows()*image.GetCols());
//                    cv::normalize(beforenormImg,  latestImage, 0, 255, cv.NORM_MINMAX)
//                }

                //cout << "img.type(): " << img.type() << endl;
                //cv::Mat convertedimg;
                //img.convertTo(convertedimg, CV_16UC1);
                //cv::imwrite( "zzz.jpg", img );

                //latestcamtime = okvis::Time(image.GetTimeStamp().seconds, image.GetTimeStamp().microSeconds*1e3);
                //timespec_get(&camtime, TIME_UTC);
                //latestcamtime = okvis::Time(camtime.tv_sec, camtime.tv_nsec);
                //okvis_estimator.addImage(latestcamtime, 0, img);

                //do the matching directly here
//                std::this_thread::sleep_for(std::chrono::microseconds(SYNC_MARGIN));
//                for(long j=gIMUframes.size()-1; j>=0 ; j--)
//                {
//                    if(gIMUframes[j].IsSync)
//                    {
//                        cout << "latest imu data with Sync: " << j << ", frame diff is " << j-last_i_index << ", gIMUframes.size() is " << gIMUframes.size() <<endl;
//                        if(!IsFirstMatchedFinished ||
//                            j-last_i_index==IMU_TO_CAM_RATIO)
//                        {
//                            IsFirstMatchedFinished=true;
//                            cout << "The " << i-1 << "th image is match with imuframe " << j << ", frame diff is " << j-last_i_index << endl;
//                            last_i_index=j;
//                            IsMatched=true;

//                        }
//                        else    IsMatched=false;
//                        break;

//                    }
//                }

//                if(!IsMatched)
//                {
//                    //Either camera skipframe or imu capture speed too fast ahead
//                    if(IsFirstMatchedFinished)
//                    {
//                        //image_to_imu_index.push_back(last_i_index+IMU_TO_CAM_RATIO);
//                        //totalwrongmatch++;
//                        last_i_index+=IMU_TO_CAM_RATIO;
//                        cout << "The " << i-1 << "th image is unmatch, but continue " << endl;
//                    }
//                    else
//                    {
//                        cout << "First cam frame cannot match, stop here, otherwise has a chance of mismatch!" <<endl;
//                        cout << "First image frame timestamp in ns: " << image.GetTimeStamp().microSeconds*1000L << endl;
//                        cout << "gIMUframes.size() is " << gIMUframes.size() <<endl;
//                        ManualStop=true;
//                        std::cin.get();
//                        //image_to_imu_index.push_back(-1); // meaning the first match still not happen
//                    }


//                }

//                //Update estimator
//                okvis::Time latestcamtime = okvis::Time(image.GetTimeStamp().seconds, image.GetTimeStamp().microSeconds*1e3);

//                //From i, update the previous imu frame to estimator until reaching the previous sync
//                for(long k=last_i_index-19; k<=last_i_index;k++)
//                {
//                    if(k<0) continue;
//                    Eigen::Vector3d gyr, acc;
//                    gyr[0] = gIMUframes[k].gyro_x;
//                    gyr[1] = gIMUframes[k].gyro_y;
//                    gyr[2] = gIMUframes[k].gyro_z;
//                    acc[0] = gIMUframes[k].acc_x;
//                    acc[1] = gIMUframes[k].acc_y;
//                    acc[2] = gIMUframes[k].acc_z;

//                    okvis::Time t_imu = latestcamtime - okvis::Duration(0, (last_i_index-k)*imu_period_in_nsec);
//                    if(t_imu>lastcamtime)
//                    {
//                        okvis_estimator.addImuMeasurement(t_imu, acc, gyr);
//                        //cout << "Finish adding the imu data to okvis.estimator with timestamp: " << t_imu << endl;
//                    }
//                    else
//                    {
//                        cout << "An imu frame is ignored since its calculated t_imu < lastcamtime" << endl;
//                    }

//                }

//                lastcamtime = latestcamtime;
//                okvis_estimator.addImage(latestcamtime, 0, latestImage);
//                cout << "Finish adding the image to okvis.estimator with timestamp: " << latestcamtime << endl;


                if(!isFirstCamCapture)
                {
                    isFirstCamCapture=true;
                    cout << "First image is captured!" << endl;
    //                timespec t;
    //                timespec_get(&first_image_timestamp, TIME_UTC);
                }

                //cout << "Finish adding the image to okvis.estimator with timestamp: " << latestcamtime << endl;
            }
            //if(BothStart) SaveImage(image, dest);

        }
        if(BothStart && i%5==0)
        {
            if (error != PGRERROR_OK)
            {
                cout << "i=" << i << ", An image frame is failed to capture!" << endl;
            }
            else
            {
                cout << "i=" << i << ", An image frame is captured!" << endl;
            }
        }

        if(BothStart) i++;
    }
    cout << "Finish CamRetrieveBuffer() thread" << endl;

    //usleep(1e6);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    ManualStop=true;
}





void ReadIMUdata(QSerialPort &serialPort, okvis::ThreadedKFVio &okvis_estimator)
{
    unsigned long long imu_period_in_nsec = 1000000000/IMU_Hz;
    unsigned long cam_period_in_nsec = 1000000000/CAM_Hz;

    int currentoffset=-1; // -1 means not yet found 0xAA, >=0 means the position away from 0xAA
    std::vector<double> imuvalue;

    long i=0, k=0; // i: valid imu frame, k: complete frame received

    unsigned int b_prev=0;
    int sum;
    double value;
    timespec imutime, tstart, tend;
    okvis::Time lastcamtime;
    bool IsWaitCamFrameCome=false;

    //serialPort.readAll(); // clear the past data in the buffer
//    if(serialPort.clear(QSerialPort::Input)) // clear the past data in the buffer
//    {
//        cout << "serialPort.clear() fails!" << endl;
//        std::cin.get();
//    }

    while (!ManualStop)
    {
        serialPort.waitForReadyRead(SERIAL_READ_TIMEOUT_TIME);
//        if(serialPort.bytesAvailable()==0)
//        {
//            usleep(1e3);
//            continue;
//        }
        //clock_gettime(CLOCK_REALTIME, &tend);
        //double retrievetime = double((long long)tend.tv_sec - (long long)tstart.tv_sec) + double((long)tend.tv_nsec-(long)tstart.tv_nsec)*1e-9;
        //cout << "serialPort.waitForReadyRead time in ms= " << retrievetime*1e3 << endl;


        QByteArray tmp;
        tmp = serialPort.readAll(); // usually take less than 0.01 ms
//        if(i%BYTE_TO_READ_RATIO==0)
//        {
//            tmp = serialPort.read(BYTE_TO_READ_PER_SERIALPORT2);
//        }
//        else
//        {
//            tmp = serialPort.read(BYTE_TO_READ_PER_SERIALPORT);
//        }

//        QByteArray tmp = serialPort.read(IMU_FRAME_LENGTH);
//        if(tmp.length()==0)
//        {
//            cout << "No new data is fetched!" << endl;
//            usleep(1e3);
//            continue;
//        }

        //if(IsProgramDebug) readData.append(tmp);
        //cout << "tmp.data(): " << tmp.data() << endl;

        //standardOutput << QObject::tr("serialPort.bytesAvailable()=%1").arg(serialPort.bytesAvailable()) << endl;
        const char* c=tmp.data();

        //standardOutput << QObject::tr("tmp.size()=%1").arg(tmp.size()) << endl;
        //if(tmp.size()==0) cout << "No data is read, the frame is skip" << endl;
        //offsetofframeintmp=0;
        for (int j=0; j<tmp.length() && i<IMU_FRAME_TO_CAPTURE; j++)
        {
//            if(k_when_first_capture<0 && isFirstCamCapture)
//            {
//                k_when_first_capture=k;
//            }
            //Detect "0xAA", get a new frame of imu data
            unsigned int b1=*c;
            auto data_in_hex = b1 & 0xff;
            //if(IsProgramDebug) standardOutput << QObject::tr("data_in_hex=%1, j=%2, b1=%3").arg(data_in_hex).arg(j).arg(b1) << endl;
            if(currentoffset<0 && data_in_hex == 0xAA)
            {
//                if(offsetofframeintmp==0)
//                {
//                    clock_gettime(CLOCK_REALTIME, &imutime);
//                }
//                else
//                {
//                    timespec_add_ns(&imutime, 2000000); // add 2ms to trick
//                }
//                offsetofframeintmp++;
                currentoffset=0;
            }
            else
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
                            //if(IsProgramDebug) standardOutput << QObject::tr("===Acc, sum=%1, value=%2").arg(sum).arg(value) << endl;
                            imuvalue.push_back(value);
                            break;

                        case 8: case 10: case 12: // gyroscope
                            sum = (b_prev<<8)+data_in_hex;
                            value = (sum>GYRO_MAX_DECIMAL?sum-65536:sum)/GYRO_LSB_DIVIDER;
                            //if(IsProgramDebug) standardOutput << QObject::tr("===Gyro, sum=%1, value=%2").arg(sum).arg(value) << endl;
                            imuvalue.push_back(value);
                            //cout << sum << endl;
                            break;

                        case 13: // whether a syn signal is sent to camera
                            //clock_gettime(CLOCK_REALTIME, &imutime);
                            timespec_get(&imutime, TIME_UTC);
//                               if(data_in_hex == 0x01)
//                               {
//                                   hasSync=1;
//                                   if(imuSyncIndex<0 && isFirstCamCapture)
//                                   {
//                                       imuSyncIndex = k;
//                                   }
//                               }
//                               else hasSync=0;
//                            if(data_in_hex == 0x01 && tmp.size() == 32 && LOG_MODE!=LOG_FOR_CALIB_IMU_INTRINSIC)
//                            {
//                                // Grab image
//                                clock_gettime(CLOCK_REALTIME, &tstart);
//                                error = cam.RetrieveBuffer(&image);
//                                clock_gettime(CLOCK_REALTIME, &tend);
//                                double retrievetime = double((long long)tend.tv_sec - (long long)tstart.tv_sec) + double((long)tend.tv_nsec-(long)tstart.tv_nsec)*1e-9;
////                                if(retrievetime > (double)cam_period_in_nsec*0.5*1e-9)
////                                {
////                                    cout << "Retrieve time > half camera period, matching fail!" << endl;
////                                    std::cin.get();
////                                    return 0;
////                                }

//                                //cout << "RetrieveBuffertime= " << retrievetime << endl;
//                                if (error != PGRERROR_OK)
//                                {
//                                    cout << "Error in RetrieveBuffer, skip this image frame" << endl;
//                                    //PrintError2(error);
//                                    //std::cin.get();
//                                    //return -1;
//                                }
//                                else
//                                {
//                                    std::thread (SaveImage, image, imagedestination).detach();
//                                    imutime = (image.GetTimeStamp().seconds-retrievetime)*1000000000L+image.GetTimeStamp().microSeconds*1000L;
//                                    //cout << "ImageTimestamp= " << imutime.seconds << std::setw(9) <<
//                                                 //std::setfill('0') << imutime.microSeconds*1000L << endl;
//                                    awayfromlastsynccounter=0;
//                                }
//                            }
//                            else awayfromlastsynccounter++;

                              if(data_in_hex == 0x01) awayfromlastsynccounter=0;
                              else awayfromlastsynccounter++;


                            break;

                        case 15: // a counter from 0 to 65535
                            sum = (b_prev<<8)+data_in_hex;
                            //if(IsProgramDebug) standardOutput << QObject::tr("Counter=%1").arg(sum) << endl;
                            break;

                        default:
                            //if(IsProgramDebug) standardOutput << QObject::tr("Parsing imu data error, j=%1, currentoffset=%2, data_in_hex=%3, re-search 0xAA !!").arg(j).arg(currentoffset).arg(data_in_hex) << endl;
                            currentoffset=-1;
                            awayfromlastsynccounter = -1000;
                            imuvalue.clear(); // remove some prev captured frames where start with 0xAA but not end in 0x55
                            break;
                    }
                }
            }
            c++;
            if(currentoffset>=0) currentoffset++;

            //Flush a new imu frame if imuvalue has more than 6 entry
            if(imuvalue.size()>=6)
            {
                //if(imuSyncIndex>0) // if the first image is captured
                //{
                    //long long t = first_image_timestamp + (k-imuSyncIndex)*imu_period_in_nsec;
//                        long long t = imutime + (awayfromlastsynccounter)*imu_period_in_nsec;
//                        fp << t <<  ",";

//                        //fp << (long long)imutime.tv_sec << std::setw(9) <<
//                        //      std::setfill('0') << (long)imutime.tv_nsec<< "," ;
//                        //fp << awayfromlastsynccounter << ",";
//                        fp << imuvalue[3] << ",";
//                        fp << imuvalue[4] << ",";
//                        fp << imuvalue[5] << ",";
//                        fp << imuvalue[0] << ",";
//                        fp << imuvalue[1] << ",";
//                        fp << imuvalue[2] << endl;
                imudata tmp;
                tmp.index=i;
                tmp.IsSync = (awayfromlastsynccounter==0);
                tmp.gyro_x = imuvalue[3];
                tmp.gyro_y = imuvalue[4];
                tmp.gyro_z = imuvalue[5];
                tmp.acc_x = imuvalue[0];
                tmp.acc_y = imuvalue[1];
                tmp.acc_z = imuvalue[2];

                //clock_gettime(CLOCK_REALTIME, &tstart);
                //cout << "imu frame timestamp: " << tstart.tv_nsec << endl;

                if(BothStart)
                {
                   gIMUframes.push_back(tmp);
                   i++;
                   if(i%100==0) cout << "i=" << i << ", An imu frame is captured!" << endl;

//                   Eigen::Vector3d gyr;
//                   for (int j = 0; j < 3; ++j) {
//                     gyr[j] = imuvalue[3+j];
//                   }

//                   Eigen::Vector3d acc;
//                   for (int j = 0; j < 3; ++j) {
//                     acc[j] = imuvalue[j];
//                   }

//                   okvis::Time t_imu;
//                   if(isFirstCamCapture)
//                   {
//                       if(lastcamtime!=latestcamtime) IsWaitCamFrameCome=false;
//                       //If cam frame still not capture but the corresponding imu frame with sync is arrived
//                       if(IsWaitCamFrameCome||(lastcamtime==latestcamtime && awayfromlastsynccounter==0))
//                       {
//                           // *0.001, avoid exceeding next correct sync timestamp
//                           t_imu = latestcamtime + okvis::Duration(0, 20*imu_period_in_nsec + awayfromlastsynccounter*imu_period_in_nsec*0.001);
//                           IsWaitCamFrameCome=true;
//                       }
//                       else
//                       {
//                           t_imu = latestcamtime + okvis::Duration(0, (awayfromlastsynccounter)*imu_period_in_nsec);
//                       }
//                       lastcamtime = latestcamtime;
//                   }
//                   else
//                   {
//                       t_imu = okvis::Time(imutime.tv_sec, imutime.tv_nsec);
//                   }


//                   // add the IMU measurement for (blocking) processing if imu timestamp + 1 sec > start
//                   // deltaT is user input argument [skip-first-second], init is 0
//                   //if (t_imu - start + okvis::Duration(1.0) > deltaT) {
//                     okvis_estimator.addImuMeasurement(t_imu, acc, gyr);
//                     cout << "Finish adding the imu data to okvis.estimator with timestamp: " << t_imu << endl;
//                   //}
                }
                //}


                imuvalue.erase (imuvalue.begin(),imuvalue.begin()+6);
                k++;
            }
        }

    }
    cout << "Finish ReadIMUdata() thread" << endl;

    //usleep(1e6);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    ManualStop=true;
}

cv::Point2d mousept(-9999,-9999), mouseclickpt(-9999,-9999);
okvis::MapPointVector landmarks, landmarks_t;
std::map<long, Eigen::Vector3d> landmarkmap;

void on_opengl(void* param)
{
    glLoadIdentity();

    glPointSize(10);
    glBegin(GL_POINTS);
    GLfloat pts[3] = {0, 0, 0};

    //Draw current landmarks as red
    for ( int i=0; i<landmarks.size(); i++ )
    {
        glColor3ub( 255, 0, 0);
        pts[0] = landmarks[i].point[0]/landmarks[i].point[3];
        pts[1] = landmarks[i].point[1]/landmarks[i].point[3];
        pts[2] = landmarks[i].point[2]/landmarks[i].point[3];
        glVertex3f(pts[0], pts[1], pts[2]);
        std::cout << pts[0] << ", " << pts[1] << ", " << pts[2] << endl;
    }

    glEnd();
    glBegin(GL_POINTS);
    //Draw old landmarks as green
    for ( int i=0; i<landmarks_t.size(); i++ )
    {
        glColor3ub( 0, 255, 0);
        pts[0] = landmarks_t[i].point[0]/landmarks_t[i].point[3];
        pts[1] = landmarks_t[i].point[1]/landmarks_t[i].point[3];
        pts[2] = landmarks_t[i].point[2]/landmarks_t[i].point[3];
        glVertex3f(pts[0], pts[1], pts[2]);
        std::cout << pts[0] << ", " << pts[1] << ", " << pts[2] << endl;
    }

    glEnd();
}

void PrintAllLandmarks()
{
    if(fp3 && landmarkmap.size()>0)
    {
        for (auto lm : landmarkmap)
        {
            //filter extreme outlier
            if(fabs(lm.second[0])<100 && fabs(lm.second[1])<100
                    && fabs(lm.second[2])<100)
            {
                fp3 << lm.second[0] << " " << lm.second[1] << " "
                     << lm.second[2] << endl;
            }
        }

    }
}

long gotinitID=-1;
double initZ=0;
double median_lm_height=0;
double lmm_max=-999, lmm_min=999, lmm_avg=0, lmm_median=0; // landmarks median statistics
std::vector<double> lmm;
class PoseViewer
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  constexpr static const double imageSize = 1000.0;

  cv::Mat _image;
  std::vector<cv::Point2d> _path;
  std::vector<double> _heights;
  double _scale = 1.0;
  double _min_x = -1.5;
  double _min_y = -1.5;
  double _min_z = -1.5;
  double _max_x = 1.5;
  double _max_y = 1.5;
  double _max_z = 1.5;
  const double _frameScale = 0.2;  // [m]
  std::atomic_bool drawing_;
  std::atomic_bool showing_;

  PoseViewer()
  {
    cv::namedWindow("Top View");
    //cv::namedWindow("3d-pt-cloud",CV_WINDOW_OPENGL|CV_WINDOW_NORMAL);
    //cv::setMouseCallback("Top View", mouse_callback);
    _image.create(imageSize, imageSize, CV_8UC3);
    drawing_ = false;
    showing_ = false;
//    _scale = 1.0;
//    _min_x = -1.5;
//    _min_y = -1.5;
  }

  const int publishlmfreq = 20;
  int publishlmcounter = 0;
  //bool printedxyz=false;
  void publishLandmarksAsCallback(const okvis::Time &t,
                            const okvis::MapPointVector &landmark_vector,
                            const okvis::MapPointVector &transferred_lm_vector) // marginalized landmarks
  {
      if(publishlmcounter%publishlmfreq==0)
      {
//          if(progress>=95 && !printedxyz)
//          {
//              for (auto lm : landmark_vector)
//              {
//                  //filter extreme outlier
//                  if(fabs(lm.point[0]/lm.point[3])<100 && fabs(lm.point[1]/lm.point[3])<100
//                          && fabs(lm.point[2]/lm.point[3])<100)
//                  {
//                      fp3 << lm.point[0]/lm.point[3] << " " << lm.point[1]/lm.point[3] << " "
//                           << lm.point[2]/lm.point[3] << endl;
//                  }

//              }
//              for (auto lm : transferred_lm_vector)
//              {
//                  //filter extreme outlier
//                  if(fabs(lm.point[0]/lm.point[3])<100 && fabs(lm.point[1]/lm.point[3])<100
//                          && fabs(lm.point[2]/lm.point[3])<100)
//                  {
//                      fp3 << lm.point[0]/lm.point[3] << " " << lm.point[1]/lm.point[3] << " "
//                           << lm.point[2]/lm.point[3] << endl;
//                  }
//              }
//              printedxyz=true;
//          }

          for (auto lm : landmark_vector)
          {
              fp2 << lm.id << ", " << lm.point[0]/lm.point[3] << ", " << lm.point[1]/lm.point[3] << ", " <<
                     lm.point[2]/lm.point[3] << ", " << lm.point[3] << ", "  << lm.quality <<
                                                ", " << lm.distance << ", " << lm.observations.size() << endl;
              landmarkmap[lm.id] = Eigen::Vector3d(lm.point[0]/lm.point[3], lm.point[1]/lm.point[3], lm.point[2]/lm.point[3]);
          }
          fp2 << endl;


      }
      publishlmcounter++;

      if(landmark_vector.size()>5)
      {
          std::vector<double> heights;
          for (auto lm : landmark_vector)
          {
              heights.push_back(lm.point[2]/lm.point[3]);
          }
          size_t n = heights.size() / 2;
          std::nth_element(heights.begin(), heights.begin()+n, heights.end());
          median_lm_height = heights[n];

          if(median_lm_height > lmm_max) lmm_max = median_lm_height;
          if(median_lm_height < lmm_min) lmm_min = median_lm_height;
          lmm_avg = (lmm_avg*(publishlmcounter-1) + median_lm_height)/publishlmcounter;
          lmm.push_back(median_lm_height);

          n = lmm.size() / 2;
          std::nth_element(lmm.begin(), lmm.begin()+n, lmm.end());
          lmm_median = lmm[n];
      }


//      landmarks.clear();
//      landmarks.insert(landmarks.begin(), landmark_vector.begin(), landmark_vector.end());

//      landmarks_t.clear();
//      landmarks_t.insert(landmarks_t.begin(), transferred_lm_vector.begin(), transferred_lm_vector.end());

      //cv::updateWindow("3d-pt-cloud");
  }


  // this we can register as a callback, so will run whether a new state is estimated
  void publishFullStateAsCallback(
      const okvis::Time &t,
      const okvis::kinematics::Transformation & T_WS, // T_WS is pose
      const Eigen::Matrix<double, 9, 1> & speedAndBiases,
      const Eigen::Matrix<double, 3, 1> &omega_S ,
      const std::vector<okvis::kinematics::Transformation,
      Eigen::aligned_allocator<okvis::kinematics::Transformation> > extrinsic,
      const bool IsInitialized)
  {

    // just append the path
    // T_WS transform physical pt from sensor(i.e. imu) coord to world coord
    Eigen::Vector3d r = T_WS.r(); // position
    Eigen::Matrix3d C = T_WS.C(); // Rotation

    if(_path.size()>0)
    {
        double dist = cv::norm(cv::Point2d(r[0], r[1])-_path.back());
        total_length_of_travel += dist;
    }
    _path.push_back(cv::Point2d(r[0], r[1]));

    if(IsInitialized && gotinitID<0)
    {
        gotinitID = _path.size()-1;
        initZ=r[2];
    }

    //also print the running path to landmark.xyz for visualization
    if(IsInitialized && fp3 && _path.size()%4 ==0)
    {
        fp3 << r[0] << " " << r[1] << " "
             << r[2] << endl;
    }

    if(fp4 && extrinsic.size()>0)
    {
        fp4 << extrinsic[0].T()(0, 0) << ", " << extrinsic[0].T()(0, 1) << ", " <<
               extrinsic[0].T()(0, 2) << ", " << extrinsic[0].r()[0] << ", " <<
               extrinsic[0].r()[1] << ", " << extrinsic[0].r()[2] << endl;
    }



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
    // x axis of imu (e.g. (1, 0, 0, 1)) will be transformed by T_WS to C.col(0)+r
    // (e_x[0]+r[0], e_x[1]+r[1]) is the projection of this axis to the bird-view plane
    Eigen::Vector3d e_x = C.col(0);
    Eigen::Vector3d e_y = C.col(1);
    Eigen::Vector3d e_z = C.col(2);

    cv::line(_image,
        convertToImageCoordinates(_path.back()),
        convertToImageCoordinates(_path.back() + cv::Point2d(e_x[0], e_x[1]) * _frameScale),
        cv::Scalar(0, 0, 255), 1, CV_AA);
    cv::line(_image,
        convertToImageCoordinates(_path.back()),
        convertToImageCoordinates(_path.back() + cv::Point2d(e_y[0], e_y[1]) * _frameScale),
        cv::Scalar(0, 255, 0), 1, CV_AA);
    cv::line(_image,
        convertToImageCoordinates(_path.back()),
        convertToImageCoordinates(_path.back() + cv::Point2d(e_z[0], e_z[1]) * _frameScale),
        cv::Scalar(255, 0, 0), 1, CV_AA);

    // some text:
    std::stringstream postext;
    postext << "position = [" << r[0] << ", " << r[1] << ", " << r[2] << "]";
    cv::putText(_image, postext.str(), cv::Point(15,15),
                cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1);

    Eigen::Vector3d ea = C.eulerAngles(0, 1, 2);

    std::stringstream rotationtext;
    rotationtext << "rotation = [" << ea[0]*(180/M_PI) << ", " << ea[1]*(180/M_PI) << ", " << ea[2]*(180/M_PI) << "]";
    cv::putText(_image, rotationtext.str(), cv::Point(15,35),
                cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1);

    std::stringstream veltext;
    veltext << "velocity = [" << speedAndBiases[0] << ", " << speedAndBiases[1] << ", " << speedAndBiases[2] << "]";
    cv::putText(_image, veltext.str(), cv::Point(15,55),
                    cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1);

    std::stringstream gyrobiastext;
    gyrobiastext << "gyrobias = [" << speedAndBiases[3] << ", " << speedAndBiases[4] << ", " << speedAndBiases[5] << "]";
    cv::putText(_image, gyrobiastext.str(), cv::Point(485,15),
                    cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1);

    std::stringstream accbiastext;
    accbiastext << "accbias = [" << speedAndBiases[6] << ", " << speedAndBiases[7] << ", " << speedAndBiases[8] << "]";
    cv::putText(_image, accbiastext.str(), cv::Point(485,35),
                    cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1);

    if(gotinitID>=0)
    {
        std::stringstream postext;
        postext << "Init position at gotinitID=" << gotinitID << " [" << _path[gotinitID].x << ", " << _path[gotinitID].y << ", " << initZ << "]";
        cv::putText(_image, postext.str(), cv::Point(485,55),
                    cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1);

        std::stringstream lmtext;
        lmtext << "Median landmarks height=" << median_lm_height ;
        cv::putText(_image, lmtext.str(), cv::Point(485,75),
                    cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1);

        std::stringstream lmmtext;
        lmmtext << "max=" << lmm_max << ", min=" << lmm_min << ", avg=" << lmm_avg << ", median=" << lmm_median ;
        cv::putText(_image, lmmtext.str(), cv::Point(485,95),
                    cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1);
    }

    if(mousept.x>-1000 && mousept.y>-1000)
    {
        std::stringstream postext;
        postext << "mouse position = [" << mousept.x << ", " << mousept.y << "]";
        cv::putText(_image, postext.str(), cv::Point(485,115),
                    cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1);
    }

    if(mouseclickpt.x>-1000 && mouseclickpt.y>-1000)
    {
        std::stringstream mctext;
        mctext << "mouse click position = [" << mouseclickpt.x << ", " << mouseclickpt.y << "]";
        cv::putText(_image, mctext.str(), cv::Point(485,135),
                    cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1);

        std::stringstream mcdifftext;
        double xdiff = fabs(mousept.x-mouseclickpt.x);
        double ydiff = fabs(mousept.y-mouseclickpt.y);
        mcdifftext << "mouse click diff = [" << xdiff << ", " << ydiff << "]" << ", dist=" << sqrt(xdiff*xdiff+ydiff*ydiff);
        cv::putText(_image, mcdifftext.str(), cv::Point(485,155),
                    cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1);
    }

    //Also output the position, angle and velocity to result.txt
    fp << r[0] << ", " << r[1] << ", " << r[2] << ", ";
    fp << ea[0] << ", " << ea[1] << ", " << ea[2] << ", ";

    fp << speedAndBiases[0] << ", " << speedAndBiases[1] << ", " << speedAndBiases[2] << ", ";
    fp << speedAndBiases[3] << ", " << speedAndBiases[4] << ", " << speedAndBiases[5] << ", ";
    fp << speedAndBiases[6] << ", " << speedAndBiases[7] << ", " << speedAndBiases[8] << endl;
    //fp << extrinsic.back().T() << endl;;

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

  cv::Point2d convertToMeters(const cv::Point2d & pointInImgCoord) const
  {
    cv::Point2d pt = cv::Point2d(pointInImgCoord.x, imageSize - pointInImgCoord.y);
    return (pt*(1/_scale))+cv::Point2d(_min_x, _min_y);
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

//      if(diff.dot(diff)<2.0)
//      {
//        _path.erase(_path.begin() + i + 1);  // clean short segment
//        _heights.erase(_heights.begin() + i + 1);
//        continue;
//      }

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


      //For the start, draw a green cross to indicate
      if(i==0)
      {
          cv::line(_image, cv::Point2d(p0.x-30, p0.y), cv::Point2d(p0.x+30, p0.y), cv::Scalar(0, 255, 0), 3);
          cv::line(_image, cv::Point2d(p0.x, p0.y-30), cv::Point2d(p0.x, p0.y+30), cv::Scalar(0, 255, 0), 3);
          cv::putText(_image, "Start", cv::Point(p0.x+20,p0.y+20),
                      cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0,255,0), 3);
      }

      //Draw a red cross to indicate the position where it got initialized
      if(i==gotinitID)
      {
          cv::line(_image, cv::Point2d(p0.x-30, p0.y), cv::Point2d(p0.x+30, p0.y), cv::Scalar(0, 0, 255), 2);
          cv::line(_image, cv::Point2d(p0.x, p0.y-30), cv::Point2d(p0.x, p0.y+30), cv::Scalar(0, 0, 255), 2);
          cv::putText(_image, "Init", cv::Point(p0.x+20,p0.y+20),
                      cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0,0,255), 2);
      }

      i++;

    }
  }

};

void mouse_callback(int  event, int  x, int  y, int  flag, void *param)
{
    PoseViewer *pv = (PoseViewer*)param;

    if (event == EVENT_LBUTTONDOWN)
    {
        mouseclickpt = pv->convertToMeters(cv::Point2d(x, y));
    }
    else if (event == EVENT_MOUSEMOVE)
    {
        mousept = pv->convertToMeters(cv::Point2d(x, y));
//        std::stringstream postext;
//        postext << "mouse position = [" << pt.x << ", " << pt.y << "]";
//        cv::putText(pv->_image, postext.str(), cv::Point(15,75),
//                    cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1);
    }
}

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

//Do the linear optimization between OKVIS output and AGV odometry
//1st arg: okvis output
//2nd arg: AGV odometry
int main2(int argc, char **argv)
{
    if (argc < 3)
    {
      std::cout << "Usage: ./" << argv[0] << " okvis_output agv_odometry";
      std::cin.get();
      return -1;
    }

    const double start_val_thd = 1e-3;
    vector<cv::Point2f> okvis, agv_odo, agv_odo_sampled;
    cv::Mat Homography_mat;

    // open the OKVIS output
    std::string line;
    std::ifstream okvis_file(argv[1]);
    if (!okvis_file.good()) {
      std::cout << "no okvis_file file found at " << argv[1]<< std::endl;
      std::cin.get();
      return -1;
    }

    int number_of_lines = 0;
    while (std::getline(okvis_file, line))
    {
      ++number_of_lines;
    }
    std::cout << "No. okvis_file lines: " << number_of_lines-1<< std::endl;

    if (number_of_lines - 1 <= 0)
    {
      std::cout << "no okvis data present in " << argv[1]<< std::endl;
      std::cin.get();
      return -1;
    }

    // set reading position to second line
    okvis_file.clear();
    okvis_file.seekg(0, std::ios::beg); // beg:: beginning of the stream
    std::getline(okvis_file, line); // skip a line

    do {
      if (!std::getline(okvis_file, line))
      {
        break;
      }

      //Data [pos_x, pos_y, pos_z, theta_x, theta_y, theta_z, vel_x, vel_y, vel_z]

      std::stringstream stream(line);
      std::string s_x, s_y;
      std::getline(stream, s_x, ',');
      std::getline(stream, s_y, ',');

      double x_pos = std::stod(s_x);
      double y_pos = std::stod(s_y);
      if(fabs(x_pos)>start_val_thd && fabs(y_pos)>start_val_thd)
      {
          okvis.emplace_back(x_pos, y_pos);
      }

    } while (true);


    // open the AGV odometry output
    std::ifstream agvodo_file(argv[2]);
    if (!agvodo_file.good()) {
      std::cout << "no agvodo_file file found at " << argv[2]<< std::endl;
      std::cin.get();
      return -1;
    }

    while (std::getline(agvodo_file, line))
    {
      ++number_of_lines;
    }
    std::cout << "No. agvodo_file lines: " << number_of_lines-1<< std::endl;

    if (number_of_lines - 1 <= 0)
    {
      std::cout << "no agv odometry data present in " << argv[2] << std::endl;
      std::cin.get();
      return -1;
    }

    // set reading position to first line
    agvodo_file.clear();
    agvodo_file.seekg(0, std::ios::beg); // beg:: beginning of the stream

    do {
      if (!std::getline(agvodo_file, line))
      {
        break;
      }

      //Data [counter time left_encoder right_encoder .... P: x_pos, y_pos, theta O: x_pos, y_pos, theta ....]
      std::stringstream stream(line.substr(line.find("P:")+2));
      std::string g_x, g_y, o_x, o_y, g_theta;
      std::getline(stream, g_x, ',');
      std::getline(stream, g_y, ',');

      double x_globalpos = std::stod(g_x);
      double y_globalpos = std::stod(g_y);

      std::getline(stream, g_theta, 'O'); // got the theta
      std::getline(stream, g_theta, ':'); // skip the O:
      std::getline(stream, o_x, ',');
      std::getline(stream, o_y, ',');

      //Only consider those points where odometry start to drive away from zero
      if(fabs(std::stod(o_x))>start_val_thd && fabs(std::stod(o_y))>start_val_thd)
      {
          agv_odo.emplace_back(x_globalpos, y_globalpos);
      }

    } while (true);

    //Sampled the agv_odo to match with the size of okvis
    std::cout << "okvis.size(): " << okvis.size() << std::endl;
    std::cout << "agv_odo.size(): " << agv_odo.size() << std::endl;
    double sampleratio = (double)(agv_odo.size())/((double)okvis.size());
    std::cout << "Sample ratio: " << sampleratio << std::endl;

    for(int i=0; i< okvis.size();i++)
    {
        int index = std::min((unsigned int)(i*sampleratio), (unsigned int)(agv_odo.size()-1));
        agv_odo_sampled.push_back(agv_odo[index]);
    }

    Homography_mat = cv::findHomography(okvis, agv_odo_sampled);
    std::cout << "Homography_mat: " << Homography_mat << std::endl;
    std::cout << "Homography_mat.type(): " << Homography_mat.type() << std::endl;

    //Transform okvis pt array using Homography_mat and save down the result
    timespec starttime;
    //clock_gettime(CLOCK_REALTIME, &starttime);
    timespec_get(&starttime, TIME_UTC);
    std::stringstream filename;
    filename << starttime.tv_sec << "alignresult.txt";
    fp.open(filename.str(), ios::out);
    if(!fp){
        std::cout<<"Fail to open file: "<<std::endl;
        std::cin.get();
    }

    fp << "wrapped_okvis_x, wrapped_okvis_y, agv_global_x, agv_global_y" << endl;

    for(int i=0; i< okvis.size();i++)
    {
        cv::Mat pt(3,1, Homography_mat.type());
        pt.at<float>(0) = okvis[i].x;
        pt.at<float>(1) = okvis[i].y;
        pt.at<float>(2) = 1;

        cv::Mat wrapped_pt = Homography_mat*pt;

        //cout << wrapped_pt.t() << endl;

        fp << wrapped_pt.at<float>(0)/wrapped_pt.at<float>(2) << ", ";
        fp << wrapped_pt.at<float>(1)/wrapped_pt.at<float>(2) << ", ";
        fp << agv_odo_sampled[i].x << ", ";
        fp << agv_odo_sampled[i].y << endl;
    }

    fp.close();

    std::cout << "Success!" << std::endl;
    std::cout << "Result is saved to " << filename.str() << std::endl;

    return 0;
}

void MatchingAndUpdateEstimator(okvis::ThreadedKFVio &okvis_estimator)
{
    //Do matching on the global vector data
    int last_i_index=0;
    bool IsMatched;
    bool IsFirstMatchedFinished=false;
    //long totalwrongmatch=0;
    char ch=0;
    //std::vector<long> image_to_imu_index;
    unsigned long long imu_period_in_nsec = 1000000000/IMU_Hz;
    okvis::Time lastcamtime;
    long imagelistcurrentsize=0;

    clock_t end, begin;
    while(!ManualStop)
    {
        if(gImageframes.size()>imagelistcurrentsize) // if new image frame arrive
        {
            begin = clock();
            //cout << "Img timestamp in microsecond: " << gImageframes.back().microSeconds << endl;
            //Check if it is a jump in the index
            if(gImageframes.size()>imagelistcurrentsize+1)
            {
                //cout << "2 image frames come too close, please start again!" << endl;
                cout << "There is a jump on gImageframes!" << endl;
                cout << "gImageframes.size() is " << gImageframes.size() <<endl;
                cout << "gIMUframes.size() is " << gIMUframes.size() <<endl;
//                cout << "imagelistcurrentsize is " << imagelistcurrentsize << endl;
//                cout << "Timestamp of latest Imageframe: " << gImageframes.back().microSeconds*1000L << endl;
//                cout << "Timestamp of latest-1 Imageframe: " << gImageframes[gImageframes.size()-2].microSeconds*1000L << endl;
//                if(gImageframes.size()>2) cout << "Timestamp of latest-2 Imageframe: " << gImageframes[gImageframes.size()-3].microSeconds*1000L << endl;
//                std::cin.get();
                //return 1;
            }

            imagelistcurrentsize = gImageframes.size();

            //Since serialPort.waitForReadyRead() need to wait for 5ms
            //usleep(SYNC_MARGIN);
            std::this_thread::sleep_for(std::chrono::microseconds(SYNC_MARGIN));
            //if(IsFirstMatchedFinished) usleep(SYNC_MARGIN); // give some margin so the matching imu frame can arrive
            //else usleep(FIRST_FRAME_MATCH_MARGIN);

            IsMatched=false;

            //Find the latest imu data in gIMUframes whose
            for(long i=gIMUframes.size()-1; i>=0 ; i--)
            {
                if(gIMUframes[i].IsSync)
                {
                    cout << "latest imu data with Sync: " << i << ", frame diff is " << i-last_i_index << ", gIMUframes.size() is " << gIMUframes.size() <<endl;
                    if(!IsFirstMatchedFinished ||
                        i-last_i_index==IMU_TO_CAM_RATIO)
                    {
//                            if(!IsFirstMatchedFinished && i>20)
//                            {
//                                cout << "First cam frame must match with imu frame <=20, otherwise has a chance of mismatch!" <<endl;
//                                std::cin.get();
//                                return -1;
//                            }
                        IsFirstMatchedFinished=true;
                        //image_to_imu_index.push_back(i);
                        cout << "The " << imagelistcurrentsize-1 << "th image is match with imuframe " << i << ", frame diff is " << i-last_i_index << endl;
                        last_i_index=i;
                        IsMatched=true;

                    }
                    else    IsMatched=false;
                    break;

                }
            }

            if(!IsMatched)
            {
                //Either camera skipframe or imu capture speed too fast ahead
                if(IsFirstMatchedFinished)
                {
                    //image_to_imu_index.push_back(last_i_index+IMU_TO_CAM_RATIO);
                    //totalwrongmatch++;
                    last_i_index+=IMU_TO_CAM_RATIO;
                    cout << "The " << imagelistcurrentsize-1 << "th image is unmatch, but continue " << endl;
                }
                else
                {
                    cout << "First cam frame cannot match, stop here, otherwise has a chance of mismatch!" <<endl;
                    cout << "First image frame timestamp in ns: " << gImageframes.back().microSeconds*1000L << endl;
                    cout << "gIMUframes.size() is " << gIMUframes.size() <<endl;
                    ManualStop=true;
                    std::cin.get();
                    //image_to_imu_index.push_back(-1); // meaning the first match still not happen
                }


            }

            //Update estimator
            okvis::Time latestcamtime = okvis::Time(gImageframes.back().seconds, gImageframes.back().microSeconds*1e3);

            //From i, update the previous imu frame to estimator until reaching the previous sync
            for(long k=last_i_index-19; k<=last_i_index;k++)
            {
                if(k<0) continue;
                Eigen::Vector3d gyr, acc;
                gyr[0] = gIMUframes[k].gyro_x;
                gyr[1] = gIMUframes[k].gyro_y;
                gyr[2] = gIMUframes[k].gyro_z;
                acc[0] = gIMUframes[k].acc_x;
                acc[1] = gIMUframes[k].acc_y;
                acc[2] = gIMUframes[k].acc_z;

                okvis::Time t_imu = latestcamtime - okvis::Duration(0, (last_i_index-k)*imu_period_in_nsec);
                if(t_imu>lastcamtime)
                {
                    okvis_estimator.addImuMeasurement(t_imu, acc, gyr);
                    //cout << "Finish adding the imu data to okvis.estimator with timestamp: " << t_imu << endl;
                }
                else
                {
                    cout << "An imu frame is ignored since its calculated t_imu < lastcamtime" << endl;
                }

            }

            lastcamtime = latestcamtime;
            okvis_estimator.addImage(latestcamtime, 0, latestImage);
            cout << "Finish adding the image to okvis.estimator with timestamp: " << latestcamtime << endl;

            //if(gImageframes.size()==(IMU_FRAME_TO_CAPTURE/IMU_TO_CAM_RATIO)) break;
            end=clock();
            cout << "Time elapse in matching (in ms): " << ((float)(end-begin)/CLOCKS_PER_SEC)*1e3 << endl;
        }

        //Check whether camera thread is already dead by examing the ratio of the
        //received cam frame and imu frame
        if((gImageframes.size()> 50 && (double)(gIMUframes.size())/(double)(gImageframes.size()) > IMU_TO_CAM_RATIO +2) ||
                (gImageframes.size() == 0 && gIMUframes.size()>1000))
        {
            cout << "Camera thread is suspected to be dead, stop" << endl;
            //t1.detach();
            ManualStop=true;
        }


        //usleep(100); // take a small snap in the empty loop
        std::this_thread::sleep_for(std::chrono::microseconds(100));

        //cout << "gImageframes.size(): " << gImageframes.size() << endl;
        //ch = cv::waitKey(10);
        if(ch == 'q' || ch == 'Q')
        {
            ManualStop=true;
        }
    }
}

// this is just a workbench. most of the stuff here will go into the Frontend class.
int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = 0;  // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
  FLAGS_colorlogtostderr = 1;

//  if (argc < 2)
//  {
//      // COMPACT_GOOGLE_LOG_ERROR.stream()
//    LOG(ERROR)<<
//    "Usage: ./" << argv[0] << " configuration-yaml-file [dataset-folder] [skip-first-seconds]";
//    return -1;
//  }

  okvis::Duration deltaT(0.0);
  if (argc == 4) {
    deltaT = okvis::Duration(atof(argv[3]));
  }

  // read configuration file
  std::string configFilename;
  if(argc<2)
  {
      configFilename = std::string(DEFAULT_CONFIG_FILE);
  }
  else
  {
      configFilename = std::string(argv[1]);
  }

  okvis::VioParametersReader vio_parameters_reader(configFilename);
  okvis::VioParameters parameters;
  vio_parameters_reader.getParameters(parameters);

  okvis::ThreadedKFVio okvis_estimator(parameters);

  PoseViewer poseViewer;
  cv::setMouseCallback("Top View", mouse_callback, &poseViewer);
  //cv::setOpenGlDrawCallback("3d-pt-cloud",on_opengl,0);
  //cout << cv::getBuildInformation() << endl;

  //set a function to be called every time a new state is estimated
  //std::bind(member_function, member_instance, ...)


  //Also output the position, angle and velocity to result.txt
  timespec starttime;
  //clock_gettime(CLOCK_REALTIME, &starttime);
  timespec_get(&starttime, TIME_UTC);
  std::stringstream filename, filename2, filename3, filename4;
  filename << starttime.tv_sec << "result.txt";
  fp.open(filename.str(), ios::out);
  if(!fp){
      cout<<"Fail to open file: "<<endl;
      std::cin.get();
  }

  fp << "pos_x, pos_y, pos_z, theta_x, theta_y, theta_z, vel_x, vel_y, vel_z, bias_gyro_x, bias_gyro_y, bias_gyro_z, bias_acc_x, bias_acc_y, bias_acc_z" << endl;



  okvis_estimator.setFullStateCallbackWithExtrinsics(
      std::bind(&PoseViewer::publishFullStateAsCallback, &poseViewer,
                std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                std::placeholders::_4, std::placeholders::_5, std::placeholders::_6));
  //So the function returned still have 4 variables


  filename2 << starttime.tv_sec << "result_landmark.txt";
  fp2.open(filename2.str(), ios::out);
  if(!fp2){
      cout<<"Fail to open file: "<<endl;
      std::cin.get();
  }

  filename3 << starttime.tv_sec << "result_all_landmark.xyz";
  fp3.open(filename3.str(), ios::out);
  if(!fp3){
      cout<<"Fail to open file: "<<endl;
      std::cin.get();
  }

  filename4 << starttime.tv_sec << "Extrinsic_T_SC.txt";
  fp4.open(filename4.str(), ios::out);
  if(!fp4){
      cout<<"Fail to open file: "<<endl;
      std::cin.get();
  }

  fp4 << "cos(theta), sin(theta), ~, T_SC(x), T_SC(y), T_SC(z)" << endl;

  fp2 << "id, x, y, z, w, quality(0-1), distance from world center, num of observations" << endl;
  okvis_estimator.setLandmarksCallback(
      std::bind(&PoseViewer::publishLandmarksAsCallback, &poseViewer,
                std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3));

  //indicates whether the addMeasurement() functions
  //should return immediately (blocking=false), or only when the processing is complete.


  //run offline by providing the dataset
  if (argc>=3)
  {
      okvis_estimator.setBlocking(true);
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
        if(gotinitID>=0) cv::waitKey(PLAY_DELAY_IN_MS);
        else cv::waitKey(10);
        poseViewer.display();

        // check if at the end
        for (size_t i = 0; i < numCameras; ++i) {
          if (cam_iterators[i] == image_names[i].end()) {
            fp.close();fp2.close();
            std::cout << "total_length_of_travel: " << total_length_of_travel << endl;
            std::cout << std::endl << "Finished. Press any key to exit." << std::endl << std::flush;
            PrintAllLandmarks();
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
          //cout << "filtered.type(): " << filtered.type() << endl;

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
              fp.close(); fp2.close();
              std::cout << "total_length_of_travel: " << total_length_of_travel << endl;
              std::cout << std::endl << "Finished. Press any key to exit." << std::endl << std::flush;
              PrintAllLandmarks();
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
          progress=int(double(counter) / double(num_camera_images) * 100);
          std::cout << "\rProgress: "
              << progress << "%  "
              << std::flush;
        }

      }
  }

  // RUN ONLINE
  else
  {
      okvis_estimator.setBlocking(false);
      std::thread t1,t2;
      QTextStream standardOutput(stdout);
      Camera cam;
      Error error;
      TriggerMode triggerMode;
      //Setup the pointgrey camera
      PrintBuildInfo2();

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



      // Connect to a camera
      cout << "Start connecting to the camera...." << endl;
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
      cout << "Start powering on the camera...." << endl;
      error = cam.WriteRegister(k_cameraPower, k_powerVal);
      if (error != PGRERROR_OK)
      {
          PrintError2(error);
          std::cin.get();
          return -1;
      }

      unsigned int regVal = 0;
      unsigned int retries = CAMERA_POWER_UP_RETRY;

      // Wait for camera to complete power-up
      do
      {
//            struct timespec nsDelay;
//            nsDelay.tv_sec = 0;
//            nsDelay.tv_nsec = (long)millisecondsToSleep * 1000000L;
//            nanosleep(&nsDelay, NULL);
          std::this_thread::sleep_for(std::chrono::milliseconds(100));

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

      if(retries<=0)
      {
          cout << "Used up all retries quota!! Continue" << endl;
      }

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

      //char new_resl[512] = "640x512";
      //strcpy(camInfo.sensorResolution, new_resl);

      //error = cam.SetVideoModeAndFrameRate(VIDEOMODE_640x480Y8, FRAMERATE_15);

      const Mode k_fmt7Mode = MODE_0;
      const PixelFormat k_fmt7PixFmt = PIXEL_FORMAT_MONO8;

      // Query for available Format 7 modes
      Format7Info fmt7Info;
      bool supported;
      fmt7Info.mode = k_fmt7Mode;
      error = cam.GetFormat7Info(&fmt7Info, &supported);
      if (error != PGRERROR_OK)
      {
          PrintError2(error);
          std::cin.get();
          return -1;
      }

      PrintFormat7Capabilities(fmt7Info);

      if ((k_fmt7PixFmt & fmt7Info.pixelFormatBitField) == 0)
      {
          // Pixel format not supported!
          cout << "Pixel format is not supported" << endl;
          std::cin.get();
          return -1;
      }

      auto imageSettings = new Format7ImageSettings();
      imageSettings->mode = k_fmt7Mode; // MODE_1 use 2x2 binning of sub-sampled to achieve faster frame rate
      imageSettings->width = CAM_IMAGE_WIDTH;
      imageSettings->height = CAM_IMAGE_HEIGHT;
      imageSettings->pixelFormat = k_fmt7PixFmt;

      bool settingsValid = false;
      Format7PacketInfo packetInfo;
      error = cam.ValidateFormat7Settings(imageSettings, &settingsValid, &packetInfo);
      if (!settingsValid)
      {
          cout << "Settings are not valid" << endl;
          std::cin.get();
          return -1;
      }

      cout << "Setting Format7 config of the camera...." << endl;
      error = cam.SetFormat7Configuration(imageSettings, packetInfo.recommendedBytesPerPacket);
      if (error != PGRERROR_OK)
      {
          PrintError2(error);
          std::cin.get();
          return -1;
      }

      // Set the shutter property of the camera
      Property prop;
      prop.type = SHUTTER;
      error = cam.GetProperty(&prop);
      if (error != PGRERROR_OK)
      {
          PrintError2(error);
          std::cin.get();
          return -1;
      }

      cout << "default Shutter time is " << fixed << setprecision(2) << prop.absValue
           << "ms" << endl;

      prop.autoManualMode = false;
      prop.absControl = true;

      const float k_shutterVal = CAMERA_SHUTTER_TIME_IN_MS * (LOG_MODE==LOG_FOR_RUN?CAMERA_SHUTTER_LOG_FOR_RUN_MULTIPLER:1.0);
      prop.absValue = k_shutterVal;

      error = cam.SetProperty(&prop);
      if (error != PGRERROR_OK)
      {
          PrintError2(error);
          std::cin.get();
          return -1;
      }

      cout << "Shutter time set to " << fixed << setprecision(2) << k_shutterVal
           << "ms" << endl;

      // Set the exposure property of the camera
      Property prop2;
      prop2.type = AUTO_EXPOSURE;
      error = cam.GetProperty(&prop2);
      if (error != PGRERROR_OK)
      {
          PrintError2(error);
          std::cin.get();
          return -1;
      }

      cout << "default AUTO_EXPOSURE is " << fixed << setprecision(2) << prop2.absValue
           << "EV" << endl;

      prop2.autoManualMode = ISCAMERA_SETTING_DEFAULT;
      prop2.absControl = true;

      const float k_exposureVal = CAMERA_EXPOSURE_IN_EV;
      prop2.absValue = k_exposureVal;

      error = cam.SetProperty(&prop2);
      if (error != PGRERROR_OK)
      {
          PrintError2(error);
          std::cin.get();
          return -1;
      }

      cout << "Exposure is set to " << fixed << setprecision(2) << k_exposureVal
           << "EV" << endl;

      // Set the gain property of the camera
      Property prop3;
      prop3.type = GAIN;
      error = cam.GetProperty(&prop3);
      if (error != PGRERROR_OK)
      {
          PrintError2(error);
          std::cin.get();
          return -1;
      }

      cout << "default GAIN is " << fixed << setprecision(2) << prop3.absValue
           << "dB" << endl;

      prop3.autoManualMode = ISCAMERA_SETTING_DEFAULT;
      prop3.absControl = true;

      const float k_GainVal = CAMERA_GAIN_IN_dB* (LOG_MODE==LOG_FOR_RUN?CAMERA_GAIN_LOG_FOR_RUN_MULTIPLER:1.0);
      prop3.absValue = k_GainVal;

      error = cam.SetProperty(&prop3);
      if (error != PGRERROR_OK)
      {
          PrintError2(error);
          std::cin.get();
          return -1;
      }

      cout << "Gain is set to " << fixed << setprecision(2) << k_GainVal
           << "dB" << endl;

      // Get current trigger settings
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

      cout << "Setting trigger mode of the camera...." << endl;
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
      config.grabTimeout = CAMERA_GRAB_TIMEOUT;
      config.numBuffers = CAMERA_NUMBER_OF_BUFFER;

      // Set the camera configuration
      cout << "Setting config of the camera...." << endl;
      error = cam.SetConfiguration(&config);
      if (error != PGRERROR_OK)
      {
          PrintError2(error);
          std::cin.get();
          return -1;
      }

      // Camera is ready, start capturing images
      cout << "Camera start capturing...." << endl;
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
      long long imutime;
      //timespec imutime;
      //TimeStamp imutime;
      int currentoffset=-1; // -1 means not yet found 0xAA, >=0 means the position away from 0xAA
      std::vector<double> imuvalue;

      int i=0, k=0; // i: valid imu frame, k: complete frame received

      unsigned int b_prev=0;
      int sum;
      double value;
      int offsetofframeintmp=0;
      int hasSync=0;
      int imuSyncIndex=-1;
      int k_when_first_capture=-1;


      //waitForReadyRead(): Blocks until new data is available for reading and the readyRead()
      //signal has been emitted, or until msecs milliseconds have passed. If msecs is -1,
      //this function will not time out.

      //Returns true if new data is available for reading; otherwise returns false
      //(if the operation timed out or if an error occurred).
      //clock_gettime(CLOCK_REALTIME, &imutime);

      timespec imucalib_starttime, tstart, tend, tmp;
      timespec_get(&imucalib_starttime, TIME_UTC);

      cout << "Now is time= " << (long long)imucalib_starttime.tv_sec << std::setw(9) <<
              std::setfill('0') << (long)imucalib_starttime.tv_nsec<< endl ;
      cout << "All data will be saved in directory /" << (long long)starttime.tv_sec << endl;
      //cout << "Main loop Start!! " << endl;

      int ttyUSBtryindex=0;
      clock_t end, begin = clock();

      QSerialPort serialPort;
      QString serialPortName = USBPORTNAME+QString::number(ttyUSBtryindex);
      serialPort.setPortName(serialPortName);

      int serialPortBaudRate = QSerialPort::Baud115200;
      serialPort.setBaudRate(serialPortBaudRate);

      if (!serialPort.open(QIODevice::ReadOnly)) {
          standardOutput << QObject::tr("Failed to open port %1, error: %2, please check if the previous window is closed").arg(serialPortName).arg(serialPort.error()) << endl;
          cout << "Press any key to try other port" << endl;
          std::cin.get();

          ttyUSBtryindex++;
          while(ttyUSBtryindex<MAX_TTY_TRY_INDEX)
          {
              serialPortName = USBPORTNAME+QString::number(ttyUSBtryindex);
              serialPort.setPortName(serialPortName);
              if(!serialPort.open(QIODevice::ReadOnly))
              {
                  standardOutput << QObject::tr("Failed to open port %1.").arg(serialPortName) << endl;
              }
              else break;

              ttyUSBtryindex++;
          }
          if(ttyUSBtryindex==MAX_TTY_TRY_INDEX)
          {
              cout << "All port fails!" << endl;
              std::cin.get();
              return 1;
          }
      }

      standardOutput << QObject::tr("Opened the serial port %1, at baudrate: %2").arg(serialPortName).arg(serialPortBaudRate) << endl;
      t2 = std::thread(ReadIMUdata, std::ref(serialPort), std::ref(okvis_estimator));
      t1 = std::thread(CamRetrieveBuffer, std::ref(cam), std::ref(okvis_estimator));

      cout << "Sleep " << SECOND_TO_SLEEP_AT_START << " seconds to let imu and camera read rate stable..." << endl << endl;
      //usleep(SECOND_TO_SLEEP_AT_START*1e6);
      std::this_thread::sleep_for(std::chrono::seconds(SECOND_TO_SLEEP_AT_START));



      //get start at the middle of 2 sync signals
      while(awayfromlastsynccounter!=IMU_TO_CAM_RATIO/2)
      {
          //usleep(1e2);
          std::this_thread::sleep_for(std::chrono::microseconds(100));
          //cout << "Try to get start at the middle of 2 sync signals..." << endl;
      }
      timespec_get(&tmp, TIME_UTC);
      BothStart=true;
      cout << "BothStart=true" << endl;
      cout << "Now timestamp= " << tmp.tv_nsec<< endl;

      std::thread(MatchingAndUpdateEstimator, std::ref(okvis_estimator)).detach();
      while(true)
      {
          okvis_estimator.display(); // show all OKVIS's camera
          cv::waitKey(PLAY_DELAY_IN_MS);
          poseViewer.display();
      }

      cout << "Main loop ends" << endl;

      // Turn trigger mode off.
      cout << "Turning trigger mode off ... " << endl;
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
      cout << "Stop capturing images ... " << endl;
      error = cam.StopCapture();
      if (error != PGRERROR_OK)
      {
          PrintError2(error);
          std::cin.get();
          return -1;
      }

      // Disconnect the camera
      cout << "Disconnecting from the camera ... " << endl;
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
