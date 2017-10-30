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
 *  Created on: Mar 10, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file stereo_triangulation.cpp
 * @brief Implementation of the triangulateFast function.
 * @author Stefan Leutenegger
 */

#include <okvis/triangulation/stereo_triangulation.hpp>
#include <okvis/kinematics/operators.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <iostream>
#include <glog/logging.h>

/// \brief okvis Main namespace of this package.
namespace okvis {

/// \brief triangulation A namespace for operations related to triangulation.
namespace triangulation {

// Triangulate the intersection of two rays.
Eigen::Vector4d triangulateFast(const Eigen::Vector3d& p1, // center of A in A coordinate
                                const Eigen::Vector3d& e1, // back project direction for keypt A in A coordinate, a unit vector
                                const Eigen::Vector3d& p2, // center of B in A coordinate
                                const Eigen::Vector3d& e2, // back project direction for keypt B in A coordinate, a unit vector
                                double sigma,
                                bool& isValid, bool& isParallel)
{
  const double initialdepthguess = 1000.0;
  const double inversecheckthreshold = 6.8523*1e-4; //  default is 1.0e-6

  isParallel = false; // This should be the default, whether e1 and e2 are parallel
  // But parallel and invalid is not the same. Points at infinity are valid and parallel.
  isValid = false; // hopefully this will be reset to true.

  // stolen and adapted from the Kneip toolchain geometric_vision/include/geometric_vision/triangulation/impl/triangulation.hpp
  Eigen::Vector3d t12 = p2 - p1;

  // check parallel
  /*if (t12.dot(e1) - t12.dot(e2) < 1.0e-12) {
    if ((e1.cross(e2)).norm() < 6 * sigma) {
      isValid = true;  // check parallel
      isParallel = true;
      return (Eigen::Vector4d((e1[0] + e2[0]) / 2.0, (e1[1] + e2[1]) / 2.0,
                              (e1[2] + e2[2]) / 2.0, 1e-2).normalized());
    }
  }*/

  Eigen::Vector2d b;
  b[0] = t12.dot(e1);
  b[1] = t12.dot(e2);

  // A = [ |e1|^2         -|e1||e2|cosr ]
  //     [ |e1||e2|cosr   -|e2|^2       ]
  // det(A) = -(|e1||e2|)^2 + (|e1||e2|)^2 * (cosr)^2 = (|e1||e2|)^2 ((cosr)^2-1)
  // since e1 and e2 are unit vector, det(A) = ((cosr)^2-1) = -(sinr)^2

  // If >=0.5 degree is allowed, then det(A) = 7.6152*1e-5
  // If >=1 degree is allowed, then det(A) = 3.0458*1e-4
  // If >=1.5 degree is allowed, then det(A) = 6.8523*1e-4
  // If >=2 degree is allowed, then det(A) = 1.2179*1e-3
  // If >=3 degree is allowed, then det(A) = 2.739*1e-3
  Eigen::Matrix2d A;
  A(0, 0) = e1.dot(e1);
  A(1, 0) = e1.dot(e2);
  A(0, 1) = -A(1, 0);
  A(1, 1) = -e2.dot(e2);

  if (A(1, 0) < 0.0)
  {
    A(1, 0) = -A(1, 0);
    A(0, 1) = -A(0, 1);
    // wrong viewing direction
  };

  bool invertible;
  Eigen::Matrix2d A_inverse;
  //The matrix will be declared invertible if the absolute value of its determinant is greater than this threshold.
  A.computeInverseWithCheck(A_inverse, invertible, inversecheckthreshold);
  Eigen::Vector2d lambda = A_inverse * b;

  if (!invertible)
  {
    isParallel = true; // let's note this.
    //LOG(INFO) << "The rays are parallel, the landmark will not be initialized";
    // parallel. that's fine. but A is not invertible. so handle it separately.
    if ((e1.cross(e2)).norm() < 6 * sigma) // if it is highly parallel
    {
       isValid = true;  // check parallel
    }

    //LOG(INFO) << "midpoint(guess): " << ((e1+e2)/2.0).transpose();
    //Just set the depth to 10m, since no better guess can be done
    return (Eigen::Vector4d((e1[0] + e2[0]) / 2.0, (e1[1] + e2[1]) / 2.0,
                            (e1[2] + e2[2]) / 2.0, (1/initialdepthguess)).normalized());
  }

  //Try to find the intersection pt of 2 rays (i.e. e1 and e2)
  //The intersection is the mid-pt of the shortest projection from 1 ray to another //A ray can be defined as r=starting_pt+direction_vector*t, for any t
  //lambda[0] and lambda[1] is the scale factor 't' which makes xm-xn = shortest projection
  Eigen::Vector3d xm = lambda[0] * e1 + p1;
  Eigen::Vector3d xn = lambda[1] * e2 + p2;
  Eigen::Vector3d midpoint = (xm + xn) / 2.0;

  // check it
  Eigen::Vector3d error = midpoint - xm;
  Eigen::Vector3d diff = midpoint - (p1 + 0.5 * t12); // diff is the height of the "triangle" for "triangulation"
  const double diff_sq = diff.dot(diff);
  const double chi2 = error.dot(error) * (1.0 / (diff_sq * sigma * sigma)); //sigma is 1StdDev of error in meter when the ray travels 1m

  isValid = true;
  if (chi2 > 9) {
    //LOG(INFO) << "triangulateFast invalid (i.e. chi2 > 9), cannot add landmark";
    //LOG(INFO) << "error: " << error.transpose() << ", diff: " << diff.transpose() << ", sigma: " << sigma << ", chi2: " << chi2;
    isValid = false;  // reject large chi2-errors
  }

  // flip if necessary (flip means z-depth calculated < 0)
  if (diff.dot(e1) < 0) {
    midpoint = (p1 + 0.5 * t12) - diff;
  }

  //LOG(INFO) << "midpoint: " << midpoint.transpose();
  //if(midpoint[2]<0 || midpoint[2]>1000) LOG(WARNING) << "triangulation wrong!";
  return Eigen::Vector4d(midpoint[0], midpoint[1], midpoint[2], 1.0).normalized();
}

}

}

