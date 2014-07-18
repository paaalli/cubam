#include "BinarySignalModel.hpp"
#include <iostream>
BinarySignalModel::BinarySignalModel() {
  mBeta = 0.5;
  mSigX = 0.8;
  mSigW = 1.0;
  mMuW = 1.0;
  mSigT = 3.0;
  mXis = 0;
  mWjs = 0;
  mTjs = 0;
  gt_prediction = 0;
  cv_prob = 0;
  use_z = false;
}

void BinarySignalModel::set_model_param(double *prm) {
  mBeta = prm[0];
  mSigX = prm[1];
  mSigW = prm[2];
  mMuW = prm[3];
  mSigT = prm[4];
}

void BinarySignalModel::get_model_param(double *prm) {
  prm[0] = mBeta;
  prm[1] = mSigX;
  prm[2] = mSigW;
  prm[3] = mMuW;
  prm[4] = mSigT;
}

void BinarySignalModel::clear_data() {
  BinaryModel::clear_data();
  clear_worker_param();
  clear_image_param();
  clear_gt_prediction();
  clear_cv_prob();
}

void BinarySignalModel::clear_worker_param() {
  delete [] mWjs; mWjs = 0;
  delete [] mTjs; mTjs = 0;
}

void BinarySignalModel::clear_image_param() {
  delete [] mXis; mXis = 0;
}

void BinarySignalModel::clear_gt_prediction() {
  delete [] gt_prediction; gt_prediction = 0;
}

//We assume cv_prob has already been assigned arrays for values. 
//Otherwise we get segment fault.
void BinarySignalModel::clear_cv_prob() {
  for (int i = mNumImgs; i >= 0; i--) {
    delete[] cv_prob[i]; cv_prob[i] = 0;
  }
  delete[] cv_prob; cv_prob = 0;
}

void BinarySignalModel::set_use_z(bool z) {
  use_z = z;
}