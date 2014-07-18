#ifndef __BinarySignalModel_hpp__
#define __BinarySignalModel_hpp__

#include "BinaryModel.hpp"

class BinarySignalModel : public BinaryModel {
public:
  BinarySignalModel();
  
  void set_model_param(double *prm);
  void get_model_param(double *prm);

  void set_use_z(bool z);
  
  void clear_data();
  
protected:
  void clear_worker_param();
  void clear_image_param();
  void clear_gt_prediction();
  void clear_cv_prob();

  bool use_z;
  int *gt_prediction;
  double **cv_prob;
  double *mXis;
  double *mWjs;
  double *mTjs;
  double mBeta;
  double mSigX;
  double mSigW;
  double mMuW;
  double mSigT;
};

#endif
