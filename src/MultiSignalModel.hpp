#ifndef __MultiSignalModel_hpp__
#define __MultiSignalModel_hpp__

#include "MultiModel.hpp"

class MultiSignalModel : public MultiModel {
public:
  MultiSignalModel();
  
  void set_model_param(double *prm);
  void get_model_param(double *prm);

  void set_use_cv(bool cv);
  void set_worker_param(double *vars);
  void set_image_param(double *xis);
  void set_gt_prediction(int *gt);
  void set_cv_prob(double **cv);

  void get_worker_param(double *vars);
  void get_image_param(double *xis);
  void get_image_prob(double *img_prob);

  void clear_data();
    
  virtual int get_worker_param_len() { return mNumWkrs*(1+mDim); }
  virtual int get_image_param_len() { return mNumImgs*mDim; }
  virtual int get_model_param_len() { return 7; }
  
  double objective();

  void optimize_gt();
  
  void gradient(double *grad);

private:
  int mDim;
  int mClasses;

protected:
  void clear_worker_param();
  void clear_image_param();
  void clear_gt_prediction();
  void clear_cv_prob();

  bool use_cv;
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
