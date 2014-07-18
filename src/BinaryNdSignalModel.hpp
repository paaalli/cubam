#ifndef __BinaryNdSignalModel_hpp__
#define __BinaryNdSignalModel_hpp__

#include "BinaryModel.hpp"

#include "BinarySignalModel.hpp"

class BinaryNdSignalModel : public BinarySignalModel {
public:
  BinaryNdSignalModel() { mDim = 2; };
  
  void set_model_param(double *prm);
  void get_model_param(double *prm);
  
  void set_worker_param(double *vars);
  void set_image_param(double *xis);
  void set_gt_prediction(int *gt);
  void set_cv_prob(double **cv);

  void get_worker_param(double *vars);
  void get_image_param(double *xis);

  void reset_worker_param();
  void reset_image_param();
  void reset_gt_prediction();
  void reset_cv_prob();
  
  void worker_objective(int wkrId, double *prm, int nprm, double* obj);
  void image_objective(int imgId, double *prm, int nprm, double* obj);
  
  virtual int get_worker_param_len() { return mNumWkrs*(1+mDim); }
  virtual int get_image_param_len() { return mNumImgs*mDim; }
  virtual int get_model_param_len() { return 6; }
  
  double objective();

  void optimize_gt();
  
  void gradient(double *grad);

private:
  int mDim;
};


#endif
