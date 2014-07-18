#ifndef __Binary1dSignalModel_hpp__
#define __Binary1dSignalModel_hpp__

#include "BinarySignalModel.hpp"

class Binary1dSignalModel : public BinarySignalModel {
public:
  void set_worker_param(double *vars);
  void set_image_param(double *xis);
  void set_gt_prediction(int *gt);

  void get_worker_param(double *vars);
  void get_image_param(double *xis);
  void set_cv_prob(double **cv);

  void reset_worker_param();
  void reset_image_param();
  void reset_gt_prediction();
  void reset_cv_prob();
  
  void worker_objective(int wkrId, double *prm, int nprm, double* obj);
  void image_objective(int imgId, double *prm, int nprm, double* obj);
  
  virtual int get_worker_param_len() { return mNumWkrs*2; }
  virtual int get_image_param_len() { return mNumImgs; }
  virtual int get_model_param_len() { return 5; }

  double objective();
  double image_objective_summed(int imgId);
  double image_objective_specified(int imgId);

  void optimize_gt();
  
  void gradient(double *grad);
  double image_gradient_summed(int imgId);
  double image_gradient_specified(int imgId);
};

#endif
