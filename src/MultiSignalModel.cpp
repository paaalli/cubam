#include "MultiSignalModel.hpp"
#include <iostream>
MultiSignalModel::MultiSignalModel() {
  mBeta = 0.5;
  mSigX = 0.8;
  mSigW = 1.0;
  mMuW = 1.0;
  mSigT = 3;
  mXis = 0;
  mWjs = 0;
  mTjs = 0;
  gt_prediction = 0;
  cv_prob = 0;
  use_cv = false;
}

void MultiSignalModel::set_model_param(double *prm) {
  mBeta = prm[0];
  mSigX = prm[1];
  mSigW = prm[2];
  mMuW = prm[3];
  mSigT = prm[4];
  if (mDataIsLoaded && (prm[5] != mDim))
    throw runtime_error("Cannot set dimension when data is loaded.");
  mDim = int(prm[5]);
  mClasses = int(prm[6]);
}

void MultiNdSignalModel::get_model_param(double *prm) {
  prm[0] = mBeta;
  prm[1] = mSigX;
  prm[2] = mSigW;
  prm[3] = mMuW;
  prm[4] = mSigT;
  prm[5] = double(mDim);
  prm[6] = double(mClasses);
}

void MultiSignalModel::clear_data() {
  MultiModel::clear_data();
  clear_worker_param();
  clear_image_param();
  clear_gt_prediction();
  clear_cv_prob();
}

void MultiSignalModel::reset_worker_param() {
  if (!mDataIsLoaded)
    throw runtime_error("You must load data before resetting parameters.");
  clear_worker_param();
  int nElements = mNumWkrs*mDim;
  mWjs = new double[nElements];
  for (int j=0; j<nElements; j++)
    mWjs[j] = 1.0;
  mTjs = new double[mNumWkrs];
  for (int j=0; j<mNumWkrs; j++)
    mTjs[j] = 0.0;
}

void MultiSignalModel::reset_image_param() {
  if (!mDataIsLoaded)
    throw runtime_error("You must load data before resetting parameters.");
  clear_image_param();
  int nElements = mNumImgs*mDim;
  mXis = new double[nElements];
  for (int i=0; i<nElements; i++)
    mXis[i] = 0.0;
}

void MultiSignalModel::reset_gt_prediction() {
  if (!mDataIsLoaded)
    throw runtime_error("You must load data before resetting parameters.");
  clear_gt_prediction();
  int nElements = mNumImgs;
  gt_prediction = new int[nElements];
  for (int i=0; i<nElements; i++)
    gt_prediction[i] = -1;
}

//We have to check whether cv_prob has been assigned a value before
//to clear it. Because it's multidimensional we need to delete the arrays
//it has as values. Attempting to do that with non assigned values will
//result in a seg fault.
void MultiSignalModel::reset_cv_prob() {
  if (!mDataIsLoaded)
    throw runtime_error("You must load data before resetting parameters.");
  if (cv_prob != 0)
    clear_cv_prob();
  int nElements = mNumImgs;
  cv_prob = new double*[nElements];
  for (int i=0; i<nElements; i++) {
    cv_prob[i] = new double[2];
    for (int j=0; j < mClasses; j++) {
      cv_prob[i][j] = double(1/mClasses)
    }
  }
}

void MultiSignalModel::set_image_param(double *xis) {
  if (!mDataIsLoaded)
    throw runtime_error("Data not loaded.");
  int nElements = mNumImgs*mDim;
  for (int i=0; i<nElements; i++)
    mXis[i] = xis[i];
}

void MultiSignalModel::set_worker_param(double *vars) {
  if (!mDataIsLoaded)
    throw runtime_error("Data not loaded.");
  int nElements = mNumWkrs*mDim;
  for (int j=0; j<nElements; j++)
    mWjs[j] = vars[j];
  for (int j=0; j<mNumWkrs; j++)
    mTjs[j] = vars[j+nElements];
}

void MultiSignalModel::set_gt_prediction(int *gt) {
  if (!mDataIsLoaded)
    throw runtime_error("Data not loaded.");
  int nElements = mNumImgs;
  for (int i=0; i<nElements; i++)
    gt_prediction[i] = gt[i];
}

void MultiSignalModel::set_cv_prob(double **cv) {
  if (!mDataIsLoaded)
    throw runtime_error("Data not loaded.");
  int nElements = mNumImgs
  for (int i=0; i<nElements; i++) {
    for (int j = 0; j < mClasses; i++)
      cv_prob[i][j] = cv[i][j];
  }
}

void MultiSignalModel::get_image_param(double *xis) {
  if (!mDataIsLoaded)
    throw runtime_error("Data not loaded.");
  int nElements = mNumImgs*mDim;
  for (int i=0; i<nElements; i++)
    xis[i] = mXis[i];
}
void MultiSignalModel::clear_worker_param() {
  delete [] mWjs; mWjs = 0;
  delete [] mTjs; mTjs = 0;
}

void MultiSignalModel::clear_image_param() {
  delete [] mXis; mXis = 0;
}

void MultiSignalModel::clear_gt_prediction() {
  delete [] gt_prediction; gt_prediction = 0;
}

//We assume cv_prob has already been assigned arrays for values. 
//Otherwise we get segment fault.
void MultiSignalModel::clear_cv_prob() {
  for (int i = mNumImgs; i >= 0; i--) {
    delete[] cv_prob[i]; cv_prob[i] = 0;
  }
  delete[] cv_prob; cv_prob = 0;
}

void MultiSignalModel::set_use_cv(bool cv) {
  use_cv = cv;
}

//Calculates the objective function for Welinder's paper. (log one [6])
double Binary1dSignalModel::objective() {
  if (!mDataIsLoaded)
    throw runtime_error("Data not loaded.");
  double obj = 0.0;
  // compute the xi prior

  for(int i=0; i<mNumImgs; i++) {
    x = new double[mClasses];
    //Initialize x array.
    for(int j=0; j < mClasses) {
      x[j] = 0.0
    }
    //Instead of 1,-1 for classes 1 and 0 we use mui = i for class i.
    for(int d=0; d<mDim; d++) {
      for(int c=0; c<mClasses; c++) {
        x[c] += (mXis[i*mDim+d]-c)*(mXis[i*mDim+d]-c);
    }

    obj += -0.5*double(mDim)*log(2.0*PI*mSigX*mSigX);
    if (!use_cv) {
      for(int k=0; k < mClasses; k++) {
        obj += log(mBeta*exp(-0.5*x[k]/(mSigX*mSigX)));
      }
    } else {
      for(int k=0; k < mClasses; k++) {
        obj += log(cv_prob[i][k]*exp(-0.5*x[k]/(mSigX*mSigX)));
      }
    }
  }

  // compute the wj prior
  for(int j=0; j<mNumWkrs; j++)
    for(int d=0; d<mDim; d++)
      obj += LOGNORM(mWjs[j*mDim+d], mMuW, mSigW);

  // compute the tj prior, PRIOR ON TJ????
  for(int j=0; j<mNumWkrs; j++)
    for (int c=0; c<mClasses; c++)
      obj += LOGNORM(mTjs[j*mClasses+c], 0.0, mSigT);


   // compute the shared terms
  int idx, i, j, lij;
  for(int k=0; k<mNumLbls; k++) {
    idx = k*3;
    i = mLabels[idx];
    j = mLabels[idx+1];
    lij = mLabels[idx+2];
    double cdfarg = 0.0;
    for(int d=0; d<mDim; d++)
      cdfarg += mXis[i*mDim+d]*mWjs[j*mDim+d];
    cdfarg -= mTjs[j][lij];
    //Really weird, Nd only wants to call cdf with value < 0. I will follow suit here
    //Dont know why though.
    if(cdfarg<0.0)
      obj += log(cdf(cdfarg));
    else
      obj += log(1.0-cdf(-cdfarg));
  
  }
  return -obj;
}

void BinaryNdSignalModel::gradient(double *grad) {
  if (!mDataIsLoaded)
    throw runtime_error("Data not loaded.");
  // assumes that *grad is a list of length mNumImgs+2*mNumWkrs
  // the order is assumed to be [xis, wjs, tjs]
  int gradLen = mDim*mNumImgs + (1+mDim)*mNumWkrs;
  for(int i=0; i<gradLen; i++)
    grad[i] = 0.0;
  // compute the xi prior gradient
  for(int i=0; i<mNumImgs; i++) {
    double x0sq = 0.0;
    double x1sq = 0.0;
    for(int d=0; d<mDim; d++) {
      x0sq += (mXis[i*mDim+d]+1.0)*(mXis[i*mDim+d]+1.0);
      x1sq += (mXis[i*mDim+d]-1.0)*(mXis[i*mDim+d]-1.0);
    }
    x0sq = NORMAL(x0sq, mSigX);
    x1sq = NORMAL(x1sq, mSigX);
    for(int d=0; d<mDim; d++)
      grad[i*mDim+d] = (mBeta*(mXis[i*mDim+d]-1.0)*x1sq + 
        (1.0-mBeta)*(mXis[i*mDim+d]+1.0)*x0sq)
        /mSigX/mSigX/ (mBeta*x1sq +(1.0-mBeta)*x0sq);
  }
  // compute the wj & tj prior gradients
  int woffset = mNumImgs*mDim;
  int toffset = woffset + mNumWkrs*mDim;
  for(int j=0; j<mNumWkrs; j++) {
    grad[toffset+j] = mTjs[j]/mSigT/mSigT;
    for(int d=0; d<mDim; d++)
      grad[woffset+j*mDim+d] = (mWjs[j*mDim+d]-mMuW)/mSigW/mSigW;
  }
  // compute the shared terms
  int idx, i, j, lij;
  for(int k=0; k<mNumLbls; k++) {
    idx = k*3;
    i = mLabels[idx];
    j = mLabels[idx+1];
    lij = mLabels[idx+2];
    double cdfarg = 0.0;
    for(int d=0; d<mDim; d++)
      cdfarg += mXis[i*mDim+d]*mWjs[j*mDim+d];
    cdfarg -= mTjs[j];
    double lambda_ij;
    if(lij == 0) {
      if(cdfarg<0.0)
        lambda_ij = -1.0/(1.0-cdf(cdfarg));
      else
        lambda_ij = -1.0/cdf(-cdfarg);
    } else {
      if(cdfarg<0.0)
        lambda_ij = 1.0/cdf(cdfarg);
      else
        lambda_ij = 1.0/(1.0-cdf(-cdfarg));
    }
    double philambda_ij = exp(-0.5*cdfarg*cdfarg)/sqrt(2.0*PI)*lambda_ij;
    // add shared components to gradients
    for(int d=0; d<mDim; d++) {
      grad[i*mDim+d] -= mWjs[j*mDim+d]*philambda_ij;
      grad[woffset+j*mDim+d] -= mXis[i*mDim+d]*philambda_ij;
    }
    grad[toffset+j] += philambda_ij;
  }
}
