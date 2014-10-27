#ifndef __MultiModel_hpp_
#define __MultiModel_hpp_

#include "Model.hpp"

class MultiModel : public Model {
public:
  MultiModel();
  
  void load_data(const char *filename);
  void clear_data();
  
protected:  
  int **mWkrLbls;
  int **mImgLbls;
  int *mLabels;
};

#endif
