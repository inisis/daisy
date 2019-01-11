#ifndef _DATA_TYPE_H_
#define _DATA_TYPE_H_

#include <iostream>
#include <utility>
#include <vector>
#include <opencv2/core/core.hpp>

using namespace std;

class BaseAttribute {
public:
    int idx;
    string name;
    float confidence;
    float value;

    float thresh_low;
    float thresh_high;
    int mappingId;
    int categoryId;
};

class AttributeTag {
public:
    int index;
    std::string tagname;
    float threshold_lower;
    float threshold_upper;
    int categoryId;
    int mappingId;
    std::string output_layer;
    int output_id;
};
#endif
