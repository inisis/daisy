//
// Created by liuhao on 18-5-4.
//

#ifndef _ATTRIBUTETAGTABLE_H
#define _ATTRIBUTETAGTABLE_H

#include "map"
#include "vector"
#include "data_type.h"

class AttributeTagTable {
public:
    AttributeTagTable() = default;

    AttributeTagTable(std::string tag_filename);

    bool loadFile(std::string tag_filename);

    std::vector<std::string> output_layervec; //那些blob要被输出
    std::vector<AttributeTag> tagtable_;  //属性结构定义

    AttributeTag getAttributeTagFromIndex(int index);


private:
    std::string tagtable_filename = "";
};

#endif //_ATTRIBUTETAGTABLE_H
