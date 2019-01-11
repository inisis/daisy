#include "AttributeTagTable.h"
#include <fstream>
#include <set>
#include <glog/logging.h>

AttributeTagTable::AttributeTagTable(std::string tag_filename) {
    loadFile(tag_filename);
}

bool AttributeTagTable::loadFile(std::string tag_filename) {
    LOG(INFO) << "Reading tag file: " << tag_filename;
    tagtable_filename = tag_filename;
    std::ifstream fp(tag_filename);
    if (!fp) {
        LOG(FATAL) << "Tag file not exist: " << tag_filename;
    }

    tagtable_.resize(0);
    while (!fp.eof()) {
        std::string tagname = "", indexstr = "", threshold_lower = "",
                threshold_upper = "", categoryId = "", mappingId = "",
                output_layer = "",  output_id="";

        fp >> tagname;
        fp >> indexstr;
        fp >> categoryId;
        fp >> mappingId;
        fp >> threshold_lower;
        fp >> threshold_upper;
        fp >> output_layer;
        fp >> output_id;
        if (tagname == "" || indexstr == "" || threshold_lower == ""
            || threshold_upper == "" || categoryId == "" || mappingId == "")
            continue;
        AttributeTag tag;
        tag.index = atoi(indexstr.c_str());
        tag.tagname = tagname;
        tag.threshold_lower = atof(threshold_lower.c_str());
        tag.threshold_upper = atof(threshold_upper.c_str());
        tag.categoryId = atoi(categoryId.c_str());
        tag.mappingId = atoi(mappingId.c_str());
        tag.output_layer = output_layer;
        tag.output_id = atoi(output_id.c_str());
        tagtable_.push_back(tag);
    }
}

AttributeTag AttributeTagTable::getAttributeTagFromIndex(int index) {
    if (tagtable_.size() > index && tagtable_[index].index == index) {
        return tagtable_[index];
    }
    for (auto tag : tagtable_) {
        if (tag.index == index)
            return tag;
    }
    if (tagtable_filename.empty())
        LOG(FATAL) << "Attribute tag did not load the tag file.";
    LOG(FATAL) << "Please check the tag.cfg [" << tagtable_filename << "], there is no such attribute index:"
               << index;
}
