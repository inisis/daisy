#include <cstdlib>
#include <string>
#include <sstream>
#include "anyconversion.h"

AnyConversion::AnyConversion(string const& value) {
    value_ = value;
}

AnyConversion::AnyConversion(const char* c) {
    value_ = c;
}

AnyConversion::AnyConversion(float d) {
    stringstream s;
    s << d;
    value_ = s.str();
}

AnyConversion::AnyConversion(double d) {
    stringstream s;
    s << d;
    value_ = s.str();
}

AnyConversion::AnyConversion(bool d) {
    stringstream s;
    if (d)
        s << "on";
    else
        s << "off";
    value_ = s.str();
}

AnyConversion::AnyConversion(int i) {
    stringstream s;
    s << i;
    value_ = s.str();
}

AnyConversion::AnyConversion(AnyConversion const& other) {
    value_ = other.value_;
}

AnyConversion& AnyConversion::operator=(AnyConversion const& other) {
    value_ = other.value_;
    return *this;
}

AnyConversion& AnyConversion::operator=(double d) {
    stringstream s;
    s << d;
    value_ = s.str();
    return *this;
}

AnyConversion& AnyConversion::operator=(float d) {
    stringstream s;
    s << d;
    value_ = s.str();
    return *this;
}

AnyConversion& AnyConversion::operator=(bool d) {
    stringstream s;
    if (d)
        s << "on";
    else
        s << "off";
    value_ = s.str();
    return *this;
}

AnyConversion& AnyConversion::operator=(int i) {
    stringstream s;
    s << i;
    value_ = s.str();
    return *this;
}

AnyConversion& AnyConversion::operator=(string const& s) {
    value_ = s;
    return *this;
}

AnyConversion::operator string() const {
    return value_;
}

AnyConversion::operator double() const {
    return atof(value_.c_str());
}

AnyConversion::operator float() const {
    return (float) atof(value_.c_str());
}

AnyConversion::operator int() const {
    return atoi(value_.c_str());
}

AnyConversion::operator bool() const {
    if (value_.compare("on") == 0 || atoi(value_.c_str()) == 1)
        return true;
    return false;
}

bool AnyConversion::empty() {
    return value_.empty();
}