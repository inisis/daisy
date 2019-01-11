
#ifndef _ANYCONVERSION_H_
#define _ANYCONVERSION_H_

#include <string>

using namespace std;

class AnyConversion {
 public:
    AnyConversion() {
    }
    ;
    explicit AnyConversion(const string&);
    explicit AnyConversion(float);
    explicit AnyConversion(double);
    explicit AnyConversion(int);
    explicit AnyConversion(bool);
    explicit AnyConversion(const char*);

    AnyConversion(const AnyConversion&);
    AnyConversion& operator=(AnyConversion const&);

    AnyConversion& operator=(float);
    AnyConversion& operator=(double);
    AnyConversion& operator=(int);
    AnyConversion& operator=(bool);
    AnyConversion& operator=(string const&);

    bool empty();

 public:
    operator string() const;
    operator float() const;
    operator double() const;
    operator int() const;
    operator bool() const;

 private:
    string value_;
};

#endif
