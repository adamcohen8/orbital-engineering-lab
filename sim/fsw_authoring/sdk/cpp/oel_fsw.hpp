#ifndef OEL_FSW_HPP
#define OEL_FSW_HPP
#include "oel_fsw.h"
#include <array>
#include <stdexcept>
#include <string>
namespace oel {
class Input {
public:
    const oel_json *raw;
    explicit Input(const oel_json *value): raw(value) {}
    int find_schema(const char *name) const { return oel_find_schema(raw,name); }
    template<size_t N> bool vector(int object,const char *key,std::array<double,N>& value) const {
        return oel_vector(raw,object,key,value.data(),N)==0;
    }
    std::string field(const char *key) const {
        char out[4096]; int token=oel_field(raw,-1,key);
        if(token<0 || oel_raw(raw,token,out,sizeof out)) throw std::runtime_error("Missing input field");
        return out;
    }
};
class FlightSoftware {
public:
    virtual ~FlightSoftware()=default;
    virtual int on_message(const std::string& operation,Input input,char *output,size_t capacity)=0;
    int run(bool checkpointable=false) {
        return oel_run([](const char* op,const oel_json* in,char* out,size_t n,void* self)->int {
            try { return static_cast<FlightSoftware*>(self)->on_message(op,Input(in),out,n); }
            catch(...) { return -1; }
        },this,checkpointable);
    }
};
}
#endif
