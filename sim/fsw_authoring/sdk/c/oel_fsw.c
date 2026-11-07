#include "oel_fsw.h"
#include <ctype.h>
#include <limits.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
#define TOKENS 32768
struct token { size_t start, end; int parent; char kind; };
struct oel_json { const char *text; struct token *tokens; int count, root; size_t at; };
static void space(oel_json *j) { while (isspace((unsigned char)j->text[j->at])) ++j->at; }
static int parse(oel_json *j, int parent, int depth) {
    int id; char c;
    space(j); c=j->text[j->at];
    if (!c || depth>64 || j->count>=TOKENS) return -1;
    id=j->count++; j->tokens[id].start=j->at; j->tokens[id].parent=parent; j->tokens[id].kind=c;
    if (c=='{' || c=='[') {
        char end=c=='{'?'}':']'; ++j->at; space(j);
        if (j->text[j->at]!=end) for (;;) {
            if (c=='{') { if(j->text[j->at]!='"' || parse(j,id,depth+1)<0) return -1;
                space(j); if(j->text[j->at++]!=':') return -1; }
            if(parse(j,id,depth+1)<0) return -1;
            space(j); if(j->text[j->at]==end) break;
            if(j->text[j->at++]!=',') return -1;
            space(j);
        }
        ++j->at;
    } else if (c=='"') {
        ++j->at;
        while(j->text[j->at] && j->text[j->at]!='"') {
            unsigned char ch=(unsigned char)j->text[j->at++];
            if(ch<32) return -1;
            if(ch=='\\') {
                char e=j->text[j->at++];
                if(e=='u') { int k; for(k=0;k<4;++k) if(!isxdigit((unsigned char)j->text[j->at++])) return -1; }
                else if(!e || !strchr("\"\\/bfnrt",e)) return -1;
            }
        }
        if(j->text[j->at++]!='"') return -1;
    } else {
        size_t begin=j->at, n;
        while(j->text[j->at] && !strchr(" \t\r\n,}]",j->text[j->at])) ++j->at;
        n=j->at-begin;
        if(!n) return -1;
        if(!((n==4 && !strncmp(j->text+begin,"true",4)) ||
             (n==5 && !strncmp(j->text+begin,"false",5)) ||
             (n==4 && !strncmp(j->text+begin,"null",4)))) {
            size_t p=begin; char *finish; double value;
            if(j->text[p]=='-') ++p;
            if(j->text[p]=='0') ++p;
            else { if(j->text[p]<'1'||j->text[p]>'9') return -1; while(isdigit((unsigned char)j->text[p])) ++p; }
            if(j->text[p]=='.') { ++p; if(!isdigit((unsigned char)j->text[p])) return -1; while(isdigit((unsigned char)j->text[p])) ++p; }
            if(j->text[p]=='e'||j->text[p]=='E') { ++p; if(j->text[p]=='+'||j->text[p]=='-') ++p; if(!isdigit((unsigned char)j->text[p])) return -1; while(isdigit((unsigned char)j->text[p])) ++p; }
            value=strtod(j->text+begin,&finish);
            if(p!=j->at || finish!=j->text+j->at || !isfinite(value)) return -1;
        }
    }
    j->tokens[id].end=j->at; return id;
}
static int equal(const oel_json *j,int t,const char *text) {
    size_t n=strlen(text);
    return t>=0 && j->tokens[t].kind=='"' && j->tokens[t].end-j->tokens[t].start==n+2 &&
        !strncmp(j->text+j->tokens[t].start+1,text,n);
}
int oel_field(const oel_json *j,int object,const char *key) {
    int i; if(object<0) object=j->root;
    if(object>=j->count || j->tokens[object].kind!='{') return -1;
    for(i=object+1;i+1<j->count && j->tokens[i].start<j->tokens[object].end;) {
        int value=i+1, next=value+1;
        if(j->tokens[i].parent!=object) return -1;
        if(equal(j,i,key)) return value;
        while(next<j->count && j->tokens[next].start<j->tokens[value].end) ++next;
        i=next;
    }
    return -1;
}
int oel_find_schema(const oel_json *j,const char *schema) {
    int i; for(i=j->root;i<j->count && j->tokens[i].start<j->tokens[j->root].end;++i)
        if(j->tokens[i].kind=='{' && equal(j,oel_field(j,i,"schema"),schema)) return i;
    return -1;
}
int oel_raw(const oel_json *j,int token,char *out,size_t cap) {
    size_t n; if(token<0) token=j->root; if(token>=j->count) return -1;
    n=j->tokens[token].end-j->tokens[token].start; if(n+1>cap) return -1;
    memcpy(out,j->text+j->tokens[token].start,n);out[n]=0;return 0;
}
int oel_vector(const oel_json *j,int object,const char *key,double *out,size_t count) {
    int t=oel_field(j,object,key),i;size_t n=0;
    if(t<0 || j->tokens[t].kind!='[') return -1;
    for(i=t+1;i<j->count && j->tokens[i].start<j->tokens[t].end;++i) {
        char *end;double v;if(j->tokens[i].parent!=t || n>=count) return -1;
        v=strtod(j->text+j->tokens[i].start,&end);
        if(end!=j->text+j->tokens[i].end || !isfinite(v)) return -1;
        out[n++]=v;
    }
    return n==count?0:-1;
}
int oel_wrench(const oel_json *j,const char *boot,const double f[3],const double t[3],char *out,size_t cap) {
    char sid[512],inv[64],clock[2048],expiry[2048],frame[512];int n,k,clock_token,ticks_token,period_token;
    unsigned long long ticks,period,increment;size_t prefix;char *finish;
    for(k=0;k<3;++k) if(!isfinite(f[k])||!isfinite(t[k])) return -1;
    if(oel_raw(j,oel_field(j,-1,"satellite_id"),sid,sizeof sid) ||
       oel_raw(j,oel_field(j,-1,"invocation_id"),inv,sizeof inv) ||
       oel_raw(j,oel_field(j,-1,"invocation_time"),clock,sizeof clock)) return -1;
    clock_token=oel_field(j,-1,"invocation_time");
    ticks_token=oel_field(j,clock_token,"ticks");period_token=oel_field(j,clock_token,"tick_period_ns");
    if(clock_token<0||ticks_token<0||period_token<0) return -1;
    errno=0;ticks=strtoull(j->text+j->tokens[ticks_token].start,&finish,10);
    if(errno||finish!=j->text+j->tokens[ticks_token].end) return -1;
    period=strtoull(j->text+j->tokens[period_token].start,&finish,10);
    if(errno||!period||finish!=j->text+j->tokens[period_token].end) return -1;
    increment=1000000000ULL/period+(1000000000ULL%period!=0);
    if(ticks>ULLONG_MAX-increment) return -1;
    prefix=j->tokens[ticks_token].start-j->tokens[clock_token].start;
    n=snprintf(expiry,sizeof expiry,"%.*s%llu%.*s",(int)prefix,clock,ticks+increment,
        (int)(j->tokens[clock_token].end-j->tokens[ticks_token].end),j->text+j->tokens[ticks_token].end);
    if(n<0||n>=(int)sizeof expiry) return -1;
    /* Frame names use the already JSON-escaped satellite identity. */
    if(strlen(sid)<2 || snprintf(frame,sizeof frame,"\"OEL/ACTUATOR/%.*s/wrench\"",(int)strlen(sid)-2,sid+1)>=(int)sizeof frame) return -1;
    n=snprintf(out,cap,
      "{\"schema\":\"oel.flight_software.output.v1\",\"contract_version\":\"1.0\",\"satellite_id\":%s,\"invocation_id\":%s,\"commands\":[{"
      "\"schema\":\"oel.flight_software.actuator_command.v1\",\"contract_version\":\"1.0\","
      "\"command_id\":{\"source_id\":\"native-fsw\",\"boot_id\":%s,\"sequence\":%s},\"satellite_id\":%s,\"actuator_id\":\"wrench\","
      "\"issued_at\":%s,\"validity\":{\"not_before\":%s,\"expires_at\":%s},"
      "\"frame\":{\"name\":%s,\"registry_version\":\"frames-v1\"},"
      "\"payload\":{\"schema\":\"ideal_wrench.v1\",\"force_n\":[%.17g,%.17g,%.17g],\"torque_n_m\":[%.17g,%.17g,%.17g]}}],"
      "\"telemetry\":[],\"requested_next_invocations\":[]}",sid,inv,boot,inv,sid,clock,clock,expiry,frame,f[0],f[1],f[2],t[0],t[1],t[2]);
    return n>=0 && (size_t)n<cap?0:-1;
}
int oel_run(oel_callback callback,void *context,int checkpointable) {
    char *line=(char*)malloc(OEL_FSW_MAX_MESSAGE+2),*out=(char*)malloc(OEL_FSW_MAX_MESSAGE+1);
    struct token *tokens=(struct token*)calloc(TOKENS,sizeof *tokens);
    int code=0,hello_done=0,booted=0;unsigned long long last_sequence=0;
#ifdef _WIN32
    _setmode(_fileno(stdin),_O_BINARY);_setmode(_fileno(stdout),_O_BINARY);
#endif
    if(!line||!out||!tokens||!callback) {code=2;goto done;}
    while(fgets(line,OEL_FSW_MAX_MESSAGE+2,stdin)) {
        oel_json j={line,tokens,0,0,0},payload,response;
        int op,seq,p,ok=0;char operation[32],sequence[64],name[32],*finish;
        unsigned long long sequence_value;size_t n=strlen(line),len;const char *terminator;
        if(!n || n>OEL_FSW_MAX_MESSAGE || line[n-1]!='\n' || parse(&j,-1,0)<0) {code=3;break;}
        space(&j);if(j.text[j.at] || j.tokens[0].kind!='{') {code=3;break;}
        if(!equal(&j,oel_field(&j,0,"protocol"),"oel.fsw.stdio.v1")) {code=3;break;}
        op=oel_field(&j,0,"operation");seq=oel_field(&j,0,"sequence");p=oel_field(&j,0,"payload");
        if(op<0||seq<0||p<0||j.tokens[op].kind!='"'||j.tokens[p].kind!='{'||
           oel_raw(&j,op,operation,sizeof operation)||oel_raw(&j,seq,sequence,sizeof sequence)) {code=3;break;}
        len=strlen(operation);if(len<2||len>31){code=3;break;}
        memcpy(name,operation+1,len-2);name[len-2]=0;
        if(sequence[0]<'1'||sequence[0]>'9') {code=3;break;}
        for(n=0;sequence[n];++n) if(!isdigit((unsigned char)sequence[n])) break;
        if(sequence[n]) {code=3;break;}
        errno=0;sequence_value=strtoull(sequence,&finish,10);
        if(errno||*finish||sequence_value<=last_sequence) {code=3;break;}
        last_sequence=sequence_value;out[0]=0;payload=j;payload.root=p;
        if(!strcmp(name,"hello")) {
            if(hello_done || !equal(&j,oel_field(&j,p,"boundary"),"oel.fsw.boundary.v1")) ok=-1;
            else {
                snprintf(out,OEL_FSW_MAX_MESSAGE,"{\"boundary\":\"oel.fsw.boundary.v1\",\"checkpointable\":%s}",checkpointable?"true":"false");
                hello_done=1;
            }
        } else if(!hello_done || (strcmp(name,"boot")&&!booted)) ok=-1;
        else if(!strcmp(name,"boot")&&booted) ok=-1;
        else if(strcmp(name,"boot")&&strcmp(name,"step")&&strcmp(name,"shutdown")&&strcmp(name,"snapshot")&&strcmp(name,"restore")) ok=-1;
        else if((!strcmp(name,"snapshot")||!strcmp(name,"restore"))&&!checkpointable) ok=-1;
        else {
            ok=callback(name,&payload,out,OEL_FSW_MAX_MESSAGE,context);
            if(!ok&&!strcmp(name,"boot")) booted=1;
        }
        if(ok) strcpy(out,"{}");
        terminator=(const char*)memchr(out,0,OEL_FSW_MAX_MESSAGE);
        if(!terminator || (size_t)(terminator-out)+strlen(operation)+strlen(sequence)+100>OEL_FSW_MAX_MESSAGE) {code=4;break;}
        /* Callbacks may return raw JSON, but malformed/unterminated payloads
           must fail inside the SDK without an unbounded string traversal. */
        response.text=out;response.tokens=tokens;response.count=0;response.root=0;response.at=0;
        if(parse(&response,-1,0)<0) {code=4;break;}
        space(&response);if(response.text[response.at]||response.tokens[0].kind!='{') {code=4;break;}
        printf("{\"protocol\":\"oel.fsw.stdio.v1\",\"sequence\":%s,\"operation\":%s,\"ok\":%s,\"payload\":%s}\n",sequence,operation,ok?"false":"true",out);
        if(fflush(stdout)) {code=5;break;}
        if(!strcmp(name,"shutdown")) break;
    }
 done: free(tokens);free(out);free(line);return code;
}
