#ifndef __BASE_H__
#define __BASE_H__

#include <stdlib.h>
#include <unistd.h>

#define V_SZ (4*V_LN)

typedef float FV __attribute__((vector_size(V_SZ)));
typedef int FI __attribute__((vector_size(V_SZ)));
typedef int BV __attribute__((vector_size(V_SZ)));

struct BagOfBits {
    union{
        FV fv;
        FI fi;
        float a[V_LN];
    };
};

#if V_LN==8 /*kludgy way of detecting avx with its fma instruction*/
#define FMA(a,b,c) (a)*(b)+(c)
#define FV_SET1(x) {(x),(x),(x),(x),(x),(x),(x),(x)}
#else
#define FMA(a,b,c) (a)*(b)+(c)
#define FV_SET1(x) {(x),(x),(x),(x)}
#endif

#define FV_ZERO() FV_SET1(0.0f)


#define FORCE_READ_INT(p,o) (*((volatile int32_t*)((p)+o)))

#define FORCE_READ(p,o) (*((volatile FV*)((p)+o)))
#define FORCE_WRITE(p,i,x) (*(volatile FV*)((p)+i))=x

template<typename T>
inline
T abortIf(T x,T val)
{
    if(x==val) {
        int w=write(2,"abortIf failure\n",16);
        abort();
    }
    return x;
}

template<typename T>
inline
void abortIfNot(T x,T val)
{
    if(x!=val) {
        int w=write(2,"abortIfNot failure\n",19);
        abort();
    }
}


//TODO: replace this with better version using machine instructions
inline
float horizontalSum(FV v)
{
    BagOfBits b;
    b.fv=v;
    if(V_LN==8){
        return b.a[0]+b.a[1]+b.a[2]+b.a[3]+b.a[4]+b.a[5]+b.a[6]+b.a[7];
    }else{
        return b.a[0]+b.a[1]+b.a[2]+b.a[3];
    }
}

inline
float horizontalMax(FV v)
{
    BagOfBits b;
    b.fv=v;
    float mx=b.a[0];
    int i;
    for(i=1;i<V_LN;++i){
        if(b.a[i]>mx){
            mx=b.a[i];
        }
    }
    return mx;
}

#define SM2(x,y) ((x)+(y))
#define SM4(x,y,z,w) SM2(SM2(x,y),SM2(z,w))
#define SM8(a,b,c,d,e,f,g,h) SM2(SM4(a,b,c,d),SM4(e,f,g,h))

#define SMF2(x,y) (FORCE_READ(x,i))+(FORCE_READ(y,i))
#define SMF4(x,y,z,w) (SMF2(x,y))+(SMF2(z,w))
#define SMF8(a,b,c,d,e,f,g,h) (SMF4(a,b,c,d))+(SMF4(e,f,g,h))

#endif /*__BASE_H__*/
