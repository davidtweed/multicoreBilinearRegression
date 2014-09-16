#include <stdio.h>
#include <math.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/mman.h>
#ifndef _GNU_SOURCE
#define _GNU_SOURCE             /* See feature_test_macros(7) */
#endif
#include <sched.h>

#include <algorithm>

#define SM2(x,y) ((x)+(y))
#define SM4(x,y,z,w) SM2(SM2(x,y),SM2(z,w))
#define SM8(a,b,c,d,e,f,g,h) SM2(SM4(a,b,c,d),SM4(e,f,g,h))

#define HOUSEHOLDER_ITER (10)
#if 4==V_LN
#define FV_ZERO() {0.0f,0.0f,0.0f,0.0f}
#else
#define FV_ZERO() {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}
#endif

#define V_SZ (4*V_LN)
typedef float FV __attribute__((vector_size(V_SZ)));

#define fma(a,b,c) (a)*(b)+(c)

#define FORCE_READ(p,o) (*((volatile FV*)(p+o)))
#define FORCE_WRITE(p,i,x) (*(volatile FV*)(p+i))=x

struct Memory {
    FV* original;
    FV* publishedPredictions[NO_CPUS];
    FV* parameterEstimates[NO_CPUS];
    FV* perThread1;
    FV* perThread2;
    FV* perThread3;
    int32_t* syncFutexes;
    int ourIdx;
};

/*
 *Add in to results the contributions from the otherUnits and multiply resulting value by multipliers.
 *otherUnits are read using FORCE_READ to get latest values wrtten by other CPUs.
 *results has already been initialized with a partal contribution.
 *Do the accumulation in 2 levels in order to reduce a bit the prec
 */
void accSumOtherContributions(FV* results,FV* ourValues,FV** otherUnits,FV *correction,FV* multipliers,int noOtherUnits,int noBlks)
{
    FV bigAcc0=FV_ZERO(),bigAcc1=FV_ZERO(),bigAcc2=FV_ZERO(),bigAcc3=FV_ZERO();
    int i=0;
    do{
        FV acc=ourValues[i];
        int unit=0;
        do{
            acc=acc+FORCE_READ(otherUnits[unit],i);
        }while (++unit<noOtherUnits);
        bigAcc0=fma((acc-correction[0]),multipliers[0],bigAcc0);
        bigAcc1=fma((acc-correction[1]),multipliers[1],bigAcc1);
        bigAcc2=fma((acc-correction[2]),multipliers[2],bigAcc2);
        bigAcc3=fma((acc-correction[3]),multipliers[3],bigAcc3);
        correction+=4;
        multipliers+=4;

    }while(++i<noBlks);
    results[0]=bigAcc0;
    results[1]=bigAcc1;
    results[2]=bigAcc2;
    results[3]=bigAcc3;
}

/*
 *Write pre-prepared values into the published array for this unit.
 *Use FORCE_WRITE to ensure results are written immediately, and use
 *a loop to minimise time values from different passes exist in the cache.
 */
void writeContributionForThisUnit(FV* publish,FV *values,int noBlks)
{
    int i=0;
    do{
        FORCE_WRITE(publish,i,values[i]);
        //++publish;
    }while(++i<noBlks);
}

/*
 *
 */
void updateContributions(FV* values,int noBlks)
{

}

/*
 *
 */
void prepareMultipliers(FV* multipliers,int noBlks)
{

}

/*solve a vector of depressed cubics*/
inline
void solveDepressedCubic(FV *result, FV p,FV q,FV x)
{
    int l;
    for(l=0;l<HOUSEHOLDER_ITER;++l){
        FV t1=3*x;
        FV t2=t1*x+p;
        FV t3=(x*x+p)*x+q;
        FV num=t3*t2;
        FV denom=t2*t2-t1*t3;
        x=x-num/denom;
/*        if(fabs((num/denom)/x)<=1e-6) break;*/
    }
    *result=x;
}


inline
void cubicsolve2(FV *result,FV p,FV q,FV x)
{
    int l;
    for(l=0;l<HOUSEHOLDER_ITER;++l){
        FV t1=3*x;
        FV fderiv=t1*x+p;
        FV fderivSq=fderiv*fderiv;
        FV f=(x*x+p)*x+q;
        FV hlfffdderiv=f*t1;
        FV num=f*(fderivSq-hlfffdderiv);
        FV fSq=f*f;
        FV denom=fderiv*(fderivSq-2*hlfffdderiv)+fSq;
        x=x-num/denom;
    }
    *result=x;
}

void* allocSharedArray(int no32BitWords)
{
    void *p=mmap(0,4*no32BitWords,PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS,0,0);
    return p;
}

void setupSharedMemoryAreas(Memory *mem)
{
    mem->original=0;
    for(int i=0;i<NO_CPUS;++i){
        mem->publishedPredictions[i]=reinterpret_cast<FV*>(allocSharedArray(0));
        mem->parameterEstimates[i]=reinterpret_cast<FV*>(allocSharedArray(0));
    }
    mem->perThread1=0;
    mem->perThread2=0;
    mem->perThread3=0;
    mem->syncFutexes=reinterpret_cast<int*>(allocSharedArray(0));
    mem->syncFutexes[0]=NO_CPUS;
    mem->ourIdx=-1;
}

void runAThread(Memory *mem)
{

}

void
fireUpForks(Memory *mem)
{
    pid_t pids[NO_CPUS];
    pid_t parentPID=getpid();
    for(int i=0;i<NO_CPUS;++i){
        pid_t thisPID;
        if((thisPID=getpid())!=0){//we're the child
            mem->ourIdx=i;
            int err=sched_setaffinity(thisPID,NO_CPUS,0);
            runAThread(mem);
            _exit(0);
        }
    }

}


int main(int argc,char* argv[])
{
    Memory mem;
    fireUpForks(&mem);

    return 0;
}
