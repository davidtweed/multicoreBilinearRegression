#include <stdio.h>
#include <math.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/mman.h>
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>

#include <algorithm>
#include <assert.h>

//=============== Macro abstractions =================

#define SM2(x,y) ((x)+(y))
#define SM4(x,y,z,w) SM2(SM2(x,y),SM2(z,w))
#define SM8(a,b,c,d,e,f,g,h) SM2(SM4(a,b,c,d),SM4(e,f,g,h))

#if L1_PENALTY
#define OUTSIDE_THRESH_UPDATE updateL1
#else
#define OUTSIDE_THRESH_UPDATE solveDepressedCubic2
#endif

#define HOUSEHOLDER_ITER (10)
#if 4==V_LN
#define FV_SET1(x) {(x),(x),(x),(x)}
#else
#define FV_SET1(x) {(x),(x),(x),(x),(x),(x),(x),(x)}
#endif
#define FV_ZERO() FV_SET1(0.0f)

#define V_SZ (4*V_LN)
typedef float FV __attribute__((vector_size(V_SZ)));
typedef int BV __attribute__((vector_size(V_SZ)));

#if V_LN==8 /*kludgy way of detecting avx with its fma instruction*/
#define fma(a,b,c) (a)*(b)+(c)
#else
#define fma(a,b,c) (a)*(b)+(c)
#endif

#define FORCE_READ(p,o) (*((volatile FV*)(p+o)))
#define FORCE_WRITE(p,i,x) (*(volatile FV*)(p+i))=x

//================ Data structures =====================

struct ControlData {
    float initLambda,lamdaStep,lambdaLIm;
};

struct Memory {
    float *sharedParams;
    FV** original;
    FV* publishedPredictions[NO_CPUS];
    FV* parameterEstimates[NO_CPUS];
    FV*** perThread1;
    FV*** perThread2;
    FV*** perThread3;
    int32_t* syncFutexes;
    int ourIdx;
};

//================== inter-CPU data reading/writing ===============

/*
 *Add in to results the contributions from the otherUnits and multiply resulting value by multipliers.
 *otherUnits are read using FORCE_READ to get latest values wrtten by other CPUs.
 *results has already been initialized with a partal contribution.
 *Note that while some "partial update skew" is inevitable, we want to do this as fast as possible
 *to minimise the amount.
 *Doing 5 accumulations at once is the most possible before hitting register spills.
 */
void accSumOtherContributions(FV* results,FV* ourValues,FV** otherUnits,FV* multipliers,int noOtherUnits,int noBlks)
{
    assert(noOtherUnits==NO_CPUS);
    FV bigAcc0=FV_ZERO(),bigAcc1=FV_ZERO(),bigAcc2=FV_ZERO(),bigAcc3=FV_ZERO(),bigAcc4=FV_ZERO();
    int i=0;
    do{
        FV acc=ourValues[i];
        int unit=0;
        do{
            acc=acc+FORCE_READ(otherUnits[unit],i);
        }while (++unit<NO_CPUS-1);
        bigAcc0=fma(acc,multipliers[0],bigAcc0);
        bigAcc1=fma(acc,multipliers[1],bigAcc1);
        bigAcc2=fma(acc,multipliers[2],bigAcc2);
        bigAcc3=fma(acc,multipliers[3],bigAcc3);
        bigAcc4=fma(acc,multipliers[4],bigAcc4);
        multipliers+=6;

    }while(++i<noBlks);
    results[0]=bigAcc0;
    results[1]=bigAcc1;
    results[2]=bigAcc2;
    results[3]=bigAcc3;
    results[4]=bigAcc4;
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
    }while(++i<noBlks);
}

//=========== Preparing "this CPU" problem components ============

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

//==================== update functions ======================

/*solve a vector of depressed cubics starting from a previous soln of x*/
//inline
void solveDepressedCubic(FV *result, FV p,FV q,FV x,float lambda,int householderIter)
{
    do{
        FV t1=3*x;
        FV t2=t1*x+p;
        FV t3=(x*x+p)*x+q;
        FV num=t3*t2;
        FV denom=t2*t2-t1*t3;
        x=x-num/denom;
/*        if(fabs((num/denom)/x)<=1e-6) break;*/
    }while(--householderIter>0);
    *result=x;
}

//inline
void solveDepressedCubic2(FV *result,FV p,FV q,FV x,float ambda,int householderIter)
{
    int l=0;
    do{
        FV t1=3*x;
        FV fderiv=t1*x+p;
        FV fderivSq=fderiv*fderiv;
        FV f=(x*x+p)*x+q;
        FV hlfffdderiv=f*t1;
        FV num=f*(fderivSq-hlfffdderiv);
        FV fSq=f*f;
        FV denom=fderiv*(fderivSq-2*hlfffdderiv)+fSq;
        x=x-num/denom;
    }while(++l<householderIter);
    *result=x;
}

//inline
void updateL1(FV *result,FV p,FV q,FV x,float lambda,int householderIter)
{
    BV gtLambda=x>=lambda;
    *result=gtLambda ? (x-lambda) : (x+lambda);
}


void formAndSolveUpdate(FV* result,FV *others,FV *corrections,FV *here,FV *params,float lambda,int noBlks,int householderIter)
{
    float thresh=lambda;
    do{
        FV t0=*result-*corrections;//we've kept some corrections here
        FV t1=*here/t0;
        FV t2=*params/t0;
        BV toKeep=(t2>=thresh) | (t2<=-thresh);//record which should be moved to zero
        //get the update which is correct outside the threshold region
        OUTSIDE_THRESH_UPDATE(result,t1,t2,*params,thresh,householderIter);
        FV zeroes=FV_ZERO();
        *result=toKeep ? (*result) : zeroes;
    }while(--noBlks>0);
}

//================= setting up data structures ===================

void* allocSharedArray(int no32BitWords)
{
    void *p=mmap(0,4*no32BitWords,PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS,0,0);
    return p;
}

template<class T>
T** add2DSuperstructure(void* raw,int noRows,int noCols)
{
    T* p=reinterpret_cast<T*>(raw);
    T** super=new T*[noRows];
    for(int i=0;i<noRows;++i){
        super[i]=&(p[i*noCols]);
    }
    return super;
}

template<class T>
T*** add3DSuperstructure(void* raw,int noRows,int noCols)
{
    T* p=reinterpret_cast<T*>(raw);
    T** super=new T*[noRows];
    for(int i=0;i<noRows;++i){
        super[i]=&(p[i*noCols]);
    }
    return &super;
}

void setupSharedMemoryAreas(Memory *mem)
{
    int r,c;
    mem->sharedParams=reinterpret_cast<float*>(allocSharedArray(1));
    mem->original=add2DSuperstructure<FV>(allocSharedArray(0),r,c);
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

void figureLinearRegressionProblemDivision() {

}

//============== running processes on CPUs ================

void exportSolution(ControlData *control,Memory *mem)
{
    for(int i=0;i<0;++i){
        fprintf(stdout,"%u %g\n",i,(double)0.0f);
    }
    fflush(stdout);

}

void runAThread(Memory *mem)
{

}

void
fireUpForks(ControlData *control,Memory *mem)
{
    pid_t pids[NO_CPUS];
    pid_t parentPID=getpid();
    for(int i=0;i<NO_CPUS;++i){
        pid_t thisPID;
        if((thisPID=getpid())!=0){//we're the child
            //it's easier if our publication array is final pointer in array
            FV* t=mem->publishedPredictions[i];
            mem->publishedPredictions[i]=mem->publishedPredictions[NO_CPUS-1];
            mem->publishedPredictions[NO_CPUS-1]=t;
            mem->ourIdx=NO_CPUS-1;
            cpu_set_t *cpu_mask=CPU_ALLOC(NO_CPUS);
            CPU_ZERO(cpu_mask);
            CPU_SET(i,cpu_mask);
            int err=sched_setaffinity(0,sizeof(cpu_set_t),cpu_mask);
            CPU_FREE(cpu_mask);
            runAThread(mem);
            _exit(0);
        }
    }
    exportSolution(control,mem);
}


int main(int argc,char* argv[])
{
    ControlData ctrl;
    Memory mem;
    fireUpForks(&ctrl,&mem);

    return 0;
}
