#include "base.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
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

const float CONVERGENCE_THRESH=1e-6;

#if L1_PENALTY
#define OUTSIDE_THRESH_UPDATE updateL1
#else
#define OUTSIDE_THRESH_UPDATE solveDepressedCubic2
#endif

#define HOUSEHOLDER_ITER (10)

inline int read32BitToFltVec(FV *r0,FV *r1, FV* r2,FV *r3,BagOfBits *p)
{
    *r0=p[0].fv;
    *r1=p[1].fv;
    *r2=p[2].fv;
    *r3=p[3].fv;
    return 4;
}

//================ Data structures =====================

struct ControlData {
    float initLambda,lambdaStep;
};

struct Memory {
    int thisLambda;
    FV** original;
    //publishedPredictions[NO_CPUS] is set to the sum of the predictions minus the goal
    FV* publishedPredictions[NO_CPUS+1];
    FV* parameterEstimates[NO_CPUS];
    BagOfBits** perThread1;
    BagOfBits*** perThread2;
    BagOfBits*** perThread3;
    int32_t* sharedLambda;
    int ourIdx;
    int noParams;//number of parameters this thread is handling
    int noNonzeroParams;
    int* nonzeroParams;//local to the core working on this*/
    bool signaledConvergence;/*has this thread converged for this value of lambda?*/
    unsigned int *localToGlobalParamID;
    int file;
};

//================== inter-CPU data reading/writing ===============

/*Publish-sum-of-published individual arrays.
 *We want these routines to be as fast as possible
 *to minimise the amount of skew in the data the
 *other threads read. Also return the total squared prediction error.
 */
float p2psum2(FV* output,FV **in,FV* target,int noBlks)
{
    FV err=FV_ZERO();
    int i=0;
    do{
        FV r=SMF2(in[0],in[1]);
        r=r-target[i];
        FORCE_WRITE(output,i,r);
        err=FMA(r,r,err);
    }while(++i<noBlks);
    return horizontalSum(err);
}

float p2psum4(FV* output,FV **in,FV* target,int noBlks)
{
    FV err=FV_ZERO();
    int i=0;
    do{
        FV r=SMF4(in[0],in[1],in[2],in[3]);
        r=r-target[i];
        FORCE_WRITE(output,i,r);
        err=FMA(r,r,err);
    }while(++i<noBlks);
    return horizontalSum(err);
}

float p2psum8(FV* output,FV **in,FV* target,int noBlks)
{
    FV err=FV_ZERO();
    int i=0;
    do{
        FV r=SMF8(in[0],in[1],in[2],in[3],in[4],in[5],in[6],in[7]);
        r=r-target[i];
        FORCE_WRITE(output,i,r);
        err=FMA(r,r,err);
    }while(++i<noBlks);
    return horizontalSum(err);
}

float p2psum16(FV* output,FV **in,FV* target,int noBlks)
{
    FV err=FV_ZERO();
    int i=0;
    do{
        FV r=SMF8(in[0],in[1],in[2],in[3],in[4],in[5],in[6],in[7]);
        r=r+SMF8(in[8],in[9],in[10],in[11],in[12],in[13],in[14],in[15]);
        r=r-target[i];
        FORCE_WRITE(output,i,r);
        err=FMA(r,r,err);
    }while(++i<noBlks);
    return horizontalSum(err);
}

/*
 *Add in to results the contributions from the otherUnits and multiply resulting value by multipliers.
 *otherUnits are read using FORCE_READ to get latest values wrtten by other CPUs.
 *results has already been initialized with a partal contribution.
 *Note that while some "partial update skew" is inevitable, we want to do this as fast as possible
 *to minimise the amount.
 *Doing 5 accumulations at once is the most possible before hitting register spills.
 */
void accSumOtherContributions(FV* results,FV* corrections,FV* residual,FV* multipliers,int noOtherUnits,int noBlks)
{
    FV bigAcc0=FV_ZERO(),bigAcc1=FV_ZERO(),bigAcc2=FV_ZERO(),bigAcc3=FV_ZERO(),bigAcc4=FV_ZERO(),bigAcc5=FV_ZERO();
    int i=0;
    do{
        FV acc=FORCE_READ(residual,i);
        bigAcc0=FMA(acc,multipliers[0],bigAcc0);
        bigAcc1=FMA(acc,multipliers[1],bigAcc1);
        bigAcc2=FMA(acc,multipliers[2],bigAcc2);
        bigAcc3=FMA(acc,multipliers[3],bigAcc3);
        bigAcc4=FMA(acc,multipliers[4],bigAcc4);
        bigAcc5=FMA(acc,multipliers[5],bigAcc5);
        multipliers+=6;
    }while(++i<noBlks);
    results[0]=bigAcc0-corrections[0];
    results[1]=bigAcc1-corrections[1];
    results[2]=bigAcc2-corrections[2];
    results[3]=bigAcc3-corrections[3];
    results[4]=bigAcc4-corrections[4];
    results[5]=bigAcc5-corrections[5];
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

//=================== general computations ===================
//Compute for fixed I's the  sum_e a_I^(e)^2
void getCoeffsSquared(float* results,FV** data,int *entries,int noEntries,int noBlks)
{
    int i=0;
    do{
        FV a0=FV_ZERO(),a1=FV_ZERO(),a2=FV_ZERO(),a3=FV_ZERO();
        int j=0;
        do{
            FV v0=data[i][j],v1=data[i][j+1],v2=data[i][j+2],v3=data[i][j+3];
            a0=FMA(v0,v0,a0);
            a1=FMA(v1,v1,a1);
            a2=FMA(v2,v2,a2);
            a3=FMA(v3,v3,a3);
            j+=4;
        }while(j<noBlks);
        results[i]=horizontalSum(a0+a1+a2+a3);
    }while(++i<noEntries);
}

//Compute for each e sum_i a_i^(e) x_i , taking advantage of knowing which x_i are zero
//to avoid unnecessary work.
//If results is full of zeros and we use full params we get the full prediction.
//If results is the old values and params is the change in parameter values then we get the
//updated prediction.
void getPrediction(FV* results,BagOfBits** data,float *params,int *entries,int noEntries,int noBlks)
{
    int j=0;
    do{
        FV acc0=FV_ZERO(),acc1=FV_ZERO(),acc2=FV_ZERO(),acc3=FV_ZERO(),acc4=FV_ZERO();
        int iIdx=0;
        do{
            int i=entries[iIdx];
            FV p=FV_SET1(params[i]);
            FV d0,d1,d2,d3;
            int adv=read32BitToFltVec(&d0,&d1,&d2,&d3,&(data[i][j]));
            acc0=FMA(d0,p,acc0);
            acc1=FMA(d1,p,acc1);
            acc2=FMA(d2,p,acc2);
            acc3=FMA(d3,p,acc3);
        }while(++iIdx<noEntries);
        FORCE_WRITE(results,j,results[j]+acc0);
        FORCE_WRITE(results,j+1,results[j+1]+acc1);
        FORCE_WRITE(results,j+2,results[j+2]+acc2);
        FORCE_WRITE(results,j+3,results[j+3]+acc3);
        j=j+4;
    }while(j<noBlks);
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



float formAndSolveUpdate(FV* result,FV *others,FV *corrections,FV *here,FV *params,float lambda,int noBlks,int* removers,int &noRemovals,int householderIter)
{
    float thresh=lambda;
    FV changes=FV_ZERO();
    do{
        FV t0=*result-*corrections;//we've kept some corrections here
        FV t1=*here/t0;
        FV t2=*params/t0;
        BV toKeep=(t2>=thresh) | (t2<=-thresh);//record which should be moved to zero
        //get the update which is correct outside the threshold region
        OUTSIDE_THRESH_UPDATE(result,t1,t2,*params,thresh,householderIter);
        FV zeroes=FV_ZERO();
        *result=toKeep ? (*result) : zeroes;
        FV diff=*result-*params;
        diff=diff*diff;
        changes=changes>diff?changes:diff;
        //copy over into removers list
    }while(--noBlks>0);
    return horizontalMax(changes);
}

void f(ControlData* control,Memory *mem)
{
    int sharedLambda=(FORCE_READ_INT(mem->sharedLambda,0))/NO_CPUS;
    if(sharedLambda!=mem->thisLambda){
        mem->thisLambda=sharedLambda;
        mem->signaledConvergence=false;
    }
    float lambda=mem->thisLambda*control->lambdaStep;
    const int noBlks=5;
    int noRemovals;
    int removers[5*V_LN];
    FV* result;
    FV *others;
    FV *corrections;
    FV *here;
    FV *params;
    int householderIter;
    float changeMagnitude=formAndSolveUpdate(result,others,corrections,here,params,lambda,noBlks,removers,noRemovals,householderIter);
    //put the results back into the correct place
    float *arr=reinterpret_cast<float*>(result);
    //now update the weights here
    int i;
    //remove new zeroes and compact the list of non-zeroParms------------------------------
    int readPt=0,copyPt;
    for(int removal=0;removal<noRemovals;++removal){
        while(mem->nonzeroParams[readPt]<removers[removal]){
            mem->nonzeroParams[copyPt++]=mem->nonzeroParams[readPt++];
        }
        assert(removers[removal]==mem->nonzeroParams[readPt]);
        ++readPt;
    }
    while(readPt<mem->noNonzeroParams){
        mem->nonzeroParams[copyPt++]=mem->nonzeroParams[readPt++];
    }
    mem->noNonzeroParams=copyPt;
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
    mem->original=add2DSuperstructure<FV>(allocSharedArray(0),r,c);
    for(int i=0;i<NO_CPUS;++i){
        mem->publishedPredictions[i]=reinterpret_cast<FV*>(allocSharedArray(0));
        mem->parameterEstimates[i]=reinterpret_cast<FV*>(allocSharedArray(0));
    }
    mem->perThread1=0;
    mem->perThread2=0;
    mem->perThread3=0;
    mem->sharedLambda=reinterpret_cast<int32_t*>(allocSharedArray(0));
    *mem->sharedLambda=0;
    mem->ourIdx=-1;
}

void figureLinearRegressionProblemDivision() {

}

//============== running processes on CPUs ================

void exportSolution(ControlData *control,Memory *mem)
{
    char buffer[65];
    int len=sprintf(buffer,"%g :: %g\n",mem->thisLambda*control->lambdaStep,0.0f);
    int written=write(mem->file,buffer,len);
    assert(written==len);
    for(int i=0;i<0;++i){
        sprintf(buffer,"%u %g\n",mem->localToGlobalParamID[i],(double)0.0f);
        written=write(mem->file,buffer,len);
        assert(written==len);
    }
    written=write(mem->file,"\n",1);
    assert(written==len);
}

void threadStep(ControlData *control,Memory *mem)
{
    //Run one iteration
    float maxChange;//=f();
    //Indicate we're happy for lambda to increase.
    if(!mem->signaledConvergence && maxChange<=CONVERGENCE_THRESH){
        __sync_fetch_and_add(mem->sharedLambda,1);
        mem->signaledConvergence=true;
        //write out optimum before we increase lambda
        exportSolution(control,mem);
    }
    //If we're the housekeeper thread update the predictions
    if(mem->ourIdx==0){
        FV* output;
        FV **in;
        FV* target;
        int noBlks;
        if(NO_CPUS==2){
            p2psum2(output,in,target,noBlks);
        }else if(NO_CPUS==4){
            p2psum4(output,in,target,noBlks);
        }else if(NO_CPUS==8){
            p2psum8(output,in,target,noBlks);
        }else if(NO_CPUS==16){
            p2psum16(output,in,target,noBlks);
        }else{
            abort();
        }
    }
}

void runAThread(ControlData *control,Memory *mem)
{
    mem->signaledConvergence=false;
    while(true){
        threadStep(control,mem);
    }
}
void
fireUpForks(ControlData *control,Memory *mem)
{
    pid_t pids[NO_CPUS];
    pid_t parentPID=getpid();
    for(int i=0;i<NO_CPUS;++i){
        pid_t thisPID;
        if((thisPID=getpid())!=0){//we're the child
            {
                char buffer[64];
                sprintf(buffer,"output_%u",i);
                mem->file=open("", O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR);
            }
            mem->ourIdx=i;
            cpu_set_t *cpu_mask=CPU_ALLOC(NO_CPUS);
            CPU_ZERO(cpu_mask);
            CPU_SET(i,cpu_mask);
            int err=sched_setaffinity(0,sizeof(cpu_set_t),cpu_mask);
            CPU_FREE(cpu_mask);
            runAThread(control,mem);
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
