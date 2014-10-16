#TYPE=1: L1, TYPE=2 : L2, othewise: L_{1/2}
TYPE=1
#Number of CPUs to be used
CPUS_AVAIL=4

COMMON=-DPENALTY=${TYPE} -DNO_CPUS=${CPUS_AVAIL}
# Non AVX/AVX2 options
OPTS=${COMMON} -mssse3 -DV_LN="(4)" -ffast-math
# AVX/AVX2 options
#OPTS=${COMMON} -mavx -mavx2 -DV_LN="(8)" -mfma -ffast-math

a.out: bilinearOpt.cpp base.h
	g++ -O3 -g ${OPTS} bilinearOpt.cpp -lm

bilinearOpt.s: bilinearOpt.cpp base.h
	g++ -O3 -S ${OPTS} bilinearOpt.cpp -lm
