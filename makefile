#OPTS=-DNO_CPUS=4 -mssse3 -DV_LN="(4)"
OPTS=-DNO_CPUS=4 -mavx -DV_LN="(8)" -mfma -ffast-math

a.out: bilinearOpt.cpp
	g++ -O3 -g ${OPTS} bilinearOpt.cpp -lm

bilinearOpt.s: bilinearOpt.cpp
	g++ -O3 -S ${OPTS} bilinearOpt.cpp -lm
