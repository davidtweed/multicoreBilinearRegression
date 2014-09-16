OPTS=-DNO_CPUS=4 -mssse3 -DV_LN="(4)"
#OPTS=-DNO_CPUS=4 -mavx -DV_LN="(8)"

it: bilinearOpt.cpp
	g++ -O3 ${OPTS} bilinearOpt.cpp -lm
