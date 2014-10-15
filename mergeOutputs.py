#!/usr/bin/env python

#An INCREDIBLY basic, hackish script to take a collection of files containing dicts output by
#the multilinearBilinearRegression program which each describe part of the solutions
# and print out a table which has
# lambda value
# total error
# number of non-zero coefficients
# and then the rest of the coefficients in numerical order, all separated by spaces.
# It should be possible to use the file in gnuplot or similar programs to produce some graphs.

import ast

def concatenate(*lists):
    return list(itertools.chain.from_iterable(lists))

if __name__=="__main__":
    noParams=int(sys.arg[1])
    dicts=[ast.literal_eval(open(n,"r")) for n in sys.argv[2:]]
    lamdas=list(set.union(*[set(d.keys()) for d in dicts]))
    lambda.sort()
    for l in lambdas:
        rowParts=[d[l] for d in dicts]
        err=max([v for (v,d) in rowParts])#only one process writes out the non-negative total error
        params=concatenate[list(d) for (_,d) in rowParts])
        params.sort()
        paramsStr=" ".join([v for (p,v) in params])
        print(rowParts,err,paramsStr)
