#!/bin/bash

for i in {0..29}
do
	./rq1.sh data/cm/ cifar10 new_loc/ localiser 1 $i > logs/new_loc/rq1.cifar10.$i.out &
	wait $!
done

#for i in {0..29}
#do
#	./rq1.sh data/fm/ fashion_mnist new_loc/ localiser 1 $i > logs/new_loc/rq1.fm.$i.out &
#	wait $!
#done

#for i in {0..29}
#do
#	./rq1.sh data/GTSRB/ GTSRB new_loc/ localiser 1 $i > logs/new_loc/rq1.gtsrb.$i.out &
#	wait $!
#done
