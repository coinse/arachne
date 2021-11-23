#!/bin/bash


for i in {0..29}
do
	# loc
	#./rq2.sh data/cm cifar10 localiser loc results/rq2/on_test $i > logs/rq2/cm/on_test/$i.out &
	#wait $!
	./rq2.sh data/cm cifar10 gradient_loss gl results/rq2/on_test $i > logs/rq2/cm/on_test/$i.gl.out &
	wait $!
	./rq2.sh data/cm cifar10 random rd results/rq2/on_test $i > logs/rq2/cm/on_test/$i.rd.out &
	wait $!
done
