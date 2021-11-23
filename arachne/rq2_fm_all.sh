#!/bin/bash


for i in {0..29}
do
	# loc
	#./rq2.sh data/fm fashion_mnist localiser loc results/rq2/on_test $i > logs/rq2/fm/on_test/$i.out &
	#wait $!
	./rq2.sh data/fm fashion_mnist gradient_loss gl results/rq2/on_test $i > logs/rq2/fm/on_test/$i.gl.out &
	wait $!
	./rq2.sh data/fm fashion_mnist random rd results/rq2/on_test $i > logs/rq2/fm/on_test/$i.rd.out &
	wait $!
done
