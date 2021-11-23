#!/bin/bash

for i in {0..29}
do
	#./rq2.sh data/gtsrb/prepared/ GTSRB localiser loc results/rq2/on_test $i > logs/rq2/gtsrb/on_test/$i.out &
	#wait $!
	./rq2.sh data/gtsrb/prepared/ GTSRB gradient_loss gl results/rq2/on_test $i > logs/rq2/gtsrb/on_test/$i.gl.out &
	wait $!
	./rq2.sh data/gtsrb/prepared/ GTSRB random rd results/rq2/on_test $i > logs/rq2/gtsrb/on_test/$i.rd.out &
	wait $!
done
