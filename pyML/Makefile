.PHONY: dist-clean clean

default: ./lib/nn.so

./obj/activation.o: ./include/nn.h ./src/activation.c
	gcc -c -fPIC -O3 -std=c11 ./src/activation.c -o ./obj/activation.o

./obj/adam_optimizer.o: ./include/nn.h ./src/adam_optimizer.c
	gcc -c -fPIC -O3 -std=c11 ./src/adam_optimizer.c -o ./obj/adam_optimizer.o

./obj/back_prop.o: ./include/nn.h ./src/back_prop.c
	gcc -c -fPIC -O3 -std=c11 ./src/back_prop.c -o ./obj/back_prop.o

./obj/epoch.o: ./include/nn.h ./src/epoch.c
	gcc -c -fPIC -O3 -std=c11 ./src/epoch.c -o ./obj/epoch.o

./obj/forw_prop.o: ./include/nn.h ./src/forw_prop.c
	gcc -c -fPIC -O3 -std=c11 ./src/forw_prop.c -o ./obj/forw_prop.o

./obj/eval_batch_sens.o: ./include/nn.h ./src/eval_batch_sens.c
	gcc -c -fPIC -O3 -std=c11 ./src/eval_batch_sens.c -o ./obj/eval_batch_sens.o

./obj/memory_management.o: ./include/nn.h ./src/memory_management.c
	gcc -c -fPIC -O3 -std=c11 ./src/memory_management.c -o ./obj/memory_management.o

./obj/predict.o: ./include/nn.h ./src/predict.c
	gcc -c -fPIC -O3 -std=c11 ./src/predict.c -o ./obj/predict.o

./obj/print.o: ./include/nn.h ./src/print.c
	gcc -c -fPIC -O3 -std=c11 ./src/print.c -o ./obj/print.o

./lib/nn.so: ./include/nn.h ./obj/activation.o ./obj/adam_optimizer.o ./obj/back_prop.o ./obj/epoch.o ./obj/eval_batch_sens.o ./obj/forw_prop.o ./obj/memory_management.o ./obj/predict.o ./obj/print.o
	mkdir -p include
	gcc --shared -fPIC -O3 -std=c11 ./obj/activation.o ./obj/adam_optimizer.o ./obj/back_prop.o ./obj/epoch.o ./obj/eval_batch_sens.o ./obj/forw_prop.o ./obj/memory_management.o ./obj/predict.o ./obj/print.o -o ./lib/nn.so

clean:
	rm ./obj/*

dist-clean:
	rm ./lib/*
