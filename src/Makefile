CC=g++
DEPS=types.h Vector.h Matrix.h BoolVector.h VarStructs.h numpy.h 
OBJ1=numstatic.o Vector.o Matrix.o BoolVector.o numpy.o main.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< -I.

test_numpy: $(OBJ1)
	$(CC) -o test_numpy $^ -I.

clean: $(OBJ1)
	rm $(OBJ1) test_numpy


