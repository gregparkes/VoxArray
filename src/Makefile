CC=g++
DEPS=types.h Vector.h Matrix.h BoolVector.h VarStructs.h numpy.h
OBJ1=numstatic.o Vector.o Matrix.o BoolVector.o
TEST_OBJ=test_matrix.o test_numpy1d.o main.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< -I.

test_numpy: $(OBJ1) $(TEST_OBJ)
	$(CC) -o test_numpy $^ -I.

clean: $(OBJ1)
	rm $(OBJ1) $(TEST_OBJ) test_numpy


