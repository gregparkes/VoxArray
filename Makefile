CC=g++
DEPS=src/types.h src/Vector.h src/Matrix.h src/BoolVector.h src/VarStructs.h src/numpy.h
OBJ1=src/numstatic.o src/Vector.o src/Matrix.o src/BoolVector.o
TEST_OBJ=src/test_matrix.o src/test_numpy1d.o src/main.o
OUT_EXEC=src/test_numpy

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< -I.

$(OUT_EXEC): $(OBJ1) $(TEST_OBJ)
	$(CC) -o $(OUT_EXEC) $^ -I.

clean: $(OBJ1)
	rm $(OBJ1) $(TEST_OBJ) $(OUT_EXEC)


