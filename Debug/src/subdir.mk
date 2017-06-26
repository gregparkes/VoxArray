################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/Matrix.cpp \
../src/Vector.cpp \
../src/main.cpp \
../src/numpy.cpp \
../src/numstatic.cpp \
../src/test_matrix.cpp \
../src/test_numpy1d.cpp 

OBJS += \
./src/Matrix.o \
./src/Vector.o \
./src/main.o \
./src/numpy.o \
./src/numstatic.o \
./src/test_matrix.o \
./src/test_numpy1d.o 

CPP_DEPS += \
./src/Matrix.d \
./src/Vector.d \
./src/main.d \
./src/numpy.d \
./src/numstatic.d \
./src/test_matrix.d \
./src/test_numpy1d.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	/usr/local/bin/g++-5 -O0 -g3 -Wall -c -fmessage-length=0 -fopenmp -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


