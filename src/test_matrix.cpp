/*
 * test_matrix.cpp
 *
 *  Created on: 21 Mar 2017
 *      Author: Greg
 */


#include <iostream>
#include <assert.h>
#include <math.h>

#include "numpy.h"

#define PRINT_STR(x) (std::cout << x << std::endl)
#define PRINT_OBJ(x) (std::cout << x.str() << std::endl)
#define CMP(x,y) (fabs(x - y) < 1E-13)

namespace tests_matrix {

	static void test_constructor()
	{
		PRINT_STR("Start Constructor");
		numpy::Matrix m(10, 10);
		PRINT_STR("Test_Matrix_Constructor :: Passed");
	}

	static void test_shape()
	{
		PRINT_STR("Start Shape");
		numpy::Matrix m(203,153);
		PRINT_STR(m.shape() );
		PRINT_STR("Test_Shape :: Passed");
	}

	static void test_str()
	{
		PRINT_STR("Start Str");
		numpy::Matrix m = numpy::zeros(3,3);
		PRINT_OBJ(m);
		PRINT_STR("Test_Str :: Passed");
	}

	static void test_empty_zeros_ones()
	{
		PRINT_STR("Start Empty_Zeros_Ones");
		numpy::Matrix m = numpy::empty(6,6);
		numpy::Matrix m2 = numpy::zeros(4,4);
		numpy::Matrix m3 = numpy::ones(4,4);
		for (int j = 0; j < 4; j++)
		{
			for (int i = 0; i < 4; i++)
			{
				//printf("%f\n", m3.data[i+j*4]);
				assert(CMP(m2.data[i+j*4],0.0));
				assert(CMP(m3.data[i+j*4],1.0));
			}
		}

		//PRINT_STR(m2.str() << std::endl << m3.str() );
		PRINT_STR("Test_Empty_Zeros_Ones :: Passed");
	}

	static void test_fill()
	{
		PRINT_STR("Start Fill");
		numpy::Matrix m = numpy::fill(4,4,5.5);
		for (int j = 0; j < 4; j++)
		{
			for (int i = 0; i < 4; i++)
			{
				assert(CMP(m.data[i+j*4], 5.5));
			}
		}

		PRINT_STR("Test_Fill :: Passed");
	}

	static void test_copy()
	{
		PRINT_STR("Start Copy");
		numpy::Matrix m = numpy::ones(6,6);
		numpy::Matrix m2 = numpy::copy(m);
		for (int i = 0; i < 6*6; i++)
		{
			assert(CMP(m.data[i],m2.data[i]));
		}

		PRINT_STR("Test_Copy :: Passed");
	}

	static void test_vectorize()
	{
		PRINT_STR("Start Vectorize");
		numpy::Matrix m = numpy::empty(4,2);
		for (int i = 0; i < 8; i++)
		{
			if (i % 2 == 0)
			{
				m.data[i] = 0.0;
			} else {
				m.data[i] = 1.0;
			}
		}
		Numpy test1 = numpy::vectorize(m);
		Numpy test2 = numpy::vectorize(m, AXIS_ROW);

		//PRINT_STR(m.str() << std::endl << test1.str() << std::endl << test2.str() );

		PRINT_STR("Test_Vectorize :: Passed");
	}

	static void test_countnonzero()
	{
		PRINT_STR("Start Count_Nonzero" );
		Mat X = numpy::zeros(6,6);
		Mat Y = numpy::ones(5,5);
		assert(numpy::count_nonzero(X) == 0);
		assert(numpy::count_nonzero(Y) == 25);

		PRINT_STR("Test_Count_Nonzero :: Passed");
	}

	static void test_empty_zero_ones_like()
	{
		PRINT_STR("Start Empty_Zero_Ones_Like");
		//@todo implement empty zero ones like test()
	}

	static void test_eye()
	{
		PRINT_STR("Start Eye" );
		//printf("got matrix\n");
		Mat X = numpy::eye(5,5);
		for (int i = 0; i < 5; i++)
		{
			assert(X.vectors[i]->data[i] == 1.0);
		}
		//PRINT_STR(X.str() );

		PRINT_STR("Test_Eye :: Passed" );
	}

	static void test_trace()
	{
		PRINT_STR("Start Trace" );
		Mat X = numpy::eye(6,6);
		assert(numpy::trace(X) == 6.0);
		X.data[0] = 3.0;
		X.data[1] = 30.0;
		assert(numpy::trace(X) == 8.0);

		PRINT_STR("Test_Trace :: Passed" );
	}

}

static void call_all_matrix_tests()
{
	using namespace tests_matrix;
	test_constructor();
	test_shape();
	test_empty_zeros_ones();
	test_str();
	test_fill();
	test_copy();
	test_vectorize();
	test_countnonzero();
	//test_empty_zero_ones_like();
	test_eye();
	test_trace();
}

