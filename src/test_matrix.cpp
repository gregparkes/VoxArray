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

namespace tests_matrix {

	static void test_constructor()
	{
		std::cout << "Start Constructor" << std::endl;
		numpy::Matrix m(10, 10);

		std::cout << std::endl << "Test_Matrix_Constructor :: Passed" << std::endl;
	}

	static void test_shape()
	{
		std::cout << "Start Shape" << std::endl;
		numpy::Matrix m(203,153);
		//std::cout << m.shape() << std::endl;

		std::cout << "Test_Shape :: Passed" << std::endl;
	}

	static void test_str()
	{
		std::cout << "Start Str" << std::endl;
		numpy::Matrix m = numpy::zeros(3,3);
		std::cout << m.str() << std::endl;

		std::cout << "Test_Str :: Passed" << std::endl;
	}

	static void test_empty_zeros_ones()
	{
		std::cout << "Start Empty_Zeros_Ones" << std::endl;
		numpy::Matrix m = numpy::empty(6,6);
		numpy::Matrix m2 = numpy::zeros(4,4);
		numpy::Matrix m3 = numpy::ones(4,4);
		for (int j = 0; j < 4; j++)
		{
			for (int i = 0; i < 4; i++)
			{
				//printf("%f\n", m3.data[i+j*4]);
				assert(m2.data[i+j*4] == 0.0);
				assert(m3.data[i+j*4] == 1.0);
			}
		}

		//std::cout << m2.str() << std::endl << m3.str() << std::endl;

		std::cout << "Test_Empty_Zeros_Ones :: Passed" << std::endl;
	}

	static void test_fill()
	{
		std::cout << "Start Fill" << std::endl;
		numpy::Matrix m = numpy::fill(4,4,5.5);
		for (int j = 0; j < 4; j++)
		{
			for (int i = 0; i < 4; i++)
			{
				assert(m.data[i+j*4] == 5.5);
			}
		}

		std::cout << "Test_Fill :: Passed" << std::endl;
	}

	static void test_copy()
	{
		std::cout << "Start Copy" << std::endl;
		numpy::Matrix m = numpy::ones(6,6);
		numpy::Matrix m2 = numpy::copy(m);
		for (int i = 0; i < 6*6; i++)
		{
			assert(m.data[i] == m2.data[i]);
		}

		std::cout << "Test_Copy :: Passed" << std::endl;
	}

	static void test_vectorize()
	{
		std::cout << "Start Vectorize" << std::endl;
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

		//std::cout << m.str() << std::endl << test1.str() << std::endl << test2.str() << std::endl;

		std::cout << "Test_Vectorize :: Passed" << std::endl;
	}

	static void test_countnonzero()
	{
		std::cout << "Start Count_Nonzero" << std::endl;
		Mat X = numpy::zeros(6,6);
		Mat Y = numpy::ones(5,5);
		assert(numpy::count_nonzero(X) == 0);
		assert(numpy::count_nonzero(Y) == 25);

		std::cout << "Test_Count_Nonzero :: Passed" << std::endl;
	}

	static void test_empty_zero_ones_like()
	{
		std::cout << "Start Empty_Zero_Ones_Like" << std::endl;
		//@todo implement empty zero ones like test()
	}

	static void test_eye()
	{
		std::cout << "Start Eye" << std::endl;
		//printf("got matrix\n");
		Mat X = numpy::eye(5,5);
		for (int i = 0; i < 5; i++)
		{
			assert(X.vectors[i]->data[i] == 1.0);
		}
		//std::cout << X.str() << std::endl;

		std::cout << "Test_Eye :: Passed" << std::endl;
	}

	static void test_trace()
	{
		std::cout << "Start Trace" << std::endl;
		Mat X = numpy::eye(6,6);
		assert(numpy::trace(X) == 6.0);
		X.data[0] = 3.0;
		X.data[1] = 30.0;
		assert(numpy::trace(X) == 8.0);

		std::cout << "Test_Trace :: Passed" << std::endl;
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

