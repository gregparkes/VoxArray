/*

------------------------------------------------------

GNU General Public License:

	Gregory Parkes, Postgraduate Student at the University of Southampton, UK.
    Copyright (C) 2017 Gregory Parkes

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

-----------------------------------------------------

	This implementation is an attempt to replicate the Python module 'numpy' which implements
	a float Numpy which can perform matrix and vector-based operations which modify every
	element in the Numpy.

    This is the C++ file, and is ANSI 90 compatible. This can be run using
    C. We also have numpy.h and Numpy.cpp. which are C++ wrapper classes
    around this foundational file.

    test_numpy1d.cpp
*/

#include <iostream>
#include <assert.h>
#include <math.h>

#include "numpy.h"

#define PRINT_STR(x) (std::cout << x << std::endl)
#define PRINT_SUCCESS(x) (std::cout << ">>> " << x << " >>>" << std::endl)
#define PRINT_FAIL(x) (std::cout << "??? " << x << " ???" << std::endl)
#define PRINT_OBJ(x) (std::cout << x.str() << std::endl)
#define CMP(x,y) (fabs(x - y) < 1E-13)
#define WEAK_CMP(x,y) (fabs(x -y) < 1E-5)

namespace tests {

	static void test_constructors()
	{
		PRINT_STR("Start Constructors");
		numpy::Vector x = numpy::Vector();
		// expect an error!
		try {
			std::cout << x.str() << std::endl;
			PRINT_FAIL("Failed error test in test_constructors()");
		} catch (const std::invalid_argument& e)
		{
			PRINT_SUCCESS("Passed null constructor test");
		} catch (...)
		{
			PRINT_FAIL("Unknown catch for test_constructors()");
		}
		// normal empty constructor
		try {
			numpy::Vector y = numpy::Vector(7);
			numpy::Vector z = numpy::Vector(5, (bool) AXIS_ROW);
		} catch (...)
		{
			PRINT_FAIL("Unknown catch for test_constructors() for normal constructor");
		}
		// constructors with few values in
		try {
			numpy::Vector a = numpy::Vector(1.0, 3.0);
			numpy::Vector b = numpy::Vector(3.0, -4.0, 2.0);
			numpy::Vector c = numpy::Vector(1.0, 2.0, -3.0, -4.0);
			PRINT_OBJ(c);
			assert(CMP(a.data[0], 1.0));
			assert(CMP(a.data[1], 3.0));
			assert(CMP(b.data[0], 3.0));
			assert(CMP(b.data[1], -4.0));
			assert(CMP(b.data[2], 2.0));
			assert(CMP(c.data[0], 1.0));
			assert(CMP(c.data[1], 2.0));
			assert(CMP(c.data[2], -3.0));
			assert(CMP(c.data[3], -4.0));
		} catch (...)
		{
			PRINT_FAIL("Unknown catch for test_constructors() for test case");
		}
		
		
		PRINT_STR("test_constructors :: Passed");
	}

	static void test_zeros()
	{
		PRINT_STR("Start Zeros");
		Numpy arr = numpy::zeros(16);
		assert(arr.n == 16);
		for (int i = 0; i < 16; i++)
		{
			assert(CMP(arr.data[i], 0.0));
		}

		PRINT_STR("Test_Zeros :: Passed");
	}

	static void test_ones()
	{
		PRINT_STR("Start Ones");
		Numpy arr = numpy::ones(16);
		assert(arr.n == 16);
		for (int i = 0; i < 16; i++)
		{
			assert(CMP(arr.data[i], 1.0));
		}

		PRINT_STR("Test_Ones :: Passed");
	}

	static void test_empty()
	{
		PRINT_STR("Start Empty");
		Numpy arr = numpy::empty(16);
		assert(arr.n == 16);
		//PRINT_STR(arr.str());
		for (int i = 0; i < 16; i++)
		{
			double val = arr.data[i];
			val *= 2;
		}
		Numpy x = numpy::Vector(4.0, 5.0, 2.0);
		Numpy y = numpy::Vector(2.0, 7.0, 5.0, 8.0);

		PRINT_STR("Test_Empty :: Passed");
	}

	static void test_empty_like()
	{
		PRINT_STR("Start Empty_Like");
		Numpy x = numpy::zeros(16);
		Numpy y = numpy::empty_like(x);
		assert(y.n == x.n);

		PRINT_STR("Test_Empty_Like :: Passed");
	}

	static void test_zeros_like()
	{
		PRINT_STR("Start Zeros_Like");
		Numpy x = numpy::zeros(16);
		Numpy y = numpy::zeros_like(x);
		assert(y.n == x.n);
		//PRINT_STR(x.str());
		for (int i = 0; i < 16; i++)
		{
			assert(CMP(y.data[i], 0.0));
		}

		PRINT_STR("Test_Zeros_Like :: Passed");
	}

	static void test_ones_like()
	{
		PRINT_STR("Start Ones_Like");
		Numpy x = numpy::zeros(16);
		Numpy y = numpy::ones_like(x);
		assert(y.n == x.n);
		for (int i = 0; i < 16; i++)
		{
			assert(CMP(y.data[i], 1.0));
		}

		PRINT_STR("Test_Ones_Like :: Passed");
	}

	static void test_fill()
	{
		PRINT_STR("Start Fill");
		Numpy arr = numpy::fill(16, 15.0);
		assert(arr.n == 16);
		for (int i = 0; i < 16; i++)
		{
			assert(CMP(arr.data[i],15.0));
		}
		PRINT_STR("Test_Fill :: Passed");
	}

	static void test_len()
	{
		PRINT_STR("Start len");
		Numpy arr = numpy::ones(10);
		assert(len(arr) == 10);
		Numpy arr2 = numpy::zeros(5);
		assert(len(arr2) == 5);
		PRINT_STR("Test_Len :: Passed");
	}

	static void test_str()
	{
		PRINT_STR("Start Str");
		Numpy x = numpy::zeros(6);
		PRINT_STR(numpy::str(x));
		Numpy y = numpy::rand(6);
		PRINT_STR(numpy::str(y));
		PRINT_STR(y.str());

		PRINT_STR("Test_Str :: Passed");
	}

	static void test_array()
	{
		PRINT_STR("Start Array");
		Numpy arr = numpy::array("0.0, 3.42, 5.4, 5.45");
		Numpy arr2 = Numpy(0.0, 3.42, 5.4, 5.45);
		//PRINT_STR(arr.str());
		assert(arr.len() == 4);
		for (int i = 0; i < arr.len(); i++)
		{
			assert(CMP(arr.data[i],arr2.data[i]));
		}
		// second test
		Numpy arr3 = numpy::array("302.45, 605.4, -321.7685, -0.44443");
		Numpy arr4 = numpy::Vector(302.45, 605.4, -321.7685, -0.44443);
		assert(arr3.len() == 4);
		for (int i = 0; i < arr3.len(); i++)
		{
			assert(CMP(arr3.data[i], arr4.data[i]));
		}

		PRINT_STR("Test_Array :: Passed");
	}

	static void test_copy()
	{
		PRINT_STR("Start Copy");
		Numpy x = numpy::ones(16);
		Numpy y = numpy::copy(x);
		assert(y.n == x.n);
		for (int i = 0; i < 16; i++)
		{
			assert(CMP(x.data[i], y.data[i]));
		}
		Numpy z = x.copy();
		for (int i = 0; i < 16; i++)
		{
			assert(CMP(x.data[i], z.data[i]));
		}
		PRINT_STR("Test_Copy :: Passed");
	}

	static void test_to_matrix()
	{
		PRINT_STR("Start To_Matrix");
		Numpy x = numpy::ones(6);
		Mat y = numpy::to_matrix(x);
		PRINT_OBJ(y);
		assert(y.nvec == 1);
		assert(y.vectors[0]->n == 6);
		for (int i = 0; i < 6; i++)
		{
			assert(CMP(y.vectors[0]->data[i], 1.0));
			assert(CMP(y.data[i], 1.0));
		}

		PRINT_STR("test_to_matrix :: Passed");
	}

	static void test_arange()
	{
		PRINT_STR("Start Arange");
		// test the single case
		Numpy a = numpy::arange(6);
		assert(a.n == 6);
		for (int i = 0; i < 6; i++)
		{
			assert(CMP(a.data[i], (double) i));
		}
		Numpy x = numpy::arange(0.0, 1.0, 0.1);
		assert(x.n == 11);
		for (int i = 0; i < 11; i++)
		{
			double y = i * 0.1;
			//printf("%lf %lf %lf %lf\n", x.data[i], y, x.data[i] - y, 1e-5);
			assert(CMP(x.data[i], y));
		}

		PRINT_STR("Test_Arange :: Passed");
	}

	static void test_take()
	{
		PRINT_STR("Start Take");
		Numpy x = numpy::arange(6);
		Numpy z = Numpy(2, 3, 4);

		// test take using another numpy array
		Numpy a = take(x, z);
		assert(a.len() == 3);
		for (int i = 0; i < a.len(); i++)
		{
			assert(CMP(a.data[i], z.data[i]));
		}

		PRINT_STR("test_take :: Passed");
	}

	static void test_rand()
	{
		PRINT_STR("Start Rand");
		Numpy x = numpy::rand(16);
		assert(x.n == 16);
		for (int i = 0; i < 16; i++)
		{
			assert(x.data[i] > 0.0 && x.data[i] < 1.0);
		}

		PRINT_STR("Test_Rand :: Passed");
	}

	static void test_randn()
	{
		PRINT_STR("Start Randn");
		Numpy x = numpy::randn(500);
		assert(x.n == 500);
		//calculate mean and hope it's close to +- 0.2 around 0.
		double count = 0.0;
		for (int i = 0; i < 500; i++)
		{
			count += x.data[i];
		}
		assert(count / 500 > -0.5 && count / 500 < 0.5);

		PRINT_STR("Test_Randn :: Passed");
	}

	

	static void test_linspace()
	{
		PRINT_STR("Start Linspace");
		Numpy x = numpy::linspace(0.0, 10.0, 11);
		assert(x.n == 11);
		for (int i = 0; i < 11; i++)
		{
			//std::cout << x.data[i] << ", " << (double) i << std::endl;
			assert(CMP(x.data[i], (double) i));
		}
		Numpy y = numpy::linspace(0.0, 5.0, 6);
		assert(y.n == 6);
		for (int i = 0; i < 6; i++)
		{
			assert(CMP(y.data[i], i));
		}
		Numpy z = numpy::linspace(0.1, 0.5, 5);
		//PRINT_STR(z.str());
		Numpy ab = numpy::linspace(0.2, 0.6, 5);
		Numpy ac = numpy::linspace(0.3, 0.7, 5);
		Numpy ad = numpy::linspace(0.12, 0.2, 10);
		//PRINT_STR(ab.str()) << ac.str()) << ad.str());

		PRINT_STR("Test_Linspace :: Passed");
	}

	static void test_abs()
	{
		PRINT_STR("Start Abs");
		Numpy x = numpy::array("5, -1.5, -3.2, 6.5, 9.8, -0.76");
		Numpy y = numpy::abs(x);
		assert(y.n == 6);
		// PRINT_STR(y.str());
		for (int i = 0; i < 6; i++)
		{
			assert(y.data[i] >= 0.0);
		}
		x.abs();
		
		for (int i = 0; i < 6; i++)
		{
			assert(x.data[i] >= 0.0);
		}

		PRINT_STR("Test_Abs :: Passed");
	}

	static void test_sum()
	{
		PRINT_STR("Start Sum");
		Numpy x = numpy::ones(12);
		double ans = numpy::sum(x);
		assert(CMP(ans, 12.0));
		double ans2 = x.sum();
		assert(CMP(ans2, 12.0));

		PRINT_STR("Test_Sum :: Passed");
	}

	static void test_all()
	{
		PRINT_STR("Start All");
		Numpy x = numpy::ones(6);
		assert(numpy::all(x));
		assert(x.all());
		Numpy y = numpy::zeros(6);
		assert(!numpy::all(y));
		assert(!y.all());

		PRINT_STR("Test_All :: Passed");
	}

	static void test_any()
	{
		PRINT_STR("Start Any");
		Numpy x = numpy::ones(6);
		assert(numpy::any(x));
		assert(x.any());
		Numpy y = numpy::zeros(6);
		assert(!numpy::any(y));
		assert(!y.any());
		Numpy z = numpy::rand(6);
		assert(numpy::any(z));
		assert(z.any());

		PRINT_STR("Test_Any :: Passed");
	}

	static void test_mean()
	{
		PRINT_STR("Start Mean");
		Numpy x = numpy::ones(6);
		assert(CMP(numpy::mean(x), 1.0));
		assert(CMP(x.mean(), 1.0));
		Numpy y = numpy::array("0.0, 0.25, 0.5, 0.75, 1.0");
		assert(CMP(numpy::mean(y), 0.5));
		assert(CMP(y.mean(), 0.5));

		PRINT_STR("Test_Mean :: Passed");
	}

	static void test_std()
	{
		PRINT_STR("Start Std");
		Numpy x = numpy::array("6.0, 2.0, 3.0, 1.0");
		//PRINT_OBJ(x);
		//PRINT_STR(numpy::std(x));
		assert(CMP(numpy::std(x), 2.1602468994693));
		assert(CMP(x.std(), 2.1602468994693));

		PRINT_STR("Test_Std :: Passed");
	}

	static void test_var()
	{
		PRINT_STR("Start Var");
		Numpy x = numpy::array("7.0,6.0,8.0,4.0,2.0,7.0,6.0,7.0,6.0,5.0");
		//PRINT_OBJ(x);
		//PRINT_STR(numpy::var(x));
		assert(WEAK_CMP(numpy::var(x), 3.0666666666667));
		assert(WEAK_CMP(x.var(), 3.0666666666667));

		PRINT_STR("Test_Var :: Passed");
	}

	static void test_argmin()
	{
		PRINT_STR("Start Argmin");
		Numpy df = numpy::array("3.0, 6.0, -2.0, 5.0, -4.0, 0.65");
		//PRINT_STR(df.str());
		//printf("%d\n",numpy::argmin(df));
		assert(numpy::argmin(df) == 4);
		assert(df.argmin() == 4);

		PRINT_STR("Test_Argmin :: Passed");
	}

	static void test_argmax()
	{
		PRINT_STR("Start Argmax");
		Numpy x = numpy::array("3.0, 6.0, -2.0, 5.0, -4.0");
		assert(numpy::argmax(x) == 1);
		assert(x.argmax() == 1);

		PRINT_STR("Test_Argmax :: Passed");
	}

	static void test_sine()
	{
		PRINT_STR("Start Sine");
		Numpy x = numpy::array("0.0, 1.570796, 3.141592, 4.712388");
		Numpy y = numpy::sin(x);
		//std::cout << y.str(7) << std::endl;
		assert(WEAK_CMP(y.data[0], 0.0));
		assert(WEAK_CMP(y.data[1], 1.0));
		assert(WEAK_CMP(y.data[2], 0.0));
		assert(WEAK_CMP(y.data[3], -1.0));
		// x.sine() doesnt copy, changes the values at reference.
		x.sin();
		assert(WEAK_CMP(x.data[0], 0.0));
		assert(WEAK_CMP(x.data[1], 1.0));
		assert(WEAK_CMP(x.data[2], 0.0));
		assert(WEAK_CMP(x.data[3], -1.0));

		PRINT_STR("Test_Sin :: Passed");
	}

	static void test_cosi()
	{
		PRINT_STR("Start Cosine");
		Numpy x = numpy::array("0.0, 1.570796, 3.14159, 4.712388");
		Numpy y = numpy::cos(x);
		//std::cout << y.str(7) << std::endl;
		assert(WEAK_CMP(y.data[0], 1.0));
		assert(WEAK_CMP(y.data[1], 0.0));
		assert(WEAK_CMP(y.data[2], -1.0));
		assert(WEAK_CMP(y.data[3], 0.0));
		// unlike numpy::cosi(x), x.cosi() doesnt copy and changes at reference.
		x.cos();
		assert(WEAK_CMP(x.data[0], 1.0));
		assert(WEAK_CMP(x.data[1], 0.0));
		assert(WEAK_CMP(x.data[2], -1.0));
		assert(WEAK_CMP(x.data[3], 0.0));

		PRINT_STR("Test_Cos :: Passed");
	}

	static void test_tang()
	{
		PRINT_STR("Start Tan");
		Numpy x = numpy::array("0.0, 45.0, 89.0, 135.0");
		Numpy y = numpy::to_radians(x).tan();
		std::cout << y.str(8) << std::endl;
		assert(WEAK_CMP(y.data[0], 0.0));
		assert(WEAK_CMP(y.data[1], 1.0));
		assert(WEAK_CMP(y.data[2], 57.29214));
		assert(WEAK_CMP(y.data[3], -1.0));

		PRINT_STR("Test_Tan :: Passed");
	}

	static void test_to_radians()
	{
		// ----------------------------------------------------------------------------------------
	}

	static void test_to_degrees()
	{
		// ---------------------------------------------------------------------------------------
	}

	static void test_expn()
	{
		PRINT_STR("Start Exp");
		Numpy x = numpy::array("0.0, 1.0");
		Numpy y = numpy::exp(x);
		assert(CMP(y.data[0], 1.0));
		assert(WEAK_CMP(y.data[1], 2.71828));
		x.exp();
		assert(CMP(x.data[0], 1.0));
		assert(WEAK_CMP(x.data[1], 2.71828));

		PRINT_STR("Test_Exp :: Passed");
	}

	static void test_operators()
	{
		PRINT_STR("Start Operators");
		Numpy x = numpy::ones(10);
		Numpy y = x.copy();
		Numpy a = numpy::fill(10, 3.0);
		
		Numpy z = numpy::zeros(10);
		
		// checking +=
		x += y;
		assert(CMP(x.data[0], 2.0));
		x += 1.0;
		assert(CMP(x.data[0], 3.0));
		x += 1;
		assert(CMP(x.data[0], 4.0));

		// checking -= 
		x -= y;
		assert(CMP(x.data[0], 3.0));
		x -= 1.0;
		assert(CMP(x.data[0], 2.0));
		x -= 1;
		assert(CMP(x.data[0], 1.0));

		// checking *=
		x *= 4.0;
		assert(CMP(x.data[0], 4.0));
		y *= 2;
		assert(CMP(y.data[0], 2.0));
		x *= y;
		assert(CMP(x.data[0], 8.0));

		// checking /=
		x /= 2.0;
		assert(CMP(x.data[0], 4.0));
		x /= y;
		assert(CMP(x.data[0], 2.0));
		y /= 2;
		assert(CMP(y.data[0], 1.0));

		// checking +
		Numpy b = x + a + y;
		Numpy c = x + 3.0 + y;
		Numpy d = x + 2 + y + 4;
		assert(CMP(b.data[0], 6.0));
		assert(CMP(c.data[0], 6.0));
		assert(CMP(d.data[0], 9.0));

		// checking -
		Numpy e = x - a - y;
		Numpy f = x - 3.0 - y;
		Numpy g = x - y - 6;
		assert(CMP(e.data[0], -2.0));
		assert(CMP(f.data[0], -2.0));
		assert(CMP(g.data[0], -5.0));

		// checking *
		Numpy h = x * a * y;
		Numpy i = x * 3.0 * y;
		Numpy j = x * y * 6;
		assert(CMP(h.data[0], 6.0));
		assert(CMP(i.data[0], 6.0));
		assert(CMP(j.data[0], 12.0));

		// checking /
		Numpy k = a / x / y;
		Numpy l = x / 2.0 / y;
		Numpy m = x / y / 2;
		assert(CMP(k.data[0], 1.5));
		assert(CMP(l.data[0], 1.0));
		assert(CMP(m.data[0], 1.0));

		// checking ^ - STILL TO DO -------------------------------------------------------------------


		PRINT_STR("Test_Operators :: Passed");
	}

	static void test_floor()
	{
		PRINT_STR("Start Floor");
		Numpy x = numpy::array("3.14, 4.54, -3.2, 7.4987");
		Numpy y = numpy::floor(x);
		Numpy answers = numpy::array("3.0, 4.0, -3.0, 7.0");
		for (int i = 0; i < 4; i++)
		{
			assert(CMP(y[i], answers[i]));
		}

		PRINT_STR("Test_Floor :: Passed");
	}

	static void test_ceil()
	{
		PRINT_STR("Start Ceil");
		Numpy x = numpy::array("3.14, 4.54, -3.2, 7.4987");
		Numpy y = numpy::ceil(x);
		Numpy answers = numpy::array("4.0, 5.0, -4.0, 8.0");
		// PRINT_STR(y.str());
		for (int i = 0; i < 4; i++)
		{
			assert(CMP(y[i], answers[i]));
		}

		PRINT_STR("Test_Ceil :: Passed");
	}

	static void test_randint()
	{
		PRINT_STR("Start Randint");
		Numpy x = numpy::randint(15, 10);
		//PRINT_STR(x.str());
		for (int i = 0; i < 15; i++)
		{
			assert(x[i] <= 10.0);
		}

		PRINT_STR("Test_Randint :: Passed");
	}

	static void test_randchoice()
	{
		PRINT_STR("Start Randchoice");
		Numpy x = numpy::randchoice(15, "-1.0, 0.0, 1.0");
		assert(x.n == 15);
		//PRINT_STR(x.str());
		for (int i = 0; i < 15; i++)
		{
			assert(CMP(x[i], -1.0) || CMP(x[i], 0.0) || CMP(x[i], 1.0));
		}

		PRINT_STR("Test_Randchoice :: Passed");
	}

	static void test_count()
	{
		PRINT_STR("Start Count");
		Numpy x = numpy::ones(10);
		assert(numpy::count(x, 1.0) == 10);

		PRINT_STR("Test_Count :: Passed");
	}

	static void test_count_nonzero()
	{
		PRINT_STR("Start Count_Nonzero");
		Numpy x = numpy::zeros(10);
		assert(numpy::count_nonzero(x) == 0);
		Numpy y = numpy::ones(10);
		assert(numpy::count_nonzero(y) == 10);

		PRINT_STR("Test_Count_Nonzero :: Passed");
	}

	static void test_cumsum()
	{
		PRINT_STR("Start Cumsum");
		Numpy x = numpy::linspace(0.0, 3.0, 4);
		// PRINT_STR(x.str() << numpy::len(x));
		Numpy y = numpy::cumsum(x);

		assert(y.data[0] == 0.0);
		assert(y.data[1] == 1.0);
		assert(y.data[2] == 3.0);
		assert(y.data[3] == 6.0);

		PRINT_STR("Test_Cumsum :: Passed");
	}

	static void test_flip()
	{
		PRINT_STR("Start Flip");
		Numpy x = numpy::linspace(0.0, 4.0, 5);
		Numpy y = numpy::flip(x);
		Numpy z = numpy::array("4.0, 3.0, 2.0, 1.0, 0.0");
		for (int i = 0; i < 5; i++)
		{
			assert(CMP(y.data[i], z.data[i]));
		}

		PRINT_STR("Test_Flip :: Passed");
	}

	static void test_logr()
	{
		PRINT_STR("Start Log");
		Numpy x = numpy::array("0.0, 1.0");
		Numpy y = numpy::log(x);
		assert(CMP(y.data[0], 1.0) < 1e-14);
		assert(WEAK_CMP(y.data[1], 2.71828) < 1e-5);
//		x.logr();
//		assert(x.data[0] - 1.0 < 1e-14);
//		assert(x.data[1] - 2.71828 < 1e-5);

		PRINT_STR("Test_Logr :: Passed");
	}

	static void test_logspace()
	{
		PRINT_STR("Start Logspace");
		Numpy x = numpy::linspace(0.0, 1.0, 11);
		Numpy y = numpy::logspace(0.0, 1.0, 11);
		for (int i = 0; i < 11; i++)
		{
			assert(CMP(y.data[i], pow(10, x[i])));
		}
		//PRINT_STR(x.str()) << y.str());

		PRINT_STR("Test_Logspace :: Passed");
	}

	static void test_cumprod()
	{
		PRINT_STR("Start Cumprod");
		Numpy x = numpy::linspace(1.0, 5.0, 5);
		Numpy y = numpy::cumprod(x);
		// PRINT_STR(y.str());
		assert(y.data[0] == 1.0);
		assert(y.data[1] == 2.0);
		assert(y.data[2] == 6.0);
		assert(y.data[3] == 24.0);
		assert(y.data[4] == 120.0);
		PRINT_STR("Test_Cumprod :: Passed");
	}

	static void test_lstrip()
	{
		PRINT_STR("Start Lstrip");
		Numpy x = numpy::linspace(0.0, 10.0, 11);
		Numpy y = numpy::lstrip(x, 4);
		assert(y.n == 7);
		//PRINT_STR(y.str());
		for (int i = 4; i < 11; i++)
		{
			assert(y.data[i-4] == i);
		}

		PRINT_STR("Test_Lstrip :: Passed");
	}

	static void test_rstrip()
	{
		PRINT_STR("Start Rstrip");
		Numpy x = numpy::linspace(0.0, 10.0, 11);
		Numpy y = numpy::rstrip(x, 5);
		assert(y.n == 6);
		//PRINT_STR(y.str());
		for (int i = 0; i < 6; i++)
		{
			assert(y.data[i] == i);
		}

		PRINT_STR("Test_Rstrip :: Passed");
	}

	static void test_hstack()
	{
		PRINT_STR("Start Vstack");
		Numpy x = numpy::arange(0.0, 1.0, 0.1);
		Numpy y = numpy::randn(5);
		Numpy z = numpy::vstack(x, y);
		//PRINT_STR(z.str());
		assert(z.n == x.n + y.n);

		PRINT_STR("Test_Vstack :: Passed");
	}

	static void test_norm()
	{
		PRINT_STR("Start Norm");
		Numpy x = numpy::linspace(1.0, 3.0, 3);
		PRINT_STR(numpy::norm(x, 2));
		assert(CMP(numpy::norm(x, _INF_NORM), 3.0));
		assert(CMP(numpy::norm(x, _ONE_NORM), 6.0));
		assert(WEAK_CMP(numpy::norm(x, _TWO_NORM), 3.74166));

		PRINT_STR("Test_Norm :: Passed");
	}

	static void test_radians()
	{
		PRINT_STR("Start Radians");
		Numpy x = numpy::array("0.0, 90.0, 180.0, 360.0");
		Numpy y = numpy::to_radians(x);
		//PRINT_STR(y.str());
		assert(WEAK_CMP(y[0], x[0]));
		assert(WEAK_CMP(y[1], M_PI / 2));
		assert(WEAK_CMP(y[2], M_PI));
		assert(WEAK_CMP(y[3], M_PI * 2));

		PRINT_STR("Test_Radians :: Passed");
	}

	static void test_degrees()
	{
		PRINT_STR("Start Degrees");
		Numpy x = numpy::array("0.0, 1.570796, 3.14159, 4.712388");
		Numpy y = numpy::to_degrees(x);
		//PRINT_STR(y.str());
		assert(y[0] - x[0] < 2.0);
		assert(y[1] - 90.0 < 2.0);
		assert(y[2] - 180.0 < 2.0);
		assert(y[3] - 360.0 < 2.0);

		PRINT_STR("Test_Degrees :: Passed");
	}

	static void test_sqroot()
	{
		PRINT_STR("Start Sqrt");
		Numpy x = numpy::array("4.0, 9.0, 16.0, 25.0");
		Numpy y = numpy::sqrt(x);
		for (int i = 0; i < 4; i++)
		{
			assert(y[i] == i+2);
		}

		PRINT_STR("Test_Sqroot :: Passed");
	}

	static void test_power()
	{
		PRINT_STR("Start Power");
		Numpy x = numpy::linspace(1.0, 5.0, 5);
		Numpy y = numpy::power(x, 2);
		for (int i = 0; i < 5; i++)
		{
			assert(CMP(y[i], x[i] * x[i]));
		}

		PRINT_STR("Test_Power :: Passed");
	}

	static void test_min()
	{
		PRINT_STR("Start Min");
		Numpy x = numpy::linspace(1.0, 10.0, 11);
		assert(numpy::min(x) == 1.0);
		Numpy y = numpy::rand(5);
		double min = y.data[0];
		for (int i = 0; i < 5; i++)
		{
			if (y[i] < min)
			{
				min = y[i];
			}
		}
		assert(CMP(min, numpy::min(y)));

		PRINT_STR("Test_Min :: Passed");
	}

	static void test_max()
	{
		PRINT_STR("Start Max");
		Numpy x = numpy::linspace(1.0, 10.0, 11);
		assert(numpy::max(x) == 10.0);
		Numpy y = numpy::rand(5);
		double max = y.data[0];
		for (int i = 0; i < 5; i++)
		{
			if (y[i] > max)
			{
				max = y[i];
			}
		}
		assert(CMP(max, numpy::max(y)));

		PRINT_STR("Test_Max :: Passed");
	}

	static void test_dot()
	{
		PRINT_STR("Start Dot");
		Numpy x = numpy::fill(4, 3.0);
		Numpy y = numpy::fill(4, 2.0);
		double z = numpy::dot(x, y);
		assert(z == 24.0);
		double z2 = x.dot(y);
		assert(z2 == 24.0);
		//PRINT_STR(z);

		PRINT_STR("Test_Dot :: Passed");
	}

	static void test_sort()
	{
		PRINT_STR("Start Sort");

		Numpy x = numpy::array("3.0, 2.0, 7.0, 5.0, 15.0, 9.0, 11.0");
		Numpy y = numpy::sort(x);
		//PRINT_STR(x.str());
		//PRINT_STR(y.str());
		for (int i = 0; i < 6; i++)
		{
			assert(y.data[i] <= y.data[i+1]);
		}

		PRINT_STR("Test_Sort :: Passed");
	}




	// end of tests here
}


static void call_all_tests()
{
	//The tests here
	using namespace tests;
	test_constructors();
	test_empty();
	test_zeros();
	test_ones();
	test_fill();
	test_len();
	test_str();
	test_array();
	test_copy();
	test_to_matrix();
	test_take();
	test_empty_like();
	test_zeros_like();
	test_ones_like();
	test_rand();
	test_randn();
	test_arange();
	test_linspace();
	test_abs();
	test_sum();
	test_all();
	test_any();
	test_mean();
	test_std();
	test_var();
	test_argmin();
	test_argmax();
	test_sine();
	test_cosi();
	test_tang();
	test_expn();
	test_operators();
	test_floor();
	test_ceil();
	test_randint();
	test_randchoice();
	test_count();
	test_count_nonzero();
	test_cumsum();
	test_flip();
	test_logr();
	test_logspace();
	test_cumprod();
	test_lstrip();
	test_rstrip();
	test_norm();
	test_radians();
	test_degrees();
	test_sqroot();
	test_power();
	test_min();
	test_max();
	test_dot();
	test_hstack();
	test_sort();
}



