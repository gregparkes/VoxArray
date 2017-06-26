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

namespace tests {

	static void test_zeros()
	{
		std::cout << "Start Zeros" << std::endl;
		Numpy arr = numpy::zeros(16);
		assert(arr.n == 16);
		for (int i = 0; i < 16; i++)
		{
			assert(arr.data[i] == 0.0);
		}

		std::cout << "Test_Zeros :: Passed" << std::endl;
	}

	static void test_ones()
	{
		std::cout << "Start Ones" << std::endl;
		Numpy arr = numpy::ones(16);
		assert(arr.n == 16);
		for (int i = 0; i < 16; i++)
		{
			assert(arr.data[i] == 1.0);
		}

		std::cout << "Test_Ones :: Passed" << std::endl;
	}

	static void test_empty()
	{
		std::cout << "Start Empty" << std::endl;
		Numpy arr = numpy::empty(16);
		assert(arr.n == 16);
		//std::cout << arr.str() << std::endl;
		for (int i = 0; i < 16; i++)
		{
			double val = arr.data[i];
			val *= 2;
		}
		Numpy x = numpy::Vector(4.0, 5.0, 2.0);
		Numpy y = numpy::Vector(2.0, 7.0, 5.0, 8.0);

		std::cout << "Test_Empty :: Passed" << std::endl;
	}

	static void test_fill()
	{
		std::cout << "Start Fill" << std::endl;
		Numpy arr = numpy::fill(16, 15.0);
		assert(arr.n == 16);
		for (int i = 0; i < 16; i++)
		{
			assert(arr.data[i] == 15.0);
		}
		std::cout << "Test_Fill :: Passed" << std::endl;
	}

	static void test_str()
	{
		std::cout << "Start Str" << std::endl;
		Numpy x = numpy::zeros(6);
		std::cout << numpy::str(x) << std::endl;
		Numpy y = numpy::rand(6);
		std::cout << numpy::str(y) << std::endl;
		std::cout << y.str() << std::endl;

		std::cout << "Test_Str :: Passed" << std::endl;
	}

	static void test_array()
	{
		std::cout << "Start Array" << std::endl;
		Numpy arr = numpy::array("0.0, 3.42, 5.4, 5.45, 2.45, 9.65");
		//std::cout << arr.str() << std::endl;
		assert(arr.n == 6);
		for (int i = 0; i < 6; i++)
		{
			assert(arr.data[i] == arr[i]);
		}

		std::cout << "Test_Array :: Passed" << std::endl;
	}

	static void test_copy()
	{
		std::cout << "Start Copy" << std::endl;
		Numpy x = numpy::ones(16);
		Numpy y = numpy::copy(x);
		assert(y.n == x.n);
		for (int i = 0; i < 16; i++)
		{
			assert(x.data[i] == y.data[i]);
		}
		Numpy z = x.copy();
		for (int i = 0; i < 16; i++)
		{
			assert(x.data[i] == z.data[i]);
		}

		std::cout << "Test_Copy :: Passed" << std::endl;
	}

	static void test_empty_like()
	{
		std::cout << "Start Empty_Like" << std::endl;
		Numpy x = numpy::zeros(16);
		Numpy y = numpy::empty_like(x);
		assert(y.n == x.n);

		std::cout << "Test_Empty_Like :: Passed" << std::endl;
	}

	static void test_zeros_like()
	{
		std::cout << "Start Zeros_Like" << std::endl;
		Numpy x = numpy::zeros(16);
		Numpy y = numpy::zeros_like(x);
		assert(y.n == x.n);
		//std::cout << x.str() << std::endl;
		for (int i = 0; i < 16; i++)
		{
			assert(y.data[i] - 0.0 < 1e-7);
		}

		std::cout << "Test_Zeros_Like :: Passed" << std::endl;
	}

	static void test_ones_like()
	{
		std::cout << "Start Ones_Like" << std::endl;
		Numpy x = numpy::zeros(16);
		Numpy y = numpy::ones_like(x);
		assert(y.n == x.n);
		for (int i = 0; i < 16; i++)
		{
			assert(y.data[i] == 1.0);
		}

		std::cout << "Test_Ones_Like :: Passed" << std::endl;
	}

	static void test_rand()
	{
		std::cout << "Start Rand" << std::endl;
		Numpy x = numpy::rand(16);
		assert(x.n == 16);
		for (int i = 0; i < 16; i++)
		{
			assert(x.data[i] > 0.0 && x.data[i] < 1.0);
		}

		std::cout << "Test_Rand :: Passed" << std::endl;
	}

	static void test_randn()
	{
		std::cout << "Start Randn" << std::endl;
		Numpy x = numpy::randn(500);
		assert(x.n == 500);
		//calculate mean and hope it's close to +- 0.2 around 0.
		double count = 0.0;
		for (int i = 0; i < 500; i++)
		{
			count += x.data[i];
		}
		assert(count / 500 > -0.5 && count / 500 < 0.5);

		std::cout << "Test_Randn :: Passed" << std::endl;
	}

	static void test_arange()
	{
		std::cout << "Start Arange" << std::endl;
		Numpy x = numpy::arange(0.0, 1.0, 0.1);
		assert(x.n == 11);
		for (int i = 0; i < 11; i++)
		{
			double y = i * 0.1;
			//printf("%lf %lf %lf %lf\n", x.data[i], y, x.data[i] - y, 1e-5);
			assert(x.data[i] - y <= 1e-14);
		}

		std::cout << "Test_Arange :: Passed" << std::endl;
	}

	static void test_linspace()
	{
		std::cout << "Start Linspace" << std::endl;
		Numpy x = numpy::linspace(0.0, 1.0, 11);
		assert(x.n == 11);
		for (int i = 0; i < 11; i++)
		{
			assert(x.data[i] - i <= 1e-14);
		}
		Numpy y = numpy::linspace(0.0, 5.0, 6);
		assert(y.n == 6);
		for (int i = 0; i < 6; i++)
		{
			assert(y.data[i] - i <= 1e-14);
		}
		Numpy z = numpy::linspace(0.1, 0.5, 5);
		//std::cout << z.str() << std::endl;
		Numpy ab = numpy::linspace(0.2, 0.6, 5);
		Numpy ac = numpy::linspace(0.3, 0.7, 5);
		Numpy ad = numpy::linspace(0.12, 0.2, 10);
		//std::cout << ab.str() << std::endl << ac.str() << std::endl << ad.str() << std::endl;

		std::cout << "Test_Linspace :: Passed" << std::endl;
	}

	static void test_abs()
	{
		std::cout << "Start Abs" << std::endl;
		Numpy x = numpy::array("5, -1.5, -3.2, 6.5, 9.8, -0.76");
		Numpy y = numpy::abs(x);
		assert(y.n == 6);
		for (int i = 0; i < 6; i++)
		{
			assert(y.data[i] >= 0);
		}
		x.abs();
		for (int i = 0; i < 6; i++)
		{
			assert(x.data[i] >= 0);
		}

		std::cout << "Test_Abs :: Passed" << std::endl;
	}

	static void test_sum()
	{
		std::cout << "Start Sum" << std::endl;
		Numpy x = numpy::ones(12);
		double ans = numpy::sum(x);
		assert(ans == 12.0);
		double ans2 = x.sum();
		assert(ans2 == 12.0);

		std::cout << "Test_Sum :: Passed" << std::endl;
	}

	static void test_all()
	{
		std::cout << "Start All" << std::endl;
		Numpy x = numpy::ones(6);
		assert(numpy::all(x));
		assert(x.all());
		Numpy y = numpy::zeros(6);
		assert(!numpy::all(y));
		assert(!y.all());

		std::cout << "Test_All :: Passed" << std::endl;
	}

	static void test_any()
	{
		std::cout << "Start Any" << std::endl;
		Numpy x = numpy::ones(6);
		assert(numpy::any(x));
		assert(x.any());
		Numpy y = numpy::zeros(6);
		assert(!numpy::any(y));
		assert(!y.any());
		Numpy z = numpy::rand(6);
		assert(numpy::any(z));
		assert(z.any());

		std::cout << "Test_Any :: Passed" << std::endl;
	}

	static void test_mean()
	{
		std::cout << "Start Mean" << std::endl;
		Numpy x = numpy::ones(6);
		assert(numpy::mean(x) - 1.0 <= 1e-14);
		assert(x.mean() - 1.0 <= 1e-14);
		Numpy y = numpy::array("0.0, 0.25, 0.5, 0.75, 1.0");
		assert(numpy::mean(y) - 0.5 <= 1e-14);
		assert(y.mean() - 0.5 <= 1e-14);

		std::cout << "Test_Mean :: Passed" << std::endl;
	}

	static void test_std()
	{
		std::cout << "Start Std" << std::endl;
		Numpy x = numpy::array("6.0, 2.0, 3.0, 1.0");
		assert(numpy::std(x) - 1.87 <= 1e-7);
		assert(x.std() - 1.87 <= 1e-7);

		std::cout << "Test_Std :: Passed" << std::endl;
	}

	static void test_var()
	{
		std::cout << "Start Var" << std::endl;
		Numpy x = numpy::array("7.0,6.0,8.0,4.0,2.0,7.0,6.0,7.0,6.0,5.0");
		assert((numpy::var(x) - 121.7) <= 1e-14);
		assert((x.var() - 121.7) <= 1e-14);

		std::cout << "Test_Var :: Passed" << std::endl;
	}

	static void test_argmin()
	{
		std::cout << "Start Argmin" << std::endl;
		Numpy df = numpy::array("3.0, 6.0, -2.0, 5.0, -4.0, 0.65");
		//std::cout << df.str() << std::endl;
		//printf("%d\n",numpy::argmin(df));
		assert(numpy::argmin(df) == 4);
		assert(df.argmin() == 4);

		std::cout << "Test_Argmin :: Passed" << std::endl;
	}

	static void test_argmax()
	{
		std::cout << "Start Argmax" << std::endl;
		Numpy x = numpy::array("3.0, 6.0, -2.0, 5.0, -4.0");
		assert(numpy::argmax(x) == 1);
		assert(x.argmax() == 1);

		std::cout << "Test_Argmax :: Passed" << std::endl;
	}

	static void test_sine()
	{
		std::cout << "Start Sine" << std::endl;
		Numpy x = numpy::array("0.0, 1.570796, 3.14159, 4.712388");
		Numpy y = numpy::sin(x);
		assert(y.data[0] == 0.0);
		assert(y.data[1] - 1.0 < 1e-5);
		assert(y.data[2] - 0.0 < 1e-5);
		assert(y.data[3] + 1.0 < 1e-5);
		// x.sine() doesnt copy, changes the values at reference.
		x.sin();
		assert(x.data[0] == 0.0);
		assert(x.data[1] - 1.0 < 1e-5);
		assert(x.data[2] - 0.0 < 1e-5);
		assert(x.data[3] + 1.0 < 1e-5);

		std::cout << "Test_Sin :: Passed" << std::endl;
	}

	static void test_cosi()
	{
		std::cout << "Start Cosine" << std::endl;
		Numpy x = numpy::array("0.0, 1.570796, 3.14159, 4.712388");
		Numpy y = numpy::cos(x);
		assert(y.data[0] == 1.0);
		assert(y.data[1] - 0.0 < 1e-5);
		assert(y.data[2] + 1.0 < 1e-5);
		assert(y.data[3] - 0.0 < 1e-5);
		// unlike numpy::cosi(x), x.cosi() doesnt copy and changes at reference.
		x.cos();
		assert(x.data[0] == 1.0);
		assert(x.data[1] - 0.0 < 1e-5);
		assert(x.data[2] + 1.0 < 1e-5);
		assert(x.data[3] - 0.0 < 1e-5);

		std::cout << "Test_Cos :: Passed" << std::endl;
	}

	static void test_tang()
	{
		std::cout << "Start Tan" << std::endl;
		Numpy x = numpy::array("0.0, 1.570796, 3.14159, 4.712388");
		Numpy y = numpy::tan(x);
		for (int i = 0; i < 4; i++)
		{
			// printf("%f %f\n", x[i], y[i]);
		}
		x.tan();

		std::cout << "Test_Tan :: Passed" << std::endl;
	}

	static void test_expn()
	{
		std::cout << "Start Exp" << std::endl;
		Numpy x = numpy::array("0.0, 1.0");
		Numpy y = numpy::exp(x);
		assert(y.data[0] - 1.0 < 1e-14);
		assert(y.data[1] - 2.71828 < 1e-5);
		x.exp();
		assert(x.data[0] - 1.0 < 1e-14);
		assert(x.data[1] - 2.71828 < 1e-5);

		std::cout << "Test_Exp :: Passed" << std::endl;
	}

	static void test_operators()
	{
		std::cout << "Start Operators" << std::endl;
		Numpy x = numpy::ones(10);
		Numpy y = x.copy();
		// check ==
		assert(x == y);
		Numpy z = numpy::zeros(10);
		// check !=
		assert(x != z);
		// check []
		y[6] = 5.0;
		y[3] = 3.0;
		// changing y
		assert(x != y);
		// check []
		assert(x[0] == 1.0);
		assert(y[6] == 5.0);
		// check +=
		x += y;
		assert(x[1] == 2.0);
		assert(y[6] == 5.0);
		x += 4.0;
		assert(x[2] == 6.0);
		// check -=
		x -= y;
		assert(x[1] == 5.0);
		x -= 4.0;
		assert(x[0] == 1.0);
		// check *=
		x *= z;
		assert(x[0] == 0.0);
		x += 1.0;
		x *= 4.0;
		assert(x[1] == 4.0);
		// check /=
		x /= y;
		assert(x[1] == 4.0);
		x /= 2.0;
		assert(x[2] == 2.0);
		//std::cout << x.str() << std::endl;

		std::cout << "Test_Operators :: Passed" << std::endl;
	}

	static void test_floor()
	{
		std::cout << "Start Floor" << std::endl;
		Numpy xz = numpy::randn(10);
		Numpy yz = numpy::floor(xz);
		//std::cout << xz.str() << std::endl;
		//std::cout << yz.str() << std::endl;
//		for (int i = 0; i < 10; i++)
//		{
//			printf("%f, ", yz[i]);
//		}

		std::cout << "Test_Floor :: Passed" << std::endl;
	}

	static void test_ceil()
	{
		std::cout << "Start Ceil" << std::endl;
		Numpy x = numpy::randint(10, 15);
		Numpy y = numpy::ceil(x);
		//std::cout << x.str() << std::endl;
		//std::cout << y.str() << std::endl;
//		for (int i = 0; i < 10; i++)
//		{
//			printf("%f, ", y[i]);
//		}

		std::cout << "Test_Ceil :: Passed" << std::endl;
	}

	static void test_randint()
	{
		std::cout << "Start Randint" << std::endl;
		Numpy x = numpy::randint(15, 10);
		//std::cout << x.str() << std::endl;
		for (int i = 0; i < 15; i++)
		{
			assert(x[i] <= 10.0);
		}

		std::cout << "Test_Randint :: Passed" << std::endl;
	}

	static void test_randchoice()
	{
		std::cout << "Start Randchoice" << std::endl;
		Numpy x = numpy::randchoice(15, "-1.0, 0.0, 1.0");
		assert(x.n == 15);
		//std::cout << x.str() << std::endl;
		for (int i = 0; i < 15; i++)
		{
			assert(x[i] == -1.0 || x[i] == 0.0 || x[i] == 1.0);
		}

		std::cout << "Test_Randchoice :: Passed" << std::endl;
	}

	static void test_count()
	{
		std::cout << "Start Count" << std::endl;
		Numpy x = numpy::ones(10);
		assert(numpy::count(x, 1.0) == 10);

		std::cout << "Test_Count :: Passed" << std::endl;
	}

	static void test_count_nonzero()
	{
		std::cout << "Start Count_Nonzero" << std::endl;
		Numpy x = numpy::zeros(10);
		assert(numpy::count_nonzero(x) == 0);
		Numpy y = numpy::ones(10);
		assert(numpy::count_nonzero(y) == 10);

		std::cout << "Test_Count_Nonzero :: Passed" << std::endl;
	}

	static void test_cumsum()
	{
		std::cout << "Start Cumsum" << std::endl;
		Numpy x = numpy::linspace(0.0, 3.0, 4);
		Numpy y = numpy::cumsum(x);
		assert(y.data[0] == 0.0);
		assert(y.data[1] == 1.0);
		assert(y.data[2] == 3.0);
		assert(y.data[3] == 6.0);

		std::cout << "Test_Cumsum :: Passed" << std::endl;
	}

	static void test_flip()
	{
		std::cout << "Start Flip" << std::endl;
		Numpy x = numpy::linspace(0.0, 4.0, 5);
		Numpy y = numpy::flip(x);
		for (int i = 0; i < 5; i++)
		{
			assert(y.data[i] == 4-i);
		}

		std::cout << "Test_Flip :: Passed" << std::endl;
	}

	static void test_logr()
	{
		std::cout << "Start Log" << std::endl;
		Numpy x = numpy::array("0.0, 1.0");
		Numpy y = numpy::log(x);
		assert(y.data[0] - 1.0 < 1e-14);
		assert(y.data[1] - 2.71828 < 1e-5);
//		x.logr();
//		assert(x.data[0] - 1.0 < 1e-14);
//		assert(x.data[1] - 2.71828 < 1e-5);

		std::cout << "Test_Logr :: Passed" << std::endl;
	}

	static void test_logspace()
	{
		std::cout << "Start Logspace" << std::endl;
		Numpy x = numpy::linspace(0.0, 1.0, 11);
		Numpy y = numpy::logspace(0.0, 1.0, 11);
		for (int i = 0; i < 11; i++)
		{
			assert(y.data[i] == pow(10, x[i]));
		}
		//std::cout << x.str() << std::endl << y.str() << std::endl;

		std::cout << "Test_Logspace :: Passed" << std::endl;
	}

	static void test_cumprod()
	{
		std::cout << "Start Cumprod" << std::endl;
		// to complete
		Numpy x = numpy::linspace(1.0, 3.0, 3);
		Numpy y = numpy::cumprod(x);
		//std::cout << y.str() << std::endl;
		assert(y.data[0] == 1.0);
		assert(y.data[1] == 2.0);
		assert(y.data[2] == 6.0);

		std::cout << "Test_Cumprod :: Passed" << std::endl;
	}

	static void test_lstrip()
	{
		std::cout << "Start Lstrip" << std::endl;
		Numpy x = numpy::linspace(0.0, 10.0, 11);
		Numpy y = numpy::lstrip(x, 4);
		assert(y.n == 7);
		//std::cout << y.str() << std::endl;
		for (int i = 4; i < 11; i++)
		{
			assert(y.data[i-4] == i);
		}

		std::cout << "Test_Lstrip :: Passed" << std::endl;
	}

	static void test_rstrip()
	{
		std::cout << "Start Rstrip" << std::endl;
		Numpy x = numpy::linspace(0.0, 10.0, 11);
		Numpy y = numpy::rstrip(x, 5);
		assert(y.n == 6);
		//std::cout << y.str() << std::endl;
		for (int i = 0; i < 6; i++)
		{
			assert(y.data[i] == i);
		}

		std::cout << "Test_Rstrip :: Passed" << std::endl;
	}

	static void test_hstack()
	{
		std::cout << "Start Vstack" << std::endl;
		Numpy x = numpy::arange(0.0, 1.0, 0.1);
		Numpy y = numpy::randn(5);
		Numpy z = numpy::vstack(x, y);
		//std::cout << z.str() << std::endl;
		assert(z.n == x.n + y.n);

		std::cout << "Test_Vstack :: Passed" << std::endl;
	}

	static void test_norm()
	{
		std::cout << "Start Norm" << std::endl;
		Numpy x = numpy::linspace(1.0, 3.0, 3);
		//std::cout << numpy::norm(x, 2) << std::endl;
		assert(numpy::norm(x, _INF_NORM) - 3.0 < 1e-14);
		assert(numpy::norm(x, _ONE_NORM) == 6.0);
		assert(numpy::norm(x, _TWO_NORM) == 1.0);

		std::cout << "Test_Norm :: Passed" << std::endl;
	}

	static void test_adjacsum()
	{
		std::cout << "Start Adjacsum" << std::endl;
		Numpy x = numpy::linspace(0.0, 1.0, 11);
		Numpy y = numpy::adjacsum(x);
		//std::cout << y.str() << std::endl;

		std::cout << "Test_Adjacsum :: Passed" << std::endl;
	}

	static void test_radians()
	{
		std::cout << "Start Radians" << std::endl;
		Numpy x = numpy::array("0.0, 90.0, 180.0, 360.0");
		Numpy y = numpy::radians(x);
		//std::cout << y.str() << std::endl;
		assert(y[0] == x[0]);
		assert(y[1] == M_PI / 2);
		assert(y[2] == M_PI);
		assert(y[3] == M_PI * 2);

		std::cout << "Test_Radians :: Passed" << std::endl;
	}

	static void test_degrees()
	{
		std::cout << "Start Degrees" << std::endl;
		Numpy x = numpy::array("0.0, 1.570796, 3.14159, 4.712388");
		Numpy y = numpy::degrees(x);
		//std::cout << y.str() << std::endl;
		assert(y[0] - x[0] < 2);
		assert(y[1] - 90.0 < 2);
		assert(y[2] - 180.0 < 2);
		assert(y[3] - 360.0 < 2);

		std::cout << "Test_Degrees :: Passed" << std::endl;
	}

	static void test_sqroot()
	{
		std::cout << "Start Sqrt" << std::endl;
		Numpy x = numpy::array("4.0, 9.0, 16.0, 25.0");
		Numpy y = numpy::sqrt(x);
		for (int i = 0; i < 4; i++)
		{
			assert(y[i] == i+2);
		}

		std::cout << "Test_Sqroot :: Passed" << std::endl;
	}

	static void test_normal()
	{
		std::cout << "Start Normal" << std::endl;
		Numpy x = numpy::normal(30, 0.0, 1.0);
		//std::cout << x.str() << std::endl << x.mean() << std::endl;

		std::cout << "Test_Normal :: Passed" << std::endl;
	}

	static void test_power()
	{
		std::cout << "Start Power" << std::endl;
		Numpy x = numpy::linspace(1.0, 5.0, 5);
		Numpy y = numpy::power(x, 2);
		for (int i = 0; i < 5; i++)
		{
			assert(y[i] == x[i] * x[i]);
		}

		std::cout << "Test_Power :: Passed" << std::endl;
	}

	static void test_min()
	{
		std::cout << "Start Min" << std::endl;
		Numpy x = numpy::linspace(1.0, 10.0, 11);
		assert(numpy::min(x) == 1.0);
		Numpy y = numpy::rand(5);
		int min = y[0];
		for (int i = 0; i < 5; i++)
		{
			if (y[i] < min)
			{
				min = y[i];
			}
		}
		assert(min - numpy::min(y) <= 1e-14);

		std::cout << "Test_Min :: Passed" << std::endl;
	}

	static void test_max()
	{
		std::cout << "Start Max" << std::endl;
		Numpy x = numpy::linspace(1.0, 10.0, 11);
		assert(numpy::max(x) == 10.0);
		Numpy y = numpy::rand(5);
		int max = y[0];
		for (int i = 0; i < 5; i++)
		{
			if (y[i] > max)
			{
				max = y[i];
			}
		}
		assert(max - numpy::max(y) <= 1e-14);

		std::cout << "Test_Max :: Passed" << std::endl;
	}

	static void test_dot()
	{
		std::cout << "Start Dot" << std::endl;
		Numpy x = numpy::fill(4, 3.0);
		Numpy y = numpy::fill(4, 2.0);
		double z = numpy::dot(x, y);
		assert(z == 24.0);
		double z2 = x.dot(y);
		assert(z2 == 24.0);
		//std::cout << z << std::endl;

		std::cout << "Test_Dot :: Passed" << std::endl;
	}

	static void test_indexing()
	{
		std::cout << "Start Indexing" << std::endl;
		Numpy x = numpy::linspace(0.0, 5.0, 6);
		//std::cout << numpy::str(x) << std::endl;
		assert(x[3] == 3.0);
		Numpy y = x.select(3);
		for (int i = 0; i < 3; i++)
		{
			assert(x[i] - 3.0 <= 1e-14);
		}
		Numpy z = x.select($, $, -1);
		//std::cout << numpy::str(z) << std::endl;
		Numpy ab = x.select($, 2);
		//std::cout << ab.str() << std::endl;
		Numpy ac = x.select(2, $);
		//std::cout << numpy::str(ac) << std::endl;
		Numpy ad = x.select(2, $, 2);
		//std::cout << numpy::str(ad) << std::endl;

		std::cout << "Test_Indexing :: Passed" << std::endl;
	}

	static void test_sort()
	{
		std::cout << "Start Sort" << std::endl;

		Numpy x = numpy::array("3.0, 2.0, 7.0, 5.0, 15.0, 9.0, 11.0");
		Numpy y = numpy::sort(x);
		//std::cout << x.str() << std::endl;
		//std::cout << y.str() << std::endl;
		for (int i = 0; i < 6; i++)
		{
			assert(y.data[i] <= y.data[i+1]);
		}

		std::cout << "Test_Sort :: Passed" << std::endl;
	}

	static void test_unique()
	{
		std::cout << "Start Unique" << std::endl;
		Numpy x = numpy::array("6.0, 7.0, 5.0, 7.0, 3.0, 6.0, 9.0, 4.0, 5.0, 6.0");
		Numpy ans = numpy::array("6.0, 7.0, 5.0, 3.0, 9.0, 4.0");
		Numpy counts = numpy::array("3.0, 2.0, 2.0, 1.0, 1.0, 1.0");
		Mat uniques = numpy::unique(x);
		//std::cout << uniques.str() << std::endl << uniques.vectors[0]->str() << std::endl;
		for (uint i = 0; i < uniques.vectors[0]->n; i++)
		{
			assert(ans.data[i] == uniques.vectors[0]->data[i]);
			assert(counts.data[i] == uniques.vectors[1]->data[i]);
		}
		std::cout << "Test_Unique :: Passed" << std::endl;
	}




	// end of tests here
}


static void call_all_tests()
{
	//The tests here
	using namespace tests;
	test_empty();
	test_zeros();
	test_ones();
	test_fill();
	test_str();
	test_array();
	test_copy();
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
	test_adjacsum();
	test_radians();
	test_degrees();
	test_sqroot();
	test_normal();
	test_power();
	test_min();
	test_max();
	test_dot();
	test_unique();
	test_hstack();
	//test_indexing();
	test_sort();
}



