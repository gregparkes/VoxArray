/*
 * numstatic.c
 *
 *  Created on: 15 Feb 2017
 *      Author: Greg
 */

#ifndef __NUMPY_static_C_F__
#define __NUMPY_static_C_F__

#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <omp.h>
#include <iostream>

#define __OMP_OPT_VALUE__ 50000
// decimal constant
#define FLT_EPSILON 1.1920929E-07F 

static inline double _absolute_(double value)
{
	return ((value < 0) ? -value : value);
}

static inline int _char_to_int_(char c)
{
	return (int) (c - '0');
}

static inline double _square_root_(double value)
{
	return sqrt(value);
}

static inline double _c_power_(double base, double exponent)
{
	return pow(base, exponent);
}

static inline double _natural_log_(double v)
{
	return log(v);
}

static inline double _sine_(double v)
{
	return sin(v);
}

static inline double _cosine_(double v)
{
	return cos(v);
}

static inline bool AlmostEqualRelativeAndAbs(double a, double b, double maxdiff,
		 double maxreldiff = FLT_EPSILON)
{
	// check if the numbers are really close.
	double diff = fabs(a - b);
	if (diff <= maxdiff)
	{
		return true;
	}

	a = fabs(a);
	b = fabs(b);
	double largest = (b > a) ? b : a;

	if (diff <= largest * maxreldiff)
	{
		return true;
	}
	return false;
}

#define CMP(x, y) AlmostEqualRelativeAndAbs(x, y, 0.005)


static double correct_degrees(double degrees)
{
	while (degrees > 360.0)
	{
		degrees -= 360.0;
	}
	while (degrees < -360.0)
	{
		degrees += 360.0;
	}
	return degrees;
}

static inline double RAD2DEG(double radians)
{
	return (correct_degrees(radians * 57.295754));
}

static inline double DEG2RAD(double degrees)
{
	return (correct_degrees(degrees) * 0.0174533);
}

/*
 * This code is taken from https://rosettacode.org/wiki/Evaluate_binomial_coefficients
 * With permission.
 */
static double _factorial_(double v)
{
	double result = v;
	double result_next;
	double pc = v;
	do
	{
		result_next = result*(pc-1);
		result = result_next;
		pc--;
	} while(pc>2);
	v = result;
	return v;
}

/*
 * This code is taken from https://rosettacode.org/wiki/Evaluate_binomial_coefficients
 * With permission.
 */
static double _binomial_coefficient_(double v, double p)
{
	if (v == p)
	{
		return 1.0;
	} else if (p == 1)
	{
		return v;
	} else if (p > v)
	{
		return 0.0;
	} else {
		return (_factorial_(v))/(_factorial_(p)*_factorial_((v - p)));
	}
}

static long _poisson_coefficient_(double lam)
{
	srand48(time(NULL));
	double limit = exp(-lam);
	double prod = drand48();
	long n;
	for (n = 0; prod >= limit; n++)
	{
		prod *= drand48();
	}
	return n;
}

/**
 Basic method to truncate a double to some significant figure.

 e.g 3.14156, 2.s.f = 3.14000
 */
static double _truncate_doub_(double value, int sigfig)
{
	int exponent = pow(10, sigfig);
	double t1 = value * exponent;
	int t_int = (int) t1; //deliberately cast to int to eliminate post.point dat
	double result = (double) (t_int) / exponent;
	return result;
}

static double* _create_empty_(unsigned int n)
{
	if (n == 0)
	{
		return 0;
	}
	double* my_empty = (double*) malloc(n * sizeof(double));
	if (my_empty == NULL)
	{
		printf("Error! Unable to allocate memory in _create_empty_(double*, unsigned int)");
		return 0;
	}
	return my_empty;
}

static int _destroy_array_(double *arr)
{
	if (arr != NULL)
	{
		free(arr);
		return 1;
	}
	return 0;
}

static int _fill_array_(double *arr, unsigned int n, double val)
{
	if (n == 0 || arr == 0)
	{
		return 0;
	}
#ifdef _OPENMP
	#pragma omp parallel for if(n>__OMP_OPT_VALUE__) schedule(static)
#endif
	for (unsigned int i = 0; i < n; i++)
	{
		arr[i] = val;
	}
	return 1;
}

static int _flip_array_(double *arr, unsigned int n)
{
	if (n == 0 || arr == 0)
	{
		return 0;
	}
	for (unsigned int i = 0; i < (int) (n / 2); i++)
	{
		//swap
		double temp = arr[i];
		arr[i] = arr[n-1-i];
		arr[n-1-i] = temp;
	}
	return 1;
}

static int _clip_array_(double *arr, unsigned int n, double a_min, double a_max)
{
	if (n == 0 || arr == 0)
	{
		return 0;
	}
#ifdef _OPENMP
	#pragma omp parallel for if(n>__OMP_OPT_VALUE__) schedule(static)
#endif
	for (unsigned int i = 0; i < n; i++)
	{
		if (arr[i] < a_min)
		{
			arr[i] = a_min;
		}
		else if (arr[i] > a_max)
		{
			arr[i] = a_max;
		}
	}
	return 1;
}

static int _absolute_array_(double *arr, unsigned int n)
{
	if (n == 0 || arr == 0)
	{
		return 0;
	}
#ifdef _OPENMP
	#pragma omp parallel for if(n>__OMP_OPT_VALUE__) schedule(static)
#endif
	for (unsigned int i = 0; i < n; i++)
	{
		arr[i] = _absolute_(arr[i]);
	}
	return 1;
}

static int _count_array_(double *arr, unsigned int n, double value)
{
	if (n == 0 || arr == 0)
	{
		return 0;
	}
	int count = 0;
#ifdef _OPENMP
	#pragma omp parallel for if(n>__OMP_OPT_VALUE__) schedule(static) reduction(+:count)
#endif
	for (unsigned int i = 0; i < n; i++)
	{
		if (arr[i] == value)
		{
			count++;
		}
	}
	return count;
}

static int _matrix_rowwise_count_(double *arr, unsigned int nvec, unsigned int ncol, unsigned int rowidx,
		double value)
{
	if (nvec == 0 || ncol == 0 || arr == 0)
	{
		return 0;
	}
	int count = 0;
#ifdef _OPENMP
	#pragma omp parallel for if(nvec>__OMP_OPT_VALUE__) schedule(static) reduction(+:count)
#endif
	for (unsigned int colidx = 0; colidx < nvec; colidx++)
	{
		if (arr[rowidx+colidx*ncol] == value)
		{
			count++;
		}
	}
	return count;
}

static int _count_nonzero_array_(double *arr, unsigned int n)
{
	int count = 0;
#ifdef _OPENMP
	#pragma omp parallel for if(n>__OMP_OPT_VALUE__) schedule(static) reduction(+:count)
#endif
	for (unsigned int i = 0; i < n; i++)
	{
		if (arr[i] != 0.0)
		{
			count++;
		}
	}
	return count;
}

static int _nonzero_array_(double *copy, double *orig, unsigned int orig_size)
{
	if (copy == 0 || orig == 0)
	{
		return 0;
	}
	unsigned int idx = 0;
	for (unsigned int i = 0; i < orig_size; i++)
	{
		if (orig[i] != 0.0)
		{
			copy[idx++] = orig[i];
		}
	}
	return 1;
}

static int _matrix_rowwise_count_nonzero_(double *arr, unsigned int nvec, unsigned int ncol,
		unsigned int rowidx)
{
	if (nvec == 0 || ncol == 0 || arr == 0)
	{
		return 0;
	}
	int count = 0;
#ifdef _OPENMP
	#pragma omp parallel for if(nvec>__OMP_OPT_VALUE__) schedule(static) reduction(+:count)
#endif
	for (unsigned int colidx = 0; colidx < nvec; colidx++)
	{
		if (arr[rowidx+colidx*ncol] != 0.0)
		{
			count++;
		}
	}
	return count;
}

static inline unsigned int _str_length_gen_(double *arr, unsigned int n,
							unsigned int dpoints)
{
	return 2 + ((dpoints+1)*n) + (2*n-2) + 1;
}

static inline unsigned int _n_digits_in_int_(int value)
{
	// determines the number of characters in this integer value
	return floor(log10(abs(value))) + 1;
}

static int _str_representation_(char *out, double *arr, unsigned int n_arr,
								unsigned int dpoints, int if_end_of_string)
{
	// Each entry is going to be 0.5454 so 6 characters long.
	// E.g [4.0000, 3.0000, 2.0000, 7.0000, 3.0000]
	// 2 + ((dpoints+2)*n) + (2*n-2)
	if (dpoints > 13)
	{
		return 0;
	}
	if (out == NULL)
	{
		printf("Out must be filled with empty slots");
		return 0;
	}
	out[0] = '[';
	int offset = 1;
	unsigned int i;
	for (i = 0; i < n_arr-1; i++)
	{
		double arr_f = _truncate_doub_(arr[i], dpoints);
		char output[dpoints+2];
		snprintf(output, dpoints+2, "%f", arr_f);
		unsigned int j;
		for (j = 0; j < dpoints+2; j++)
		{
			out[(offset)+j] = output[j];
		}
		offset += dpoints;
		out[++offset] = ',';
		out[++offset] = ' ';
		offset++;
	}
	double arr_f = _truncate_doub_(arr[i], dpoints);
	char output[dpoints+2];
	snprintf(output, dpoints+2, "%f", arr_f);
	unsigned int j;
	for (j = 0; j < dpoints+2; j++)
	{
		out[(offset)+j] = output[j];
	}
	offset += dpoints;
	out[++offset] = ']';
	if (if_end_of_string) {
		out[++offset] = '\0';
	}
	return 1;
}

static int _str_shape_func_(char* out, unsigned int val1, unsigned int val2,
		unsigned int dig_1, unsigned int dig_2, unsigned int len)
{
	out[0] = '(';
	unsigned int i;
	char intchar[15];
	sprintf(intchar, "%d", val1);
	for (i = 0; i < dig_1; i++)
	{
		out[1+i] = intchar[i];
	}
	out[1+dig_1] = ',';
	char intchar2[15];
	sprintf(intchar2, "%d", val2);
	for (i = 0; i < dig_2; i++)
	{
		out[2+dig_1+i] = intchar2[i];
	}
	out[len-2] = ')';
	out[len-1] = '\0';
	return 1;
}

static int _copy_array_(double *copy, double *orig, unsigned int n)
{
	// orig initialized, copy is empty array.
	if (n == 0 || copy == 0 || orig == 0)
	{
		return 0;
	}
#ifdef _OPENMP
	#pragma omp parallel for if(n>__OMP_OPT_VALUE__) schedule(static)
#endif
		for (unsigned int i = 0; i < n; i++)
		{
			copy[i] = orig[i];
		}
	return 1;
}

static int _matrix_copy_row_(double *copy, double *orig, unsigned int nvec, unsigned int ncol, unsigned int rowidx)
{
	if (ncol == 0 || copy == 0 || orig == 0 || nvec == 0)
	{
		return 0;
	}
#ifdef _OPENMP
	#pragma omp parallel for if(nvec>__OMP_OPT_VALUE__) schedule(static)
#endif
	for (unsigned int colidx = 0; colidx < nvec; colidx++)
	{
		copy[rowidx+colidx*ncol] = orig[rowidx+colidx*ncol];
	}
	return 1;
}

static int _normal_distrib_(double *arr, unsigned int n, double mean,
							double sd)
{
	if (n == 0 || arr == 0)
	{
		return 0;
	}
	srand48(time(NULL));

#ifdef _OPENMP
	#pragma omp parallel for if(n>__OMP_OPT_VALUE__) schedule(static)
#endif
	for (unsigned int i = 0; i < n; i++)
	{
		double x, y, r;
		do
		{
			x = 2.0 * drand48() - 1;
			y = 2.0 * drand48() - 1;
			r = x*x + y*y;
		}
		while (r == 0.0 || r > 1.0);
		{
			double j = x * sqrt(-2.0*log(r)/r);
			arr[i] = j*sd + mean;
		}
	}
	return 1;
}

static int _floor_array_(double *arr, unsigned int n)
{
	if (n == 0 || arr == 0)
	{
		return 0;
	}
#ifdef _OPENMP
	#pragma omp parallel for if(n>__OMP_OPT_VALUE__) schedule(static)
#endif
	for (unsigned int i = 0; i < n; i++)
	{
		double arr_f = _truncate_doub_(arr[i], 0);
		arr[i] = arr_f;
	}
	return 1;
}

static bool _all_true_(double *arr, unsigned int n)
{
	for (unsigned int i = 0; i < n; i++)
	{
		if (!arr[i])
		{
			return false;
		}
	}
	return true;
}

static bool _any_true_(double *arr, unsigned int n)
{
	for (unsigned int i = 0; i < n; i++)
	{
		if (arr[i])
		{
			return true;
		}
	}
	return false;
}

static double _summation_array_(double *arr, unsigned int n)
{
	double total = arr[0];
#ifdef _OPENMP
	#pragma omp parallel for if(n>__OMP_OPT_VALUE__) schedule(static) reduction(+:total)
#endif
	for (unsigned int i = 1; i < n; i++)
	{
		total += arr[i];
	}
	return total;
}

static double _matrix_rowwise_summation_(double *arr, unsigned int nvec, unsigned int ncol, unsigned int rowidx)
{
	double total = arr[rowidx];
#ifdef _OPENMP
	#pragma omp parallel for if(nvec>__OMP_OPT_VALUE__) schedule(static) reduction(+:total)
#endif
	for (unsigned int colidx = 1; colidx < nvec; colidx++)
	{
		total += arr[rowidx+colidx*ncol];
	}
	return total;
}

static double _std_array_(double *arr, unsigned int n)
{
	double m = _summation_array_(arr, n) / n;
	double s = 0.0;
#ifdef _OPENMP
	#pragma omp parallel for if(n>__OMP_OPT_VALUE__) schedule(static) reduction(+:s)
#endif
	for (unsigned int i = 0; i < n; i++)
	{
		s += (arr[i] - m) * (arr[i] - m);
	}
	return _square_root_(s / (n-1));
}

static double _matrix_rowwise_std_(double *arr, unsigned int nvec, unsigned int ncol, unsigned int rowidx)
{
	double m = _matrix_rowwise_summation_(arr, nvec, ncol, rowidx) / nvec;
	double s = 0.0;
#ifdef _OPENMP
	#pragma omp parallel for if(nvec>__OMP_OPT_VALUE__) schedule(static) reduction(+:s)
#endif
	for (unsigned int colidx = 0; colidx < nvec; colidx++)
	{
		s += (arr[rowidx+colidx*ncol] - m) * (arr[rowidx+colidx*ncol] - m);
	}
	return _square_root_(s / (nvec - 1));
}

static double _var_array_(double *arr, unsigned int n)
{
	double m = _summation_array_(arr, n) / n;
	double s = 0.0;
#ifdef _OPENMP
	#pragma omp parallel for if(n>__OMP_OPT_VALUE__) schedule(static) reduction(+:s)
#endif
	for (unsigned int i = 0; i < n; i++)
	{
		s += (arr[i] - m) * (arr[i] - m);
	}
	return (s / (n - 1));
}

static double _matrix_rowwise_var_(double *arr, unsigned int nvec, unsigned int ncol, unsigned int rowidx)
{
	double m = _matrix_rowwise_summation_(arr, nvec, ncol, rowidx) / nvec;
	double s = 0.0;
#ifdef _OPENMP
	#pragma omp parallel for if(nvec>__OMP_OPT_VALUE__) schedule(static) reduction(+:s)
#endif
	for (unsigned int colidx = 0; colidx < nvec; colidx++)
	{
		s += (arr[rowidx+colidx*ncol] - m) * (arr[rowidx+colidx*ncol] - m);
	}
	return s / (nvec - 1);
}

static double _absolute_summation_array_(double *arr, unsigned int n)
{
	double total = _absolute_(arr[0]);
#ifdef _OPENMP
	#pragma omp parallel for if(n>__OMP_OPT_VALUE__) schedule(static) reduction(+:total)
#endif
	for (unsigned int i = 1; i < n; i++)
	{
		total += _absolute_(arr[i]);
	}
	return total;
}

static double _absolute_matrix_rowwise_summation_(double *arr, unsigned int nvec, unsigned int ncol,
													unsigned int rowidx)
{
	double total = _absolute_(arr[rowidx]);
#ifdef _OPENMP
	#pragma omp parallel for if(nvec>__OMP_OPT_VALUE__) schedule(static) reduction(+:total)
#endif
	for (unsigned int colidx = 1; colidx < nvec; colidx++)
	{
		total += _absolute_(arr[rowidx+colidx*nvec]);
	}
	return total;
}

static double _prod_array_(double *arr, unsigned int n)
{
	double prod = arr[0];
#ifdef _OPENMP
	#pragma omp parallel for if(n>__OMP_OPT_VALUE__) schedule(static) reduction(*:prod)
#endif
	for (unsigned int i = 1; i < n; i++)
	{
		prod *= arr[i];
	}
	return prod;
}

static double _matrix_rowwise_prod_(double *arr, unsigned int nvec, unsigned int ncol, unsigned int rowidx)
{
	double total = arr[rowidx];
#ifdef _OPENMP
	#pragma omp parallel for if(nvec>__OMP_OPT_VALUE__) schedule(static) reduction(+:total)
#endif
	for (unsigned int colidx = 1; colidx < nvec; colidx++)
	{
		total *= arr[rowidx+colidx*ncol];
	}
	return total;
}

static int _cumulative_sum_(double *zeros, double *oldarr, unsigned int n)
{
	if (n == 0 || zeros == 0 || oldarr == 0)
	{
		return 0;
	}
	// std::cout << "Getting here" << std::endl;
#ifdef _OPENMP
	#pragma omp parallel for if((n)>__OMP_OPT_VALUE__) schedule(static)
#endif
	for (unsigned int j = 0; j < n; j++)
	{
		for (unsigned int i = j; i > 0; i--)
		{
			zeros[j] = zeros[j] + oldarr[i];
		}
	}
	return 1;
}

static int _cumulative_prod_(double *zeros, double *oldarr, unsigned int n)
{
	if (n == 0 || zeros == 0 || oldarr == 0)
	{
		return 0;
	}
	// std::cout << "Getting here" << std::endl;
#ifdef _OPENMP
	#pragma omp parallel for if((n)>__OMP_OPT_VALUE__) schedule(static)
#endif
	for (unsigned int j = 0; j < n; j++)
	{
		for (unsigned int i = j; i > 0; i--)
		{
			zeros[j] *= oldarr[i];
		}
	}
	return 1;
}

static int _ceil_array_(double *arr, unsigned int n)
{
	if (n == 0 || arr == 0)
	{
		return 0;
	}
#ifdef _OPENMP
	#pragma omp parallel for if(n>__OMP_OPT_VALUE__) schedule(static)
#endif
	for (unsigned int i = 0; i < n; i++)
	{
		if (arr[i] < 0)
		{
			double arr_f = _truncate_doub_(arr[i], 0);
			arr[i] = arr_f - 1;
		} else
		{
			double arr_f = _truncate_doub_(arr[i], 0);
			arr[i] = arr_f + 1;
		}
	}
	return 1;
	// to be implemented
}

static double _min_value_(double *arr, unsigned int n)
{
	double mina = arr[0];
#ifdef _OPENMP
	#pragma omp parallel for default(none) shared(arr,n) if(n>__OMP_OPT_VALUE__) \
	reduction(min:mina)
#endif
	for (unsigned int i = 1; i < n; i++)
	{
		if (arr[i] < mina)
		{
			mina = arr[i];
		}
	}
	return mina;
}

static double _matrix_rowwise_min_value_(double *arr, unsigned int nvec, unsigned int ncol, unsigned int rowidx)
{
	double mina = arr[rowidx];
#ifdef _OPENMP
	#pragma omp parallel for if(nvec>__OMP_OPT_VALUE__) reduction(min:mina)
#endif
	for (unsigned int colidx = 1; colidx < nvec; colidx++)
	{
		if (arr[rowidx+colidx*ncol] > mina)
		{
			mina = arr[rowidx+colidx*ncol];
		}
	}
	return mina;
}

static unsigned int _min_index_(double* arr, unsigned int n)
{
	double minval;
	unsigned int minidx = 0;
	minval = arr[0];
	for (unsigned int i = 0; i < n; i++)
	{
		if (arr[i] < minval)
		{
			minval = arr[i];
			minidx = i;
		}
	}
	return minidx;
}

static unsigned int _matrix_rowwise_min_index_(double *arr, unsigned int nvec, unsigned int ncol,
												unsigned int rowidx)
{
	double minval = arr[rowidx];
	unsigned int minidx = rowidx;
#ifdef _OPENMP
	#pragma omp parallel for if(nvec>__OMP_OPT_VALUE__) schedule(static) reduction(min:minval,minidx)
#endif
	for (unsigned int colidx = 1; colidx < nvec; colidx++)
	{
		if (arr[rowidx+colidx*ncol] < minval)
		{
			minval = arr[rowidx+colidx*ncol];
			minidx = rowidx+colidx*ncol;
		}
	}
	return minidx;
}

static double _max_value_(double *arr, unsigned int n)
{
	double maxa = arr[0];
	unsigned int i;
#ifdef _OPENMP
	#pragma omp parallel for default(none) private(i) shared(arr,n) if(n>__OMP_OPT_VALUE__) \
	reduction(max:maxa)
#endif
	for (i = 1; i < n; i++)
	{
		if (arr[i] > maxa)
		{
			maxa = arr[i];
		}
	}
	return maxa;
}

static double _matrix_rowwise_max_value_(double *arr, unsigned int nvec, unsigned int ncol, unsigned int rowidx)
{
	double maxa = arr[rowidx];
#ifdef _OPENMP
	#pragma omp parallel for if(nvec>__OMP_OPT_VALUE__) reduction(max:maxa)
#endif
	for (unsigned int colidx = 1; colidx < nvec; colidx++)
	{
		if (arr[rowidx+colidx*ncol] > maxa)
		{
			maxa = arr[rowidx+colidx*ncol];
		}
	}
	return maxa;
}

static unsigned int _max_index_(double* arr, unsigned int n)
{
	double maxv = arr[0];
	double maxi = 0;
	unsigned int i;
#ifdef _OPENMP
	#pragma omp parallel for default(none) private(i) shared(arr,n) if(n>__OMP_OPT_VALUE__) \
	reduction(max:maxv,maxi)
#endif
	for (i = 1; i < n; i++)
	{
		if (arr[i] > maxv)
		{
			maxv = arr[i];
			maxi = i;
		}
	}
	return maxi;
}

static unsigned int _matrix_rowwise_max_index_(double *arr, unsigned int nvec, unsigned int ncol,
												unsigned int rowidx)
{
	double maxval = arr[rowidx];
	unsigned int maxidx = rowidx;
#ifdef _OPENMP
	#pragma omp parallel for if(nvec>__OMP_OPT_VALUE__) schedule(static) reduction(min:maxval,maxidx)
#endif
	for (unsigned int colidx = 1; colidx < nvec; colidx++)
	{
		if (arr[rowidx+colidx*ncol] > maxval)
		{
			maxval = arr[rowidx+colidx*ncol];
			maxidx = rowidx+colidx*ncol;
		}
	}
	return maxidx;
}

static int _sine_array_(double *arr, unsigned int n)
{
	if (n == 0 || arr == 0)
	{
		return 0;
	}
	unsigned int i;
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (i = 0; i < n; i++)
	{
		arr[i] = sin(arr[i]);
	}
	return 1;
}

static int _cos_array_(double *arr, unsigned int n)
{
	if (n == 0 || arr == 0)
	{
		return 0;
	}
	unsigned int i;
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (i = 0; i < n; i++)
	{
		arr[i] = cos(arr[i]);
	}
	return 1;
}

static int _tan_array_(double *arr, unsigned int n)
{
	if (n == 0 || arr == 0)
	{
		return 0;
	}
	unsigned int i;
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (i = 0; i < n; i++)
	{
		arr[i] = tan(arr[i]);
	}
	return 1;
}

static int _to_radians_array_(double *arr, unsigned int n)
{
	if (n == 0 || arr == 0)
	{
		return 0;
	}
	unsigned int i;
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (i = 0; i < n; i++)
	{
		arr[i] = DEG2RAD(arr[i]);
	}	
	return 1;
}

static int _to_degrees_array_(double *arr, unsigned int n)
{
	if (n == 0 || arr == 0)
	{
		return 0;
	}
	unsigned int i;
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (i = 0; i < n; i++)
	{
		arr[i] = RAD2DEG(arr[i]);
	}	
	return 1;
}

static int _exp_array_(double *arr, unsigned int n)
{
	if (n == 0 || arr == 0)
	{
		return 0;
	}
	unsigned int i;
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (i = 0; i < n; i++)
	{
		arr[i] = exp(arr[i]);
	}
	return 1;
}

static int _log10_array_(double *arr, unsigned int n)
{
	if (n == 0 || arr == 0)
	{
		return 0;
	}
	unsigned int i;
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (i = 0; i < n; i++)
	{
		arr[i] = log(arr[i]);
	}
	return 1;
}

static int _pow_array_(double *arr, unsigned int n, double exponent)
{
	if (n == 0 || arr == 0 || exponent < 0)
	{
		return 0;
	}
	unsigned int i;
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (i = 0; i < n; i++)
	{
		arr[i] = pow(arr[i], exponent);
	}
	return 1;
}

static int _pow_base_array_(double *arr, unsigned int n, double base)
{
	if (n == 0 || arr == 0 || base < 0)
	{
		return 0;
	}
	unsigned int i;
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (i = 0; i < n; i++)
	{
		arr[i] = pow(base, arr[i]);
	}
	return 1;
}

static double _vector2_norm_(double *arr, unsigned int n, unsigned int p = 2)
{
	_pow_array_(arr, n, p);
	double sumpow = _summation_array_(arr, n);
	return _c_power_(sumpow, 1/p);
}

static void swap(double *a, double *b)
{
	double temp = *a;
	*a = *b;
	*b = temp;
}

static int _partition_(double *arr, const int left, const int right) {
    int mid = left + (right - left) / 2;
    double pivot = arr[mid];
    // move the mid point value to the front.
    swap(&arr[mid],&arr[left]);
    int i = left + 1;
    int j = right;
    while (i <= j) {
        while(i <= j && arr[i] <= pivot) {
            i++;
        }

        while(i <= j && arr[j] > pivot) {
            j--;
        }

        if (i < j) {
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i-1],&arr[left]);
    return i - 1;
}

static void _quicksort_(double* arr, int left, int right)
{
	if (left >= right) {
		return;
	}

	int part = _partition_(arr, left, right);
	_quicksort_(arr, left, part - 1);
	_quicksort_(arr, part + 1, right);
}

// where out == null, in is a string, size is length of in
// input expected - "0.75, 0.54, 0.23, -0.78, -0.001, 0.43" etc
// warning, size WILL be modified to reflect the size of OUT
static double* _parse_string_to_array_(const char *in, int *size)
{
	// we count the number of commas to guess out out array length.
	int countcom = 0;
	int i;
	for (i = 0; i < *size; i++)
	{
		// we'll check that there are only numbers, commas, dots, and the minus sign, else quit
		if (in[i] == ',' || in[i] == ' ' || in[i] == '.' || in[i] == '-' ||
				((int) (in[i] - '0') >= 0 && (int) (in[i] - '0') <= 9) ||
				in[i] == '\n' || in[i] == '\0')
		{
			// extract commas
			if (in[i] == ',')
			{
				countcom++;
			}
		} else {
			return NULL;
		}
	}
	// guess is commas + 1
	int guessn = countcom + 1;
	// allocate memory for double array
	double *arr = _create_empty_(guessn);
	if (arr == NULL)
	{
		return NULL;
	}
	int idx = 0;
	// arr should hold some space now
	for (i = 0; i < guessn; i++)
	{
		//extract number, somehow
		char number[24];
		int j;
		// extract each of the numerical characters and place in a char array
		for (j = 0; j < 24; j++)
		{
			int a = (int) (in[idx] - '0');
			// if the character is 0-9, a dot or a minus...
			if ((a >= 0 && a <= 9) || in[idx] == '.' || in[idx] == '-')
			{
				number[j] = in[idx];
				// else if it's a comma or end of string character...
			} else if (in[idx] == ',' || in[idx] == '\0')
			{

				number[j] = '\0';
				arr[i] = strtod(number, NULL);
				idx++;
				break;
				// else if we have a space...
			} else if (in[idx] == ' ')
			{
				j--;
				//ignore
			} else {
				return 0;
			}
			idx++;
		}
	}
	*size = guessn;
	return arr;
}

static inline double _determinant_2x_(double a, double b, double c, double d)
{
	return (a*d - b*c);
}

static inline double _determinant_3x_(double a, double b, double c, double d, double e, double f,
										double g, double h, double i)
{
	return (a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h);
}

static int _transpose_square_matrix_(double *arr, unsigned int nvec, unsigned int ncol)
{
	// points on the diagonal do not move.
	if (nvec == 0 || arr == 0 || ncol == 0)
	{
		return 0;
	}
	// if we're dealing with a square matrix, we don't have to do any reallocations
	for (unsigned int y = 0; y < nvec-1; y++)
	{
		for (unsigned int x = y+1; x < ncol; x++)
		{
			swap(&arr[x+y*ncol], &arr[x*ncol+y]);
		}
	}
	return 1;
}

static double _vector_dot_array_(double *arr1, double *arr2, unsigned int n)
{
	if (n == 0 || arr1 == 0 || arr2 == 0)
	{
		return 0;
	}
	double la_dot = 0.0;
#ifdef _OPENMP
	#pragma omp parallel for if(n>__OMP_OPT_VALUE__) schedule(static) reduction(+:la_dot)
#endif
	for (unsigned int i = 0; i < n; i++)
	{
		la_dot += arr1[i] * arr2[i];
	}
	return la_dot;
}

static double _matrix_rowwise_dot_(double *matrix, double *vec, unsigned int nvec,
									unsigned int n, unsigned int rowidx)
{
	if (n == 0 || matrix == 0 || vec == 0 || nvec == 0)
	{
		return 0;
	}
	double la_dot = 0.0;
#ifdef _OPENMP
	#pragma omp parallel for if(nvec>__OMP_OPT_VALUE__) schedule(static) reduction(+:la_dot)
#endif
	for (unsigned int i = 0; i < nvec; i++)
	{
		la_dot += matrix[rowidx+i*n] * vec[i];
	}
	return la_dot;
}

static int _swap_row_(double *matrix, unsigned int nvec, unsigned int ncol, unsigned int row1,
		unsigned int row2)
{
	for (unsigned int y = 0; y < nvec; y++)
	{
		swap(&matrix[row1+y*ncol], &matrix[row2+y*ncol]);
	}
	return 1;
}

static int _gaussian_elimination_(double *matrix, unsigned int n)
{
	// must be NxN matrix square
	int row_switch_count = 0;
	// calculate each row's magnitude
	int rowmag[n];
	for (unsigned int i = 0; i < n; i++)
	{
		rowmag[i] = _absolute_(_matrix_rowwise_summation_(matrix+(i*n), n, n, i));
	}
	double min_v = rowmag[0];
	double max_v = rowmag[0];
	int min_i = 0;
	int max_i = 0;
	for (unsigned int i = 1; i < n; i++)
	{
		if (rowmag[i] < min_v)
		{
			min_v = rowmag[i];
			min_i = i;
		}
		if (rowmag[i] < max_v)
		{
			max_v = rowmag[i];
			max_i = i;
		}
	}
	// make sure the smallest row is top
	if (min_i > 0 || max_i == 0)
	{
		_swap_row_(matrix, n, n, min_i, max_i);
		row_switch_count++;
	}
	for (unsigned int k = 0; k < n-1; k++)
	{
		for (unsigned int j = k+1; j < n; j++)
		{
			if (matrix[j+k*n] != 0.0)
			{
				// for every element on the column 0, zero it by selecting the negative divide
				double value = -(matrix[j+k*n] / matrix[k*n]);
				// for every element on selected row, substract by denominator
				for (unsigned int i = k; i < n; i++)
				{
					matrix[j+i*n] += value*matrix[k+i*n];
				}
			}
		}
	}
	return row_switch_count;
}

// given two values, this method will return the multiplicative number associated with the number
// of eigenvalues, e.g
// 5, -4 is phrased as (5-lambda)(-4-lambda) -> -20 + 5lambda - 4lambda + lambda^2.
// this method will return 1lambda - 1. i.e telling you that from these two numbers, expanding the brackets
// gives +1 lambda.
static double _eigenvalue_det_(double a, double b)
{
	return -a - b;
}

/**

---------------------- VALUE-APPLIED MATHEMATICS -----------------------------

*/

static int _add_array_(double *arr, unsigned int n, double value)
{
	if (arr == 0 || n == 0)
	{
		return 0;
	}
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (unsigned int i = 0; i < n; i++)
	{
		arr[i] += value;
	}	
	return 1;
}

static int _sub_array_(double *arr, unsigned int n, double value)
{
	if (arr == 0 || n == 0)
	{
		return 0;
	}
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (unsigned int i = 0; i < n; i++)
	{
		arr[i] -= value;
	}	
	return 1;
}

static int _mult_array_(double *arr, unsigned int n, double value)
{
	if (arr == 0 || n == 0)
	{
		return 0;
	}
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (unsigned int i = 0; i < n; i++)
	{
		arr[i] *= value;
	}	
	return 1;
}

static int _div_array_(double *arr, unsigned int n, double value)
{
	if (arr == 0 || n == 0 || CMP(value, 0.0))
	{
		return 0;
	}
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (unsigned int i = 0; i < n; i++)
	{
		arr[i] /= value;
	}	
	return 1;
}


/****

---------------------- ELEMENT-WISE MATHEMATICS -------------------------------

*/

// adds right to left, then returns
static int _element_add_(double *left, double *right, unsigned int n)
{
	if (n == 0 || left == 0 || right == 0)
	{
		return 0;
	}
	// assumes left and right are same size and DOES NOT CHECK!
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (unsigned int i = 0; i < n; i++)
	{
		left[i] += right[i];
	}
	return 1;
}

static int _element_sub_(double *left, double *right, unsigned int n)
{
	if (n == 0 || left == 0 || right == 0)
	{
		return 0;
	}
	// assumes left and right are same size and DOES NOT CHECK!
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (unsigned int i = 0; i < n; i++)
	{
		left[i] -= right[i];
	}
	return 1;
}

static int _element_mult_(double *left, double *right, unsigned int n)
{
	if (n == 0 || left == 0 || right == 0)
	{
		return 0;
	}
	// assumes left and right are same size and DOES NOT CHECK!
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (unsigned int i = 0; i < n; i++)
	{
		left[i] *= right[i];
	}
	return 1;
}

static int _element_div_(double *left, double *right, unsigned int n)
{
	if (n == 0 || left == 0 || right == 0)
	{
		return 0;
	}
	// assumes left and right are same size and DOES NOT CHECK!
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (unsigned int i = 0; i < n; i++)
	{
		left[i] /= right[i];
	}
	return 1;
}

#endif






