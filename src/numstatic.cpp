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
#include <stdexcept>

/* DEFINITIONS */

#define __OMP_OPT_VALUE__ 50000
#define FLT_EPSILON 1.1920929E-07F
#define INVALID(x) (throw std::invalid_argument(x))
#define RANGE(x) (throw std::range_error(x))
#define INVALID_AXIS() (throw std::invalid_argument("axis must be 0 or 1"))


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

static inline double _uniform_rand_()
{
	return (rand() / (RAND_MAX + 1.0));
}

static inline double _uniform_M_to_N_(double M, double N)
{
	return (M + (rand() / (RAND_MAX / (N-M))));
}

template <typename T> static inline void swap(T *a, T *b)
{
	T temp = *a;
	*a = *b;
	*b = temp;
}

/** Given an integer, determine how many characters long it is (using logarithms of exponents)*/
static unsigned int _integer_char_length_(double a)
{
	return (unsigned int) (log10(fabs(a))) + 1;
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

static bool _is_integer_(double d)
{
	int trun = (int) d;
	// recast and compare
	return CMP(((double) trun), d);
}

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
static double _binomial_coefficient_(double n, double x)
{
	if (n == x)
	{
		return 1.0;
	} else if (x == 1)
	{
		return n;
	} else if (x > n)
	{
		return 0.0;
	} else {
		return (_factorial_(n))/(_factorial_(x)*_factorial_((n - x)));
	}
}

static double _distribution_binomial_(double n, double p, uint pi)
{
	return (_binomial_coefficient_(n, pi) * _c_power_(p, pi) * (_c_power_((1-p), (n-pi))));
}

static long _poisson_generator_knuth_(double lam)
{

	double L = exp(-lam);
	long k = 0;
	double p = 1.0;
	do
	{
		k++;
		p *= _uniform_rand_();
	} while (p > L);
	return k - 1;
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

template <typename T> static T* _create_empty_(unsigned int n)
{
	if (n != 0)
	{
		T* my_empty = (T*) malloc(n * sizeof(T));
		if (my_empty != NULL)
		{
			return my_empty;
		}
		else
		{
			printf("Error! Unable to allocate memory in _create_empty_(unsigned int)");
			return 0;
		}
	}
	else return 0;
}

static int _destroy_array_(void *arr)
{
	if (arr != NULL)
	{
		free(arr);
		return 1;
	}
	return 0;
}

template <typename T> static int _fill_array_(T *arr, unsigned int n, T val)
{
	if (n != 0 && arr != NULL)
	{
		T *idx = &arr[0];
		T *end = idx + n;
		while (idx < end)
		{
			*idx++ = val;
		}
		return 1;
	}
	else return 0;
}

template <typename T> static int _flip_array_(T *arr, unsigned int n)
{
	if (n != 0 && arr != NULL)
	{
		T *p1 = &arr[0];
		T *p2 = p1 + n - 1;
		for( /* void */; p1 < p2; p1++, p2--)
		{
			swap<T>(p1, p2);
		}
		return 1;
	}
	else return 0;
}

static int _clip_array_(double *arr, unsigned int n, double a_min, double a_max)
{
	if (n != 0 && arr != NULL)
	{
		double *p1 = &arr[0];
		double *end = p1 + n;
		for (/* void */; p1 < end; p1++)
		{
			if (*p1 < a_min)
			{
				*p1 = a_min;
			}
			else if (*p1 > a_max)
			{
				*p1 = a_max;
			}
		}
		return 1;
	}
	else return 0;
}

static int _absolute_array_(double *arr, unsigned int n)
{
	if (n != 0 && arr != NULL)
	{
		double *p1 = &arr[0];
		double *end = p1 + n;
		for (/* void */; p1 < end; p1++)
		{
			*p1 = _absolute_(*p1);
		}
		return 1;
	}
	else return 0;
}

static int _count_array_(double *arr, unsigned int n, double value)
{
	if (n != 0 && arr != NULL)
	{
		int count = 0;
		double *p1 = &arr[0];
		double *end = p1 + n;
	/*
	#ifdef _OPENMP
		#pragma omp parallel for if(n>__OMP_OPT_VALUE__) schedule(static) reduction(+:count)
	#endif
	*/
		for (/* void */; p1 < end; p1++)
		{
			if (CMP(*p1, value))
			{
				count++;
			}
		}
		return count;
	}
	else return -1;
}

static int _bincount_array_(double *bins, double *arr, unsigned int nbins, unsigned int narr)
{
	/* Here we assume bins is all zeros, arr is the numbers to count, 
	and n refers to each array size */
	if (nbins != 0 && narr != 0 && bins != NULL && arr != NULL)
	{
		unsigned int i;
		double *p1 = &arr[0];
		double *p1end = p1 + narr;
		for (/* void */; p1 < p1end; p1++)
		{
			// add one to bins where it's cast index == arr[i]
			(*(bins + ((int) *p1)))++;
		}
		return 1;
	}

	else return 0;
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
	if (arr != NULL && n != 0)
	{
		int count = 0;
		double *p1 = &arr[0];
		double *end = p1 + n;
			/*
#ifdef _OPENMP
	#pragma omp parallel for if(n>__OMP_OPT_VALUE__) schedule(static) reduction(+:count)
#endif
*/
		for (/* void */; p1 < end; p1++)
		{
			if (!CMP(*p1, 0.0))
			{
				count++;
			}
		}
		return count;
	}
	else return -1;
}

static int _nonzero_array_(double *copy, double *orig, unsigned int orig_size)
{
	if (copy != NULL && orig != NULL && orig_size != 0)
	{
		double *p1 = &copy[0];
		double *p2 = &orig[0];
		double *pend = p2 + orig_size;
		for (/* void */; p2 < pend; p2++)
		{
			if (!CMP(*p2, 0.0))
			{
				*p1++ = *p2;
			}
		}
		return 1;
	}
	else return 0;
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

static inline unsigned int _str_bool_length_gen_(unsigned int n)
{
	return 2 + (3*(n-1)) + 2;
}

static inline unsigned int _n_digits_in_int_(int value)
{
	// determines the number of characters in this integer value
	return floor(log10(abs(value))) + 1;
}

static unsigned int _str_length_int_gen_(double *arr, unsigned int n)
{
	unsigned int i;
	unsigned int total = 0;
	for (i = 0; i < n; i++)
	{
		total += _integer_char_length_(arr[i]);
	}
	return total + 2 + ((n - 1) * 2);
} 

static int _bool_representation_(char *out, bool *in, unsigned int n_in)
{
	if (out == NULL)
	{
		printf("Out must be filled with empty slots");
		return 0;
	}
	char *p1 = &out[0];
	bool *p2 = &in[0];
	bool *in_end = p2 + n_in;
	*p1++ = '[';
	while ( p2 < in_end )
	{
		if (*p2++)
		{
			*p1++ = '1';
		}
		else
		{
			*p1++ = '0';
		}
		*p1++ = ',';
		*p1++ = ' ';
	}
	*p1++ = (char) ((int) (*(in_end-1)));
	*p1++ = ']';
	*p1 = '\0';
	return 1;
}

static int _int_representation_(char *out, double *arr, unsigned int n_arr,
								int if_end_of_string, bool row_based = false)
{
	if (out == NULL)
	{
		printf("Out must be filled with empty slots");
		return 0;
	}
	out[0] = '[';
	int offset = 1;
	unsigned int i;
	for (i = 0; i < n_arr; i++)
	{
		unsigned int c_length = _integer_char_length_(arr[i]);
		if (arr[i] < 0)
		{
			c_length++;
		}
		// if the length of number is 1, then convert directly to char.
		if (c_length == 1)
		{
			// convert int to char
			out[offset] = ((int) arr[i]) + '0';
		}
		else
		{
			char output[c_length];
			sprintf(output, "%d", (int) arr[i]);
			unsigned int j;
			for (j = 0; j < c_length; j++)
			{
				out[(offset)+j] = output[j];
			}
			offset += (c_length-1);
		}
		if (i < (n_arr - 1))
		{
			out[++offset] = ',';
			out[++offset] = ' ';
			offset++;
		}
	}
	out[++offset] = ']';
	if (row_based)
	{
		out[++offset] = '.';
		out[++offset] = 'T';
	}
	// if end of string char, add it
	if (if_end_of_string) {
		out[++offset] = '\0';
	}
	return 1;
}

static int _str_representation_(char *out, double *arr, unsigned int n_arr,
								unsigned int dpoints, int if_end_of_string, 
								bool row_based = false)
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
	for (i = 0; i < n_arr; i++)
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
		// unless we are at the end, add on our comma and whitespace padding
		if (i < n_arr-1)
		{
			out[++offset] = ',';
			out[++offset] = ' ';
			offset++;
		}
	}
	// add on final bracket
	out[++offset] = ']';
	// add on .T if transposed
	if (row_based)
	{
		out[++offset] = '.';
		out[++offset] = 'T';
	}
	// if end of string char, add it
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


template <typename T> static int _copy_array_(T *copy, T *orig, unsigned int n)
{
	// orig initialized, copy is empty array.
	if (n != 0 && copy != NULL && orig != NULL)
	{
		T *p1 = &copy[0];
		T *p2 = &orig[0];
		T *end = (p1+n);
		while (p1 < end)
		{
			*p1++ = *p2++;
		}
		return 1;
	}
	else return 0;
}

/* Where copy must be the same size as indices, not orig */
template <typename T> static int _copy_from_index_array_(T *copy, T *orig, double *indices, 
	unsigned int size)
{
	if (copy != NULL && orig != NULL && indices != NULL && size != 0)
	{
		T *p1 = &copy[0];
		double *p3 = &indices[0];
		T *end = p1 + size;
		for (/* void */; p1 < end; p1++, p3++)
		{
			int idx = (int) (*p3);
			if (idx > -1)
			{
				*p1 = *(orig + idx);
			}
			else
			{
				return 0;
			}
		}
		return 1;
	}
	else return 0;
}

template <typename T> static int _copy_from_mask_array_(T *copy, T *orig, bool *mask, unsigned int size,
	bool keep_shape)
{
	if (copy != NULL && orig != NULL && mask != NULL && size != 0)
	{
		T *p1 = &copy[0];
		T *p2 = &orig[0];
		bool *p3 = &mask[0];
		T *end = orig + size;
		if (keep_shape)
		{
			for (/* void */; p2 < end; p1++, p2++, p3++)
			{
				if (*p3)
				{
					*p1 = *p2;
				}
				else
				{
					*p1 = 0;
				}
			}
		}
		else
		{
			for (/* void */; p2 < end; p2++, p3++)
			{
				if (*p3)
				{
					*p1++ = *p2;
				}
			}
		}
		return 1;
	}
	else return 0;
}

static int _rand_array_(double *arr, unsigned int n)
{
	if (n != 0 && arr != NULL)
	{
		double *p1 = &arr[0];
		double *end = p1 + n;
		unsigned int i;
		for (/* void */; p1 < end; p1++)
		{
			*p1 = _uniform_rand_();
		}
		return 1;
	}
	else return 0;
}

static int _normal_distrib_(double *arr, unsigned int n, double mean,
							double sd)
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
		double x, y, r;
		do
		{
			x = 2.0 * _uniform_rand_() - 1;
			y = 2.0 * _uniform_rand_() - 1;
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

static int _randint_array_(double *arr, unsigned int n, unsigned int max)
{
	if (n == 0 || max == 0)
	{
		return 0;
	}
	unsigned int i;
#ifdef _OPENMP
	#pragma omp parallel for if(n>__OMP_OPT_VALUE__) schedule(static)
#endif
	for (i = 0; i < n; i++)
	{
		arr[i] = _uniform_rand_() * max;
		if (arr[i] < 0)
		{
			arr[i] = _truncate_doub_(arr[i], 0) - 1;
		} else
		{
			arr[i] = _truncate_doub_(arr[i], 0) + 1;
		}
	}
	return 1;
}

static int _binomial_array_(double *out, uint n, uint n_trials, double p)
{
	if (out == 0 || n == 0 || n_trials == 0)
	{
		return 0;
	}
	unsigned int i, j;
#ifdef _OPENMP
	#pragma omp parallel for if(n>__OMP_OPT_VALUE__) schedule(static)
#endif
	for (i = 0; i < n; i++)
	{
		uint n_successes = 0;
		// for each trial, simulate the success
		for (j = 0; j < n_trials; j++)
		{
			if (_uniform_rand_() < p)
			{
				n_successes++;
			}
		}
		out[i] = n_successes;
	}
	return 1;
}

static int _poisson_array_(double * out, uint n, double lam)
{
	if (out == 0 || n == 0 || lam < 0)
	{
		return 0;
	}
	unsigned int i;
#ifdef _OPENMP
	#pragma omp parallel for if(n>__OMP_OPT_VALUE__) schedule(static)
#endif
	for (i = 0; i < n; i++)
	{
		out[i] = _poisson_generator_knuth_(lam);
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

static int _boolean_summation_array_(bool *arr, unsigned int n)
{
	if (arr == 0 || n == 0)
	{
		return -1;
	}
	unsigned int total = 0;
	unsigned int i;
#ifdef _OPENMP
	#pragma omp parallel for if(n>__OMP_OPT_VALUE__) schedule(static) reduction(+:total)
#endif
	for (i = 0; i < n; i++)
	{
		if (arr[i])
		{
			total++;
		}
	}
	return total;
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
	#pragma omp parallel for if(nvec>__OMP_OPT_VALUE__) schedule(static) reduction(*:total)
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
	#pragma omp parallel for if(n>__OMP_OPT_VALUE__) schedule(static)
#endif
	for (unsigned int j = 0; j < n; j++)
	{
		for (unsigned int i = j; i > 0; i--)
		{
			zeros[j] += oldarr[i];
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
#ifdef _OPENMP
	#pragma omp parallel for if(n>__OMP_OPT_VALUE__) schedule(static) reduction(min:minval,minidx)
#endif
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
	double summer = _summation_array_(arr, n);
	return _c_power_(summer, 1.0 / p);
}



static int _partition_(double *arr, const int left, const int right) {
    int mid = left + (right - left) / 2;
    double pivot = arr[mid];
    // move the mid point value to the front.
    swap<double>(&arr[mid], &arr[left]);
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
            swap<double>(&arr[i], &arr[j]);
        }
    }
    swap<double>(&arr[i-1],&arr[left]);
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
	double *arr = _create_empty_<double>(guessn);
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
			swap<double>(&arr[x+y*ncol], &arr[x*ncol+y]);
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
		swap<double>(&matrix[row1+y*ncol], &matrix[row2+y*ncol]);
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

static int _equals_array_(bool *out, double *arr, unsigned int n, double value)
{
	if (out == 0 || arr == 0 || n == 0)
	{
		return 0;
	}
	unsigned int i;
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (i = 0; i < n; i++)
	{
		out[i] = CMP(arr[i], value);
	}
	return 1;
}

static int _not_equals_array_(bool *out, double *arr, unsigned int n, double value)
{
	if (out == 0 || arr == 0 || n == 0)
	{
		return 0;
	}
	unsigned int i;
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (i = 0; i < n; i++)
	{
		out[i] = !CMP(arr[i], value);
	}
	return 1;
}

static int _less_than_array_(bool *out, double *arr, unsigned int n, double value,
	bool include_equals)
{
	if (out == 0 || arr == 0 || n == 0)
	{
		return 0;
	}
	unsigned int i;
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (i = 0; i < n; i++)
	{
		if (include_equals)
		{
			out[i] = (arr[i] <= value);
		}
		else
		{
			out[i] = (arr[i] < value);
		}
	}
	return 1;
}

static int _greater_than_array_(bool *out, double *arr, unsigned int n, double value,
	bool include_equals)
{
	if (out == 0 || arr == 0 || n == 0)
	{
		return 0;
	}
	unsigned int i;
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (i = 0; i < n; i++)
	{
		if (include_equals)
		{
			out[i] = (arr[i] >= value);
		}
		else
		{
			out[i] = (arr[i] > value);
		}
	}
	return 1;
}

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

static int _element_div_(double *out, double *in, unsigned int n)
{
	if (n == 0 || out == 0 || in == 0)
	{
		return 0;
	}
	// assumes left and right are same size and DOES NOT CHECK!
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (unsigned int i = 0; i < n; i++)
	{
		// cannot divide by 0!
		if (CMP(in[i],0.0))
		{
			return 0;
		} else {
			out[i] /= in[i];
		}
	}
	return 1;
}

static int _element_equals_(bool *out, double *left, double *right, unsigned int n)
{
	if (out == 0 || left == 0 || right == 0 || n == 0)
	{
		return 0;
	}
	unsigned int i;
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (i = 0; i < n; i++)
	{
		out[i] = CMP(left[i],right[i]);
	}
	return 1;
}

static int _element_not_equals_(bool *out, double *left, double *right, unsigned int n)
{
	if (out == 0 || left == 0 || right == 0 || n == 0)
	{
		return 0;
	}
	unsigned int i;
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (i = 0; i < n; i++)
	{
		out[i] = !CMP(left[i],right[i]);
	}
	return 1;
}

static int _element_less_than_(bool *out, double *left, double *right, unsigned int n,
	bool include_equals)
{
	if (out == 0 || left == 0 || right == 0 || n == 0)
	{
		return 0;
	}
	unsigned int i;
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (i = 0; i < n; i++)
	{
		if (include_equals)
		{
			out[i] = (left[i] <= right[i]);
		}
		else
		{
			out[i] = (left[i] < right[i]);
		}
	}
	return 1;
}

static int _element_greater_than_(bool *out, double *left, double *right, unsigned int n,
	bool include_equals)
{
	if (out == 0 || left == 0 || right == 0 || n == 0)
	{
		return 0;
	}
	unsigned int i;
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (i = 0; i < n; i++)
	{
		if (include_equals)
		{
			out[i] = (left[i] >= right[i]);
		}
		else
		{
			out[i] = (left[i] > right[i]);
		}
	}
	return 1;
}

static int _element_not_(bool *data, unsigned int n)
{
	if (data == 0 || n == 0)
	{
		return 0;
	}
	unsigned int i;
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (i = 0; i < n; i++)
	{
		data[i] = !data[i];
	}
	return 1;
}

static int _element_and_(bool *left, bool *right, unsigned int n)
{
	if (left == 0 || right == 0 || n == 0)
	{
		return 0;
	}
	unsigned int i;
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (i = 0; i < n; i++)
	{
		left[i] = (left[i] & right[i]);
	}
	return 1;
}

static int _element_or_(bool *left, bool *right, unsigned int n)
{
	if (left == 0 || right == 0 || n == 0)
	{
		return 0;
	}
	unsigned int i;
#ifdef _OPENMP
	#pragma omp parallel for schedule(static) if(n>__OMP_OPT_VALUE__)
#endif
	for (i = 0; i < n; i++)
	{
		left[i] = (left[i] | right[i]);
	}
	return 1;
}



#endif






