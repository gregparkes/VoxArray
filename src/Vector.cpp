/*
 * Vec1f.cpp
 *
 *  Created on: 18 Feb 2017
 *      Author: Greg
 */

#ifndef __VEC1f_cpp__
#define __VEC1f_cpp__

#include <stdexcept>
#include <cstring>

#include "Vector.h"
#include "Matrix.h"
#include "numstatic.cpp"


// Here we define class-specific instance methods
namespace numpy {

/********************************************************************************************


	NON-CLASS MEMBER FUNCTIONS 


*///////////////////////////////////////////////////////////////////////////////////////////


	Vector empty(uint n)
	{
		Vector np(n);
		return np;
	}

	Vector empty_like(const Vector& rhs)
	{
		return Vector(rhs.n);
	}

	Vector zeros(uint n)
	{
		Vector np(n);
		if (!_fill_array_(np.data, n, 0.0))
		{
			throw std::invalid_argument("Fill Error");
		}
		return np;
	}

	Vector zeros_like(const Vector& rhs)
	{
		Vector np(rhs.n);
		_fill_array_(np.data, np.n, 0.0);
		return np;
	}

	Vector ones(uint n)
	{
		Vector np(n);
		if (!_fill_array_(np.data, n, 1.0))
		{
			throw std::invalid_argument("Fill Error");
		}
		return np;
	}

	Vector ones_like(const Vector& rhs)
	{
		Vector np(rhs.n);
		_fill_array_(np.data, np.n, 1.0);
		return np;
	}

	Vector fill(uint n, double val)
	{
		Vector np(n);
		if (!_fill_array_(np.data, n, val))
		{
			throw std::invalid_argument("Fill Error");
		}
		return np;
	}

	char* str(const Vector& rhs, uint dpoints)
	{
		unsigned int str_len = _str_length_gen_(rhs.data, rhs.n, dpoints);
		char *strg = new char[str_len];
		if (!_str_representation_(strg, rhs.data, rhs.n, dpoints, 1))
		{
			throw std::invalid_argument("Problem with creating string representation");
		}
		return strg;
	}

	uint len(const Vector& rhs)
	{
		return rhs.n;
	}

	Vector array(const char *input)
	{
		if (input == null)
		{
			throw std::invalid_argument("input must not be null");
		}
		int n = strlen(input);
		Vector np;
		// now we somehow parse the string
		if (n > -1)
		{
			double *vals = _parse_string_to_array_(input, &n);
			if (vals == null)
			{
				throw std::runtime_error("Unable to parse string into array");
			}
			np.data = vals;
			np.n = n;
		} else {
			throw std::range_error("n must be > -1");
		}
		return np;
	}

	Vector copy(const Vector& rhs)
	{
		Vector np(rhs.n);
		if (!_copy_array_(np.data, rhs.data, rhs.n))
		{
			throw std::invalid_argument("copy failed!");
		}
		np.column = rhs.column;
		np.flag_delete = rhs.flag_delete;
		return np;
	}

	Vector rand(uint n)
	{
		if (n == 0)
		{
			throw std::range_error("n cannot = 0");
		}
		Vector np(n);
		_rand_array_(np.data, np.n);
		return np;
	}

	Vector randn(uint n)
	{
		if (n == 0)
		{
			throw std::range_error("n cannot = 0");
		}
		Vector np(n);
		if (!_normal_distrib_(np.data, n, 0.0, 1.0))
		{
			throw std::invalid_argument("Error with creating normal distribution");
		}
		return np;
	}

	Vector normal(uint n, double mean, double sd)
	{
		if (n == 0)
		{
			throw std::range_error("n cannot = 0");
		}
		Vector np(n);
		if (!_normal_distrib_(np.data, n, mean, sd))
		{
			throw std::invalid_argument("Error with creating normal distribution");
		}
		return np;
	}

	Vector randint(uint n, uint max)
	{
		if (n == 0)
		{
			throw std::range_error("n cannot = 0");
		}
		if (max == 0)
		{
			throw std::range_error("max cannot = 0");
		}
		Vector np(n);
		_randint_array_(np.data, n, max);
		return np;
	}

	Vector randchoice(uint n, const char *values)
	{
		if (n == 0)
		{
			throw std::range_error("n cannot = 0");
		}
		if (values == null)
		{
			throw std::invalid_argument("input must not be null");
		}
		int strn = strlen(values);
		Vector np(n);
		// now we somehow parse the string

		double *vals = _parse_string_to_array_(values, &strn);
		if (vals == null)
		{
			throw std::runtime_error("Unable to parse string into array");
		}
		srand48(time(NULL));
		for (uint i = 0; i < n; i++)
		{
			// create random float
			double idx_f = drand48() * strn;
			// pass to an array and floor it (i.e 2.90 becomes 2, an index for values*)
			int idx = (int) _truncate_doub_(idx_f, 0);
			// set data using index and values
			np.data[i] = vals[idx];
		}
		return np;
	}

	Vector binomial(const Vector& n, const Vector& p)
	{
		if (n.n != p.n)
		{
			throw std::range_error("n and p vectors must be same length");
		}
		Vector np = empty_like(n);
		_binomial_array_(np.data, n.data, p.data, np.n);
		return np;
	}

	Vector poisson(const Vector& lam)
	{
		Vector res = empty_like(lam);
		for (uint i = 0; i < lam.n; i++)
		{
			res.data[i] = _poisson_coefficient_(lam.data[i]);
		}
		return res;
	}

	Vector nonzero(const Vector& rhs)
	{
		int cnz = _count_nonzero_array_(rhs.data, rhs.n);
		Vector np(cnz);
		_nonzero_array_(np.data, rhs.data, rhs.n);
		return np;
	}

	Vector flip(const Vector& rhs)
	{
		Vector np = copy(rhs);
		_flip_array_(np.data, np.n);
		return np;
	}

	Vector vstack(const Vector& lhs, const Vector& rhs)
	{
		Vector np(lhs.n + rhs.n);
		for (uint i = 0; i < lhs.n; i++)
		{
			np.data[i] = lhs.data[i];
		}
		for (uint i = 0; i < rhs.n; i++)
		{
			np.data[i+lhs.n] = rhs.data[i];
		}
		return np;
	}

	Vector vectorize(const Matrix& rhs, uint axis)
	{
		Vector np(rhs.nvec*rhs.vectors[0]->n);
		if (axis == 0)
		{
		#ifdef _OPENMP
			#pragma omp parallel for if(rhs.nvec>100000) schedule(static)
		#endif
			for (uint y = 0; y < rhs.nvec; y++)
			{
				for (uint x = 0; x < rhs.vectors[y]->n; x++)
				{
					np.data[x+y*rhs.vectors[y]->n] = rhs.vectors[y]->data[x];
				}
			}
		} else if (axis == 1)
		{
		#ifdef _OPENMP
			#pragma omp parallel for if(rhs.vectors[0]->n>100000) schedule(static)
		#endif
			for (uint x = 0; x < rhs.vectors[0]->n; x++)
			{
				for (uint y = 0; y < rhs.nvec; y++)
				{
					np.data[y+x*rhs.nvec] = rhs.vectors[y]->data[x];
				}
			}
		} else {
			throw std::invalid_argument("axis must be 0 or 1.");
		}
		return np;
	}

	Vector arange(double start, double end, double step)
	{
		if (step <= 0)
		{
			throw std::range_error("step cannot be <= 0");
		}
		if (start > end)
		{
			//swap them
			double temp = start;
			start = end;
			end = temp;
		}
		uint n = (uint) ((end - start) / step) + 1;
		Vector np(n);
		np.data[0] = start;
		np.data[n-1] = end;
	#ifdef _OPENMP
		#pragma omp parallel for default(none) shared(np,n,start,step) if(n>100000) schedule(static)
	#endif
		for (uint i = 1; i < n-1; i++)
		{
			np.data[i] = start + step * i;
		}
		return np;
	}

	Vector linspace(double start, double end, uint n)
	{
		if (n == 0)
		{
			throw std::invalid_argument("n cannot be <= 0");
		}
		if (start > end)
		{
			//swap them
			double temp = start;
			start = end;
			end = temp;
		}
		Vector np(n);
		np.data[0] = start;
		np.data[n-1] = end;
		double step = (end-start) / (n-1);
	#ifdef _OPENMP
		#pragma omp parallel for default(none) shared(np,n,start,step) if(n>100000) schedule(static)
	#endif
		for (uint i = 1; i < n-1; i++)
		{
			np.data[i] = start + step * i;
		}
		return np;
	}

	Vector logspace(double start, double end, uint n)
	{
		Vector np = linspace(start, end, n);
		_pow_base_array_(np.data, np.n, 10.0);
		return np;
	}

	Vector lstrip(const Vector& rhs, uint idx)
	{
		if (idx == 0)
		{
			throw std::invalid_argument("idx cannot be <= 0");
		}
		if (idx >= rhs.n)
		{
			throw std::range_error("idx cannot be > rhs size");
		}
		Vector np(rhs.n - idx);
		_copy_array_(np.data, rhs.data+idx, rhs.n-idx);
		return np;
	}

	Vector rstrip(const Vector& rhs, uint idx)
	{
		if (idx == 0)
		{
			throw std::invalid_argument("idx cannot be <= 0");
		}
		if (idx >= rhs.n)
		{
			throw std::range_error("idx cannot be > rhs size");
		}
		Vector np(idx+1);
		_copy_array_(np.data, rhs.data, idx+1);
		return np;
	}

	Vector clip(const Vector& rhs, double a_min, double a_max)
	{
		Vector np = copy(rhs);
		_clip_array_(np.data, np.n, a_min, a_max);
		return np;
	}

	Vector floor(const Vector& rhs)
	{
		Vector np = copy(rhs);
		if (!_floor_array_(np.data, np.n))
		{
			throw std::invalid_argument("Unable to floor array.");
		}
		return np;
	}

	Vector ceil(const Vector& rhs)
	{
		Vector np = copy(rhs);
		if (!_ceil_array_(np.data, np.n))
		{
			throw std::invalid_argument("Unable to ceil array.");
		}
		return np;
	}

	int count(const Vector& rhs, double value)
	{
		return _count_array_(rhs.data, rhs.n, value);
	}

	int count_nonzero(const Vector& rhs)
	{
		return _count_nonzero_array_(rhs.data, rhs.n);
	}

	bool isColumn(const Vector& rhs)
	{
		return rhs.column;
	}

	bool isRow(const Vector& rhs)
	{
		return ! rhs.column;
	}

	Vector abs(const Vector& rhs)
	{
		Vector np = copy(rhs);
		_absolute_array_(np.data, np.n);
		return np;
	}

	double sum(const Vector& rhs)
	{
		return _summation_array_(rhs.data, rhs.n);
	}

	double mean(const Vector& rhs)
	{
		return sum(rhs) / rhs.n;
	}

	double median(const Vector& rhs, bool isSorted)
	{
		if (isSorted)
		{
			return rhs.data[rhs.n / 2];
		}
		else
		{
			Vector s = sort(rhs);
			return s.data[s.n / 2];
		}
	}

	double std(const Vector& rhs)
	{
		return _std_array_(rhs.data, rhs.n);
	}

	double var(const Vector& rhs)
	{
		return _var_array_(rhs.data, rhs.n);
	}

	double prod(const Vector& rhs)
	{
		return _prod_array_(rhs.data, rhs.n);
	}

	Vector cumsum(const Vector& rhs)
	{
		Vector np = zeros(rhs.n);
		if (!_cumulative_sum_(np.data, rhs.data, rhs.n))
		{
			throw std::invalid_argument("cumsum failed!");
		}
		return np;
	}

	Vector adjacsum(const Vector& rhs)
	{
		// Default - Does not wrap adjacency.

		Vector np(rhs.n);
		np.data[0] = rhs.data[0] + rhs.data[1];
		np.data[rhs.n-1] = rhs.data[rhs.n-1] + rhs.data[rhs.n-2];
		for (uint i = 1; i < rhs.n-1; i++)
		{
			np.data[i] = rhs.data[i] + rhs.data[i-1] + rhs.data[i+1];
		}
		return np;
	}

	Vector cumprod(const Vector& rhs)
	{
		Vector np = ones(rhs.n);
		if (!_cumulative_prod_(np.data, rhs.data, rhs.n))
		{
			throw std::invalid_argument("cumprod failed!");
		}
		return np;
	}

	double trapz(const Vector& y, double dx)
	{
		double total = 0.0;
		for (uint i = 1; i < y.n-1; i++)
		{
			total += 2*y.data[i];
		}
		return ((dx/2) * (y.data[0] + total + y.data[y.n-1]));
	}

	bool all(const Vector& rhs)
	{
		return _all_true_(rhs.data, rhs.n);
	}

	bool any(const Vector& rhs)
	{
		return _any_true_(rhs.data, rhs.n);
	}

	double min(const Vector& rhs)
	{
		return _min_value_(rhs.data, rhs.n);
	}

	uint argmin(const Vector& rhs)
	{
		return _min_index_(rhs.data, rhs.n);
	}

	double max(const Vector& rhs)
	{
		return _max_value_(rhs.data, rhs.n);
	}

	uint argmax(const Vector& rhs)
	{
		return _max_index_(rhs.data, rhs.n);
	}

	double cov(const Vector& v, const Vector& w)
	{
		return dot(v, w) / (v.n - 1);
	}

	double norm(const Vector& rhs, int order)
	{
		if (order == _ONE_NORM)
		{
			return _absolute_summation_array_(rhs.data, rhs.n);
		} else if (order >= 2)
		{
			// Create a copy for the C function to use (to power data on)
			Vector cp = copy(rhs);
			// The Euclidean norm - the normal norm, or norm-2
			return _vector2_norm_(cp.data, cp.n, order);
		} else if (order == _INF_NORM)
		{
			// order = 3 - infinity norm
			return max(rhs);
		}
		else
		{
			throw std::invalid_argument("order must be 1,2..,n or inf (-1)");
		}
	}

	Vector diag(const Matrix& rhs)
	{
		Vector v = empty(rhs.nvec);
		for (uint i = 0; i < rhs.nvec; i++)
		{
			v.data[i] = rhs.vectors[i]->data[i];
		}
		return v;
	}

	double dot(const Vector& lhs, const Vector& rhs)
	{
		if (lhs.n != rhs.n)
		{
			throw std::range_error("lhs must be the same size as the rhs vector");
		}
		// apparently the dot of column.row is the same result as column.column or row.row
		return _vector_dot_array_(lhs.data, rhs.data, rhs.n);
	}

	double inner(const Vector& v, const Vector& w)
	{
		return dot(v, w);
	}

	double magnitude(const Vector& v)
	{
		return _square_root_(dot(v, v));
	}

	Vector normalized(const Vector& v)
	{
		Vector np = copy(v);
		np *= (1.0 / magnitude(v));
		return np;
	}

	Vector standardize(const Vector& v)
	{
		Vector np = copy(v);
		// remove mean
		np -= mean(np);
		// scale std
		np /= std(np);
		return np;
	}

	Vector minmax(const Vector& v)
	{
		Vector np = copy(v);
		double curr_max = np.max();
		double curr_min = np.min();
		return ((np - curr_min) / (curr_max - curr_min));
	}

	Vector cross(const Vector& v, const Vector& w)
	{
		if (v.n != 3 || w.n != 3)
		{
			throw std::logic_error("v and w must have a length == 3");
		}
		// 3d
		double o1, o2, o3;
		o1 = v.data[1] * w.data[2] - w.data[1] * v.data[2];
		o2 = w.data[0] * v.data[2] - v.data[0] * w.data[2];
		o3 = v.data[0] * w.data[1] - w.data[0] * v.data[1];
		return Vector(o1, o2, o3);
	}

	Vector sin(const Vector& rhs)
	{
		Vector np = copy(rhs);
		if (!_sine_array_(np.data, np.n))
		{
			throw std::invalid_argument("Unable to sine-ify array.");
		}
		return np;
	}

	Vector cos(const Vector& rhs)
	{
		Vector np = copy(rhs);
		if (!_cos_array_(np.data, np.n))
		{
			throw std::invalid_argument("Unable to cos-ify array.");
		}
		return np;
	}

	Vector tan(const Vector& rhs)
	{
		Vector np = copy(rhs);
		if (!_tan_array_(np.data, np.n))
		{
			throw std::invalid_argument("Unable to tan-ify array.");
		}
		return np;
	}

	Vector to_radians(const Vector& rhs)
	{
		Vector v = copy(rhs);
		if (!_to_radians_array_(v.data, v.n))
		{
			throw std::invalid_argument("Unable to convert to radians.");
		}
		return v;
	}

	Vector to_degrees(const Vector& rhs)
	{
		Vector v = copy(rhs);
		if (!_to_degrees_array_(v.data, v.n))
		{
			throw std::invalid_argument("Unable to convert to degrees.");
		}
		return v;
	}

	Vector exp(const Vector& rhs)
	{
		Vector np = copy(rhs);
		if (!_exp_array_(np.data, np.n))
		{
			throw std::invalid_argument("Unable to exp-ify array.");
		}
		return np;
	}

	Vector log(const Vector& rhs)
	{
		Vector np = copy(rhs);
		if (!_log10_array_(np.data, np.n))
		{
			throw std::invalid_argument("Unable to log-ify array.");
		}
		return np;
	}

	Vector sqrt(const Vector& rhs)
	{
		Vector np = copy(rhs);
		if (!_pow_array_(np.data, np.n, 0.5))
		{
			throw std::invalid_argument("Unable to sqrt-ify array.");
		}
		return np;
	}

	Vector power(double base, const Vector& exponent)
	{
		Vector np(exponent.n);
		for (uint i = 0; i < np.n; i++)
		{
			np.data[i] = pow(base, exponent.data[i]);
		}
		return np;
	}
	Vector power(const Vector& base, double exponent)
	{
		Vector np = copy(base);
		if (!_pow_array_(np.data, np.n, exponent))
		{
			throw std::invalid_argument("Unable to exp-ify array.");
		}
		return np;
	}
	Vector power(const Vector& base, const Vector& exponent)
	{
		if (base.n != exponent.n)
		{
			throw std::invalid_argument("base size and exponent size must be equal");
		}
		Vector np(base.n);
	#ifdef _OPENMP
		#pragma omp parallel for schedule(static) if(np.n>100000)
	#endif
		for (uint i = 0; i < np.n; i++)
		{
			np.data[i] = pow(base.data[i], exponent.data[i]);
		}
		return np;
	}

	Vector transpose(const Vector& rhs)
	{
		// we essentially do nothing apart from create a copy
		Vector np = copy(rhs);
		np.column = !np.column;
		return np;
	}

	Vector sort(const Vector& rhs, uint sorter)
	{
		Vector np = copy(rhs);
		_quicksort_(np.data, 0, np.n-1);
		if (sorter == SORT_DESCEND)
		{
			np.flip();
		}
		return np;
	}

	Vector rotate_vector2d(const Vector& v, double degrees)
	{
		if (v.n != 2)
		{
			throw std::invalid_argument("v must be of length 2!");
		}
		degrees = DEG2RAD(degrees);
		double s = _sine_(degrees);
		double c = _cosine_(degrees);

		Vector np = empty(2);
		np.data[0] = (v.data[0] * c) - (v.data[1] * s);
		np.data[1] = (v.data[0] * s) + (v.data[1] * c);
		return np;
	}

	double angle(const Vector& l, const Vector& r)
	{
		return acosf(dot(l, r) / sqrtf(dot(l,l) * dot(r,r)));
	}

	Vector project(const Vector& length, const Vector& direction)
	{
		if (length.n != direction.n)
		{
			throw std::invalid_argument("length and direction size must be the same!");
		}
		double d = dot(length, direction);
		double mag_sq = dot(direction, direction);
		return (direction * (d / mag_sq));
	}

	Vector perpendicular(const Vector& length, const Vector& dir)
	{
		return length - project(length, dir);
	}

	Vector reflection(const Vector& source, const Vector& normal)
	{
		return source - normal * (dot(source, normal) * 2.0);
	}


/********************************************************************************************


		CLASS & MEMBER FUNCTIONS 


*///////////////////////////////////////////////////////////////////////////////////////////

	Vector::Vector()
	{
#ifdef _CUMPY_DEBUG_
		printf("constructing empty vector %x\n", this);
#endif
		this->n = 0;
		this->column = true;
		this->data = null;
		this->flag_delete = false;
	}

	Vector::Vector(uint n, bool column)
	{
#ifdef _CUMPY_DEBUG_
		printf("constructing vector normal %x\n", this);
#endif
		if (n == 0)
		{
			throw std::range_error("n cannot = 0");
		}
		this->n = n;
		this->column = column;
		data = _create_empty_(n);
		if (data == NULL)
		{
			throw std::runtime_error("Unable to allocate memory");
		}
		this->flag_delete = true;
	}

	Vector::Vector(double val1, double val2, double val3, double val4)
	{
#ifdef _CUMPY_DEBUG_
		printf("constructing vector values %x\n", this);
#endif
		this->column = true;
		this->flag_delete = true;
		if (val4 - 99999.99999 > 1e-14)
		{
			this->n = 4;
		} else if (val3 - 99999.99999 > 1e-14)
		{
			this->n = 3;
		} else {
			this->n = 2;
		}
		data = _create_empty_(n);
		data[0] = val1;
		data[1] = val2;
		if (n > 2)
		{
			data[2] = val3;
			if (n > 3)
			{
				data[3] = val4;
			}
		}
	}

	Vector::~Vector()
	{
		if (flag_delete && data != NULL)
		{
#ifdef _CUMPY_DEBUG_
			printf("deleting vector %x\n", this);
#endif
			if (!_destroy_array_(data))
			{
				throw std::invalid_argument("Unable to destroy array");
			}
		}

	}

	char* Vector::str(uint dpoints)
	{
		unsigned int str_len = _str_length_gen_(data, n, dpoints);
		char *strg = new char[str_len];
		if (!_str_representation_(strg, data, n, dpoints, 1))
		{
			throw std::invalid_argument("Problem with creating string representation");
		}
		return strg;
	}

	Vector Vector::copy()
	{
		Vector np(n);
		if (!_copy_array_(np.data, data, n))
		{
			throw std::invalid_argument("copy failed!");
		}
		column = np.column;
		flag_delete = np.flag_delete;
		return np;
	}

	Vector& Vector::flip()
	{
		_flip_array_(data,n);
		return *this;
	}

	Vector& Vector::clip(double a_min, double a_max)
	{
		_clip_array_(data, n, a_min, a_max);
		return *this;
	}

	Vector& Vector::abs()
	{
		_absolute_array_(data, n);
		return *this;
	}

	double Vector::sum()
	{
		return _summation_array_(data, n);
	}

	double Vector::prod()
	{
		return _prod_array_(data,n);
	}

	bool Vector::all()
	{
		return _all_true_(data,n);
	}

	bool Vector::any()
	{
		return _any_true_(data,n);
	}

	uint Vector::count(double value)
	{
		return _count_array_(data,n,value);
	}

	uint Vector::count_nonzero()
	{
		return _count_nonzero_array_(data, n);
	}

	double Vector::mean()
	{
		return sum() / n;
	}

	double Vector::std()
	{
		return _std_array_(data,n);
	}

	double Vector::var()
	{
		return _var_array_(data,n);
	}

	uint Vector::argmin()
	{
		return _min_index_(data, n);
	}

	uint Vector::argmax()
	{
		return _max_index_(data, n);
	}

	double Vector::norm(int order)
	{
		if (order == 0 || order < -1 )
		{
			throw std::invalid_argument("order must be 1,2..,n or inf");
		}
		if (order == 1)
		{
			return _absolute_summation_array_(data,n);
		} else if (order >= 2)
		{
			// Create a copy for the ^ power operator call
			Vector cp = copy();
			// The Euclidean norm - the normal norm, or norm-2
			_pow_array_(cp.data, cp.n, order);
			return _c_power_(cp.sum(), 1/order);
		} else
		{
			// order = 3 - infinity norm
			return max();
		}
	}

	double Vector::min()
	{
		return _min_value_(data, n);
	}

	double Vector::max()
	{
		return _max_value_(data, n);
	}

	Vector& Vector::sin()
	{
		if (! _sine_array_(data, n))
		{
			throw std::invalid_argument("Unable to sine-ify array.");
		}
		return *this;
	}

	Vector& Vector::cos()
	{
		if (! _cos_array_(data, n))
		{
			throw std::invalid_argument("Unable to cos-ify array.");
		}
		return *this;
	}

	Vector& Vector::tan()
	{
		if (! _tan_array_(data, n))
		{
			throw std::invalid_argument("Unable to tan-ify array.");
		}
		return *this;
	}

	Vector& Vector::exp()
	{
		if (! _exp_array_(data, n))
		{
			throw std::invalid_argument("Unable to exp-ify array.");
		}
		return *this;
	}

	Vector& Vector::log()
	{
		if (! _log10_array_(data, n))
		{
			throw std::invalid_argument("Unable to log-ify array.");
		}
		return *this;
	}

	Vector& Vector::sqrt()
	{
		if (! _pow_array_(data, n, 0.5))
		{
			throw std::invalid_argument("Unable to sqrt-ify array.");
		}
		return *this;
	}


	Vector& Vector::to_radians()
	{
		if (!_to_radians_array_(data, n))
		{
			throw std::invalid_argument("Unable to convert to radians.");
		}
		return *this;
	}

	Vector& Vector::to_degrees()
	{
		if (!_to_degrees_array_(data, n))
		{
			throw std::invalid_argument("Unable to convert to radians.");
		}
		return *this;
	}

	Vector& Vector::pow_base(double base)
	{
		if (! _pow_base_array_(data, n, base))
		{
			throw std::invalid_argument("Unable to pow-ify array.");
		}
		return *this;
	}

	Vector& Vector::pow_exp(double exponent)
	{
		if (! _pow_array_(data, n, exponent))
		{
			throw std::invalid_argument("Unable to pow-ify array.");
		}
		return *this;
	}

	Vector& Vector::fill(double value)
	{
		_fill_array_(data, n, value);
		return *this;
	}

	Vector& Vector::floor()
	{
		if (!_floor_array_(data, n))
		{
			throw std::invalid_argument("Unable to ceil array.");
		}
		return *this;
	}

	Vector& Vector::ceil()
	{
		if (!_ceil_array_(data, n))
		{
			throw std::invalid_argument("Unable to ceil array.");
		}
		return *this;
	}

	double Vector::dot(const Vector& rhs)
	{
		if (n != rhs.n)
		{
			throw std::range_error("lhs must be the same size as the rhs vector");
		}
		if (!rhs.column & column)
		{
			// if it is abT, not aTb, it cannot be a dot product, it is matrix product instead
			throw std::logic_error("rhs cannot be a row-vector if this is a column vector, use matrix product instead");
		}
		return _vector_dot_array_(data, rhs.data, n);
	}

	double Vector::magnitude()
	{
		return _square_root_(dot(*this));
	}

	double Vector::distance(const Vector& rhs)
	{
		if (n != rhs.n)
		{
			throw std::range_error("lhs must be the same size as the rhs vector");
		}
		return _square_root_(dot(*this - rhs));
	}

	void Vector::normalize()
	{
		*this *= (1.0 / magnitude());
	}

	Vector& Vector::T()
	{
		column = !column;
		return *this;
	}

	Vector& Vector::sort(uint sorter)
	{
		_quicksort_(data, 0, n-1);
		if (sorter == SORT_DESCEND)
		{
			flip();
		}
		return *this;
	}

	//  --------------- Instance Operator overloads ------------------------


	Vector& Vector::operator+=(const Vector& rhs)
	{
		if (rhs.n != n)
		{
			throw std::invalid_argument("rhs size != array size");
		}
		_element_add_(data, rhs.data, n);
		return *this;
	}
	Vector& Vector::operator+=(double value)
	{
		_add_array_(data, n, value);
		return *this;
	}
	Vector& Vector::operator+=(int value)
	{
		_add_array_(data, n, (double) value);
		return *this;
	}

	Vector& Vector::operator-=(const Vector& rhs)
	{
		if (rhs.n != n)
		{
			throw std::invalid_argument("rhs size != array size");
		}
		_element_sub_(data, rhs.data, n);
		return *this;
	}
	Vector& Vector::operator-=(double value)
	{
		_sub_array_(data, n, value);
		return *this;
	}
	Vector& Vector::operator-=(int value)
	{
		_sub_array_(data, n, (double) value);
		return *this;
	}

	Vector& Vector::operator*=(const Vector& rhs)
	{
		if (rhs.n != n)
		{
			throw std::invalid_argument("rhs size != array size");
		}
		if (!column & rhs.column)
		{
			// if lhs is row, rhs is column, xTy or yTx, then it is the same as a dot product
			throw std::logic_error("You are multiplying a column by row vector, call dot() method instead");
		} else if (column & !rhs.column)
		{
			// if lhs is column, rhs is row, xyT or yxT, then it is the same as matrix product
			throw std::logic_error("You are multiplying a row by column vector, call matrix product instead");
		} else {
			_element_mult_(data, rhs.data, n);
		}
		return *this;
	}
	Vector& Vector::operator*=(double value)
	{
		_mult_array_(data, n, value);
		return *this;
	}
	Vector& Vector::operator*=(int value)
	{
		_mult_array_(data, n, (double) value);
		return *this;
	}

	Vector& Vector::operator/=(const Vector& rhs)
	{
		if (rhs.n != n)
		{
			throw std::invalid_argument("rhs size != array size");
		}
		if (!_element_div_(data, rhs.data, n))
		{
			throw std::logic_error("cannot divide by zero!");
		}
		return *this;
	}
	Vector& Vector::operator/=(double value)
	{
		if (!_div_array_(data, n, value))
		{
			throw std::logic_error("cannot divide by zero!");
		}
		return *this;
	}
	Vector& Vector::operator/=(int value)
	{
		if (!_div_array_(data, n, (double) value))
		{
			throw std::logic_error("cannot divide by zero!");
		}
		return *this;
	}

	/*
	* THIS SECTION WILL NOW FOCUS ON +, -, *, / AND ^ GLOBAL OPERATORS 
	*
	* IN ASSOCIATION WITH VECTORS.
	*/

	Vector operator+(const Vector& lhs, double value)
	{
		Vector np = _copy_vector_(lhs);
		np += value;
		return np;
	}
	Vector operator+(double value, const Vector& rhs)
	{
		Vector np = _copy_vector_(rhs);
		np += value;
		return np;
	}
	Vector operator+(const Vector& lhs, const Vector& rhs)
	{
		if (lhs.n != rhs.n)
		{
			throw std::range_error("lhs and rhs vector not the same size");
		}
		Vector np = _copy_vector_(lhs);
		np += rhs;
		return np;
	}
	Vector operator+(const Vector& lhs, int value)
	{
		Vector np = _copy_vector_(lhs);
		np += (double) value;
		return np;
	}
	Vector operator+(int lhs, const Vector& rhs)
	{
		Vector np = _copy_vector_(rhs);
		np += (double) lhs;
		return np;
	}

	Vector operator-(const Vector& lhs, int value)
	{
		Vector np = _copy_vector_(lhs);
		np -= (double) value;
		return np;
	}
	Vector operator-(const Vector& lhs, double value)
	{
		Vector np = _copy_vector_(lhs);
		np -= value;
		return np;
	}
	Vector operator-(const Vector& lhs, const Vector& rhs)
	{
		if (lhs.n != rhs.n)
		{
			throw std::range_error("lhs and rhs vector not same size");
		}
		Vector np = _copy_vector_(lhs);
		np -= rhs;
		return np;
	}

	Vector operator*(const Vector& lhs, double value)
	{
		Vector np = _copy_vector_(lhs);
		np *= value;
		return np;
	}
	Vector operator*(double value, const Vector& rhs)
	{
		Vector np = _copy_vector_(rhs);
		np *= value;
		return np;
	}
	Vector operator*(const Vector& lhs, const Vector& rhs)
	{
		if (lhs.n != rhs.n)
		{
			throw std::range_error("lhs and rhs vector not same size");
		}
		Vector np = _copy_vector_(lhs);
		np *= rhs;
		return np;
	}
	Vector operator*(const Vector& lhs, int value)
	{
		Vector np = _copy_vector_(lhs);
		np *= (double) value;
		return np;
	}
	Vector operator*(int lhs, const Vector& rhs)
	{
		Vector np = _copy_vector_(rhs);
		np *= (double) lhs;
		return np;
	}

	Vector operator/(const Vector& lhs, int value)
	{
		if (value == 0)
		{
			throw std::invalid_argument("cannot divide by 0!");
		}
		Vector np = _copy_vector_(lhs);
		np /= (double) value;
		return np;
	}
	Vector operator/(const Vector& lhs, double value)
	{
		if (CMP(value, 0.0))
		{
			throw std::invalid_argument("cannot divide by 0!");
		}
		Vector np = _copy_vector_(lhs);
		np /= value;
		return np;
	}
	Vector operator/(const Vector& lhs, const Vector& rhs)
	{
		if (lhs.n != rhs.n)
		{
			throw std::range_error("lhs and rhs vector not same size");
		}
		Vector np = _copy_vector_(lhs);
		np /= rhs;
		return np;
	}

	Vector operator^(const Vector& base, double exponent)
	{
		Vector np = _copy_vector_(base);
		_pow_array_(np.data, np.n, exponent);
		return np;
	}
	Vector operator^(double base, const Vector& exponent)
	{
		Vector np = _copy_vector_(exponent);
		_pow_base_array_(np.data, np.n, base);
		return np;
	}
	Vector operator^(const Vector& base, const Vector& exponent)
	{
		if (base.n != exponent.n)
		{
			throw std::range_error("base and exponent vector not same size");
		}
		Vector np = _copy_vector_(base);
		for (uint i = 0; i < base.n; i++)
		{
			np.data[i] = _c_power_(base.data[i], exponent.data[i]);
		}
		return np;
	}

	/********************************************************************************************


		PRIVATE ACCESSORY FUNCTIONS 


	*///////////////////////////////////////////////////////////////////////////////////////////

	Vector _copy_vector_(const Vector& v)
	{
		Vector np(v.n);
		_copy_array_(np.data, v.data, v.n);
		np.column = v.column;
		np.flag_delete = v.flag_delete;
		return np;
	}


}

#endif

