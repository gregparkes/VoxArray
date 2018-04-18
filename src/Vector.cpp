/*
 * Vector.cpp
 *
 *  Created on: 18 Feb 2017
 *      Author: Greg
 */

#ifndef __VEC1f_cpp__
#define __VEC1f_cpp__

#include <stdexcept>
#include <cstring>

#include "Vector.h"
#include "BoolVector.h"
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
		if (!_fill_array_<double>(np.data, n, 0.0))
		{
			INVALID("Fill Error");
		}
		return np;
	}

	Vector zeros_like(const Vector& rhs)
	{
		Vector np(rhs.n);
		_fill_array_<double>(np.data, np.n, 0.0);
		return np;
	}

	Vector ones(uint n)
	{
		Vector np(n);
		if (!_fill_array_<double>(np.data, n, 1.0))
		{
			INVALID("Fill Error");
		}
		return np;
	}

	Vector ones_like(const Vector& rhs)
	{
		Vector np(rhs.n);
		_fill_array_<double>(np.data, np.n, 1.0);
		return np;
	}

	Vector fill(uint n, double val)
	{
		Vector np(n);
		if (!_fill_array_<double>(np.data, n, val))
		{
			INVALID("Fill Error");
		}
		return np;
	}

	uint len(const Vector& rhs)
	{
		return rhs.n;
	}

	char* str(const Vector& rhs, uint dpoints, bool represent_float)
	{
		if (rhs.data == NULL)
		{
			INVALID("in str(), data is pointing to no values (NULL)");
		}

		unsigned int str_len;
		// if we are dealing in floats, calculate the length accordingly, else integerize
		if (represent_float)
		{
			str_len = _str_length_gen_(rhs.data, rhs.n, dpoints);
		}
		else
		{
			str_len = _str_length_int_gen_(rhs.data, rhs.n);
		}

		// if we are row-based, add a ".T" at the end to indicate transpose row.
		if (!rhs.column)
		{
			str_len += 2;
		}
		char *strg = new char[str_len];
		if (represent_float)
		{
			if (!_str_representation_(strg, rhs.data, rhs.n, dpoints, 1, !rhs.column))
			{
				INVALID("Problem with creating float string representation");
			}
		}
		else
		{
			if (!_int_representation_(strg, rhs.data, rhs.n, 1, !rhs.column))
			{
				INVALID("Problem with creating int string representation");
			}
		}
		// now create string according to rules of length.
		return strg;
	}

	Vector array(const char *input)
	{
		if (input == null)
		{
			INVALID("input must not be null");
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
			RANGE("n must be > -1");
		}
		return np;
	}

	Vector copy(const Vector& rhs)
	{
		Vector np(rhs.n);
		if (!_copy_array_<double>(np.data, rhs.data, rhs.n))
		{
			INVALID("copy failed!");
		}
		np.column = rhs.column;
		np.flag_delete = rhs.flag_delete;
		return np;
	}

	Matrix to_matrix(const Vector& rhs)
	{
		Matrix m = empty(1, rhs.n);
		if (!_copy_array_<double>(m.data, rhs.data, rhs.n))
		{
			INVALID("copy failed!");
		}
		return m;
	}

	Mask to_mask(const Vector& rhs)
	{
		return Mask(rhs);
	}

	Vector take(const Vector& a, const Vector& indices)
	{
		// our maximum index cannot be greater than the size!
		if (max(indices) >= a.n)
		{
			INVALID("Max IDX cannot be >= the size of our array");
		}
		Vector res = empty_like(indices);
		if (!_copy_from_index_array_<double>(res.data, a.data, indices.data, res.n))
		{
			INVALID("Unable to copy from index vector in 'take()'.");
		}
		return res;
	}

	Vector where(const Vector& a, const Mask& m, bool keep_shape)
	{
		if (a.n != m.n)
		{
			INVALID("array and mask must be the same size!");
		}
		// calculate the new size of out array if not keep shape
		uint newn = 0;
		if (keep_shape)
		{
			newn = a.n;
		}
		else
		{
			newn = _boolean_summation_array_(m.data, m.n);
		}
		// create new results vector
		Vector res = empty(newn);
		// if we keep the shape, simply copy across and set to 0. - else keep_shape returns resdata
		if (!_copy_from_mask_array_<double>(res.data, a.data, m.data, m.n, keep_shape))
		{
			INVALID("Unable to copy from mask vector in 'where()'.");
		}
		return res;
	}

	Vector rand(uint n)
	{
		if (n == 0)
		{
			RANGE("n cannot = 0");
		}
		Vector np(n);
		_rand_array_(np.data, np.n);
		return np;
	}

	Vector randn(uint n)
	{
		if (n == 0)
		{
			RANGE("n cannot = 0");
		}
		Vector np(n);
		if (!_normal_distrib_(np.data, n, 0.0, 1.0))
		{
			INVALID("Error with creating normal distribution");
		}
		return np;
	}

	Vector normal(uint n, double mean, double sd)
	{
		if (n == 0)
		{
			RANGE("n cannot = 0");
		}
		Vector np(n);
		if (!_normal_distrib_(np.data, n, mean, sd))
		{
			INVALID("Error with creating normal distribution");
		}
		return np;
	}

	Vector randint(uint n, uint max)
	{
		if (n == 0)
		{
			RANGE("n cannot = 0");
		}
		if (max == 0)
		{
			RANGE("max cannot = 0");
		}
		Vector np(n);
		_randint_array_(np.data, n, max);
		return np;
	}

	Vector randchoice(uint n, const char *values)
	{
		if (n == 0)
		{
			RANGE("n cannot = 0");
		}
		if (values == null)
		{
			INVALID("input must not be null");
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

	Vector randchoice(uint n, const double* array, uint arr_size)
	{
		if (n == 0 || arr_size == 0)
		{
			RANGE("n cannot = 0 in ranchoice()");
		}
		if (array == NULL)
		{
			INVALID("array cannot be null in randchoice()");
		}
		Vector np(n);
		srand48(time(NULL));
		for (uint i = 0; i < n; i++)
		{
			double idx_f = drand48() * arr_size;
			int idx = (int) _truncate_doub_(idx_f, 0);
			np.data[i] = array[idx];
		}
		return np;
	}

	Vector randchoice(uint n, const Vector& r)
	{
		if (n == 0)
		{
			RANGE("n cannot = 0 in ranchoice()");
		}
		Vector np = empty_like(r);
		srand48(time(NULL));
		for (uint i = 0; i < n; i++)
		{
			double idx_f = drand48() * r.n;
			int idx = (int) _truncate_doub_(idx_f, 0);
			np.data[i] = r.data[idx];
		}
		return np;
	}

	Vector binomial(uint n, double p, uint size)
	{
		if (p < 0.0 || p > 1.0)
		{
			INVALID("p in binomial() must be between 0 and 1");
		}
		if (size == 0 || n == 0)
		{
			INVALID("size/n in binomial() must not = 0");
		}
		Vector np(size);
		if (!_binomial_array_(np.data, size, n, p))
		{
			INVALID("unable to generate binomial distribution");
		}
		return np;
	}

	Vector poisson(double lam, uint size)
	{
		if (lam < 0)
		{
			INVALID("lam in poisson() must be >= 0");
		}
		if (size == 0)
		{
			INVALID("size of array in poisson() cannot = 0");
		}
		Vector res = empty(size);
		_poisson_array_(res.data, size, lam);
		return res;
	}

	Vector sample(const Vector& rhs, uint n)
	{
		// create arange vector of same size, add 1 so we don't handle zero.
		Vector indices = arange(rhs.n) + 1;
		// shuffle indices randomly using the Durstenfled-Fisher-Yates algorithm.
		_durstenfeld_fisher_yates_<double>(indices.data, indices.n);
		// use mask on rhs to sample values where mask (indices < n) does not contain 0.
		return where(rhs, indices <= (double) n);
	}

	Vector nonzero(const Vector& rhs)
	{
		int cnz = _count_nonzero_array_(rhs.data, rhs.n);
		Vector np = empty(cnz);
		_nonzero_array_(np.data, rhs.data, rhs.n);
		return np;
	}

	Vector flip(const Vector& rhs)
	{
		Vector np = copy(rhs);
		_flip_array_<double>(np.data, np.n);
		return np;
	}

	Vector shuffle(const Vector& rhs)
	{
		Vector np = copy(rhs);
		_durstenfeld_fisher_yates_<double>(np.data, np.n);
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

	Matrix concat(const Vector& lhs, const Vector& rhs, uint axis)
	{
		if (axis == 0)
		{
			return to_matrix(vstack(lhs, rhs));
		}
		else if (axis == 1)
		{
			return hstack(lhs, rhs);
		} else { INVALID_AXIS(); }
	}

	Vector vectorize(const Matrix& rhs, uint axis)
	{
		Vector np(rhs.nvec*rhs.vectors[0]->n);
		if (axis == 0)
		{
			for (uint y = 0; y < rhs.nvec; y++)
			{
				for (uint x = 0; x < rhs.vectors[y]->n; x++)
				{
					np.data[x+y*rhs.vectors[y]->n] = rhs.vectors[y]->data[x];
				}
			}
		} else if (axis == 1)
		{
			for (uint x = 0; x < rhs.vectors[0]->n; x++)
			{
				for (uint y = 0; y < rhs.nvec; y++)
				{
					np.data[y+x*rhs.nvec] = rhs.vectors[y]->data[x];
				}
			}
		} else { INVALID_AXIS(); }

		return np;
	}

	Vector arange(uint end)
	{
		if (end < 0)
		{
			RANGE("in arange(), end must be >= 0");
		}
		// cast end to an unsigned int
		uint n = end;
		Vector v(n);
		for (uint i = 0; i < n; i++)
		{
			v.data[i] = (double) i;
		}
		return v;
	}

	Vector arange(double start, double end, double step)
	{
		if (step <= 0)
		{
			RANGE("step cannot be <= 0");
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
			INVALID("n cannot be <= 0");
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
			INVALID("idx cannot be <= 0");
		}
		if (idx >= rhs.n)
		{
			RANGE("idx cannot be > rhs size");
		}
		Vector np(rhs.n - idx);
		_copy_array_<double>(np.data, rhs.data+idx, rhs.n-idx);
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
		_copy_array_<double>(np.data, rhs.data, idx+1);
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
			INVALID("Unable to floor array.");
		}
		return np;
	}

	Vector ceil(const Vector& rhs)
	{
		Vector np = copy(rhs);
		if (!_ceil_array_(np.data, np.n))
		{
			INVALID("Unable to ceil array.");
		}
		return np;
	}

	int count(const Vector& rhs, double value)
	{
		return _count_array_(rhs.data, rhs.n, value);
	}

	Vector bincount(const Vector& rhs)
	{
		// 1. obtain maximum value
		// 2. create array based on max value (not greater than FLT_EPSILON or 100k)
		// 3. Go through rhs and add each occurence to the appropriate index.

		int max_v = (int) max(rhs);

		if (max_v >= 100000)
		{
			RANGE("in bincount() the max value is above 100k - too large!");
		}
		// cast to uint and create array
		Vector res = zeros(max_v+1);
		if (!_bincount_array_(res.data, rhs.data, res.n, rhs.n))
		{
			INVALID("Unable to perform bincount on array");
		}
		return res;
	}

	Matrix unique(const Vector& rhs, bool get_counts)
	{
		// first, calculate the number of unique values.
		Vector unique = zeros(rhs.n);
		uint counter = 1;
		unique.data[0] = rhs.data[0];
		for (uint i = 1; i < rhs.n; i++)
		{
			bool flag = true;
			for (uint j = 0; j < counter; j++)
			{
				if (CMP(rhs.data[i], rhs.data[j]))
				{
					// not unique
					flag = false;
					break;
				}
			}
			if (flag) {
				unique.data[counter++] = rhs.data[i];
			}
			// unique
		}
		// strip away excess space for concise vector.
		Vector newunique = rstrip(unique, counter - 1);
		if (!get_counts) //if we don't include counts - convert to matrix and return.
		{
			return to_matrix(newunique);
		}
		else // add counts vector and concat with hstack
		{
			Vector counts = empty_like(newunique);
			for (uint i = 0; i < newunique.n; i++)
			{
				counts[i] = count(rhs, newunique[i]);
			}
			// printf("%s\n%s\n", newunique.str(), counts.str());
			return hstack(newunique, counts);
		}
		
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

	Vector add(const Vector& a, const Vector& b)
	{
		return (a + b);
	}

	Vector sub(const Vector& a, const Vector& b)
	{
		return (a - b);
	}

	Vector mult(const Vector& a, const Vector& b)
	{
		return (a * b);
	}

	Vector div(const Vector& a, const Vector& b)
	{
		return (a / b);
	}

	double sum(const Vector& rhs)
	{
		return _summation_array_(rhs.data, rhs.n);
	}

	double mean(const Vector& rhs)
	{
		return sum(rhs) / rhs.n;
	}

	double median(const Vector& rhs, bool isSorted, bool partial_sort)
	{
		// if we don't want the partial_sort by default.
		if (!partial_sort)
		{
			Vector cp = copy(rhs);
			return _median_array_(cp.data, cp.n, isSorted);
		}
		else
		{
			return _median_array_(rhs.data, rhs.n, isSorted);
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

	double percentile(const Vector& rhs, double q)
	{
		// convert q into K to therefore use ksmallest
		uint K = (uint) ((q / 100) * rhs.n);
		// check to ensure K is not greater than n
		if (K < rhs.n)
		{
			return ksmallest(rhs, K);
		}
		else
		{
			INVALID("K generated cannot be >= n in percentile()");
		}
		
	}

	Vector percentiles(const Vector& rhs, const Vector& q)
	{
		// compute K for all q.
		Vector res = empty_like(q);
		for (uint i = 0; i < res.n; i++)
		{
			uint K = (uint) ((q.data[i]/100.0) * rhs.n);
			if (K < rhs.n)
			{
				res[i] = ksmallest(rhs, K);
			}
			else
			{
				INVALID("K generated cannot be >= n in percentiles()");
			}
		}
		return res;
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
			INVALID("cumsum failed!");
		}
		return np;
	}

	Vector cumprod(const Vector& rhs)
	{
		Vector np = ones(rhs.n);
		if (!_cumulative_prod_(np.data, rhs.data, rhs.n))
		{
			INVALID("cumprod failed!");
		}
		return np;
	}

	bool all(const Vector& rhs)
	{
		return _all_true_<double>(rhs.data, rhs.n);
	}

	bool any(const Vector& rhs)
	{
		return _any_true_<double>(rhs.data, rhs.n);
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

	double ksmallest(const Vector& rhs, uint K, bool is_sorted, bool partial_sort)
	{
		if (K == 1)
		{
			return min(rhs);
		}
		else if (!is_sorted)
		{
			if (!partial_sort)
			{
				Vector cp = copy(rhs);
				return _quickselect_(cp.data, 0, cp.n-1, K);
			}
			else
			{
				return _quickselect_(rhs.data, 0, rhs.n-1, K);
			}
		}
		else
		{
			return rhs.data[K];
		}
	}

	double klargest(const Vector& rhs, uint K, bool is_sorted, bool partial_sort)
	{
		if (K == 1)
		{
			return max(rhs);
		}
		else if (!is_sorted)
		{
			if (!partial_sort)
			{
				Vector cp = copy(rhs);
				return _quickselect_(cp.data, 0, cp.n-1, cp.n - K + 1);
			}
			else
			{
				return _quickselect_(rhs.data, 0, rhs.n-1, rhs.n - K + 1);
			}
		}
		else
		{
			return rhs.data[rhs.n - K];
		}
	}

	Vector nsmallest(const Vector& rhs, uint N)
	{
		// based on log(n) if this is bigger than idx (in most cases) we 
		// sort the whole array then pick.
		// else use k-smallest
		if ((int) (log10(rhs.n)) > N)
		{
			Vector r_sorted = sort(rhs);
			return rstrip(r_sorted, N-1);
		}
		else
		{
			//k-smallest
			Vector res = empty(N);
			res[0] = min(rhs);
			for (int k = 1; k < N; k++)
			{
				res[k] = ksmallest(rhs, k);
			}
			return res;
		}
	}

	Vector nlargest(const Vector& rhs, uint N)
	{
		if ((int) (log10(rhs.n)) > N)
		{
			Vector r_sorted = sort(rhs);
			return lstrip(r_sorted, rhs.n - N);
		}
		else
		{
			//k-smallest
			Vector res = empty(N);
			res[0] = max(rhs);
			for (int k = 1; k < N; k++)
			{
				res[k] = klargest(rhs, k);
			}
			return res;
		}
	}

	double cov(const Vector& v, const Vector& w)
	{
		if (v.n == w.n)
		{
			return _cov_array_(v.data, w.data, v.n);
		}
		else
		{
			INVALID("v and w must be the same size in cov(v, w)");
		}
	}

	double corr(const Vector& v, const Vector& w)
	{
		return cov(v, w) / (_square_root_(var(v) * var(w)));
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
			INVALID("order must be 1,2..,n or inf (-1)");
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
			RANGE("lhs must be the same size as the rhs vector");
		}
		// apparently the dot of column.row is the same result as column.column or row.row
		return _vector_dot_array_(lhs.data, rhs.data, rhs.n);
	}

	double inner(const Vector& v, const Vector& w)
	{
		return _vector_dot_array_(v.data, w.data, v.n);
	}

	double magnitude(const Vector& v)
	{
		return _square_root_(dot(v, v));
	}

	Vector normalize(const Vector& v)
	{
		Vector np = copy(v);
		_mult_array_(np.data, np.n, 1.0 / magnitude(v));
		return np;
	}

	Vector standardize(const Vector& v)
	{
		Vector np = copy(v);
		// remove mean
		_sub_array_(np.data, np.n, _summation_array_(np.data, np.n) / np.n);
		// scale std
		_div_array_(np.data, np.n, _std_array_(np.data, np.n));
		return np;
	}

	Vector minmax(const Vector& v)
	{
		double curr_max = _max_value_(v.data, v.n);
		double curr_min = _min_value_(v.data, v.n);
		Vector cp = copy(v);
		// subtract min from copy
		_sub_array_(cp.data, cp.n, curr_min);
		// divide by difference in max/min
		_div_array_(cp.data, cp.n, (curr_max - curr_min));
		return cp;
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
			INVALID("Unable to sine-ify array.");
		}
		return np;
	}

	Vector cos(const Vector& rhs)
	{
		Vector np = copy(rhs);
		if (!_cos_array_(np.data, np.n))
		{
			INVALID("Unable to cos-ify array.");
		}
		return np;
	}

	Vector tan(const Vector& rhs)
	{
		Vector np = copy(rhs);
		if (!_tan_array_(np.data, np.n))
		{
			INVALID("Unable to tan-ify array.");
		}
		return np;
	}

	Vector to_radians(const Vector& rhs)
	{
		Vector v = copy(rhs);
		if (!_to_radians_array_(v.data, v.n))
		{
			INVALID("Unable to convert to radians.");
		}
		return v;
	}

	Vector to_degrees(const Vector& rhs)
	{
		Vector v = copy(rhs);
		if (!_to_degrees_array_(v.data, v.n))
		{
			INVALID("Unable to convert to degrees.");
		}
		return v;
	}

	Vector exp(const Vector& rhs)
	{
		Vector np = copy(rhs);
		if (!_exp_array_(np.data, np.n))
		{
			INVALID("Unable to exp-ify array.");
		}
		return np;
	}

	Vector log(const Vector& rhs)
	{
		Vector np = copy(rhs);
		if (!_log10_array_(np.data, np.n))
		{
			INVALID("Unable to log-ify array.");
		}
		return np;
	}

	Vector sqrt(const Vector& rhs)
	{
		Vector np = copy(rhs);
		if (!_pow_array_(np.data, np.n, 0.5))
		{
			INVALID("Unable to sqrt-ify array.");
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
			INVALID("Unable to exp-ify array.");
		}
		return np;
	}
	Vector power(const Vector& base, const Vector& exponent)
	{
		if (base.n != exponent.n)
		{
			INVALID("base size and exponent size must be equal");
		}
		Vector np(base.n);
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
		_quicksort_(np.data, 0, np.n-1, !((bool) sorter));
		return np;
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
			RANGE("n cannot = 0");
		}
		this->n = n;
		// false/0 = column, true/1 = row
		this->column = !column;
		data = _create_empty_<double>(n);
		if (data == NULL)
		{
			throw std::runtime_error("Unable to allocate memory");
		}
		this->flag_delete = true;
	}

	Vector::Vector(double v1, double v2)
	{
#ifdef _CUMPY_DEBUG_
		printf("constructing vector set2 %x\n", this);
#endif	
		if (isinf(v1) || isinf(v2))
		{
			INVALID("values cannot = infinity in Vector(v1, v2)");
		}
		this->n = 2;
		this->column = true;
		data = _create_empty_<double>(n);
		if (data == NULL)
		{
			throw std::runtime_error("Unable to allocate memory");
		}
		data[0] = v1;
		data[1] = v2;
		this->flag_delete = true;
	}

	Vector::Vector(double v1, double v2, double v3)
	{
#ifdef _CUMPY_DEBUG_
		printf("constructing vector set3 %x\n", this);
#endif	
		if (isinf(v1) || isinf(v2) || isinf(v3))
		{
			INVALID("values cannot = infinity in Vector(v1, v2, v3)");
		}
		this->n = 3;
		this->column = true;
		data = _create_empty_<double>(n);
		if (data == NULL)
		{
			throw std::runtime_error("Unable to allocate memory");
		}
		data[0] = v1;
		data[1] = v2;
		data[2] = v3;
		this->flag_delete = true;
	}

	Vector::Vector(double v1, double v2, double v3, double v4)
	{
#ifdef _CUMPY_DEBUG_
		printf("constructing vector set4 %x\n", this);
#endif	
		if (isinf(v1) || isinf(v2) || isinf(v3) || isinf(v4))
		{
			INVALID("values cannot = infinity in Vector(v1, v2, v3, v4)");
		}
		this->n = 4;
		this->column = true;
		data = _create_empty_<double>(n);
		if (data == NULL)
		{
			throw std::runtime_error("Unable to allocate memory");
		}
		data[0] = v1;
		data[1] = v2;
		data[2] = v3;
		data[3] = v4;
		this->flag_delete = true;
	}

	Vector::Vector(double *array, uint size)
	{
#ifdef _CUMPY_DEBUG_
		printf("constructing vector array set %x\n", this);
#endif
		if (array == NULL)
		{
			INVALID("array in Vector() is empty");
		}
		if (size <= 0)
		{
			INVALID("size in Vector() = 0");
		}
		this->n = size;
		this->column = true;
		data = _create_empty_<double>(n);
		if (data == NULL)
		{
			throw std::runtime_error("Unable to allocate memory");
		}
		if (!_copy_array_<double>(data, array, size))
		{
			INVALID("Unable to copy array in Vector()");
		}
		this->flag_delete = true;
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
				INVALID("Unable to destroy array");
			}
		}
	}

	
	bool Vector::isFloat()
	{
		if (data == NULL)
		{
			INVALID("in isFloat() data = null pointer");
		}
		for (uint i = 0; i < n; i++)
		{
			if (_is_integer_(data[i]))
			{
				return false;
			}
		}
		return true;
	}

	bool Vector::isInteger()
	{
		if (data == NULL)
		{
			INVALID("in isFloat() data = null pointer");
		}
		for (uint i = 0; i < n; i++)
		{
			if (!_is_integer_(data[i]))
			{
				return false;
			}
		}
	}

	char* Vector::str(uint dpoints, bool represent_float)
	{
		if (data == NULL)
		{
			INVALID("in str(), data is pointing to no values (NULL)");
		}
		unsigned int str_len;
		// if we are dealing in floats, calculate the length accordingly, else integerize
		if (represent_float)
		{
			str_len = _str_length_gen_(data, n, dpoints);
		}
		else
		{
			str_len = _str_length_int_gen_(data, n);
		}

		// if we are row-based, add a ".T" at the end to indicate transpose row.
		if (!column)
		{
			str_len += 2;
		}
		char *strg = new char[str_len];
		if (represent_float)
		{
			if (!_str_representation_(strg, data, n, dpoints, 1, !column))
			{
				INVALID("Problem with creating float string representation");
			}
		}
		else
		{
			if (!_int_representation_(strg, data, n, 1, !column))
			{
				INVALID("Problem with creating int string representation");
			}
		}
		// now create string according to rules of length.
		return strg;
	}

	Vector Vector::copy()
	{
		Vector np(n);
		if (!_copy_array_<double>(np.data, data, n))
		{
			INVALID("copy failed!");
		}
		column = np.column;
		flag_delete = np.flag_delete;
		return np;
	}

	Matrix Vector::to_matrix()
	{
		Matrix m = empty(1, n);
		if (!_copy_array_<double>(m.data, data, n))
		{
			INVALID("copy failed!");
		}
		return m;
	}

	Mask Vector::to_mask()
	{
		return Mask(*this);
	}

	Vector& Vector::flip()
	{
		_flip_array_<double>(data,n);
		return *this;
	}

	Vector& Vector::shuffle()
	{
		_durstenfeld_fisher_yates_<double>(data, n);
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
		return _all_true_<double>(data,n);
	}

	bool Vector::any()
	{
		return _any_true_<double>(data,n);
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
			INVALID("order must be 1,2..,n or inf");
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
			INVALID("Unable to sine-ify array.");
		}
		return *this;
	}

	Vector& Vector::cos()
	{
		if (! _cos_array_(data, n))
		{
			INVALID("Unable to cos-ify array.");
		}
		return *this;
	}

	Vector& Vector::tan()
	{
		if (! _tan_array_(data, n))
		{
			INVALID("Unable to tan-ify array.");
		}
		return *this;
	}

	Vector& Vector::exp()
	{
		if (! _exp_array_(data, n))
		{
			INVALID("Unable to exp-ify array.");
		}
		return *this;
	}

	Vector& Vector::log()
	{
		if (! _log10_array_(data, n))
		{
			INVALID("Unable to log-ify array.");
		}
		return *this;
	}

	Vector& Vector::sqrt()
	{
		if (! _pow_array_(data, n, 0.5))
		{
			INVALID("Unable to sqrt-ify array.");
		}
		return *this;
	}


	Vector& Vector::to_radians()
	{
		if (!_to_radians_array_(data, n))
		{
			INVALID("Unable to convert to radians.");
		}
		return *this;
	}

	Vector& Vector::to_degrees()
	{
		if (!_to_degrees_array_(data, n))
		{
			INVALID("Unable to convert to radians.");
		}
		return *this;
	}

	Vector& Vector::pow_base(double base)
	{
		if (! _pow_base_array_(data, n, base))
		{
			INVALID("Unable to pow-ify array.");
		}
		return *this;
	}

	Vector& Vector::pow_exp(double exponent)
	{
		if (! _pow_array_(data, n, exponent))
		{
			INVALID("Unable to pow-ify array.");
		}
		return *this;
	}

	Vector& Vector::fill(double value)
	{
		_fill_array_<double>(data, n, value);
		return *this;
	}

	Vector& Vector::floor()
	{
		if (!_floor_array_(data, n))
		{
			INVALID("Unable to ceil array.");
		}
		return *this;
	}

	Vector& Vector::ceil()
	{
		if (!_ceil_array_(data, n))
		{
			INVALID("Unable to ceil array.");
		}
		return *this;
	}

	double Vector::dot(const Vector& rhs)
	{
		if (n != rhs.n)
		{
			RANGE("lhs must be the same size as the rhs vector");
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
			RANGE("lhs must be the same size as the rhs vector");
		}
		return _square_root_(dot(*this - rhs));
	}

	Vector& Vector::normalize()
	{
		// calculate magnitude as sqrt of dot
		double mag = magnitude();
		// multiply array by 1.0 / mag
		_mult_array_(data, n, 1.0 / mag);
		return *this;
	}

	Vector& Vector::standardize()
	{
		// substract the mean from array
		_sub_array_(data, n, _summation_array_(data, n) / n);
		// divide by standard deviation
		_div_array_(data, n, _std_array_(data, n));
		return *this;
	}

	Vector& Vector::T()
	{
		column = !column;
		return *this;
	}

	Vector& Vector::sort(uint sorter)
	{
		_quicksort_(data, 0, n-1, !((bool) sorter));
		return *this;
	}

	// -------------Function operators -------------------------

	Vector& Vector::add(const Vector& rhs)
	{
		if (rhs.n != n)
		{
			INVALID("rhs size != array size");
		}
		_element_add_(data, rhs.data, n);
		return *this;
	}
	Vector& Vector::add(double value)
	{
		_add_array_(data, n, value);
		return *this;
	}
	Vector& Vector::add(int value)
	{
		_add_array_(data, n, (double) value);
		return *this;
	}

	Vector& Vector::sub(const Vector& rhs)
	{
		if (rhs.n != n)
		{
			INVALID("rhs size != array size");
		}
		_element_sub_(data, rhs.data, n);
		return *this;
	}
	Vector& Vector::sub(double value)
	{
		_sub_array_(data, n, value);
		return *this;
	}
	Vector& Vector::sub(int value)
	{
		_sub_array_(data, n, (double) value);
		return *this;
	}

	Vector& Vector::mult(const Vector& rhs)
	{
		if (rhs.n != n)
		{
			INVALID("rhs size != array size");
		}
		_element_mult_(data, rhs.data, n);
		return *this;
	}
	Vector& Vector::mult(double value)
	{
		_mult_array_(data, n, value);
		return *this;
	}
	Vector& Vector::mult(int value)
	{
		_mult_array_(data, n, (double) value);
		return *this;
	}

	Vector& Vector::div(const Vector& rhs)
	{
		if (rhs.n != n)
		{
			INVALID("rhs size != array size");
		}
		_element_div_(data, rhs.data, n);
		return *this;
	}
	Vector& Vector::div(double value)
	{
		_div_array_(data, n, value);
		return *this;
	}
	Vector& Vector::div(int value)
	{
		_div_array_(data, n, (double) value);
		return *this;
	}

	Mask Vector::lt(const Vector& r)
	{
		return ((*this) < r);
	}
	Mask Vector::lt(double r)
	{
		return ((*this) < r);
	}
	Mask Vector::lt(int r)
	{
		return ((*this) < r);
	}

	Mask Vector::lte(const Vector& r)
	{
		return ((*this) <= r);
	}
	Mask Vector::lte(double r)
	{
		return ((*this) <= r);
	}
	Mask Vector::lte(int r)
	{
		return ((*this) <= r);
	}

	Mask Vector::mt(const Vector& r)
	{
		return ((*this) > r);
	}
	Mask Vector::mt(double r)
	{
		return ((*this) > r);
	}
	Mask Vector::mt(int r)
	{
		return ((*this) > r);
	}

	Mask Vector::mte(const Vector& r)
	{
		return ((*this) >= r);
	}
	Mask Vector::mte(double r)
	{
		return ((*this) >= r);
	}
	Mask Vector::mte(int r)
	{
		return ((*this) >= r);
	}

	Mask Vector::eq(const Vector& r)
	{
		return ((*this) == r);
	}
	Mask Vector::eq(double r)
	{
		return ((*this) == r);
	}
	Mask Vector::eq(int r)
	{
		return ((*this) == r);
	}

	Mask Vector::neq(const Vector& r)
	{
		return ((*this) != r);
	}
	Mask Vector::neq(double r)
	{
		return ((*this) != r);
	}
	Mask Vector::neq(int r)
	{
		return ((*this) != r);
	}


	//  --------------- Instance Operator overloads ------------------------

	Vector Vector::operator[](const Mask& rhs)
	{
		if (rhs.n != n)
		{
			INVALID("rhs size != array size");
		}
		uint res_n = _boolean_summation_array_(rhs.data, rhs.n);
		// new array size is the sum of trues.
		Vector res = empty(res_n);
		if (!_copy_from_mask_array_<double>(res.data, data, rhs.data, rhs.n, false))
		{
			INVALID("Unable to copy from mask vector in 'operator[]'.");
		}
		return res;
	}

	Vector& Vector::operator+=(const Vector& rhs)
	{
		if (rhs.n != n)
		{
			INVALID("rhs size != array size");
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
			INVALID("rhs size != array size");
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
			INVALID("rhs size != array size");
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
			RANGE("rhs size != array size");
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
			RANGE("lhs and rhs vector not the same size");
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
			RANGE("lhs and rhs vector not same size");
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
			RANGE("lhs and rhs vector not same size");
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
			throw std::logic_error("cannot divide by 0!");
		}
		Vector np = _copy_vector_(lhs);
		np /= (double) value;
		return np;
	}
	Vector operator/(const Vector& lhs, double value)
	{
		if (CMP(value, 0.0))
		{
			throw std::logic_error("cannot divide by 0!");
		}
		Vector np = _copy_vector_(lhs);
		np /= value;
		return np;
	}
	Vector operator/(const Vector& lhs, const Vector& rhs)
	{
		if (lhs.n != rhs.n)
		{
			RANGE("lhs and rhs vector not same size");
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
			RANGE("base and exponent vector not same size");
		}
		Vector np = _copy_vector_(base);
		for (uint i = 0; i < base.n; i++)
		{
			np.data[i] = _c_power_(base.data[i], exponent.data[i]);
		}
		return np;
	}

	Mask operator==(const Vector& l, const Vector& r)
	{
		if (l.n != r.n)
		{
			RANGE("l and r not same size!");
		}
		Mask res(l.n);
		_element_equals_(res.data, l.data, r.data, l.n);
		return res;
	}
	Mask operator==(const Vector& l, double r)
	{
		Mask res(l.n);
		_equals_array_(res.data, l.data, l.n, r);
		return res;
	}
	Mask operator==(const Vector& l, int r)
	{
		Mask res(l.n);
		_equals_array_(res.data, l.data, l.n, (double) r);
		return res;
	}

	Mask operator!=(const Vector& l, const Vector& r)
	{
		if (l.n != r.n)
		{
			RANGE("l and r not same size!");
		}
		Mask res(l.n);
		_element_not_equals_(res.data, l.data, r.data, l.n);
		return res;
	}
	Mask operator!=(const Vector& l, double r)
	{
		Mask res(l.n);
		_not_equals_array_(res.data, l.data, l.n, r);
		return res;
	}
	Mask operator!=(const Vector& l, int r)
	{
		Mask res(l.n);
		_not_equals_array_(res.data, l.data, l.n, (double) r);
		return res;
	}

	Mask operator<(const Vector& l, const Vector& r)
	{
		if (l.n != r.n)
		{
			RANGE("l and r not same size!");
		}
		Mask res(l.n);
		_element_less_than_(res.data, l.data, r.data, l.n, false);
		return res;
	}
	Mask operator<(const Vector& l, double r)
	{
		Mask res(l.n);
		_less_than_array_(res.data, l.data, l.n, r, false);
		return res;
	}
	Mask operator<(const Vector& l, int r)
	{
		Mask res(l.n);
		_less_than_array_(res.data, l.data, l.n, (double) r, false);
		return res;
	}

	Mask operator<=(const Vector& l, const Vector& r)
	{
		if (l.n != r.n)
		{
			RANGE("l and r not same size!");
		}
		Mask res(l.n);
		_element_less_than_(res.data, l.data, r.data, l.n, true);
		return res;
	}
	Mask operator<=(const Vector& l, double r)
	{
		Mask res(l.n);
		_less_than_array_(res.data, l.data, l.n, r, true);
		return res;
	}
	Mask operator<=(const Vector& l, int r)
	{
		Mask res(l.n);
		_less_than_array_(res.data, l.data, l.n, (double) r, true);
		return res;
	}

	Mask operator>(const Vector& l, const Vector& r)
	{
		if (l.n != r.n)
		{
			RANGE("l and r not same size!");
		}
		Mask res(l.n);
		_element_greater_than_(res.data, l.data, r.data, l.n, false);
		return res;
	}
	Mask operator>(const Vector& l, double r)
	{
		Mask res(l.n);
		_greater_than_array_(res.data, l.data, l.n, r, false);
		return res;
	}
	Mask operator>(const Vector& l, int r)
	{
		Mask res(l.n);
		_greater_than_array_(res.data, l.data, l.n, (double) r, false);
		return res;
	}

	Mask operator>=(const Vector& l, const Vector& r)
	{
		if (l.n != r.n)
		{
			RANGE("l and r not same size!");
		}
		Mask res(l.n);
		_element_greater_than_(res.data, l.data, r.data, l.n, true);
		return res;
	}
	Mask operator>=(const Vector& l, double r)
	{
		Mask res(l.n);
		_greater_than_array_(res.data, l.data, l.n, r, true);
		return res;
	}
	Mask operator>=(const Vector& l, int r)
	{
		Mask res(l.n);
		_greater_than_array_(res.data, l.data, l.n, (double) r, true);
		return res;
	}

	/********************************************************************************************


		PRIVATE ACCESSORY FUNCTIONS 


	*///////////////////////////////////////////////////////////////////////////////////////////

	Vector _copy_vector_(const Vector& v)
	{
		Vector np(v.n);
		_copy_array_<double>(np.data, v.data, v.n);
		np.column = v.column;
		np.flag_delete = v.flag_delete;
		return np;
	}


}

#endif

