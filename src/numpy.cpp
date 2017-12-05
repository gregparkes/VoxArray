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

    numpy.cpp
*/

#ifndef __cplusplus_numpycpp__
#define __cplusplus_numpycpp__

#include <stdexcept>
#include <cstring>
#include <iostream>

#include "numpy.h"
#include "numstatic.cpp"

namespace numpy {


	/*
	 * The static functions here. ---------------------------------
	 */

	Vector empty(uint n)
	{
		Vector np(n);
		return np;
	}

	Matrix empty(uint ncol, uint nrow)
	{
		Matrix m(ncol,nrow);
		return m;
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

	Matrix zeros(uint ncol, uint nrow)
	{
		Matrix m(ncol,nrow);
		if (!_fill_array_(m.data, ncol*nrow, 0.0))
		{
			throw std::invalid_argument("Fill Error");
		}
		return m;
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

	Matrix ones(uint ncol, uint nrow)
	{
		Matrix m(ncol,nrow);
		if (!_fill_array_(m.data, ncol*nrow, 1.0))
		{
			throw std::invalid_argument("Fill Error");
		}
		return m;
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

	Matrix fill(uint ncol, uint nrow, double val)
	{
		Matrix m(ncol,nrow);
		if (!_fill_array_(m.data, ncol*nrow, val))
		{
			throw std::invalid_argument("Fill Error");
		}
		return m;
	}

	char* shape(const Matrix& rhs)
	{
		// determine length of each integer in characters.
				// then it's (int1,int2) so character lengths + 3, allocate memory
		if (rhs.vectors[0]->n == 0 || rhs.nvec == 0)
		{
			throw std::invalid_argument("column or row length cannot = 0");
		}
		uint n_digit_row = _n_digits_in_int_(rhs.vectors[0]->n);
		uint n_digit_col = _n_digits_in_int_(rhs.nvec);
		char* strg = new char[n_digit_row + n_digit_col + 4];
		if (!_str_shape_func_(strg, rhs.vectors[0]->n, rhs.nvec, n_digit_row, n_digit_col,
				n_digit_row + n_digit_col + 4))
		{
			throw std::invalid_argument("Problem with creating string representation");
		}
		return strg;
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

	char* str(const Matrix& rhs, uint dpoints)
	{
		uint str_len = _str_length_gen_(rhs.vectors[0]->data, rhs.vectors[0]->n, dpoints);
		uint total_length = str_len + 2; // one at start, then \n and space
		char* strg = new char[total_length*rhs.nvec+1];
		char* ptr = strg;
		*ptr++ = '[';
		for (uint i = 0; i < rhs.nvec-1; i++)
		{
			if (!_str_representation_(ptr, rhs.vectors[i]->data,
								rhs.vectors[i]->n, dpoints, 0))
			{
				throw std::invalid_argument("Problem with creating string representation");
			}
			ptr += str_len-1;
			*ptr++ = '\n';
			*ptr++ = ' ';
		}
		if (!_str_representation_(ptr, rhs.vectors[rhs.nvec-1]->data,
										rhs.vectors[rhs.nvec-1]->n, dpoints, 1))
		{
			throw std::invalid_argument("Problem with creating string representation");
		}
		ptr += str_len-1;
		*ptr = ']';
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
		return np;
	}

	Matrix copy(const Matrix& rhs)
	{
		Matrix m(rhs.nvec, rhs.vectors[0]->n);
		uint nbyn = rhs.nvec*rhs.vectors[0]->n;
		if (!_copy_array_(m.data, rhs.data, nbyn))
		{
			throw std::invalid_argument("copy failed!");
		}
		uint i;
		// update new vectors on other parameters
	#ifdef _OPENMP
		#pragma omp parallel for if(rhs.nvec>100000) schedule(static)
	#endif
		for (i = 0; i < rhs.nvec; i++)
		{
			m.vectors[i]->column = rhs.vectors[i]->column;
			m.vectors[i]->flag_delete = rhs.vectors[i]->flag_delete;
			m.vectors[i]->n = rhs.vectors[i]->n;
		}
		return m;
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

	Vector nonzero(const Vector& rhs)
	{
		int cnz = _count_nonzero_array_(rhs.data, rhs.n);
		Vector np(cnz);
		_nonzero_array_(np.data, rhs.data, rhs.n);
		return np;
	}

	Vector nonzero(const Matrix& rhs)
	{
		int cnz = _count_nonzero_array_(rhs.data, rhs.nvec*rhs.vectors[0]->n);
		Vector np(cnz);
		_nonzero_array_(np.data, rhs.data, rhs.nvec*rhs.vectors[0]->n);
		return np;
	}

	Matrix unique(const Vector& rhs)
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
				if (rhs.data[i] == rhs.data[j])
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
		// now create our matrix, and count our variables.
		Vector newunique = rstrip(unique, counter - 1);
		Vector counts = empty_like(newunique);
		for (uint i = 0; i < newunique.n; i++)
		{
			counts[i] = count(rhs, newunique[i]);
		}
		// printf("%s\n%s\n", newunique.str(), counts.str());
		return hstack(newunique, counts);
	}

	Vector flip(const Vector& rhs)
	{
		Vector np(rhs.n);
		uint i;
	#ifdef _OPENMP
		#pragma omp parallel for if(rhs.n>100000) schedule(static)
	#endif
		for (i = 0; i < rhs.n; i++)
		{
			np.data[i] = rhs.data[rhs.n-1-i];
		}
		return np;
	}

	Matrix flip(const Matrix& rhs, uint axis)
	{
		Matrix result = copy(rhs);
		if (axis == 0)
		{
		#ifdef _OPENMP
			#pragma omp parallel for if(rhs.nvec>100000) schedule(static)
		#endif
			for (uint i = 0; i < rhs.nvec; i++)
			{
				result.vectors[i]->flip();
			}
		} else if (axis == 1)
		{
		#ifdef _OPENMP
			#pragma omp parallel for if(rhs.nvec>100000) schedule(static)
		#endif
			for (uint i = 0; i < rhs.nvec; i++)
			{
				for (uint j = 0; j < rhs.vectors[0]->n; j++)
				{
					result.vectors[i]->data[j] = rhs.vectors[i]->data[rhs.vectors[0]->n-1-j];
				}
			}
		} else
		{
			throw std::invalid_argument("axis must be 0 or 1");
		}
		return result;
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

	Vector empty_like(const Vector& rhs)
	{
		return Vector(rhs.n);
	}

	Matrix empty_like(const Matrix& rhs)
	{
		return Matrix(rhs.nvec, rhs.vectors[0]->n);
	}

	Vector zeros_like(const Vector& rhs)
	{
		Vector np(rhs.n);
		_fill_array_(np.data, np.n, 0.0);
		return np;
	}

	Matrix zeros_like(const Matrix& rhs)
	{
		Matrix res(rhs.nvec, rhs.vectors[0]->n);
		_fill_array_(rhs.data, rhs.nvec*rhs.vectors[0]->n, 0.0);
		return res;
	}

	Vector ones_like(const Vector& rhs)
	{
		Vector np(rhs.n);
		_fill_array_(np.data, np.n, 1.0);
		return np;
	}

	Matrix ones_like(const Matrix& rhs)
	{
		Matrix res(rhs.nvec, rhs.vectors[0]->n);
		_fill_array_(rhs.data, rhs.nvec*rhs.vectors[0]->n, 1.0);
		return res;
	}

	Vector rand(uint n)
	{
		if (n == 0)
		{
			throw std::range_error("n cannot = 0");
		}
		Vector np(n);
		srand48(time(NULL));
	#ifdef _OPENMP
		#pragma omp parallel for if(n>100000) schedule(static)
	#endif
		for (uint i = 0; i < n; i++)
		{
			np.data[i] = drand48();
		}
		return np;
	}

	Matrix rand(uint ncol, uint nrow)
	{
		if (ncol == 0 || nrow == 0)
		{
			throw std::range_error("ncol or nrow cannot = 0");
		}
		Matrix res(ncol,nrow);
		srand48(time(NULL));
	#ifdef _OPENMP
		#pragma omp parallel for if(ncol*nrow>100000) schedule(static)
	#endif
		for (uint i = 0; i < ncol*nrow; i++)
		{
			res.data[i] = drand48();
		}
		return res;
	}

	// with sd = 1.0, mean = 0.0
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

	Matrix randn(uint ncol, uint nrow)
	{
		if (ncol == 0 || nrow == 0)
		{
			throw std::range_error("ncol or nrow cannot = 0");
		}
		Matrix res(ncol,nrow);
		if (!_normal_distrib_(res.data, ncol*nrow, 0.0, 1.0))
		{
			throw std::invalid_argument("Error with creating normal distribution");
		}
		return res;
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

	Matrix normal(uint ncol, uint nrow, double mean, double sd)
	{
		if (ncol == 0 || nrow == 0)
		{
			throw std::range_error("ncol or nrow cannot = 0");
		}
		Matrix res(ncol,nrow);
		if (!_normal_distrib_(res.data, ncol*nrow, mean, sd))
		{
			throw std::invalid_argument("Error with creating normal distribution");
		}
		return res;
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
		srand48(time(NULL));
		Vector np(n);
	#ifdef _OPENMP
		#pragma omp parallel for if(n>100000) schedule(static)
	#endif
		for (uint i = 0; i < n; i++)
		{
			np.data[i] = drand48() * max;
		}
		np.ceil();
		return np;
	}

	Matrix randint(uint ncol, uint nrow, uint max)
	{
		if (ncol == 0 || nrow == 0)
		{
			throw std::range_error("ncol or nrow cannot = 0");
		}
		if (max == 0)
		{
			throw std::range_error("max cannot = 0");
		}
		srand48(time(NULL));
		Matrix res(ncol,nrow);
	#ifdef _OPENMP
		#pragma omp parallel for if((ncol*nrow)>100000) schedule(static)
	#endif
		for (uint i = 0; i < ncol*nrow; i++)
		{
			res.data[i] = drand48() * max;
		}
		if (!_ceil_array_(res.data, ncol*nrow))
		{
			throw std::invalid_argument("Unable to ceil array");
		}
		return res;
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
	#ifdef _OPENMP
		#pragma omp parallel for if(n>100000) schedule(static)
	#endif
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

	Matrix randchoice(uint ncol, uint nrow, const char* values)
	{
		if (ncol == 0 || nrow == 0)
		{
			throw std::range_error("ncol or nrow cannot = 0");
		}
		if (values == null)
		{
			throw std::invalid_argument("input must not be null");
		}
		int strn = strlen(values);
		Matrix res(ncol,nrow);
		// now we somehow parse the string

		double *vals = _parse_string_to_array_(values, &strn);
		if (vals == null)
		{
			throw std::runtime_error("Unable to parse string into array");
		}
		srand48(time(NULL));
	#ifdef _OPENMP
		#pragma omp parallel for if(ncol*nrow>100000) schedule(static)
	#endif
		for (uint i = 0; i < ncol*nrow; i++)
		{
			// create random float
			double idx_f = drand48() * strn;
			// pass to an array and floor it (i.e 2.90 becomes 2, an index for values*)
			int idx = (int) _truncate_doub_(idx_f, 0);
			// set data using index and values
			res.data[i] = vals[idx];
		}
		return res;
	}

	uint binomial(uint n, double p)
	{
		return (uint) _binomial_coefficient_(n,p);
	}

	Vector binomial(const Vector& n, const Vector& p)
	{
		if (n.n != p.n)
		{
			throw std::range_error("n and p vectors must be same length");
		}
		Vector np(n.n);
	#ifdef _OPENMP
		#pragma omp parallel for if(n.n>100000) schedule(static) shared(n,p)
	#endif
		for (uint i = 0; i < n.n; i++)
		{
			np.data[i] = _binomial_coefficient_(n.data[i], p.data[i]);
		}
		return np;
	}

	long poisson(double lam)
	{
		return _poisson_coefficient_(lam);
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

	Matrix clip(const Matrix& rhs, double a_min, double a_max)
	{
		Matrix res = copy(rhs);
		_clip_array_(res.data, res.nvec*res.vectors[0]->n, a_min, a_max);
		return res;
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

	Matrix floor(const Matrix& rhs)
	{
		Matrix m = copy(rhs);
		if (!_floor_array_(m.data, m.nvec*m.vectors[0]->n))
		{
			throw std::invalid_argument("Unable to floor matrix.");
		}
		return m;
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

	Matrix ceil(const Matrix& rhs)
	{
		Matrix m = copy(rhs);
		if (!_ceil_array_(m.data, m.nvec*m.vectors[0]->n))
		{
			throw std::invalid_argument("Unable to ceil matrix.");
		}
		return m;
	}

	int count(const Vector& rhs, double value)
	{
		return _count_array_(rhs.data, rhs.n, value);
	}

	int count(const Matrix& rhs, double value)
	{
		return _count_array_(rhs.data, rhs.nvec*rhs.vectors[0]->n, value);
	}

	Vector count(const Matrix& rhs, double value, uint axis)
	{
		if (axis == 0)
		{
			Vector v(rhs.nvec);
			for (uint i = 0; i < rhs.nvec; i++)
			{
				v.data[i] = rhs.vectors[i]->count(value);
			}
			return v;
		}
		else if (axis == 1)
		{
			Vector v(rhs.vectors[0]->n);
			for (uint i = 0; i < rhs.vectors[0]->n; i++)
			{
				v.data[i] = _matrix_rowwise_count_(rhs.data, rhs.nvec, rhs.vectors[0]->n, i, value);
			}
			return v;
		}
		else
		{
			throw std::invalid_argument("axis must be 0 or 1");
		}
	}

	bool isColumn(const Vector& rhs)
	{
		return rhs.column;
	}

	bool isRow(const Vector& rhs)
	{
		return ! rhs.column;
	}

	int count_nonzero(const Vector& rhs)
	{
		return _count_nonzero_array_(rhs.data, rhs.n);
	}

	int count_nonzero(const Matrix& rhs)
	{
		return _count_nonzero_array_(rhs.data, rhs.nvec*rhs.vectors[0]->n);
	}

	Vector count_nonzero(const Matrix& rhs, uint axis)
	{
		if (axis == 0)
		{
			Vector np(rhs.nvec);
			for (uint i = 0; i < rhs.nvec; i++)
			{
				np.data[i] = rhs.vectors[i]->count_nonzero();
			}
			return np;
		} else if (axis == 1)
		{
			Vector np(rhs.vectors[0]->n);
			for (uint i = 0; i < rhs.vectors[0]->n; i++)
			{
				np.data[i] = _matrix_rowwise_count_nonzero_(rhs.data, rhs.nvec, rhs.vectors[0]->n, i);
			}
			return np;
		} else
		{
			throw std::invalid_argument("axis must be 0 or 1.");
		}
	}

	Vector abs(const Vector& rhs)
	{
		Vector np = copy(rhs);
		_absolute_array_(rhs.data, rhs.n);
		return np;
	}

	Matrix abs(const Matrix& rhs)
	{
		Matrix res = copy(rhs);
		_absolute_array_(res.data, res.nvec*res.vectors[0]->n);
		return res;
	}

	double sum(const Vector& rhs)
	{
		return _summation_array_(rhs.data, rhs.n);
	}

	double sum(const Matrix& rhs)
	{
		return _summation_array_(rhs.data, rhs.nvec*rhs.vectors[0]->n);
	}

	Vector sum(const Matrix& rhs, uint axis)
	{
		if (axis == 0)
		{
			Vector res(rhs.nvec);
			for (uint i = 0; i < rhs.nvec; i++)
			{
				res.data[i] = rhs.vectors[i]->sum();
			}
			return res;
		} else if (axis == 1)
		{
			Vector res(rhs.vectors[0]->n);
			for (uint i = 0; i < rhs.vectors[0]->n; i++)
			{
				res.data[i] = _matrix_rowwise_summation_(rhs.data, rhs.nvec, rhs.vectors[0]->n, i);
			}
			return res;
		} else
		{
			throw std::invalid_argument("axis must be 0 or 1");
		}

	}

	double prod(const Vector& rhs)
	{
		return _prod_array_(rhs.data, rhs.n);
	}

	double prod(const Matrix& rhs)
	{
		return _prod_array_(rhs.data, rhs.nvec*rhs.vectors[0]->n);
	}

	Vector prod(const Matrix& rhs, uint axis)
	{
		if (axis == 0)
		{
			Vector res(rhs.nvec);
			for (uint i = 0; i < rhs.nvec; i++)
			{
				res.data[i] = rhs.vectors[i]->prod();
			}
			return res;
		} else if (axis == 1)
		{
			Vector res(rhs.vectors[0]->n);
			for (uint i = 0; i < rhs.vectors[0]->n; i++)
			{
				res.data[i] = _matrix_rowwise_prod_(rhs.data, rhs.nvec, rhs.vectors[0]->n, i);
			}
			return res;
		} else
		{
			throw std::invalid_argument("axis must be 0 or 1");
		}
	}

	Vector cumsum(const Vector& rhs)
	{
		Vector np = zeros(rhs.n);
		_cumulative_sum_(np.data, rhs.data, rhs.n);
		return np;
	}

	Matrix cumsum(const Matrix& rhs, uint axis)
	{
		if (axis == 0)
		{
			Matrix m = zeros_like(rhs);
		#ifdef _OPENMP
			#pragma omp parallel for if((rhs.nvec)>100000) schedule(static)
		#endif
			for (uint i = 0; i < rhs.nvec; i++)
			{
				_cumulative_sum_(m.vectors[i]->data, rhs.vectors[i]->data, rhs.vectors[i]->n);
			}
			return m;
		} else if (axis == 1)
		{
			Matrix m = copy(rhs);
		#ifdef _OPENMP
			#pragma omp parallel for if((rhs.vectors[0]->n)>100000) schedule(static)
		#endif
			for (uint i = 0; i < rhs.vectors[0]->n; i++)
			{
				for (uint j = 0; j < rhs.nvec; j++)
				{
					for (uint k = j; k >= 0; k--)
					{
						m.data[j] += rhs.data[k];
					}
				}
			}
			return m;
		} else
		{
			throw std::invalid_argument("axis must be 0 or 1");
		}
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

	Matrix adjacsum(const Matrix& rhs)
	{
		Matrix m = empty_like(rhs);
		uint nrow = rhs.vectors[0]->n;
		m.data[0] = rhs.data[0] + rhs.data[1] + rhs.data[nrow];
		m.data[(rhs.nvec-1)*nrow] = rhs.data[(rhs.nvec-1)*nrow] + rhs.data[1+(rhs.nvec-1)*nrow] +
									rhs.data[(rhs.nvec-2)*nrow];
		m.data[rhs.nvec-1] = rhs.data[rhs.nvec-1] + rhs.data[rhs.nvec-2] + rhs.data[rhs.nvec*2];
		m.data[(rhs.nvec-1)*nrow+(nrow-1)] = rhs.data[(rhs.nvec-1)*nrow+(nrow-1)] + rhs.data[(rhs.nvec-1)*nrow+(nrow-2)] +
											 rhs.data[(rhs.nvec-2)*nrow+(nrow-1)];
		for (uint i = 1; i < rhs.nvec-1; i++)
		{
			m.data[i] = rhs.data[i] + rhs.data[i-1] + rhs.data[i+1] + rhs.data[i+nrow];
			m.data[i+(rhs.nvec-1)*nrow] = rhs.data[i+(rhs.nvec-1)*nrow] + rhs.data[(i-1)+(rhs.nvec-1)*nrow] +
					rhs.data[(i+1)+(rhs.nvec-1)*nrow] + rhs.data[i+(rhs.nvec-2)*nrow];
		}
		for (uint i = 1; i < nrow-1; i++)
		{
			m.data[i*nrow] = rhs.data[i*nrow] + rhs.data[1+i*nrow] + rhs.data[i*(nrow-1)] + rhs.data[i*(nrow+1)];
			m.data[(rhs.nvec-1)+i*nrow] = m.data[(rhs.nvec-1)+i*nrow] + rhs.data[(rhs.nvec-2)+i*nrow] +
					rhs.data[(rhs.nvec-1)+i*(nrow-1)] + rhs.data[(rhs.nvec-1)+i*(nrow+1)];
		}
		for (uint y = 2; y < rhs.nvec-2; y++)
		{
			for (uint x = 2; x < nrow-2; x++)
			{
				m.data[x+y*nrow] = rhs.data[x+y*nrow] + rhs.data[x+1+y*nrow] + rhs.data[x-1+y*nrow] +
								   rhs.data[x+(y+1)*nrow] + rhs.data[x+(y-1)*nrow];
			}
		}
		return m;
	}

	Vector cumprod(const Vector& rhs)
	{
		Vector np = zeros(rhs.n);
		np.data[0] = rhs.data[0];
		for (uint j = 1; j < rhs.n; j++)
		{
			np.data[j] = rhs.data[j] * rhs.data[j-1];
		}
		return np;
	}

	double trapz(const Vector& y, const Vector& x, double dx)
	{
		double total = 0.0;
		for (uint i = 1; i < y.n-1; i++)
		{
			total += 2*y.data[i];
		}
		return (dx/2) * (y.data[0] + total + y.data[y.n-1]);
	}

	bool all(const Vector& rhs)
	{
		return _all_true_(rhs.data, rhs.n);
	}

	bool all(const Matrix& rhs)
	{
		return _all_true_(rhs.data, rhs.nvec*rhs.vectors[0]->n);
	}

	bool any(const Vector& rhs)
	{
		return _any_true_(rhs.data, rhs.n);
	}

	bool any(const Matrix& rhs)
	{
		return _any_true_(rhs.data, rhs.nvec*rhs.vectors[0]->n);
	}

	double min(const Vector& rhs)
	{
		return _min_value_(rhs.data, rhs.n);
	}

	double min(const Matrix& rhs)
	{
		return _min_value_(rhs.data, rhs.nvec*rhs.vectors[0]->n);
	}

	Vector min(const Matrix& rhs, uint axis)
	{
		if (axis == 0)
		{
			Vector v = empty(rhs.nvec);
			for (uint i = 0; i < rhs.nvec; i++)
			{
				v.data[i] = rhs.vectors[i]->min();
			}
			return v;
		} else if (axis == 1)
		{
			Vector v = empty(rhs.vectors[0]->n);
			for (uint i = 0; i < rhs.vectors[0]->n; i++)
			{
				v.data[i] = _matrix_rowwise_min_value_(rhs.data, rhs.nvec, rhs.vectors[i]->n, i);
			}
			return v;
		} else
		{
			throw std::invalid_argument("axis must be 0 or 1");
		}
	}

	double max(const Vector& rhs)
	{
		return _max_value_(rhs.data, rhs.n);
	}

	double max(const Matrix& rhs)
	{
		return _max_value_(rhs.data, rhs.nvec*rhs.vectors[0]->n);
	}

	Vector max(const Matrix& rhs, uint axis)
	{
		if (axis == 0)
		{
			Vector v = empty(rhs.nvec);
			for (uint i = 0; i < rhs.nvec; i++)
			{
				v.data[i] = rhs.vectors[i]->max();
			}
			return v;
		} else if (axis == 1)
		{
			Vector v = empty(rhs.vectors[0]->n);
			for (uint i = 0; i < rhs.vectors[0]->n; i++)
			{
				v.data[i] = _matrix_rowwise_max_value_(rhs.data, rhs.nvec, rhs.vectors[i]->n, i);
			}
			return v;
		} else
		{
			throw std::invalid_argument("axis must be 0 or 1");
		}
	}

	double mean(const Vector& rhs)
	{
		return sum(rhs) / rhs.n;
	}

	Vector mean(const Matrix& rhs, uint axis)
	{
		if (axis == 0)
		{
			Vector v = empty(rhs.nvec);
			for (uint i = 0; i < rhs.nvec; i++)
			{
				v.data[i] = rhs.vectors[i]->mean();
			}
			return v;
		} else if (axis == 1)
		{
			Vector v = empty(rhs.vectors[0]->n);
			for (uint i = 0; i < rhs.vectors[0]->n; i++)
			{
				v.data[i] = _matrix_rowwise_summation_(rhs.data, rhs.nvec, rhs.vectors[i]->n, i) / rhs.nvec;
			}
			return v;
		} else
		{
			throw std::invalid_argument("axis must be 0 or 1");
		}
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

	Vector std(const Matrix& rhs, uint axis)
	{
		if (axis == 0)
		{
			Vector v = empty(rhs.nvec);
			for (uint i = 0; i < rhs.nvec; i++)
			{
				v.data[i] = rhs.vectors[i]->std();
			}
			return v;
		} else if (axis == 1)
		{
			Vector v = empty(rhs.vectors[0]->n);
			for (uint i = 0; i < rhs.vectors[0]->n; i++)
			{
				v.data[i] = _matrix_rowwise_std_(rhs.data, rhs.nvec, rhs.vectors[i]->n, i);
			}
			return v;
		} else
		{
			throw std::invalid_argument("axis must be 0 or 1");
		}
	}

	double var(const Vector& rhs)
	{
		return _var_array_(rhs.data, rhs.n);
	}

	Vector var(const Matrix& rhs, uint axis)
	{
		if (axis == 0)
		{
			Vector v = empty(rhs.nvec);
			for (uint i = 0; i < rhs.nvec; i++)
			{
				v.data[i] = rhs.vectors[i]->var();
			}
			return v;
		} else if (axis == 1)
		{
			Vector v = empty(rhs.vectors[0]->n);
			for (uint i = 0; i < rhs.vectors[0]->n; i++)
			{
				v.data[i] = _matrix_rowwise_var_(rhs.data, rhs.nvec, rhs.vectors[i]->n, i);
			}
			return v;
		} else
		{
			throw std::invalid_argument("axis must be 0 or 1");
		}
	}

	double cov(const Vector& v, const Vector& w)
	{
		return dot(v, w) / (v.n - 1);
	}

	Matrix cov(const Matrix& A)
	{
		Matrix m = empty(A.nvec, A.nvec);
		// calculate diagonals as variance.
		for (uint i = 0; i < m.nvec; i++)
		{
			m.data[i+i*m.nvec] = m.vectors[i]->var();
		}
		// calculate covariances.
		for (uint i = 0; i < m.nvec-1; i++)
		{
			for (uint j = i+1; j < m.nvec; j++)
			{
				m.vectors[i]->data[j] = m.vectors[j]->data[i] = cov(*A.vectors[i], *A.vectors[j]);
			}
		}
		return m;
	}

	Matrix corr(const Matrix& A)
	{
		Matrix m = cov(A);
		Matrix res = copy(m);
		for (uint j = 0; j < m.nvec; j++)
		{
			for (uint i = 0; i < m.nvec; i++)
			{
				if (i == j)
				{
					if (m.data[i+j*m.nvec] == 0.0)
					{
						throw std::logic_error("Covariance diagonal elements cannot = 0!");
					}
					else
					{
						res.data[i+j*m.nvec] = 1.0;
					}
				}
				else
				{
					res.data[i+j*m.nvec] = m.data[i+j*m.nvec] / (_square_root_(m.data[i+i*m.nvec])*
											_square_root_(m.data[j+j*m.nvec]));
				}
			}
		}
		return res;
	}

	uint argmin(const Vector& rhs)
	{
		return _min_index_(rhs.data, rhs.n);
	}

	Vector argmin(const Matrix& rhs, uint axis)
	{
		if (axis == 0)
		{
			Vector v = empty(rhs.nvec);
			for (uint i = 0; i < rhs.nvec; i++)
			{
				v.data[i] = rhs.vectors[i]->argmin();
			}
			return v;
		}
		else if (axis == 1)
		{
			Vector v = empty(rhs.vectors[0]->n);
			for (uint i = 0; i < v.n; i++)
			{
				v.data[i] = _matrix_rowwise_min_index_(rhs.data, rhs.nvec, rhs.vectors[0]->n, i);
			}
			return v;
		}
		else
		{
			throw std::invalid_argument("axis must be 0 or 1");
		}
	}

	uint argmax(const Vector& rhs)
	{
		return _max_index_(rhs.data, rhs.n);
	}

	Vector argmax(const Matrix& rhs, uint axis)
	{
		if (axis == 0)
		{
			Vector v = empty(rhs.nvec);
			for (uint i = 0; i < rhs.nvec; i++)
			{
				v.data[i] = rhs.vectors[i]->argmax();
			}
			return v;
		}
		else if (axis == 1)
		{
			Vector v = empty(rhs.vectors[0]->n);
			for (uint i = 0; i < v.n; i++)
			{
				v.data[i] = _matrix_rowwise_max_index_(rhs.data, rhs.nvec, rhs.vectors[0]->n, i);
			}
			return v;
		}
		else
		{
			throw std::invalid_argument("axis must be 0 or 1");
		}
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

	double norm(const Matrix& rhs, int order)
	{
		// check that matrix is square
		if (rhs.nvec != rhs.vectors[0]->n)
		{
			throw std::invalid_argument("rhs matrix must be square");
		}
		if (order == _ONE_NORM)
		{
			Vector res(rhs.nvec);
			for (uint i = 0; i < rhs.nvec; i++)
			{
				res.data[i] = _absolute_summation_array_(rhs.vectors[i]->data, rhs.vectors[i]->n);
			}
			return res.max();
		}
		else if (order == _INF_NORM)
		{
			Vector res(rhs.vectors[0]->n);
			for (uint i = 0; i < rhs.vectors[0]->n; i++)
			{
				res.data[i] = _absolute_matrix_rowwise_summation_(rhs.data, rhs.nvec, rhs.vectors[0]->n, i);
			}
			return res.max();
		}
		else
		{
			throw std::invalid_argument("order must be 1 or inf");
		}
	}

	Matrix diag(const Vector& rhs)
	{
		Matrix m = zeros(rhs.n, rhs.n);
		for (uint i = 0; i < rhs.n; i++)
		{
			m.vectors[i]->data[i] = rhs.data[i];
		}
		return m;
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

	Matrix tril(const Matrix& rhs, bool diag)
	{
		Matrix m = zeros_like(rhs);
		if (diag) {
			for (uint i = 0; i < rhs.nvec; i++)
			{
				m.vectors[i]->data[i] = rhs.vectors[i]->data[i];
			}
		}
		for (uint i = 1; i < rhs.nvec; i++)
		{
			for (uint j = i; j < rhs.vectors[0]->n; j++)
			{
				m.vectors[i]->data[j] = rhs.vectors[i]->data[j];
			}
		}
		return m;
	}

	Matrix triu(const Matrix& rhs, bool diag)
	{
		Matrix m = zeros_like(rhs);
		if (diag) {
			for (uint i = 0; i < rhs.nvec; i++)
			{
				m.vectors[i]->data[i] = rhs.vectors[i]->data[i];
			}
		}
		for (uint i = 1; i < m.vectors[0]->n; i++)
		{
			for (uint j = i; j < rhs.nvec; j++)
			{
				m.vectors[i]->data[j] = rhs.vectors[i]->data[j];
			}
		}
		return m;
	}

	double trace(const Matrix& rhs)
	{
		if (rhs.nvec != rhs.vectors[0]->n)
		{
			throw std::range_error("rhs must be a square-matrix");
		}
		double result = 0.0;
	#ifdef _OPENMP
		#pragma omp parallel for schedule(static) default(none) shared(rhs) reduction(+:result) \
		if(rhs.nvec>100000)
	#endif
		for (uint i = 0; i < rhs.nvec; i++)
		{
			result += rhs.data[i+i*rhs.nvec];
		}
		return result;
	}

	double inner(const Vector& v, const Vector& w)
	{
		return dot(v, w);
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

	Vector dot(const Matrix& lhs, const Vector& rhs)
	{
		if (rhs.n != lhs.nvec)
		{
			throw std::range_error("matrix MxN must be the same size as the rhs vector Nx1");
		}
		if (!rhs.column)
		{
			throw std::logic_error("rhs cannot be a row-vector for matrix-vector dot.");
		}
		Vector np(lhs.vectors[0]->n);
		for (uint i = 0; i < np.n; i++)
		{
			// lhs.nvec must = rhs.n - hence it is used in this method
			np.data[i] = _matrix_rowwise_dot_(lhs.data, rhs.data, lhs.nvec, lhs.vectors[0]->n, i);
		}
		return np;
	}

	Matrix dot(const Matrix& A, const Matrix& B)
	{
		if (A.nvec != B.vectors[0]->n)
		{
			throw std::range_error("matrix A n-cols must = matrix B n-rows");
		}
		if (A.vectors[0]->n != B.nvec)
		{
			throw std::range_error("matrix A n-rows must = matrix B n-cols");
		}
		Matrix m(A.vectors[0]->n, A.vectors[0]->n);
		// matrix-matrix dot always form square products
		for (uint j = 0; j < m.nvec; j++)
		{
			for (uint i = 0; i < m.nvec; i++)
			{
				// calculate rowwise dot for each row with a given j vector
				m.data[i+j*A.vectors[0]->n] = _matrix_rowwise_dot_(A.data, B.vectors[j]->data,
														A.nvec, A.vectors[0]->n, i);
			}
		}
		return m;
	}

	Matrix outer(const Vector& v, const Vector& w)
	{
		// v and w can be different lengths
		if (!v.column || w.column)
		{
			throw std::logic_error("v must be a column vector AND w must be a row-vector");
		}
		// w dictates number of vectors, v dictates length of vectors
		Matrix m(w.n, v.n);
		for (uint y = 0; y < w.n; y++)
		{
			for (uint x = 0; x < v.n; x++)
			{
				m.data[x+y*v.n] = v.data[x] * w.data[y];
			}
		}
		return m;
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

	Matrix eye(uint ncol, uint nrow)
	{
		//printf("got matrix\n");
		Matrix result = zeros(ncol,nrow);

		uint smallestn = ncol < nrow ? ncol : nrow;
	#ifdef _OPENMP
		#pragma omp parallel for schedule(static) if(smallestn>100000)
	#endif
		for (uint i = 0; i < smallestn; i++)
		{
			result.data[i+i*nrow] = 1.0;
		}
		return result;
	}

	double det(const Matrix& rhs)
	{
		// if matrix is not square, leave
		if (rhs.nvec != rhs.vectors[0]->n)
		{
			throw std::invalid_argument("matrix rhs must be square size!");
		}
		// if 2x2, use method
		if (rhs.nvec == 2)
		{
			return _determinant_2x_(rhs.data[0],rhs.data[1],rhs.data[2],rhs.data[3]);
		}
		else if (rhs.nvec == 3)
		{ // if 3x3, use method
			return _determinant_3x_(rhs.data[0],rhs.data[1],rhs.data[2],rhs.data[3],
										  rhs.data[4],rhs.data[5],rhs.data[6],rhs.data[7],
										  rhs.data[8]);
		}
		else
		{ // else use a nxn method
			// copy into a fresh matrix
			Matrix m = copy(rhs);
			// use gaussian elimination to find trace
			int switch_count = _gaussian_elimination_(m.data, m.nvec);
			// the determinant is the product of the diagonals of this reduced matrix.
			// take into account the number of switches and multiply by -1 for every switch count.
			if (switch_count % 2 != 0)
			{
				return trace(m) * -1;
			}
			else
			{
				return trace(m);
			}
		}
	}

	Matrix eig(const Matrix& A)
	{
		// first compute eigenvalues
		if (A.nvec != A.vectors[0]->n)
		{
			throw std::logic_error("rhs must be a square-matrix");
		}
		// solve through power iteration for eigenvalues
		int n_simulations = 100;
		Vector b_k = rand(A.vectors[0]->n);

		for (uint i = 0; i < n_simulations; i++)
		{
			// dot product matrix-vector
			Vector bk_1 = dot(A, b_k);
			// calculate the norm
			double bk_norm = norm(bk_1);
			// re-normalise the norm
			b_k = (bk_1 / bk_norm);
		}
		// vector b_k should now contain eigenvalues
	}

	MATRIX_COMPLEX2& lu(const Matrix& A)
	{
		// set diag of L = 1
		Matrix L = eye(A.nvec, A.vectors[0]->n);
		// U is copied from A to begin
		Matrix U = copy(A);
		uint ncol = A.vectors[0]->n;
		for (unsigned int k = 0; k < A.nvec-1; k++)
		{
			for (unsigned int j = k+1; j < ncol; j++)
			{
				if (A.data[j+k*ncol] != 0.0)
				{
					// for every element on the column 0, zero it by selecting the negative divide
					double value = -(U.data[j+k*ncol] / U.data[k*ncol]);
					// for every element on selected row, substract by denominator
					for (unsigned int i = k; i < A.nvec; i++)
					{
						U.data[j+i*ncol] += value*A.data[k+i*ncol];
					}
					L.data[j+k*ncol] = -value;
				}
			}
		}
		// returning a matrix complex
		MATRIX_COMPLEX2 mcx(L, U);
		return mcx;
	}

	Vector solve(const Matrix& A, const Vector& b)
	{
		// b must be same length as A.
		if (b.n != A.nvec && b.n != A.vectors[0]->n)
		{
			throw std::range_error("A matrix must have the same number of vectors as b length");
		}
		// LU decomposition
		MATRIX_COMPLEX2 mcx = lu(A);
		Matrix L = *mcx.J;
		Matrix U = *mcx.K;
		// forward-backward substitution using

		// Ly = b - solve by forward substitution
		Vector y = empty_like(b);
		y.data[0] = b.data[0] / L.data[0];
		for (uint i = 1; i < A.vectors[0]->n; i++)
		{
			double tot = 0.0;
			for (uint j = 0; j < i; j++)
			{
				tot += L.data[i+j*L.vectors[0]->n]*y.data[j];
			}
			y.data[i] = b.data[i] - tot;
		}
		// now solve Ux = y by backward substitution
		Vector x = empty_like(b);
		x.data[x.n-1] = y.data[y.n-1] / L.data[L.nvec*L.vectors[0]->n-1];
		for (uint i = U.nvec-2; i >= 0; i--)
		{
			double tot = 0.0;
			for (uint j = U.vectors[0]->n-1; j > i; j--)
			{
				tot += U.data[i+j*U.vectors[0]->n]*x.data[j];
			}
			x.data[i] = y.data[i] - tot;
		}
		// finally return
		return x;
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

	Matrix sin(const Matrix& rhs)
	{
		Matrix m = copy(rhs);
		if (!_sine_array_(m.data, m.nvec*m.vectors[0]->n))
		{
			throw std::invalid_argument("Unable to sine-ify matrix.");
		}
		return m;
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

	Matrix cos(const Matrix& rhs)
	{
		Matrix m = copy(rhs);
		if (!_cos_array_(m.data, m.nvec*m.vectors[0]->n))
		{
			throw std::invalid_argument("Unable to cos-ify matrix.");
		}
		return m;
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

	Matrix tan(const Matrix& rhs)
	{
		Matrix m = copy(rhs);
		if (!_tan_array_(m.data, m.nvec*m.vectors[0]->n))
		{
			throw std::invalid_argument("Unable to tan-ify matrix.");
		}
		return m;
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

	Matrix exp(const Matrix& rhs)
	{
		Matrix m = copy(rhs);
		if (!_exp_array_(m.data, m.nvec*m.vectors[0]->n))
		{
			throw std::invalid_argument("Unable to exp-ify matrix.");
		}
		return m;
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

	Matrix log(const Matrix& rhs)
	{
		Matrix m = copy(rhs);
		if (!_log10_array_(m.data, m.nvec*m.vectors[0]->n))
		{
			throw std::invalid_argument("Unable to log-ify matrix.");
		}
		return m;
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

	Matrix sqrt(const Matrix& rhs)
	{
		Matrix m = copy(rhs);
		if (!_pow_array_(m.data, m.nvec*m.vectors[0]->n, 0.5))
		{
			throw std::invalid_argument("Unable to sqrt-ify matrix.");
		}
		return m;
	}

	Vector radians(const Vector& rhs)
	{
		Vector np(rhs.n);
		for (uint i = 0; i < rhs.n; i++)
		{
			np.data[i] = rhs.data[i] * (M_PI / 180);
		}
		return np;
	}

	Vector degrees(const Vector& rhs)
	{
		Vector np(rhs.n);
		for (uint i = 0; i < rhs.n; i++)
		{
			np.data[i] = rhs.data[i] * (180 / M_PI);
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

	Matrix transpose(const Matrix& rhs)
	{
		if (rhs.nvec == rhs.vectors[0]->n)
		{
			// square matrices are much easier to handle, since it's a straight diagonal swap.
			Matrix m = copy(rhs);
			_transpose_square_matrix_(m.data, m.nvec, m.vectors[0]->n);
			return m;
		}
		else
		{
			// we will allocate m to be the transposed-shape of rhs, and then manually enter the raw
			// data * array to transpose each element across. this should mean we won't need to
			// allocate or deallocate more memory after this.
			Matrix m = empty(rhs.vectors[0]->n, rhs.nvec);
			// we create another pointer starting at index 0 of m.
			double *ptr = m.data;
			for (uint x = 0; x < rhs.vectors[0]->n; x++)
			{
				// for each index in the x direction of the new matrix, set our pointer location to
				// the corresponding x element from rhs on the left-handside
				*ptr++ = rhs.data[x];
				// for each remaining y given x, add our pointer along and slot in values from rhs into m.
				for (uint y = 1; y < rhs.nvec; y++)
				{
					*ptr++ = rhs.data[x+y*rhs.vectors[0]->n];
				}
			}
			return m;
		}

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

	Matrix sort(const Matrix& rhs, uint axis, uint sorter)
	{
		if (axis == 0)
		{
			Matrix m = copy(rhs);
			for (uint i = 0; i < m.nvec; i++)
			{
				m.vectors[i]->sort(sorter);
			}
			return m;
		}
		else if (axis == 1)
		{
			//axis=1
			// rather than re-write our quicksorter - we just transpose the matrix
			Matrix m_T = transpose(rhs);
			for (uint i = 0; i < m_T.nvec; i++)
			{
				m_T.vectors[i]->sort(sorter);
			}
			return transpose(m_T);
		}
		else
		{
			throw std::invalid_argument("axis must be 0 or 1");
		}
	}

	Matrix hstack(const Vector& lhs, const Vector& rhs)
	{
		if (lhs.n != rhs.n)
		{
			throw std::invalid_argument("lhs and rhs length must be the same!");
		}
		Matrix result = empty(2, lhs.n);
	#ifdef _OPENMP
		#pragma omp parallel for schedule(static) if(lhs>100000)
	#endif
		for (uint x = 0; x < lhs.n; x++)
		{
			result.data[x] = lhs.data[x];
			result.data[x+lhs.n] = rhs.data[x];
		}
		return result;
	}

	/* --------------------------------------------------------------------------------------- *
	 *
	 * Now we handle our global operator overloads of +, -, *, / etc. This applies to all
	 * classes of Vector, Matrix and higher dimensions. Special rules apply when we multiply
	 * transposes etc, but for the most part, this provides extended vector operations.
	 *
	 ----------------------------------------------------------------------------------------*/

	// Operator Overloads - creates on the stack

	Vector operator+(const Vector& lhs, double value)
	{
		Vector np = copy(lhs);
		np += value;
		return np;
	}
	Vector operator+(double value, const Vector& rhs)
	{
		Vector np = copy(rhs);
		np += value;
		return np;
	}
	Vector operator+(const Vector& lhs, const Vector& rhs)
	{
		if (lhs.n != rhs.n)
		{
			throw std::range_error("lhs and rhs vector not the same size");
		}
		Vector np = copy(lhs);
		np += rhs;
		return np;
	}

	Vector operator-(const Vector& lhs, double value)
	{
		Vector np = copy(lhs);
		np -= value;
		return np;
	}
	Vector operator-(const Vector& lhs, const Vector& rhs)
	{
		if (lhs.n != rhs.n)
		{
			throw std::range_error("lhs and rhs vector not same size");
		}
		Vector np = copy(lhs);
		np -= rhs;
		return np;
	}

	Vector operator*(const Vector& lhs, double value)
	{
		Vector np = copy(lhs);
		np *= value;
		return np;
	}
	Vector operator*(double value, const Vector& rhs)
	{
		Vector np = copy(rhs);
		np *= value;
		return np;
	}
	Vector operator*(const Vector& lhs, const Vector& rhs)
	{
		if (lhs.n != rhs.n)
		{
			throw std::range_error("lhs and rhs vector not same size");
		}
		Vector np = copy(lhs);
		np *= rhs;
		return np;
	}

	Vector operator/(const Vector& lhs, double value)
	{
		Vector np = copy(lhs);
		np /= value;
		return np;
	}
	Vector operator/(const Vector& lhs, const Vector& rhs)
	{
		if (lhs.n != rhs.n)
		{
			throw std::range_error("lhs and rhs vector not same size");
		}
		Vector np = copy(lhs);
		np /= rhs;
		return np;
	}

	Vector operator^(const Vector& base, double exponent)
	{
		Vector np = copy(base);
		_pow_array_(np.data, np.n, exponent);
		return np;
	}
	Vector operator^(double base, const Vector& exponent)
	{
		Vector np = copy(exponent);
		_pow_base_array_(np.data, np.n, base);
		return np;
	}
	Vector operator^(const Vector base, const Vector& exponent)
	{
		Vector np = copy(base);
		for (uint i = 0; i < base.n; i++)
		{
			np.data[i] = _c_power_(base.data[i], exponent.data[i]);
		}
		return np;
	}

	bool operator<(const Vector& lhs, double value)
	{
		return (sum(lhs) < value);
	}
	bool operator<(double value, const Vector& rhs)
	{
		return (value < sum(rhs));
	}
	bool operator<(const Vector& lhs, const Vector& rhs)
	{
		return (sum(lhs) < sum(rhs));
	}

	bool operator>(const Vector& lhs, double value)
	{
		return (sum(lhs) > value);
	}
	bool operator>(double value, const Vector& rhs)
	{
		return (value > sum(rhs));
	}
	bool operator>(const Vector& lhs, const Vector& rhs)
	{
		return (sum(lhs) > sum(rhs));
	}



}

#endif


