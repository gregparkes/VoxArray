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
*/

/*
 * Matrix.cpp
 *
 *  Created on: 21 Mar 2017
 *      Author: Greg
 */

#ifndef __MATRIX_cpp__
#define __MATRIX_cpp__

#include <cstring>

#include "Vector.h"
#include "Matrix.h"
#include "VarStructs.h"
#include "numstatic.cpp"


// extra definitions to make life easier.
#define N_COLS (vectors[0]->n)
#define N_ROWS(x) (x.vectors[0]->n)
#define BUFFSIZE(x) (x->nvec*x->vectors[0]->n)


namespace numpy {

/********************************************************************************************


	NON-CLASS MEMBER FUNCTIONS 


*///////////////////////////////////////////////////////////////////////////////////////////


	Matrix empty(uint ncol, uint nrow)
	{
		Matrix m(ncol,nrow);
		return m;
	}

	Matrix zeros(uint ncol, uint nrow)
	{
		Matrix m(ncol,nrow);
		if (!_fill_array_(m.data, ncol*nrow, 0.0))
		{
			INVALID("Fill Error");
		}
		return m;
	}

	Matrix ones(uint ncol, uint nrow)
	{
		Matrix m(ncol,nrow);
		if (!_fill_array_(m.data, ncol*nrow, 1.0))
		{
			INVALID("Fill Error");
		}
		return m;
	}

	Matrix fill(uint ncol, uint nrow, double val)
	{
	  Matrix m(ncol,nrow);
		if (!_fill_array_(m.data, ncol*nrow, val))
		{
			INVALID("Fill Error");
		}
		return m;
	}

	char* shape(const Matrix& rhs)
	{
		// determine length of each integer in characters.
				// then it's (int1,int2) so character lengths + 3, allocate memory
		if (rhs.vectors[0]->n == 0 || rhs.nvec == 0)
		{
			INVALID("column or row length cannot = 0");
		}
		uint n_digit_row = _n_digits_in_int_( N_ROWS(rhs) );
		uint n_digit_col = _n_digits_in_int_(rhs.nvec);
		char* strg = new char[n_digit_row + n_digit_col + 4];
		if (!_str_shape_func_(strg, N_ROWS(rhs), rhs.nvec, n_digit_row, n_digit_col,
				n_digit_row + n_digit_col + 4))
		{
			INVALID("Problem with creating string representation");
		}
		return strg;
	}

	char* str(const Matrix& rhs, uint dpoints)
	{
		uint str_len = _str_length_gen_(rhs.vectors[0]->data, N_ROWS(rhs), dpoints);
		uint total_length = str_len + 2; // one at start, then \n and space
		char* strg = new char[total_length*rhs.nvec+1];
		char* ptr = strg;
		*ptr++ = '[';
		for (uint i = 0; i < rhs.nvec-1; i++)
		{
			if (!_str_representation_(ptr, rhs.vectors[i]->data,
								rhs.vectors[i]->n, dpoints, 0))
			{
				INVALID("Problem with creating string representation");
			}
			ptr += str_len-1;
			*ptr++ = '\n';
			*ptr++ = ' ';
		}
		if (!_str_representation_(ptr, rhs.vectors[rhs.nvec-1]->data,
										rhs.vectors[rhs.nvec-1]->n, dpoints, 1))
		{
			INVALID("Problem with creating string representation");
		}
		ptr += str_len-1;
		*ptr = ']';
		return strg;
	}

	Matrix copy(const Matrix& rhs)
	{
		Matrix m(rhs.nvec, N_ROWS(rhs));
		uint nbyn = _fullsize_(rhs);
		if (!_copy_array_(m.data, rhs.data, nbyn))
		{
			INVALID("copy failed!");
		}
		uint i;
		// update new vectors on other parameters
		for (i = 0; i < rhs.nvec; i++)
		{
			m.vectors[i]->column = rhs.vectors[i]->column;
			m.vectors[i]->flag_delete = rhs.vectors[i]->flag_delete;
			m.vectors[i]->n = rhs.vectors[i]->n;
		}
		return m;
	}

	Vector nonzero(const Matrix& rhs)
	{
		int cnz = _count_nonzero_array_(rhs.data, _fullsize_(rhs));
		Vector np(cnz);
		_nonzero_array_(np.data, rhs.data, _fullsize_(rhs));
		return np;
	}

	Matrix flip(const Matrix& rhs, uint axis)
	{
		Matrix result = copy(rhs);
		if (axis == 0)
		{
			for (uint i = 0; i < rhs.nvec; i++)
			{
				result.vectors[i]->flip();
			}
		} else if (axis == 1)
		{
			for (uint i = 0; i < rhs.nvec; i++)
			{
				for (uint j = 0; j < rhs.vectors[0]->n; j++)
				{
					result.vectors[i]->data[j] = rhs.vectors[i]->data[rhs.vectors[0]->n-1-j];
				}
			}
		} else { INVALID_AXIS(); }

		return result;
	}

	Matrix empty_like(const Matrix& rhs)
	{
		return Matrix(rhs.nvec, rhs.vectors[0]->n);
	}

	Matrix zeros_like(const Matrix& rhs)
	{
		Matrix res(rhs.nvec, rhs.vectors[0]->n);
		_fill_array_(rhs.data, rhs.nvec*rhs.vectors[0]->n, 0.0);
		return res;
	}

	Matrix ones_like(const Matrix& rhs)
	{
		Matrix res(rhs.nvec, rhs.vectors[0]->n);
		_fill_array_(rhs.data, rhs.nvec*rhs.vectors[0]->n, 1.0);
		return res;
	}

	Matrix rand(uint ncol, uint nrow)
	{
		if (ncol == 0 || nrow == 0)
		{
			RANGE("ncol or nrow cannot = 0");
		}
		Matrix res(ncol,nrow);
		_rand_array_(res.data, ncol*nrow);
		return res;
	}

	Matrix randn(uint ncol, uint nrow)
	{
		if (ncol == 0 || nrow == 0)
		{
			RANGE("ncol or nrow cannot = 0");
		}
		Matrix res(ncol,nrow);
		if (!_normal_distrib_(res.data, ncol*nrow, 0.0, 1.0))
		{
			INVALID("Error with creating normal distribution");
		}
		return res;
	}

	Matrix normal(uint ncol, uint nrow, double mean, double sd)
	{
		if (ncol == 0 || nrow == 0)
		{
			RANGE("ncol or nrow cannot = 0");
		}
		Matrix res(ncol,nrow);
		if (!_normal_distrib_(res.data, ncol*nrow, mean, sd))
		{
			INVALID("Error with creating normal distribution");
		}
		return res;
	}

	Matrix randint(uint ncol, uint nrow, uint max)
	{
		if (ncol == 0 || nrow == 0)
		{
			RANGE("ncol or nrow cannot = 0");
		}
		if (max == 0)
		{
			RANGE("max cannot = 0");
		}
		Matrix res(ncol,nrow);
		_randint_array_(res.data, ncol*nrow, max);
		return res;
	}

	Matrix randchoice(uint ncol, uint nrow, const char* values)
	{
		if (ncol == 0 || nrow == 0)
		{
			RANGE("ncol or nrow cannot = 0");
		}
		if (values == null)
		{
			INVALID("input must not be null");
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
		#pragma omp parallel for if(ncol*nrow>__OMP_OPT_VALUE__) schedule(static)
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

	long poisson(double lam)
	{
		return _poisson_coefficient_(lam);
	}

	Matrix clip(const Matrix& rhs, double a_min, double a_max)
	{
		Matrix res = copy(rhs);
		_clip_array_(res.data, _fullsize_(rhs), a_min, a_max);
		return res;
	}

	Matrix floor(const Matrix& rhs)
	{
		Matrix m = copy(rhs);
		if (!_floor_array_(m.data, _fullsize_(m)))
		{
			throw std::invalid_argument("Unable to floor matrix.");
		}
		return m;
	}

	Matrix ceil(const Matrix& rhs)
	{
		Matrix m = copy(rhs);
		if (!_ceil_array_(m.data, _fullsize_(m)))
		{
			throw std::invalid_argument("Unable to ceil matrix.");
		}
		return m;
	}

	int count(const Matrix& rhs, double value)
	{
		return _count_array_(rhs.data, _fullsize_(rhs), value);
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
		else { INVALID_AXIS(); }
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
		} else { INVALID_AXIS(); }
	}

	Matrix abs(const Matrix& rhs)
	{
		Matrix res = copy(rhs);
		_absolute_array_(res.data, res.nvec*res.vectors[0]->n);
		return res;
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
		} else { INVALID_AXIS(); }
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
		} else { INVALID_AXIS(); }
	}

	Matrix cumsum(const Matrix& rhs, uint axis)
	{
		if (axis == 0)
		{
			Matrix m = zeros_like(rhs);
			for (uint i = 0; i < rhs.nvec; i++)
			{
				_cumulative_sum_(m.vectors[i]->data, rhs.vectors[i]->data, rhs.vectors[i]->n);
			}
			return m;
		} else if (axis == 1)
		{
			Matrix m = copy(rhs);
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
		} else { INVALID_AXIS(); }
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

	bool all(const Matrix& rhs)
	{
		return _all_true_(rhs.data, _fullsize_(rhs));
	}

	bool any(const Matrix& rhs)
	{
		return _any_true_(rhs.data, _fullsize_(rhs));
	}

	double min(const Matrix& rhs)
	{
		return _min_value_(rhs.data, _fullsize_(rhs));
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
		} else { INVALID_AXIS(); }
	}

	double max(const Matrix& rhs)
	{
		return _max_value_(rhs.data, _fullsize_(rhs));
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
		} else { INVALID_AXIS(); }
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
		} else { INVALID_AXIS(); }
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
		} else { INVALID_AXIS(); }
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
		} else { INVALID_AXIS(); }
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
					if (CMP(m.data[i+j*m.nvec], 0.0))
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
		} else { INVALID_AXIS(); }
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
		} else { INVALID_AXIS(); }
	}

	double norm(const Matrix& rhs, int order)
	{
		// check that matrix is square
		if (rhs.nvec != rhs.vectors[0]->n)
		{
			RANGE("rhs matrix must be square");
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
			RANGE("rhs must be a square-matrix");
		}
		double result = 0.0;
		for (uint i = 0; i < rhs.nvec; i++)
		{
			result += rhs.data[i+i*rhs.nvec];
		}
		return result;
	}

	Vector dot(const Matrix& lhs, const Vector& rhs)
	{
		if (rhs.n != lhs.nvec)
		{
			RANGE("matrix MxN must be the same size as the rhs vector Nx1");
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
			RANGE("matrix A n-cols must = matrix B n-rows");
		}
		if (A.vectors[0]->n != B.nvec)
		{
			RANGE("matrix A n-rows must = matrix B n-cols");
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

	Matrix eye(uint ncol, uint nrow)
	{
		//printf("got matrix\n");
		Matrix result = zeros(ncol,nrow);

		uint smallestn = ncol < nrow ? ncol : nrow;
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
			RANGE("matrix rhs must be square size!");
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
			RANGE("rhs must be a square-matrix");
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

	MATRIX_COMPLEX2 lu(const Matrix& A)
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
			RANGE("A matrix must have the same number of vectors as b length");
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

	Matrix sin(const Matrix& rhs)
	{
		Matrix m = copy(rhs);
		if (!_sine_array_(m.data, m.nvec*m.vectors[0]->n))
		{
			INVALID("Unable to sine-ify matrix.");
		}
		return m;
	}

	Matrix cos(const Matrix& rhs)
	{
		Matrix m = copy(rhs);
		if (!_cos_array_(m.data, m.nvec*m.vectors[0]->n))
		{
			INVALID("Unable to cos-ify matrix.");
		}
		return m;
	}

	Matrix tan(const Matrix& rhs)
	{
		Matrix m = copy(rhs);
		if (!_tan_array_(m.data, m.nvec*m.vectors[0]->n))
		{
			INVALID("Unable to tan-ify matrix.");
		}
		return m;
	}

	Matrix to_radians(const Matrix& rhs)
	{
		Matrix m = copy(rhs);
		if (!_to_radians_array_(m.data, m.nvec*m.vectors[0]->n))
		{
			INVALID("Unable to convert to radians.");
		}
		return m;
	}

	Matrix to_degrees(const Matrix& rhs)
	{
		Matrix m = copy(rhs);
		if (!_to_degrees_array_(m.data, m.nvec*m.vectors[0]->n))
		{
			INVALID("Unable to convert to degrees.");
		}
		return m;
	}

	Matrix exp(const Matrix& rhs)
	{
		Matrix m = copy(rhs);
		if (!_exp_array_(m.data, m.nvec*m.vectors[0]->n))
		{
			INVALID("Unable to exp-ify matrix.");
		}
		return m;
	}

	Matrix log(const Matrix& rhs)
	{
		Matrix m = copy(rhs);
		if (!_log10_array_(m.data, m.nvec*m.vectors[0]->n))
		{
			INVALID("Unable to log-ify matrix.");
		}
		return m;
	}

	Matrix sqrt(const Matrix& rhs)
	{
		Matrix m = copy(rhs);
		if (!_pow_array_(m.data, m.nvec*m.vectors[0]->n, 0.5))
		{
			INVALID("Unable to sqrt-ify matrix.");
		}
		return m;
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
		} else { INVALID_AXIS(); }
	}

	Matrix hstack(const Vector& lhs, const Vector& rhs)
	{
		if (lhs.n != rhs.n)
		{
			INVALID("lhs and rhs length must be the same!");
		}
		Matrix result = empty(2, lhs.n);
		for (uint x = 0; x < lhs.n; x++)
		{
			result.data[x] = lhs.data[x];
			result.data[x+lhs.n] = rhs.data[x];
		}
		return result;
	}



/********************************************************************************************


	CLASS MEMBER FUNCTIONS 


*///////////////////////////////////////////////////////////////////////////////////////////


	Matrix::Matrix(uint ncol, uint nrow)
	{
		//printf("constructing MATRIX %x\n", this);
		if (ncol == 0)
		{
			throw std::range_error("ncol cannot = 0");
		}
		if (nrow == 0)
		{
			throw std::range_error("nrow cannot = 0");
		}
		this->nvec = ncol;
		// now allocate the full-block memory.
		data = _create_empty_(nrow*ncol);
		if (data == NULL)
		{
			throw std::runtime_error("Unable to allocate memory");
		}
		// create a set of pointers for each vector to jump to
		this->vectors = new Vector*[ncol];
		// now initialise all the objects as empty, and set their pointers to the respective block
		for (uint i = 0; i < ncol; i++)
		{
			vectors[i] = new Vector();
			vectors[i]->n = nrow;
			vectors[i]->data = (data + i*nrow);
		}
	}

	Matrix::~Matrix()
	{
		//printf("deleting MATRIX %x\n", this);
		for (uint i = 0; i < nvec; i++)
		{
			delete vectors[i];
		}
		if (!_destroy_array_(data))
		{
			throw std::invalid_argument("Unable to destroy array");
		}
	}

	char* Matrix::str(uint dpoints)
	{
		uint str_len = _str_length_gen_(vectors[0]->data, N_COLS, dpoints);
		uint total_length = str_len + 2; // one at start, then \n and space
		char* strg = new char[total_length*nvec+1];
		char* ptr = strg;
		*ptr++ = '[';
		for (uint i = 0; i < nvec-1; i++)
		{
			if (!_str_representation_(ptr, vectors[i]->data,
								vectors[i]->n, dpoints, 0))
			{
				throw std::invalid_argument("Problem with creating string representation");
			}
			ptr += str_len-1;
			*ptr++ = '\n';
			*ptr++ = ' ';
		}
		if (!_str_representation_(ptr, vectors[nvec-1]->data,
										vectors[nvec-1]->n, dpoints, 1))
		{
			throw std::invalid_argument("Problem with creating string representation");
		}
		ptr += str_len-1;
		*ptr = ']';
		return strg;
	}

	char* Matrix::shape()
	{
		// determine length of each integer in characters.
		// then it's (int1,int2) so character lengths + 3, allocate memory
		if (N_COLS == 0 || nvec == 0)
		{
			throw std::invalid_argument("column or row length cannot = 0");
		}
		uint n_digit_row = _n_digits_in_int_(N_COLS);
		uint n_digit_col = _n_digits_in_int_(nvec);
		char* strg = new char[n_digit_row + n_digit_col + 4];
		if (!_str_shape_func_(strg, N_COLS, nvec, n_digit_row, n_digit_col,
				n_digit_row + n_digit_col + 4))
		{
			throw std::invalid_argument("Problem with creating string representation");
		}
		return strg;
	}

	uint Matrix::nfloats()
	{
		return (nvec * N_COLS);
	}

	Matrix Matrix::copy()
	{
		Matrix m(nvec, N_COLS);
		uint nbyn = BUFFSIZE(this);
		if (!_copy_array_(m.data, data, nbyn))
		{
			throw std::invalid_argument("copy failed!");
		}
		// update new vectors on other parameters
		for (uint i = 0; i < nvec; i++)
		{
			m.vectors[i]->column = vectors[i]->column;
			m.vectors[i]->flag_delete = vectors[i]->flag_delete;
			m.vectors[i]->n = vectors[i]->n;
		}
		return m;
	}

	double& Matrix::ix(uint i, uint j)
	{
		return data[i+j*vectors[0]->n];
	}

	Matrix& Matrix::flip(uint axis)
	{
		if (axis == 0)
		{
		#ifdef _OPENMP
			#pragma omp parallel for if(nvec>100000) schedule(static)
		#endif
			for (uint i = 0; i < nvec; i++)
			{
				vectors[i]->flip();
			}
		} else if (axis == 1)
		{
		#ifdef _OPENMP
			#pragma omp parallel for if(nvec>100000) schedule(static)
		#endif
			for (uint i = 0; i < nvec; i++)
			{
				for (uint j = 0; j < N_COLS; j++)
				{
					vectors[i]->data[j] = vectors[i]->data[N_COLS-1-j];
				}
			}
		} else
		{
			throw std::invalid_argument("axis must be 0 or 1");
		}
		return *this;
	}

	Matrix& Matrix::clip(double a_min, double a_max)
	{
		_clip_array_(data, BUFFSIZE(this), a_min, a_max);
		return *this;
	}

	double Matrix::sum()
	{
		return _summation_array_(data, BUFFSIZE(this));
	}

	Vector Matrix::sum(uint axis)
	{
		if (axis == 0)
		{
			Vector res(nvec);
			for (uint i = 0; i < nvec; i++)
			{
				res.data[i] = vectors[i]->sum();
			}
			return res;
		} else if (axis == 1)
		{
			Vector res(N_COLS);
			for (uint i = 0; i < N_COLS; i++)
			{
				res.data[i] = _matrix_rowwise_summation_(data, nvec, N_COLS, i);
			}
			return res;
		} else
		{
			throw std::invalid_argument("axis must be 0 or 1");
		}
	}

	double Matrix::prod()
	{
		return _prod_array_(data, BUFFSIZE(this));
	}

	Vector Matrix::prod(uint axis)
	{
		if (axis == 0)
		{
			Vector res(nvec);
			for (uint i = 0; i < nvec; i++)
			{
				res.data[i] = vectors[i]->prod();
			}
			return res;
		} else if (axis == 1)
		{
			Vector res(N_COLS);
			for (uint i = 0; i < N_COLS; i++)
			{
				double total = 1.0;
				for (uint j = 0; j < nvec; j++)
				{
					total *= vectors[j]->data[i];
				}
				res.data[i] = total;
			}
			return res;
		} else
		{
			throw std::invalid_argument("axis must be 0 or 1");
		}
	}

	Matrix& Matrix::abs()
	{
		_absolute_array_(data, BUFFSIZE(this));
		return *this;
	}

	bool Matrix::all()
	{
		return _all_true_(data, BUFFSIZE(this));
	}

	bool Matrix::any()
	{
		return _any_true_(data, BUFFSIZE(this));
	}

	uint Matrix::count(double value)
	{
		return _count_array_(data, BUFFSIZE(this), value);
	}

	Vector Matrix::count(double value, uint axis)
	{
		if (axis == 0)
		{
			Vector v(nvec);
			for (uint i = 0; i < nvec; i++)
			{
				v.data[i] = vectors[i]->count(value);
			}
			return v;
		}
		else if (axis == 1)
		{
			Vector v(N_COLS);
			for (uint i = 0; i < N_COLS; i++)
			{
				uint count = 0;
				for (uint j = 0; j < nvec; j++)
				{
					if (vectors[j]->data[i] == value)
					{
						count++;
					}
				}
				v.data[i] = count;
			}
			return v;
		}
		else
		{
			throw std::invalid_argument("axis must be 0 or 1");
		}
	}

	uint Matrix::count_nonzero()
	{
		return _count_nonzero_array_(data, BUFFSIZE(this));
	}

	Vector Matrix::count_nonzero(uint axis)
	{
		if (axis == 0)
		{
			Vector np(nvec);
			for (uint i = 0; i < nvec; i++)
			{
				np.data[i] = vectors[i]->count_nonzero();
			}
			return np;
		} else if (axis == 1)
		{
			Vector np(N_COLS);
			for (uint i = 0; i < N_COLS; i++)
			{
				np.data[i] = _matrix_rowwise_count_nonzero_(data, nvec, N_COLS, i);
			}
			return np;
		} else
		{
			throw std::invalid_argument("axis must be 0 or 1.");
		}
	}

	double Matrix::mean()
	{
		return _summation_array_(data, BUFFSIZE(this)) / (BUFFSIZE(this));
	}

	Vector Matrix::mean(uint axis)
	{
		if (axis == 0)
		{
			Vector v(nvec);
			for (uint i = 0; i < nvec; i++)
			{
				v.data[i] = vectors[i]->mean();
			}
			return v;
		} else if (axis == 1)
		{
			Vector v(N_COLS);
			for (uint i = 0; i < N_COLS; i++)
			{
				v.data[i] = _matrix_rowwise_summation_(data, nvec, vectors[i]->n, i) / nvec;
			}
			return v;
		} else
		{
			throw std::invalid_argument("axis must be 0 or 1");
		}
	}

	Vector Matrix::std(uint axis)
	{
		if (axis == 0)
		{
			Vector v(nvec);
			for (uint i = 0; i < nvec; i++)
			{
				v.data[i] = vectors[i]->std();
			}
			return v;
		} else if (axis == 1)
		{
			Vector v(N_COLS);
			for (uint i = 0; i < N_COLS; i++)
			{
				v.data[i] = _matrix_rowwise_std_(data, nvec, vectors[i]->n, i);
			}
			return v;
		} else
		{
			throw std::invalid_argument("axis must be 0 or 1");
		}
	}

	Vector Matrix::var(uint axis)
	{
		if (axis == 0)
		{
			Vector v(nvec);
			for (uint i = 0; i < nvec; i++)
			{
				v.data[i] = vectors[i]->var();
			}
			return v;
		} else if (axis == 1)
		{
			Vector v(N_COLS);
			for (uint i = 0; i < N_COLS; i++)
			{
				v.data[i] = _matrix_rowwise_var_(data, nvec, vectors[i]->n, i);
			}
			return v;
		} else
		{
			throw std::invalid_argument("axis must be 0 or 1");
		}
	}

	Matrix& Matrix::fill(double value)
	{
		_fill_array_(data, BUFFSIZE(this), value);
		return *this;
	}


	Matrix& Matrix::floor()
	{
		_floor_array_(data, BUFFSIZE(this));
		return *this;
	}

	Matrix& Matrix::ceil()
	{
		_ceil_array_(data, BUFFSIZE(this));
		return *this;
	}

	Matrix& Matrix::sort(uint axis, uint sorter)
	{
		if (axis == 0)
		{
			for (uint i = 0; i < nvec; i++)
			{
				vectors[i]->sort(sorter);
			}
			return *this;
		}
		else if (axis == 1)
		{
			//@todo Fix row-wise matrix sort
			throw std::runtime_error("We currently have no implementation for sorting this matrix locally by row");
		}
		else
		{
			throw std::invalid_argument("axis must be 0 or 1");
		}
	}

	Vector Matrix::argmin(uint axis)
	{
		if (axis == 0)
		{
			Vector v(nvec);
			for (uint i = 0; i < nvec; i++)
			{
				v.data[i] = vectors[i]->argmin();
			}
			return v;
		}
		else if (axis == 1)
		{
			Vector v(N_COLS);
			for (uint i = 0; i < v.n; i++)
			{
				v.data[i] = _matrix_rowwise_min_index_(data, nvec, N_COLS, i);
			}
			return v;
		}
		else
		{
			throw std::invalid_argument("axis must be 0 or 1");
		}
	}

	Vector Matrix::argmax(uint axis)
	{
		if (axis == 0)
		{
			Vector v(nvec);
			for (uint i = 0; i < nvec; i++)
			{
				v.data[i] = vectors[i]->argmax();
			}
			return v;
		}
		else if (axis == 1)
		{
			Vector v(N_COLS);
			for (uint i = 0; i < v.n; i++)
			{
				v.data[i] = _matrix_rowwise_max_index_(data, nvec, N_COLS, i);
			}
			return v;
		}
		else
		{
			throw std::invalid_argument("axis must be 0 or 1");
		}
	}

	double Matrix::min()
	{
		return _min_value_(data, BUFFSIZE(this));
	}

	Vector Matrix::min(uint axis)
	{
		if (axis == 0)
		{
			Vector v(nvec);
			for (uint i = 0; i < nvec; i++)
			{
				v.data[i] = vectors[i]->min();
			}
			return v;
		} else if (axis == 1)
		{
			Vector v(N_COLS);
			for (uint i = 0; i < N_COLS; i++)
			{
				v.data[i] = _matrix_rowwise_min_value_(data, nvec, N_COLS, i);
			}
			return v;
		} else
		{
			throw std::invalid_argument("axis must be 0 or 1");
		}
	}

	double Matrix::max()
	{
		return _max_value_(data, BUFFSIZE(this));
	}

	Vector Matrix::max(uint axis)
	{
		if (axis == 0)
		{
			Vector v(nvec);
			for (uint i = 0; i < nvec; i++)
			{
				v.data[i] = vectors[i]->max();
			}
			return v;
		} else if (axis == 1)
		{
			Vector v(N_COLS);
			for (uint i = 0; i < N_COLS; i++)
			{
				v.data[i] = _matrix_rowwise_max_value_(data, nvec, vectors[i]->n, i);
			}
			return v;
		} else
		{
			throw std::invalid_argument("axis must be 0 or 1");
		}
	}

	double Matrix::norm(int order)
	{
		if (nvec != N_COLS)
		{
			throw std::invalid_argument("rhs matrix must be square");
		}
		if (order == _ONE_NORM)
		{
			Vector res(nvec);
			for (uint i = 0; i < nvec; i++)
			{
				res.data[i] = _absolute_summation_array_(vectors[i]->data, vectors[i]->n);
			}
			return res.max();
		}
		else if (order == _INF_NORM)
		{
			Vector res(N_COLS);
			for (uint i = 0; i < N_COLS; i++)
			{
				res.data[i] = _absolute_matrix_rowwise_summation_(data, nvec, N_COLS, i);
			}
			return res.max();
		}
		else
		{
			throw std::invalid_argument("order must be 1 or inf");
		}
	}

	Matrix& Matrix::sin()
	{
		if (! _sine_array_(data, BUFFSIZE(this)))
		{
			throw std::invalid_argument("Unable to log-ify matrix.");
		}
		return *this;
	}

	Matrix& Matrix::cos()
	{
		if (! _cos_array_(data, BUFFSIZE(this)))
		{
			throw std::invalid_argument("Unable to log-ify matrix.");
		}
		return *this;
	}

	Matrix& Matrix::tan()
	{
		if (! _tan_array_(data, BUFFSIZE(this)))
		{
			throw std::invalid_argument("Unable to log-ify matrix.");
		}
		return *this;
	}

	Matrix& Matrix::to_radians()
	{
		if (!_to_radians_array_(data, BUFFSIZE(this)))
		{
			throw std::invalid_argument("Unable to convert to radians.");
		}
		return *this;
	}

	Matrix& Matrix::to_degrees()
	{
		if (!_to_degrees_array_(data, BUFFSIZE(this)))
		{
			throw std::invalid_argument("Unable to convert to degrees.");
		}
		return *this;
	}

	Matrix& Matrix::exp()
	{
		if (! _exp_array_(data, BUFFSIZE(this)))
		{
			throw std::invalid_argument("Unable to log-ify matrix.");
		}
		return *this;
	}

	Matrix& Matrix::log()
	{
		if (! _log10_array_(data, BUFFSIZE(this)))
		{
			throw std::invalid_argument("Unable to log-ify matrix.");
		}
		return *this;
	}

	Matrix& Matrix::sqrt()
	{
		if (! _pow_array_(data, BUFFSIZE(this), 0.5))
		{
			throw std::invalid_argument("Unable to sqrt-ify matrix.");
		}
		return *this;
	}

	bool Matrix::operator==(const Matrix& rhs)
	{
		if (rhs.nvec != nvec || rhs.vectors[0]->n != N_COLS) return false;
		for (uint i = 0; i < nvec*N_COLS; i++)
		{
			if (!CMP(rhs.data[i], data[i]))
			{
				return false;
			}
		}
		return true;
	}

	bool Matrix::operator!=(const Matrix& rhs)
	{
		if (rhs.nvec != nvec || rhs.vectors[0]->n != N_COLS) return false;
		for (uint i = 0; i < nvec*N_COLS; i++)
		{
			if (!CMP(rhs.data[i], data[i]))
			{
				return true;
			}
		}
		return false;
	}

	Matrix& Matrix::operator+=(const Matrix& rhs)
	{
		if (rhs.nvec != nvec || rhs.vectors[0]->n != N_COLS)
		{
			throw std::invalid_argument("lhs and rhs are not the same size.");
		}
		_element_add_(data, rhs.data, _fullsize_(rhs));
		return *this;
	}
	Matrix& Matrix::operator+=(double value)
	{
		_add_array_(data, BUFFSIZE(this), value);
		return *this;
	}
	Matrix& Matrix::operator+=(int value)
	{
		_add_array_(data, BUFFSIZE(this), (double) value);
		return *this;
	}

	Matrix& Matrix::operator-=(const Matrix& rhs)
	{
		if (rhs.nvec != nvec || rhs.vectors[0]->n != N_COLS)
		{
			throw std::invalid_argument("lhs and rhs are not the same size.");
		}
		_element_sub_(data, rhs.data, _fullsize_(rhs));
		return *this;
	}
	Matrix& Matrix::operator-=(double value)
	{
		_sub_array_(data, BUFFSIZE(this), value);
		return *this;
	}
	Matrix& Matrix::operator-=(int value)
	{
		_sub_array_(data, BUFFSIZE(this), (double) value);
		return *this;
	}

	Matrix& Matrix::operator*=(const Matrix& rhs)
	{
		if (rhs.nvec != nvec || rhs.vectors[0]->n != N_COLS)
		{
			throw std::invalid_argument("lhs and rhs are not the same size.");
		}
		_element_mult_(data, rhs.data, _fullsize_(rhs));
		return *this;
	}
	Matrix& Matrix::operator*=(double value)
	{
		_mult_array_(data, BUFFSIZE(this), value);
		return *this;
	}
	Matrix& Matrix::operator*=(int value)
	{
		_mult_array_(data, BUFFSIZE(this), (double) value);
		return *this;
	}

	Matrix& Matrix::operator/=(const Matrix& rhs)
	{
		if (rhs.nvec != nvec || rhs.vectors[0]->n != N_COLS)
		{
			throw std::invalid_argument("lhs and rhs are not the same size.");
		}
		_element_div_(data, rhs.data, _fullsize_(rhs));
		return *this;
	}
	Matrix& Matrix::operator/=(double value)
	{
		_div_array_(data, BUFFSIZE(this), value);
		return *this;
	}
	Matrix& Matrix::operator/=(int value)
	{
		_div_array_(data, BUFFSIZE(this), (double) value);
		return *this;
	}

	/**
	* NOW COMPLETING OPERATOR OVERLOADS NOT IN CLASS BOUNDARY
	*/


	Matrix operator+(const Matrix& l, const Matrix& r)
	{
		if (l.nvec != r.nvec || l.vectors[0]->n != r.vectors[0]->n)
		{
			throw std::invalid_argument("lhs and rhs are not the same size.");
		}
		Matrix m = _copy_matrix_(l);
		m += r;
		return m;
	}
	Matrix operator+(const Matrix& l, double r)
	{
		Matrix m = _copy_matrix_(l);
		m += r;
		return m;
	}
	Matrix operator+(const Matrix& l, int r)
	{
		Matrix m = _copy_matrix_(l);
		m += r;
		return m;
	}
	Matrix operator+(double l, const Matrix& r)
	{
		Matrix m = _copy_matrix_(r);
		m += l;
		return m;
	}
	Matrix operator+(int l, const Matrix& r)
	{
		Matrix m = _copy_matrix_(r);
		m += l;
		return m;
	}

	Matrix operator-(const Matrix& l, const Matrix& r)
	{
		if (l.nvec != r.nvec || l.vectors[0]->n != r.vectors[0]->n)
		{
			throw std::invalid_argument("lhs and rhs are not the same size.");
		}
		Matrix m = _copy_matrix_(l);
		m -= r;
		return m;
	}
	Matrix operator-(const Matrix& l, double r)
	{
		Matrix m = _copy_matrix_(l);
		m -= r;
		return m;
	}
	Matrix operator-(const Matrix& l, int r)
	{
		Matrix m = _copy_matrix_(l);
		m -= r;
		return m;
	}

	Matrix operator*(const Matrix& l, const Matrix& r)
	{
		if (l.nvec != r.nvec || l.vectors[0]->n != r.vectors[0]->n)
		{
			throw std::invalid_argument("lhs and rhs are not the same size.");
		}
		Matrix m = _copy_matrix_(l);
		m *= r;
		return m;
	}
	Matrix operator*(const Matrix& l, double r)
	{
		Matrix m = _copy_matrix_(l);
		m *= r;
		return m;
	}
	Matrix operator*(const Matrix& l, int r)
	{
		Matrix m = _copy_matrix_(l);
		m *= r;
		return m;
	}
	Matrix operator*(double l, const Matrix& r)
	{
		Matrix m = _copy_matrix_(r);
		m *= l;
		return m;
	}
	Matrix operator*(int l, const Matrix& r)
	{
		Matrix m = _copy_matrix_(r);
		m *= l;
		return m;
	}

	Matrix operator/(const Matrix& l, const Matrix& r)
	{
		if (l.nvec != r.nvec || l.vectors[0]->n != r.vectors[0]->n)
		{
			throw std::invalid_argument("lhs and rhs are not the same size.");
		}
		Matrix m = _copy_matrix_(l);
		m /= r;
		return m;
	}
	Matrix operator/(const Matrix& l, double r)
	{
		if (CMP(r, 0.0))
		{
			throw std::invalid_argument("cannot divide by 0!");
		}
		Matrix m = _copy_matrix_(l);
		m /= r;
		return m;
	}
	Matrix operator/(const Matrix& l, int r)
	{
		if (r == 0)
		{
			throw std::invalid_argument("cannot divide by 0!");
		}
		Matrix m = _copy_matrix_(l);
		m /= r;
		return m;
	}



	/** ---------------ACCESSORY METHOD! ------------------- */


	Matrix _copy_matrix_(const Matrix& m)
	{
		Matrix cp(m.nvec, m.vectors[0]->n);
		_copy_array_(cp.data, m.data, _fullsize_(m));
		for (uint i = 0; i < m.nvec; i++)
		{
			cp.vectors[i]->column = m.vectors[i]->column;
			cp.vectors[i]->flag_delete = m.vectors[i]->flag_delete;
			cp.vectors[i]->n = m.vectors[i]->n;
		}
		return cp;
	}

	uint _fullsize_(const Matrix& m)
	{
		 return m.nvec * m.vectors[0]->n;
	}
}

#endif
