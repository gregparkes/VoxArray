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

#include <stdexcept>

#include "Matrix.h"
#include "numstatic.cpp"

#define N_COLS (vectors[0]->n)
#define BUFFSIZE(x) (x->nvec*x->vectors[0]->n)


namespace numpy {

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
