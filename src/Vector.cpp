/*
 * Vec1f.cpp
 *
 *  Created on: 18 Feb 2017
 *      Author: Greg
 */

#ifndef __VEC1f_cpp__
#define __VEC1f_cpp__

#include <stdexcept>

#include "Vector.h"
#include "numstatic.cpp"


// Here we define class-specific instance methods
namespace numpy {

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

	double* Vector::raw_data()
	{
		return data;
	}

	uint Vector::len()
	{
		return n;
	}

	Vector Vector::select(uint start)
	{
		if (start >= n)
		{
			throw std::invalid_argument("start must be -1 < start < n");
		}
		if (start == 0)
		{
			return copy();
		} else {
			// get a strip return
			Vector np(n - start);
			for (uint i = start, j = 0; i < n; i++, j++)
			{
				np.data[j] = data[i];
			}
			return np;
		}
	}

	Vector Vector::select(uint start, uint end, int step)
	{
		if (start >= n)
		{
			throw std::invalid_argument("start must be -1 < start < n");
		}
		if (end >= n)
		{
			throw std::invalid_argument("end must be -1 < end < n");
		}
		if (step == 0)
		{
			throw std::invalid_argument("step must not = 0");
		}
		if (end == $)
		{
			// select all from end
			end = n;
		}
		int newn = (int) (_truncate_doub_((end - start) / _absolute_(step), 1));
		if (start == $ && end != n && (step == 1 || step == -1))
		{
			newn += 1;
			end += 1;
		}
		Vector np(newn);
		// if we have negative step, swap end and start
		if (step < 0)
		{
			// reverse
			for (uint i = (end - 1), j = 0; i >= start; i+=step, j++)
			{
				np.data[j] = data[i];
			}
		} else {
			for (uint i = start, j = 0; i < end; j++, i+=step)
			{
				np.data[j] = data[i];
			}
		}
		return np;
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

	bool Vector::isColumn()
	{
		return column;
	}

	bool Vector::isRow()
	{
		return ! column;
	}

	uint Vector::count_nonzero()
	{
		return _count_nonzero_array_(data, n);
	}

	double Vector::mean()
	{
		return _summation_array_(data,n) / n;
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
		for (uint i = 0; i < n; i++)
		{
			data[i] = value;
		}
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
		double dottie = 0;
		for (uint i = 0; i < n; i++)
		{
			dottie += data[i] * rhs.data[i];
		}
		return dottie;
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
		for (uint i = 0; i < n; i++)
		{
			data[i] += rhs.data[i];
		}
		return *this;
	}
	Vector& Vector::operator+=(double value)
	{
		for (uint i = 0; i < n; i++)
		{
			data[i] += value;
		}
		return *this;
	}
	Vector& Vector::operator+=(int value)
	{
		double v = (double) value;
		for (uint i = 0; i < n; i++)
		{
			data[i] += v;
		}
		return *this;
	}


	Vector& Vector::operator-=(const Vector& rhs)
	{
		if (rhs.n != n)
		{
			throw std::invalid_argument("rhs size != array size");
		}
		for (uint i = 0; i < n; i++)
		{
			data[i] -= rhs.data[i];
		}
		return *this;
	}
	Vector& Vector::operator-=(double value)
	{
		for (uint i = 0; i < n; i++)
		{
			data[i] -= value;
		}
		return *this;
	}
	Vector& Vector::operator-=(int value)
	{
		double v = (double) value;
		for (uint i = 0; i < n; i++)
		{
			data[i] -= v;
		}
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
			for (uint i = 0; i < n; i++)
			{
				// else just multiply rhs index with lhs index
				data[i] *= rhs.data[i];
			}
		}
		return *this;
	}
	Vector& Vector::operator*=(double value)
	{
		for (uint i = 0; i < n; i++)
		{
			data[i] *= value;
		}
		return *this;
	}
	Vector& Vector::operator*=(int value)
	{
		double v = (double) value;
		for (uint i = 0; i < n; i++)
		{
			data[i] *= v;
		}
		return *this;
	}

	Vector& Vector::operator/=(const Vector& rhs)
	{
		if (rhs.n != n)
		{
			throw std::invalid_argument("rhs size != array size");
		}
		for (uint i = 0; i < n; i++)
		{
			data[i] /= rhs.data[i];
		}
		return *this;
	}
	Vector& Vector::operator/=(double value)
	{
		for (uint i = 0; i < n; i++)
		{
			data[i] /= value;
		}
		return *this;
	}
	Vector& Vector::operator/=(int value)
	{
		double v = (double) value;
		for (uint i = 0; i < n; i++)
		{
			data[i] /= v;
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

	/*
		ACCESSORY FUNCTIONS
	*/ 

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

