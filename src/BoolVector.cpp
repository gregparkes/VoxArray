/*

------------------------------------------------------

GNU General Public License:

    Gregory Parkes, Postgraduate Student at the University of Southampton, UK.
    Copyright (C) 2017-18 Gregory Parkes

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
    C. We also have Numpy.cpp. which are C++ wrapper classes
    around this foundational file.

    BoolVector.cpp
*/


#ifndef __BoolVector_cpp__
#define __BoolVector_cpp__

#include <stdexcept>

#include "BoolVector.h"
#include "Vector.h"
#include "numstatic.cpp"

namespace numpy {

/********************************************************************************************

    STATIC FUNCTIONS 

*///////////////////////////////////////////////////////////////////////////////////////////

uint sum(const Mask& rhs)
{
	return (uint) (_boolean_summation_array_(rhs.data, rhs.n));
}

double mean(const Mask& rhs)
{
	return (double) (sum(rhs) / rhs.n);
}


/********************************************************************************************

        CLASS & MEMBER FUNCTIONS 

*///////////////////////////////////////////////////////////////////////////////////////////

	Mask::Mask()
	{
#ifdef _CUMPY_DEBUG_
		printf("constructing empty vector %x\n", this);
#endif
		this->n = 0;
		this->data = null;
		this->flag_delete = false;
	}

    Mask::Mask(uint n)
    {
#ifdef _CUMPY_DEBUG_
		printf("constructing mask normal %x\n", this);
#endif
		if (n == 0)
		{
			RANGE("n cannot = 0");
		}
		this->n = n;
		data = _create_empty_<bool>(n);
		if (data == NULL)
		{
			throw std::runtime_error("Unable to allocate memory");
		}
		this->flag_delete = true;
    }

    Mask::Mask(const Vector& r)
    {
#ifdef _CUMPY_DEBUG_
		printf("constructing mask from vector %x\n", this);
#endif
		if (r.n == 0)
		{
			RANGE("n cannot = 0");
		}
		this->n = r.n;
		data = _create_empty_<bool>(this->n);

		if (data == NULL)
		{
			throw std::runtime_error("Unable to allocate memory");
		}
		this->flag_delete = true;
		// convert doubles to true or false.
		for (uint i = 0; i < this->n; i++)
		{
			if (CMP(r.data[i], 0.0))
			{
				data[i] = false;
			} 
			else
			{
				data[i] = true;
			}
		}
    }

    Mask::~Mask()
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

    char* Mask::str()
    {
    	// we represent in terms of 0/1 e.g [1, 0, 0, 1]
    	unsigned int str_len = _str_bool_length_gen_(n);
    	char *strg = new char[str_len];
    	if (!_bool_representation_(strg, data, n))
		{
			INVALID("Problem with creating bool representation");
		}
		return strg;
    }

    bool* Mask::raw_data()
    {
    	return data;
    }

    uint Mask::len()
    {
    	return n;
    }

    Mask Mask::copy()
    {
    	Mask m(n);
		if (!_copy_array_<bool>(m.data, data, n))
		{
			INVALID("copy failed!");
		}
		flag_delete = m.flag_delete;
		return m;
    }

    Mask& Mask::logical_not()
    {
    	// perform element-wise not operation
    	_element_not_(data, n);
    	return *this;
    }

    Mask& Mask::logical_and(const Mask& rhs)
    {
    	if (n != rhs.n)
		{
			INVALID("masks must be the same size.");
		}
    	_element_and_(data, rhs.data, n);
    	return *this;
    }

    Mask& Mask::logical_or(const Mask& rhs)
    {
    	if (n != rhs.n)
		{
			INVALID("masks must be the same size.");
		}
    	_element_or_(data, rhs.data, n);
    	return *this;
    }

    uint Mask::sum()
    {
    	return (uint) (_boolean_summation_array_(data, n));
    }

    double Mask::mean()
    {
    	return ((double) sum()) / n;
    }

    Vector Mask::bincount()
    {
    	Vector res = empty(2);
    	// the number of trues is the summation of (1)
    	// the number of false is therefore the inverse of trues
    	int all_trues = _boolean_summation_array_(data, n);
    	if (all_trues != -1)
    	{
    		res[1] = all_trues;
	    	res[0] = (n - all_trues);
    	}
    	else
    	{
    		res[1] = 0;
    		res[0] = n;
    	}
    	return res;
    }

	/*  --------------- Instance Operator overloads ------------------------ */

	Mask& Mask::operator~()
	{
		_element_not_(data, n);
		return *this;
	}

	Mask& Mask::operator&(const Mask& rhs)
	{
		if (n != rhs.n)
		{
			INVALID("masks must be the same size.");
		}
		_element_and_(data, rhs.data, n);
		return *this;
	}

	Mask& Mask::operator|(const Mask& rhs)
	{
		if (n != rhs.n)
		{
			INVALID("masks must be the same size.");
		}
		_element_or_(data, rhs.data, n);
		return *this;
	}




}

#endif
