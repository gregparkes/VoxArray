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

*/

/*
 * BaseArray.cpp
 *
 * Base Array class template implementation. 

 *  Created on: 18 Feb 2018
 *      Author: Greg
 */

#ifndef __VOX_BASEARRAY_CPP__
#define __VOX_BASEARRAY_CPP__

#include <stdexcept>
#include <cstring>

#include "BaseArray.h"
#include "numstatic.cpp"

namespace numpy {


    template <class T> BaseArray<T>::BaseArray()
    {
#ifdef _CUMPY_DEBUG_
		printf("constructing empty array %x\n", this);
#endif
    	this->data = NULL;
    	this->n = 0;
    	this->_init_flag = false;
    } 

    template <class T> BaseArray<T>::BaseArray(uint n)
    {
#ifdef _CUMPY_DEBUG_
		printf("constructing empty array %x\n", this);
#endif
		if (n == 0)
		{
			RANGE("n cannot = 0");
		}
		this->n = n;
		this->data = new T[n];
		if (this->data == NULL)
		{
			throw std::runtime_error("Unable to allocate memory");
		}
        this->_init_flag = true;
    }

    template <class T> BaseArray<T>::BaseArray(T* values, uint n)
    {
#ifdef _CUMPY_DEBUG_
		printf("constructing array set %x\n", this);
#endif
		if (values == NULL)
		{
			INVALID("array in BaseArray() is empty");
		}
		if (n == 0)
		{
			INVALID("size in BaseArray() = 0");
		}
		this->n = n;
		this->data = new T[n];
		if (data == NULL)
		{
			throw std::runtime_error("Unable to allocate memory");
		}
		// copy across
		if (!_copy_array_<T>(this->data, values, n))
		{
			INVALID("Unable to copy array in BaseArray()");
		}
        this->_init_flag = true;
    }

    template <class T> BaseArray<T>::~BaseArray()
    {
    	if (this->_init_flag && this->data != NULL)
    	{
	#ifdef _CUMPY_DEBUG_
			printf("deleting array %x\n", this);
	#endif
			delete[] data;
    	}
    }

    template <class T> BaseArray<T>& BaseArray<T>::flip()
    {
    	_flip_array_<T>(this->data, this->n);
    	return *this;
    }

    template <class T> BaseArray<T>& BaseArray<T>::shuffle()
    {
    	_durstenfeld_fisher_yates_<T>(this->data, this->n);
    	return *this;
    }

    template <class T> BaseArray<T>& BaseArray<T>::fill(T value)
    {
    	_fill_array_<T>(this->data, this->n, value);
    	return *this;
    }

    template <class T> uint BaseArray<T>::count(T value)
    {
    	return _count_array_<T>(this->data, this->n, value);
    }




}


#endif