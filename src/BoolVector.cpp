
#ifndef __BoolVector_cpp__
#define __BoolVector_cpp__

#include <stdexcept>

#include "BoolVector.h"
#include "Vector.h"
#include "numstatic.cpp"

namespace numpy {

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
			throw std::range_error("n cannot = 0");
		}
		this->n = n;
		data = _create_empty_bool_(n);
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
			throw std::range_error("n cannot = 0");
		}
		this->n = r.n;
		data = _create_empty_bool_(this->n);

		if (data == NULL)
		{
			throw std::runtime_error("Unable to allocate memory");
		}
		this->flag_delete = true;
		// convert doubles to true or false.
		for (uint i = 0; i < n; i++)
		{
			data[i] = (bool) r.data[i];
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
				throw std::invalid_argument("Unable to destroy array");
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
			throw std::invalid_argument("Problem with creating bool representation");
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
		if (!_copy_bool_(m.data, data, n))
		{
			throw std::invalid_argument("copy failed!");
		}
		flag_delete = m.flag_delete;
		return m;
    }

    uint Mask::sum()
    {
    	uint count = 0;
    	for (uint i = 0; i < n; i++)
    	{
    		if (data[i])
    		{
    			count++;
    		}
    	}
    	return count;
    }

    double Mask::mean()
    {
    	return ((double) sum()) / n;
    }

}

#endif