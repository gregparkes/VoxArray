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
 * BaseArray.h
 *
 * Base Array class template. 

 *  Created on: 18 Feb 2018
 *      Author: Greg
 */

#ifndef __VOX_BASEARRAY__
#define __VOX_BASEARRAY__

#include "types.h"

namespace numpy {

    template <class T> class BaseArray
    {
    public:

        /* VARIABLES */
        T *data;
        uint n;
        bool _init_flag;


        /* CONSTRUCTOR/DESTRUCTOR */
        BaseArray();
        BaseArray(uint n);
        BaseArray(T* values, uint n);
        virtual ~BaseArray();

        /* INLINE METHODS */
        inline T* raw_data() { return data; }
        inline uint len() { return n; }
        inline T& ix(int i) { return data[i]; }
        inline T& operator[](int i) { return data[i]; }
        inline bool is_init() { return _init_flag; }

        /* OTHER FUNCTIONS */
        virtual BaseArray& flip();
        virtual BaseArray& shuffle();
        virtual BaseArray& fill(T value);
        virtual uint count(T value);

        /* forced to implement str() */
        //virtual char* str() = 0;

    };

}

/* INCLUDE CPP FILE TO PREVENT COMPILE ERRORS */
#include "BaseArray.cpp"


 #endif