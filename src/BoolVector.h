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

    BoolVector.h
*/

#ifndef __cplusplus_BOOLVECTOR__
#define __cplusplus_BOOLVECTOR__

#include "types.h"

namespace numpy {

/********************************************************************************************

    FUNCTION DECLARATIONS 

*///////////////////////////////////////////////////////////////////////////////////////////

    /**
     Calculate the number of true values (count)

     @param rhs : the mask
     @return Count of true values
    */
    uint sum(const Mask& rhs);

    /**
     Calculate the mean of true values as a proportion of the array

     @param rhs : the mask
     @return Mean
    */
    double mean(const Mask& rhs);

     /**
     Convert the mask to a vector (as 0.0s and 1.0s)

     @param rhs : the mask to convert
     @return Vector object <created on the stack>
    */
    Vector to_vector(const Mask& rhs);

    /**
     Compares elements in a and b checks that all of elements in a are found in b.

     @param a : the base vector with values
     @param b : the vector to compare to (must be same size or smaller than a)
     @return Mask array of size (a) <created on the stack>
    */
    Mask isin(const Vector& a, const Vector& b);

    /**
     Tests whether any elements in the mask evaluate to True.

     @param m : the mask to evaluate
     @return True or False
    */
    bool any(const Mask& m);

    /**
     Tests whether all elements in the mask evaluate to True.

     @param m : the mask to evaluate
     @return True or False
    */
    bool all(const Mask& m);


/********************************************************************************************

        CLASS & MEMBER FUNCTIONS 

*///////////////////////////////////////////////////////////////////////////////////////////

class Mask
{
    public:

        /**
         Contructs an empty object, with no memory allocated.

         WARNING: Do not use unless you know what you are doing!

         Used by other objects for optimization.

         This object will likely crash if any method is called after using this constructor
         */
        Mask();

        /**
         Contructs an empty mask with uninitialized values.
         */
        Mask(uint n);

        /**
         Constructs an array filled with falses (0) or true (1+)

         @param n : size of mask
         @param f : [0 or 1..MAX]
        */
        Mask(uint n, uint f);

        /**
         Construct with vector input; computes true/false values from it, creating mask of same size
         */
        Mask(const Vector& r);

        /**
         Destructor.
        */
        ~Mask();

        /**
         String representation of the mask.

         @return The corresponding string. <created on heap, must be deleted>
        */
        char* str();

        /**
         Gives the user access to the raw array which stores the values.

         WARNING: Use at your own risk!

         @return The raw data - no memory allocated
         */
        bool* raw_data();

        /**
         Returns the length of the mask

         @return Length
        */
        uint len();

        // direct indexing
        inline bool& operator[](int idx) { return data[idx]; }

        /**
         Copies this mask.

         @return The new mask object. <created on the stack>
         */
        Mask copy();

        /**
         Computes the logical_not of every element in the array

         @return The reference to this object. <object not created>
        */
        Mask& logical_not();

        /**
         Computes the logical_and of every element in the array

         @return The reference to this object. <object not created>
        */
        Mask& logical_and(const Mask& rhs);

        /**
         Computes the logical_or of every element in the array

         @return The reference to this object. <object not created>
        */
        Mask& logical_or(const Mask& rhs);

        /**
         Calculate the sum of the true values present.

         @return Sum.
        */
        uint sum();

        /**
         Calculate the mean of the true values present.

         @return Mean.
        */
        double mean();

        /**
         Calculate the number of true (1) and false (0) present in array.

         @return Vector(2) with [0] as false, [1] as true.
        */
        Vector bincount();

        /**
         Convert the mask to a vector (as 0.0s and 1.0s)

         @return Vector object <created on the stack>
        */
        Vector to_vector();

        /* --------- OPERATOR OVERLOADS ---------------- */

        Mask& operator~();

        Mask& operator&(const Mask& rhs);

        Mask& operator|(const Mask& rhs);

    /* ----------------- PUBLIC VARIABLES -------------------------- */

    bool *data;
    uint n;
    bool flag_delete;

};  




}

#endif
