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
    C. We also have Numpy.cpp. which are C++ wrapper classes
    around this foundational file.

    Vec1f.h
*/

/*
 * Vec1f.h
 *
 *  Created on: 18 Feb 2017
 *      Author: Greg
 */

#ifndef __VEC1F_H_
#define __VEC1F_H_

namespace numpy {

#include "types.h"

class Vector
{
	public:
		/**
		 Contructs an empty object, with no memory allocated.

		 WARNING: Do not use unless you know what you are doing!

		 Used by Matrix.h for optimization.

		 This object will likely crash if any method is called after using this constructor
		 */
		Vector();
		/**
		 Contructs an empty array with uninitialized values.
		 */
		Vector(uint n, bool column = AXIS_COLUMN);

		/**
		 Some constructor that simply input values, such as basic 2,3,4-D vectors
		 */
		Vector(double val1, double val2, double val3 = 99999.99999, double val4 = 99999.99999);

		/**
		 Deletes memory.
		 */
		~Vector();

		/**
		 Returns a string representation of the object.

		 e.g "[0.00, 1.00, 2.00, 3.00, 4.00]"
		 @param dpoints (optional) : sets the number of values after decimal point to keep
		 @return The corresponding string. <created on heap, must be deleted>
		 */
		char* str(uint dpoints = 5);

		/**
		 Gives the user access to the raw array which stores the values.

		 WARNING: Use at your own risk!

		 @return The raw data - no memory allocated
		 */
		double* raw_data();

		/**
		 Returns the length of the vector

		 @return Length
		*/
		uint len();

		/**
		 Indexes the vector and returns a splice of the selection.

		 e.g select(3) returns all from element 3 onwards.
		 	 select($) returns a copy

		 @param start (optional) : indicates where to start copying from indexwise.
		 @return The new select array <created on the stack>
		 */
		Vector select(uint start = $);

		/**
		 Indexes the vector and returns a splice of the selection.

		 e.g select(3) returns all from element 3 onwards.
			 select($, 6) returns all from 0 to 6
			 select(3, $) returns all from 3 to n-1
			 select(2, $, 2) returns all from 2 to n-1 skipping even elements

		 @param start : indicates where to start copying from indexwise, or $ for all
		 @param end : indicates which index to stop copying at, or $ for all
		 @param step (optional) : indicates the step size, default is 1, will not accept $.
		 @return The new select array <created on the stack>
		 */
		Vector select(uint start, uint end, int step = 1);

		/**
		 Copies an array object.

		 @return The new array object. <created on the stack>
		 */
		Vector copy();

		/**
		 Flips the elements in the array into a new copied array.

		 e.g [1.0, 2.0, 3.0] -> [3.0, 2.0, 1.0]

		 @return The reference to this object. <object not created>
		 */
		Vector& flip();

		/**
		 Clips (limits) the vector. Given an interval, values that fall outside of the
		 interval are set to the interval.

		 e.g clip(x, 0.0, 1.0) -> sets all values 0 < x < 1.

		 @param a_min : minimum interval
		 @param a_max : maximum interval
		 @return The clipped array.
		 */
		Vector& clip(double a_min, double a_max);

		/**
		 Updates all of the elements to positive.

		 e.g [-1.0, -5.0, 2.0] -> [1.0, 5.0, 2.0]

		 @return The reference to this object. <object not created>
		 */
		Vector& abs();

		/**
		 Sums (adds together) all the elements in the array.

		 e.g [1.0, 2.0, 3.0] -> 6.0

		 @return The sum of the array
		 */
		double sum();

		/**
		 Calculates the product of the elements in the array.

		 e.g [1.0, 2.0, 3.0, 4.0] -> 1*2*3*4 = 24.0

		 @return The product of the array
		 */
		double prod();

		/**
		 Tests whether all elements in the vector evaluate to True.

		 @return True or False
		 */
		bool all();

		/**
		 Tests whether any elements in the vector evaluate to True.

		 @return True or False
		 */
		bool any();

		/**
		 Counts how many elements in the vector are 'value'.

		 @param value : The value to count
		 @return The number of instances of value
		 */
		uint count(double value);

		/**
		 Returns true if the vector is a column-vector. (Standard)

		 @return True or False
		 */
		bool isColumn();

		/**
		 Returns true if the vector is a row-vector. (Transposed)

		 @return True or False
		 */
		bool isRow();

		/**
		 Counts the number of non-zeros in array (mode).

		 @param rhs : the array to count
		 @return The count of non-zeros in array
		 */
		uint count_nonzero();

		/**
		 Calculates the average mean of the vector.

		 e.g (a + b + , ..., + n) / N

		 @return The mean of the vector
		 */
		double mean();

		/**
		 Calculates the standard deviation (sd) of the vector.

		e.g sqrt((1 / N)(a-m^2 + b-m^2 + , ..., + n-m^2))

		 @return The std of the vector
		 */
		double std();

		/**
		 Calculates the variance of the vector.

		 e.g (a-m^2 + b-m^2 + , ..., + n-m^2) / (N-1)

		 @return The variance of the vector
		*/
		double var();

		/**
		 Returns the smallest index in the vector.

		 @return The smallest index
		 */
		uint argmin();

		/**
		 Returns the largest index in the vector.

		 @return The largest index
		 */
		uint argmax();

		/**
		 Calculates the vector-norm of the vector.

		 @param order : must be -> order >= 1 or -1 (infinity)
						use _INF_NORM, _ONE_NORM, _TWO_NORM ideally
		 @return The norm of the vector
		 */
		double norm(int order = _TWO_NORM);

		/**
		 Returns the smallest value in the vector.

		 @return Smallest value
		 */
		double min();

		/**
		 Returns the largest value in the vector.

		 @return Largest value
		 */
		double max();

		/**
		 Applies the sine function to the vector.

		 @return The reference to this object. <object not created>
		 */
		Vector& sin();

		/**
		 Applies the cosine function to the vector.

		 @return The reference to this object. <object not created>
		 */
		Vector& cos();

		/**
		 Applies the tangent function to the vector.

		 @return The reference to this object. <object not created>
		 */
		Vector& tan();

		/**
		 Applies the exponential function to the vector.

		 @return The reference to this object. <object not created>
		 */
		Vector& exp();

		/**
		 Creates an (applied) log_10 copy of the vector.

		 @return The reference to this object. <object not created>
		 */
		Vector& log();

		/**
		 Creates an (applied) square root copy of the vector.

		 @return The reference to this object. <object not created>
		 */
		Vector& sqrt();

		/**
		 Creates an (applied) power copy of the vector.

		 e.g -> base ^ this[i], (for all indices).

		 @param base (optional): the base value, default = 2
		 @return The reference to this object. <object not created>
		 */
		Vector& pow_base(double base = 2.0);
		/**
		 Creates an (applied) power copy of the vector.

		 e.g -> this[i] ^ exponent, (for all indices).

		 @param exponent (optional) : power value, default = 2
		 @return The reference to this object. <object not created>
		 */
		Vector& pow_exp(double exponent = 2.0);

		/**
		 Applies a value to the vector.

		 @param value : the value to fill the vector with.
		 @return The reference to this object. <object not created>
		 */
		Vector& fill(double value);

		/**
		 Applies floor operation on the elements to nearest whole number.

		 e.g [1.54, 1.86, 2.23] -> [1.00, 1.00, 2.00]

		 @return The reference to this object. <object not created>
		 */
		Vector& floor();

		/**
		 Applies cell operation on the elements to nearest whole number.

		 e.g [1.54, 1.86, 2.23] -> [2.00, 2.00, 3.00]

		 @return The reference to this object. <object not created>
		 */
		Vector& ceil();

		/**
		 Sort the elements in ascending/descending order

		 @param sorter (optional) : indicates which direction to sort the values, ascending or descending.
		 @return The reference to this object. <object not created>
		 */
		Vector& sort(uint sorter = SORT_ASCEND);

		/**
		 Computes the dot product from this vector and rhs vector

		 @param rhs : the other vector
		 @return Dot product
		 */
		double dot(const Vector& rhs);

		/**
		 Transpose the vector from column->row or vice versa.

		 @return The reference to this object. <object not created>
		 */
		Vector& T();

		// Operator Overloads

		// direct indexing
		double& operator[](uint idx);
		// indexing using the select struct

		Vector& operator=(double value);
		Vector& operator=(const Vector& rhs);

		bool operator==(const Vector& rhs);
		bool operator!=(const Vector& rhs);
		Vector& operator+=(const Vector& rhs);
		Vector& operator+=(double value);
		Vector& operator-=(const Vector& rhs);
		Vector& operator-=(double value);
		Vector& operator*=(const Vector& rhs);
		Vector& operator*=(double value);
		Vector& operator/=(const Vector& rhs);
		Vector& operator/=(double value);

	// variables to be publicly accessed.

	double *data;
	uint n;
	bool column, flag_delete;

};

}


#endif /* VEC1F_H_ */
