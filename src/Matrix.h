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

    Matrix.h
*/

/*
 * Matrix.h
 *
 *  Created on: 18 Feb 2017
 *      Author: Greg
 */

#ifndef __MATRIX_H_
#define __MATRIX_H_

#include "types.h"
#include "Vector.h"

namespace numpy {

class Matrix
{
	public:

		/**
		 	 Contructs an empty matrix with uninitialized values.
		 */
		Matrix(uint ncol, uint nrow);
		/**
		 	 Deletes memory.
		 */
		~Matrix();

		/**
		 Returns a string representation of the object.

		 e.g "[[0.00, 1.00, 2.00, 3.00, 4.00]
		 	   [0.00, 1.00, 2.00, 3.00, 4.00]
		 	   [0.00, 1.00, 2.00, 3.00, 4.00]]"

		 @param dpoints (optional) : sets the number of values after decimal point to keep
		 @return The corresponding string. <created on heap, must be deleted>
		 */
		char* str(uint dpoints = 5);

		/**
		 Returns the shape of the matrix, e.g (3,4)

		 @return The shape of the matrix. <created on heap, must be deleted>
		*/
		char* shape();

		/**
		 Returns the length of saved memory stored for the entire matrix.

		@return the number of doubles in memory
		*/
		uint nfloats();

		/**
		 Copies a matrix object.

		 @return The new matrix object. <created on the stack>
		 */
		Matrix copy();

		/**
		 Retrieve an element from the matrix.

		 @param i : index corresponding to the row number
		 @param j : index corresponding to the column number
		 @return Value at location [i,j]
		*/
		inline double& ix(uint i, uint j) { return data[i+j*vectors[0]->n]; }

		/**
		 Flips the columns/rows in the matrix.

		 e.g [1.0, 2.0, 3.0] -> [3.0, 2.0, 1.0] for each row

		 @param axis : either 0 (column-wise) or 1 (row-wise)
		 @return The reference to this object. <object not created>
		 */
		Matrix& flip(uint axis = 0);

		/**
		 Clips (limits) the matrix. Given an interval, values that fall outside of the
		 interval are set to the interval.

		 e.g clip(x, 0.0, 1.0) -> sets all values 0 < x < 1.

		 @param a_min : minimum interval
		 @param a_max : maximum interval
		 @return The clipped matrix.
		 */
		Matrix& clip(double a_min, double a_max);

		/**
		 Sums (adds together) all the elements in the matrix.

		 @return The sum of the matrix
		 */
		double sum();

		/**
		 Sums (adds together) all the rows or columns in the matrix.

		 @param axis : either 0 (column-wise) or 1 (row-wise)
		 @return The vector sums of the matrix. <created on the stack>
		 */
		Vector sum(uint axis);

		/**
		 Calculates the product of the elements in the matrix.

		 @return The product of the array
		 */
		double prod();

		/**
		 Calculates the product of the rows or columns in the matrix.

		 @param axis : either 0 (column-wise) or 1 (row-wise)
		 @return The vector sums of the matrix <created on the stack>
		 */
		Vector prod(uint axis);

		/**
		 Updates all of the elements to positive.
		 e.g [-1.0, -5.0, 2.0] -> [1.0, 5.0, 2.0]

		 @return The reference to this object. <object not created>
		 */
		Matrix& abs();

		/**
		 Tests whether all elements in the matrix evaluate to True.
		 @return True or False
		 */
		bool all();

		/**
		 Tests whether any elements in the matrix evaluate to True.
		 @return True or False
		 */
		bool any();

		/**
		 Counts how many times 'value' appears in the matrix.

		 @param value : The value to count
		 @return The number of instances of value
		 */
		uint count(double value);

		/**
		 Counts how many times 'value' appears in each column/row of the matrix.

		 @param value : The value to count
		 @param axis : either 0 (column-wise) or 1 (row-wise)
		 @return The number of instances of value
		 */
		Vector count(double value, uint axis);

		/**
		 Counts the number of non-zeros in matrix (mode).

		 @return The count of non-zeros in matrix
		 */
		uint count_nonzero();

		/**
		  Counts the number of non-zeros in matrix, per row/column(mode).

		 @param axis : either 0 (column-wise) or 1 (row-wise)
		 @return The number of instances of value
		 */
		Vector count_nonzero(uint axis);

		/**
		 Calculates the average mean of the entire matrix.

		 @return The mean of the matrix
		 */
		double mean();

		/**
		 Calculates the average mean of the matrix per column/row.

		 e.g (a + b + , ..., + n) / N per row/column

		 @param axis : either 0 (column-wise) or 1 (row-wise)
		 @return The mean of the matrix
		 */
		Vector mean(uint axis);

		/**
		 Calculates the standard deviation of the matrix.

		 e.g sqrt((1 / N)(a-m^2 + b-m^2 + , ..., + n-m^2))

		 @return The std of the matrix
		 */
		double std();

		/**
		 Calculates the standard deviation of the matrix per column/row.

		 e.g sqrt((1 / N)(a-m^2 + b-m^2 + , ..., + n-m^2)) per row/column

		 @param axis : either 0 (column-wise) or 1 (row-wise)
		 @return The std of the matrix
		 */
		Vector std(uint axis);

		/**
		 Calculates the variance of the matrix.

		 e.g (a-m^2 + b-m^2 + , ..., + n-m^2) / (N-1)

		 @return The variance of the matrix
		*/
		double var();

		/**
		 Calculates the variance of the matrix per column/row.

		 e.g (a-m^2 + b-m^2 + , ..., + n-m^2) / (N-1) per col/row

		 @param axis : either 0 (column-wise) or 1 (row-wise)
		 @return The variance of the matrix
		*/
		Vector var(uint axis);

		/**
		 Returns the smallest index in the matrix.

		 @return The smallest index
		 */
		double argmin();

		/**
		 Returns the smallest index in the matrix per column/row.

		 @param axis : either 0 (column-wise) or 1 (row-wise)
		 @return The smallest index
		 */
		Vector argmin(uint axis);

		/**
		 Returns the largest index in the matrix.

		 @return The largest index
		 */
		double argmax();

		/**
		 Returns the largest index in the matrix per column/row.

		 @param axis : either 0 (column-wise) or 1 (row-wise)
		 @return The largest index
		 */
		Vector argmax(uint axis);

		/**
		 Returns the smallest value in the matrix.

		 @return Smallest value
		 */
		double min();

		/**
		 Returns the smallest value in the matrix per column/row.

		 @param axis : either 0 (column-wise) or 1 (row-wise)
		 @return Smallest value
		 */
		Vector min(uint axis);

		/**
		 Returns the largest value in the matrix.

		 @return Largest value
		 */
		double max();

		/**
		 Returns the largest value in the matrix per column/row.

		 @param axis : either 0 (column-wise) or 1 (row-wise)
		 @return Largest value
		 */
		Vector max(uint axis);

		/*
		 Calculates the matrix-norm of the matrix.

		 order 1; ||A||1 = max_j \sum_i=1^n |a_ij|
		 order inf; ||A||inf = max_i \sum_j=1^n |a_ij|

		 @param order (optional) : must be order = 1 or infinity, where 1 = absolute column
			 sum and infinity = absolute row sum
		 @return The norm of the matrix
		 */
		double norm(int order = _ONE_NORM);

		/**
		 Applies the sine function to the matrix.

		 @return The reference to this object. <object not created>
		 */
		Matrix& sin();

		/**
		 Applies the cosine function to the matrix.

		 @return The reference to this object. <object not created>
		 */
		Matrix& cos();

		/**
		 Applies the tangent function to the matrix.

		 @return The reference to this object. <object not created>
		 */
		Matrix& tan();

		/**
		 Converts the matrix from degrees to radians.

		 @return The reference to this object. <object not created>
		 */
		Matrix& to_radians();

		/**
		 Converts the matrix from radians to degrees.

		 @return The reference to this object. <object not created>
		 */
		Matrix& to_degrees();

		/**
		 Applies the exponential function to the matrix.

		 @return The reference to this object. <object not created>
		 */
		Matrix& exp();

		/**
		 Creates an (applied) log_10 of the matrix.

		 @return The reference to this object. <object not created>
		 */
		Matrix& log();

		/**
		 Creates an (applied) square root of the matrix.

		 @return The reference to this object. <object not created>
		 */
		Matrix& sqrt();

		/**
		 Creates an (applied) power of the matrix.
 @todo Implement powbase()
		 e.g -> base ^ this[i], (for all indices).

		 @param base (optional): the base value, default = 2
		 @return The reference to this object. <object not created>
		 */
		Matrix& pow_base(double base = 2.0);
		/**
		 Creates an (applied) power copy of the matrix.

		 e.g -> this[i] ^ exponent, (for all indices).
 @todo Implement powexp()
		 @param exponent (optional): the exponent value, default = 2
		 @return The reference to this object. <object not created>
		 */
		Matrix& pow_exp(double exponent = 2.0);

		/**
		 Applies a value to the vector.

		 @param value : the value to fill the matrix with.
		 @return The reference to this object. <object not created>
		 */
		Matrix& fill(double value);

		/**
		 Applies floor operation on the matrix to nearest whole number.

		 e.g [1.54, 1.86, 2.23] -> [1.00, 1.00, 2.00]

		 @return The reference to this object. <object not created>
		 */
		Matrix& floor();

		/**
		 Applies ceil operation on the matrix to nearest whole number.

		 e.g [1.54, 1.86, 2.23] -> [2.00, 2.00, 3.00]

		 @return The reference to this object. <object not created>
		 */
		Matrix& ceil();

		/**
		 Sort the elements in matrix in ascending/descending order by column/row

		 @param axis (optional) : either 0 (column-wise) or 1 (row-wise)
		 @param sorter (optional) : indicates which direction to sort the values, ascending or descending.
		 @return The reference to this object. <object not created>
		 */
		Matrix& sort(uint axis = 0, uint sorter = SORT_ASCEND);

		/**
		 Sort the matrix row/column-wise by a single row/column's values

		 e.g sort_by(0, 0)

		 [[1, 2, 2],	 [[3, 0, 0],
		  [2, 2, 1], ->   [2, 2, 1],
		  [3, 0, 0]]	  [1, 2, 2]]

		 @param index : the index to sort on
		 @param axis (optional) : either 0 (column-wise) or 1 (row-wise)
		 @param sorter (optional) : indicates which direction to sort the values, ascending or descending.
		 @return The reference to this object. <object not created>
		*/
		Matrix& sort_by(uint index, uint axis = 0, uint sorter = SORT_ASCEND);

		/**
		 * OPERATOR OVERLOADS
		 */

		bool operator==(const Matrix& rhs);
		bool operator!=(const Matrix& rhs);
		Matrix& operator+=(const Matrix& rhs);
		//Matrix& operator+=(const Vector& rhs);
		Matrix& operator+=(double value);
		Matrix& operator+=(int value);
		Matrix& operator-=(const Matrix& rhs);
		//Matrix& operator-=(const Vector& rhs);
		Matrix& operator-=(double value);
		Matrix& operator-=(int value);
		Matrix& operator*=(const Matrix& rhs);
		//Matrix& operator*=(const Vector& rhs);
		Matrix& operator*=(double value);
		Matrix& operator*=(int value);
		Matrix& operator/=(const Matrix& rhs);
		//Matrix& operator/=(const Vector& rhs);
		Matrix& operator/=(double value);
		Matrix& operator/=(int value);

	// variables to be publicly accessed.

	Vector **vectors;
	uint nvec;
	double *data;

};

Matrix operator+(const Matrix& l, const Matrix& r);
//Matrix operator+(const Matrix& l, const Vector& r);
Matrix operator+(const Matrix& l, double r);
Matrix operator+(const Matrix& l, int r);
Matrix operator+(double l, const Matrix& r);
Matrix operator+(int l, const Matrix& r);

Matrix operator-(const Matrix& l, const Matrix& r);
//Matrix operator-(const Matrix& l, const Vector& r);
Matrix operator-(const Matrix& l, double r);
Matrix operator-(const Matrix& l, int r);

Matrix operator*(const Matrix& l, const Matrix& r);
//Matrix operator*(const Matrix& l, const Vector& r);
Matrix operator*(const Matrix& l, double r);
Matrix operator*(const Matrix& l, int r);
Matrix operator*(double l, const Matrix& r);
Matrix operator*(int l, const Matrix& r);

Matrix operator/(const Matrix& l, const Matrix& r);
//Matrix operator/(const Matrix& l, const Vector& r);
Matrix operator/(const Matrix& l, double r);
Matrix operator/(const Matrix& l, int r);


/** ACCESSORY METHODS */

Matrix _copy_matrix_(const Matrix& m);
uint _fullsize_(const Matrix& m);

}

#endif
