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

    numpy.h
*/

#ifndef __cplusplus_NUMPY__
#define __cplusplus_NUMPY__

#include "types.h"
#include "Vector.h"
#include "Matrix.h"

// various structures
#include "VarStructs.h"

namespace numpy {

	/* ---------------------------------------------------------------

	 Here we define static methods to be accessed as long as the user
	 is using the numpy namespace. This is the ideal way of creating arrays
	 to be manipulated.

	 * ---------------------------------------------------------------
	 */

	/**
	 Returns an array object with un-zeroed elements in.

	 @param n : the size of the array to create.
	 @return The array object. <created on the stack>
	 */
	Vector empty(uint n);

	/**
	 Returns a matrix object with un-zeroed elements in.

	 @param ncols : the number of columns to create.
	 @param nrows : the number of rows to create.
	 @return The array object. <created on the stack>
	 */
	Matrix empty(uint ncols, uint nrows);

	/**
	 Returns an array object with zeroed elements in.

	 @param n : the size of the array to create.
	 @return The new array object. <created on the stack>
	 */
	Vector zeros(uint n);

	/**
	 Returns a matrix object with zeroed elements in.

	 @param ncols : the number of columns to create.
	 @param nrows : the number of rows to create.
	 @return The array object. <created on the stack>
	 */
	Matrix zeros(uint ncols, uint nrows);

	/**
	 Returns an array object with all elements = 1 in.

	 @param n : the size of the array to create.
	 @return The new array object. <created on the stack>
	 */
	Vector ones(uint n);

	/**
	 Returns a matrix object with all elements = 1 in.

	 @param ncols : the number of columns to create.
	 @param nrows : the number of rows to create.
	 @return The array object. <created on the stack>
	 */
	Matrix ones(uint ncols, uint nrows);

	/**
	 Returns an array object with filled elements in.

	 @param n : the size of the array to create.
	 @param val : the value to fill the array with.
	 @return The new array object. <created on the stack>
	 */
	Vector fill(uint n, double val);

	/**
	 Returns a matrix object with filled elements in.

	 @param ncols : the number of columns to create.
	 @param nrows : the number of rows to create.
	 @param val : the value to fill the matrix with
	 @return The new array object. <created on the stack>
	 */
	Matrix fill(uint ncols, uint nrows, double val);

	/**
	 Returns the length of the vector

	 @param rhs : the vector
	 @return Length
	*/
	uint len(const Vector& rhs);

	/**
	 Returns the shape of the matrix in string representation

	 @param rhs : the matrix
	 @return shape
	*/
	char* shape(const Matrix& rhs);

	/**
	Returns a string representation of the object.

	e.g "[0.00, 1.00, 2.00, 3.00, 4.00]"

	@param rhs : the array to represent.
	@param dpoints (optional) : number of decimal places to keep in each value
	@return The corresponding string. <created on heap, must be deleted>
	 */
	char* str(const Vector& rhs, uint dpoints = 5);

	/**
	Returns a string representation of the matrix.

	e.g "[0.00, 1.00, 2.00, 3.00, 4.00]"

	@param rhs : the matrix to represent.
	@param dpoints (optional) : number of decimal places to keep in each value
	@return The corresponding string. <created on heap, must be deleted>
	 */
	char* str(const Matrix& rhs, uint dpoints = 5);

	/**
	 Returns an array object with set elements in.
	 e.g Numpy x = numpy::array("0.75, 0.34, 20.34, 0.67, -0.65");

	 @param values : list of values to add, same format as normal syntax except as string.
	 @return The new array object. <created on the stack>
	 */
	Vector array(const char* input);

	/**
	 Copies an array object.

	 @param rhs : the array to copy.
	 @return The new array object. <created on the stack>
	 */
	Vector copy(const Vector& rhs);

	/**
	 Copies a matrix object.

	 @param rhs : the matrix to copy.
	 @return The new matrix object. <created on the stack>
	 */
	Matrix copy(const Matrix& rhs);

	/**
	 Return a copy of the array flattened into 1-dimension, either column/row wise.

	 @param rhs : the matrix
	 @param axis (optional) : either 0 (column-wise) or 1 (row-wise)
	 @return Flattened array
	 */
	Vector vectorize(const Matrix& rhs, uint axis = 0);

	/**
	 Extracts all the non-zero values from an array and copies them.

	 @param rhs : the array to extract from.
	 @return The new non-zero array object. <created on the heap, must be deleted>
	 */
	Vector nonzero(const Vector& rhs);

	/**
	 Extracts all the non-zero values from a matrix and copies them.

	 @param rhs : the matrix to extract from.
	 @return The new non-zero matrix object. <created on the heap, must be deleted>
	 */
	Vector nonzero(const Matrix& rhs);

	/**
	 Finds all of the unique elements in the vector.
	 1 - The first column contains all the unique elements.
	 2 - The second column contains all of the counts of those elements.

	 @param rhs : the vector
	 @return New matrix with all unique elements.
	 */
	Matrix unique(const Vector& rhs);

	/**
	 Flips the elements in the array into a new copied array.

	 e.g [1.0, 2.0, 3.0] -> [3.0, 2.0, 1.0]

	 @param rhs : the array to flip.
	 @return The flipped array object. <created on the stack>
	 */
	Vector flip(const Vector& rhs);

	/**
	 Flips the columns/rows in the matrix.

	 e.g [1.0, 2.0, 3.0] -> [3.0, 2.0, 1.0] for each row

	 @param rhs : the matrix to flip.
	 @param axis (optional) : either 0 (column-wise) or 1 (row-wise)
	 @return The flipped matrix. <created on the stack>
	 */
	Matrix flip(const Matrix& rhs, uint axis = 0);

	/**
	Concatenates two arrays together, with the right vector joining
	to the right of the left vector.

	e.g [1.0, 2.0] + [3.0, 4.0] = [1.0, 2.0, 3.0, 4.0]

	@param lhs : the left-hand side vector
	@param rhs : the right-hand side vector
	@return The new array object <created on the stack>
	 */
	Vector vstack(const Vector& lhs, const Vector& rhs);

	/**
	 Creates a copy of a vector with unfilled elements.

	 @param rhs : the array to copy
	 @return The new array object <created on the stack>
	 */
	Vector empty_like(const Vector& rhs);

	/**
	 Creates a copy of a matrix with unfilled elements.

	 @param rhs : the matrix to copy
	 @return The new matrix object <created on the stack>
	 */
	Matrix empty_like(const Matrix& rhs);

	/**
	 Creates a copy of a vector with zeroed elements.

	 @param rhs : the array to copy
	 @return The new array object <created on the stack>
	 */
	Vector zeros_like(const Vector& rhs);

	/**
	 Creates a copy of a matrix with zeroed elements.

	 @param rhs : the matrix to copy
	 @return The new matrix object <created on the stack>
	 */
	Matrix zeros_like(const Matrix& rhs);

	/**
	 Creates a copy of a vector with oned elements.

	 @param rhs : the array to copy
	 @return The new array object <created on the stack>
	 */
	Vector ones_like(const Vector& rhs);

	/**
	 Creates a copy of a matrix with oned elements.

	 @param rhs : the matrix to copy
	 @return The new matrix object <created on the stack>
	 */
	Matrix ones_like(const Matrix& rhs);

	/**
	 Creates a vector with random floats of uniform
	 distribution N[0, 1].

	 @param n (optional) : the size of the desired array, default = 1 (i.e one random number)
	 @return The new array object <created on the stack>
	 */
	Vector rand(uint n = 1);

	/**
	 Creates a matrix with random floats of uniform
	 distribution N[0, 1].

	 @param ncols : the number of columns to make
	 @param nrows : the number of rows to make
	 @return The new matrix object <created on the stack>
	 */
	Matrix rand(uint ncols, uint nrows);

	/**
	 Creates a vector with random floats in line with
	 the normal (Gaussian) distribution - or bell-shaped
	 curve. Default mean = 0.0, variance = 1.0.

	-> p(x|mu,sig^2) = (1/(sqrt(2 sig^2 pi))e(-((x-u)^2/2 sig^2))

	 @param n (optional) : the size of the desired array, default = 1
	 @return The new array object <created on the stack>
	 */
	Vector randn(uint n = 1);

	/**
	 Creates a matrix with random floats in line with
	 the normal (Gaussian) distribution - or bell-shaped
	 curve. Default mean = 0.0, variance = 1.0.

	 -> p(x|mu,sig^2) = (1/(sqrt(2 sig^2 pi))e(-((x-u)^2/2 sig^2))

	 @param ncols : the number of columns to make
	 @param nrows : the number of rows to make
	 @return The new matrix object <created on the stack>
	 */
	Matrix randn(uint ncols, uint nrows);

	/**
	 Creates a vector with random floats in line with
	 the normal (Gaussian) distribution - or bell-shaped
	 curve.

	-> p(x|mu,sig^2) = (1/(sqrt(2 sig^2 pi))e(-((x-u)^2/2 sig^2))

	 @param n : the size of the desired array
	 @param mean : the center of the distribution
	 @param sd : standard deviation of the distribution
	 @return The new array object <created on the stack>
	 */
	Vector normal(uint n, double mean, double sd);

	/**
	 Creates a matrix with random floats in line with
	 the normal (Gaussian) distribution - or bell-shaped
	 curve.

	-> p(x|mu,sig^2) = (1/(sqrt(2 sig^2 pi))e(-((x-u)^2/2 sig^2))

	 @param ncols : the number of columns to make
	 @param nrows : the number of rows to make
	 @param mean : the center of the distribution
	 @param sd : standard deviation of the distribution
	 @return The new matrix object <created on the stack>
	 */
	Matrix normal(uint ncols, uint nrows, double mean, double sd);

	/**
	 Creates a vector with random integers in N[0, max]

	 @param n : the size of the desired array
	 @param max : the max number to generate
	 @return The new array object <created on the stack>
	 */
	Vector randint(uint n, uint max);

	/**
	 Creates a matrix with random integers in N[0, max]

	 @param ncols : the number of columns to make
	 @param nrows : the number of rows to make
	 @param max : the max number to generate
	 @return The new matrix object <created on the stack>
	 */
	Matrix randint(uint ncols, uint nrows, uint max);

	/**
	 Creates a vector with random elements from the
	 'values' array, in uniform distribution [0, 1].

	 e.g -> double xs[4] = {1.0, 3.0, 2.0, 0.75};
	 	 -> Numpy y = randchoice(10, xs, 4);

	 @param n : the size of the desired array.
	 @param values : pool of values to choose from.
	 @return The new array object <created on the stack>
	 */
	Vector randchoice(uint n, const char* values);

	/**
	 Creates a matrix with random elements from the
	 'values' array, in uniform distribution [0, 1].

	 e.g -> double xs[4] = {1.0, 3.0, 2.0, 0.75};
		 -> Numpy y = randchoice(10, xs, 4);

	 @param ncols : the number of columns to make
	 @param nrows : the number of rows to make
	 @param values : pool of values to choose from.
	 @return The new matrix object <created on the stack>
	 */
	Matrix randchoice(uint ncols, uint nrows, const char* values);

	/**
	 Creates a binomial response (as int) from the Binomial Distribution.

	 e.g binomial(10,0.5) simulates flipping a coin with n parameter = 10, most probable outcome is 5.

	 @param n : parameter of the distribution
	 @param p : the probability of getting 1 (success), in N[0, 1]
	 @return The integer generated from n and p.
	 */
	uint binomial(uint n = 1, double p = 0.5);

	/**
	 Creates a vector binomial response (as int) from the Binomial Distribution.

	 @param n : array parameter of the distribution
	 @param p : the array probability of getting 1 (success), in N[0, 1]
	 @return The binomial vector response (integers) <created on the stack>
	 */
	Vector binomial(const Vector& n, const Vector& p);

	/**
	 Draws a value from a Poisson Distribution.

	 @param lam : expectation of interval
	 @return value from distribution.
	 */
	long poisson(double lam = 1.0);

	/**
	 Draws samples from a Poisson Distribution.

	 @param lam : vector of expectation intervals
	 @return Poisson Vector <created on the stack>
	 */
	Vector poisson(const Vector& lam);

	/**
	 Creates a vector with evenly-spaced elements from
	 start to end. Array size dictated by step and
	 (end-start).

	 e.g arange(0.0, 0.5, 0.1) =
	 	 [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

	 @param start : the first element
	 @param end : the last element
	 @param step (optional) : the size of the each step, default value = 1.0
	 @return The new array object <created on the stack>
	 */
	Vector arange(double start, double end, double step = 1.0);

	/**
	 Creates a vector with evenly-spaced elements from
	 start to end.

	 e.g linspace(0.0, 0.5, 6) =
		 [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

	 @param start : the first element
	 @param end : the last element
	 @param n : the size of the array.
	 @return The new array object <created on the stack>
	 */
	Vector linspace(double start, double end, uint n);

	/**
	 Creates a vector with logarithmically-spaced elements from
	 start to end.

	 @param start : the first element
	 @param end : the last element
	 @param n : the size of the array.
	 @return The new array object <created on the stack>
	 */
	Vector logspace(double start, double end, uint n);

	/**
	 Creates a vector by stripping away all the values
	 left of the index selected from rhs (including index).

	 e.g [0.1, 0.3, 0.5, 0.7] (1) -> [0.3, 0.5, 0.7]

	 @param rhs : the array to strip from
	 @param idx : the index (from left) to start copying from
	 @return The new array object <created on the stack>
	 */
	Vector lstrip(const Vector& rhs, uint idx);

	/**
	 Creates a vector by stripping away all the values
	 right of the index selected from rhs (including index).

	 e.g [0.1, 0.3, 0.5, 0.7] (1) -> [0.1, 0.3]

	 @param rhs : the array to strip from
	 @param idx : the index (from right) to start copying from
	 @return The new array object <created on the stack>
	 */
	Vector rstrip(const Vector& rhs, uint idx);

	/**
	 Clips (limits) the vector. Given an interval, values that fall outside of the
	 interval are set to the interval.

	 e.g clip(x, 0.0, 1.0) -> sets all values 0 < x < 1.

	 @param rhs : array to clip
	 @param a_min : minimum interval
	 @param a_max : maximum interval
	 @return The clipped array. <created on the stack>
	 */
	Vector clip(const Vector& rhs, double a_min, double a_max);

	/**
	 Clips (limits) the matrix. Given an interval, values that fall outside of the
	 interval are set to the interval.

	 e.g clip(x, 0.0, 1.0) -> sets all values 0 < x < 1.

	 @param rhs : matrix to clip
	 @param a_min : minimum interval
	 @param a_max : maximum interval
	 @return The clipped matrix. <created on the stack>
	 */
	Matrix clip(const Matrix& rhs, double a_min, double a_max);

	/**
	 Copies a vector and floors the elements to nearest whole number.

	 e.g [1.54, 1.86, 2.23] -> [1.00, 1.00, 2.00]

	 @param rhs : the array to floor.
	 @return The new array object <created on the stack>
	 */
	Vector floor(const Vector& rhs);

	/**
	 Copies a matrix and floors the elements to nearest whole number.

	 @param rhs : the matrix to floor.
	 @return The new matrix object <created on the stack>
	 */
	Matrix floor(const Matrix& rhs);

	/**
	 Copies a vector and ceils the elements to nearest whole number.

	 e.g [1.54, 1.86, 2.23] -> [2.00, 2.00, 3.00]

	 @param rhs : the array to ceil.
	 @return The new array object <created on the stack>
	 */
	Vector ceil(const Vector& rhs);

	/**
	 Copies a matrix and ceils the elements to nearest whole number.

	 @param rhs : the matrix to ceil.
	 @return The new matrix object <created on the stack>
	 */
	Matrix ceil(const Matrix& rhs);

	/**
	 Counts the number of instances 'value' appears in array (mode).

	 @param rhs : the array to count
	 @param value : the value to find
	 @return The count of value in array
	 */
	int count(const Vector& rhs, double value);

	/**
	 Counts the number of instances 'value' appears in matrix (mode).

	 @param rhs : the matrix to count
	 @param value : the value to find
	 @return The count of value in matrix
	 */
	int count(const Matrix& rhs, double value);

	/**
	 Counts how many times 'value' appears in each column/row of the matrix.

	 @param value : The value to count
	 @param axis : either 0 (column-wise) or 1 (row-wise)
	 @return The number of instances of value
	 */
	Vector count(const Matrix& rhs, double value, uint axis);

	/**
	 Returns true if the vector is a column-vector. (Standard)

	 @param rhs : the vector
	 @return True or False
	 */
	bool isColumn(const Vector& rhs);

	/**
	 Returns true if the vector is a row-vector. (Transposed)

	 @param rhs : the vector
	 @return True or False
	 */
	bool isRow(const Vector& rhs);

	/**
	 Counts the number of non-zeros in array (mode).

	 @param rhs : the array to count
	 @return The count of non-zeros in array
	 */
	int count_nonzero(const Vector& rhs);

	/**
	 Counts the number of non-zeros in matrix (mode).

	 @param rhs : the matrix to count
	 @return The count of non-zeros in matrix
	 */
	int count_nonzero(const Matrix& rhs);

	/**
	 Counts the number of non-zeros in matrix (mode), per column/row.

	 @param rhs : the matrix to count
	 @param axis : either 0 or 1 for column/row
	 @return The count of non-zeros in matrix, in vector form per col/row
	 */
	Vector count_nonzero(const Matrix& rhs, uint axis);

	/**
	 Copies an array and sets all values to positive.

	 e.g [-1.0, -5.0, 2.0] -> [1.0, 5.0, 2.0]
	 y = |x|

	 @param rhs : the array to absolute
	 @return The new array object <created on the stack>
	 */
	Vector abs(const Vector& rhs);

	/**
	 Copies a matrix and sets all values to positive.

	 @param rhs : the matrix to absolute
	 @return The new matrix object <created on the stack>
	 */
	Matrix abs(const Matrix& rhs);

	/**
	 Sums (adds together) all the elements in the array.

	 e.g [1.0, 2.0, 3.0] -> 6.0

	 @param rhs : the array to sum
	 @return The sum of the array
	 */
	double sum(const Vector& rhs);
	/**
	 Sums (adds together) all the elements in the matrix.

	 @param rhs : the matrix to sum
	 @return The total sum of the matrix
	 */
	double sum(const Matrix& rhs);
	/**
	 Sums (adds together) each row/column in the matrix.

	 @param rhs : the matrix to sum
	 @param axis : either 0 (column-wise) or 1 (row-wise)
	 @return The total sum of the matrix columns/rows
	 */
	Vector sum(const Matrix& rhs, uint axis);

	/**
	 Calculates the product of the elements in the array.

	 e.g [1.0, 2.0, 3.0, 4.0] -> 1*2*3*4 = 24.0

	 @param rhs : the array to product
	 @return The product of the array
	 */
	double prod(const Vector& rhs);

	/**
	 Calculates the product of all the elements in the matrix.

	 @param rhs : the matrix to product
	 @return The total product of the matrix
	 */
	double prod(const Matrix& rhs);

	/**
	 Calculates the product of each row/column in the matrix.

	 @param rhs : the matrix to product
	 @param axis : either 0 (column-wise) or 1 (row-wise)
	 @return The total product of the matrix columns/rows
	 */
	Vector prod(const Matrix& rhs, uint axis);

	/**
	 Calculates the cumulative sum of the array into a new array.

	 e.g [1.0, 2.0, 3.0] -> [1.0, 3.0, 6.0]
	 -> a, a+b, a+b+c, a+b+c+d, ... , a+b+..+n

	 @param rhs : the array to sum
	 @return The new array cumulatively summed <created on the stack>
	 */
	Vector cumsum(const Vector& rhs);

	/**
	 Calculates the cumulative sum of the matrix, column/row, into a new array.

	 e.g [1.0, 2.0, 3.0] -> [1.0, 3.0, 6.0]
	 -> a, a+b, a+b+c, a+b+c+d, ... , a+b+..+n

	 @param rhs : the matrix to sum
	 @param axis (optional) : either 0 (column-wise) or 1 (row-wise)
	 @return The new matrix cumulatively summed <created on the stack>
	 */
	Matrix cumsum(const Matrix& rhs, uint axis = 0);

	/**
	 Calculates the adjacent sum of the array into a new array.
	 Does not wrap around, i.e idx [0] does not factor in [n-1] or
	 vice versa.

	 e.g [1.0, 2.0, 3.0] -> [3.0, 6.0, 5.0]
	 -> a+b, a+b+c, b+c+d, c+d

	 @param rhs : the array to sum
	 @return The new array adjacently summed <created on the stack>
	 */
	Vector adjacsum(const Vector& rhs);

	/**
	 Calculates the adjacent sum of the matrix into a new matrix.
	 Does not wrap around, i.e idx [0] does not factor in [n-1] or
	 vice versa.

	 e.g [1.0, 2.0, 3.0]	[5.0, 7.0, 8.0]
	 	 [2.0, 1.0, 3.0] =  [8.0, 9.0, 9.0]
	 	 [4.0, 1.0, 2.0] 	[7.0, 8.0, 6.0]

	 @param rhs : the matrix to sum
	 @return The new matrix adjacently summed <created on the stack>
	 */
	Matrix adjacsum(const Matrix& rhs);

	/**
	 Calculates the cumulative product and copies into new array.

	 e.g [1.0, 2.0, 3.0] -> [1.0, 2.0, 6.0]

	 @param rhs : the array to product
	 @return The new array product. <created on the stack>
	 */
	Vector cumprod(const Vector& rhs);

	/**
	 Calculates the cumulative product of the matrix per row/column and copies into new array.

	 e.g [1.0, 2.0, 3.0] -> [1.0, 2.0, 6.0]
@todo Implement cumprod()
	 @param rhs : the matrix to product
	 @param axis (optional) : either 0 (column-wise) or 1 (row-wise)
	 @return The new matrix rray product. <created on the stack>
	 */
	Matrix cumprod(const Matrix& rhs, uint axis = 0);

	/**
	 Integrate along the vector using the composite trapezoidal rule. Integral y(x).

	 @param y : y vector
	 @param x : x input vector (if left null, assumes x is evenly spaced)
	 @param dx : spacing between sample points (default 1)
	 @return Integral
	 */
	double trapz(const Vector& y, const Vector& x = null, double dx = 1.0);

	/**
	 Tests whether all elements in the vector evaluate to True.

	 @param rhs : the vector to evaluate
	 @return True or False
	 */
	bool all(const Vector& rhs);

	/**
	 Tests whether all elements in the matrix evaluate to True.

	 @param rhs : the matrix to evaluate
	 @return True or False
	 */
	bool all(const Matrix& rhs);

	/**
	 Tests whether any elements in the vector evaluate to True.

	 @param rhs : the vector to evaluate
	 @return True or False
	 */
	bool any(const Vector& rhs);

	/**
	 Tests whether any elements in the matrix evaluate to True.

	 @param rhs : the matrix to evaluate
	 @return True or False
	 */
	bool any(const Matrix& rhs);

	/**
	 Returns the smallest value in the vector.

	 @param rhs : the vector to evaluate
	 @return Smallest value
	 */
	double min(const Vector& rhs);
	/**
	 Returns the smallest value in the matrix.

	 @param rhs : the matrix to evaluate
	 @return Smallest value
	 */
	double min(const Matrix& rhs);
	/**
	 Returns the smallest value in the matrix, per column/row.

	 @param rhs : the matrix to evaluate
	 @param axis : either 0 (column-wise) or 1 (row-wise)
	 @return Smallest values
	 */
	Vector min(const Matrix& rhs, uint axis);

	/**
	 Returns the largest value in the vector.

	 @param rhs : the vector to evaluate
	 @return Largest value
	 */
	double max(const Vector& rhs);
	/**
	 Returns the largest value in the matrix.

	 @param rhs : the matrix to evaluate
	 @return Largest value
	 */
	double max(const Matrix& rhs);
	/**
	 Returns the largest value in the matrix, per column/row.

	 @param rhs : the matrix to evaluate
	 @param axis : either 0 (column-wise) or 1 (row-wise)
	 @return Largest value
	 */
	Vector max(const Matrix& rhs, uint axis);

	/**
	 Calculates the average mean of the vector.

	 e.g (a + b + , ..., + n) / N

	 @param rhs : the vector
	 @return The mean of the vector
	 */
	double mean(const Vector& rhs);

	/**
	 Calculates the average mean of the matrix, per column/row.

	 e.g (a + b + , ..., + n) / N

	 @param rhs : the matrix
	 @param axis (optional) : either 0 (column-wise) or 1 (row-wise)
	 @return The means of the matrix
	 */
	Vector mean(const Matrix& rhs, uint axis = 0);

	/**
	 Finds the median value in the vector. By default we assume the vector is
	 unordered, so this is a linear operation. Make sure the flag is set to true
	 if it is sorted, to maximise efficiency.

	 @param rhs : the vector
	 @param isSorted (optional) : default to false
	 @return Median value
	 */
	double median(const Vector& rhs, bool isSorted = false);

	/**
	 Finds the median value per column/row in the matrix. By default we assume the
	 vector(s) are unordered, so this is a linear operation. Make sure the flag is set
	 to true if it is sorted, to maximise efficiency.
@todo Implement median()
	 @param rhs : the matrix
	 @param axis (optional) : either 0 (column-wise) or 1 (row-wise)
	 @param isSorted (optional) : default to false
	 @return Median values
	 */
	Vector median(const Matrix& rhs, uint axis = 0, bool isSorted = false);

	/**
	 Calculates the standard deviation (sd) of the vector.

	e.g sqrt((1 / N)(a-m^2 + b-m^2 + , ..., + n-m^2))

	 @param rhs : the vector
	 @return The standard deviation (sd) of the vector
	 */
	double std(const Vector& rhs);

	/**
	 Calculates the standard deviation (sd) of the matrix, per column/row.

	e.g sqrt((1 / N)(a-m^2 + b-m^2 + , ..., + n-m^2))

	 @param rhs : the matrix
	 @param axis (optional) : either 0 (column-wise) or 1 (row-wise)
	 @return The standard deviations (sd) of the matrix
	 */
	Vector std(const Matrix& rhs, uint axis = 0);

	/**
	 Calculates the variance of the vector.

	 e.g (a-m^2 + b-m^2 + , ..., + n-m^2) / (N-1)

	 @param rhs : the vector
	 @return The variance of the vector
	*/
	double var(const Vector& rhs);

	/**
	 Calculates the variance of the matrix, per column/row.

	 e.g (a-m^2 + b-m^2 + , ..., + n-m^2) / (N-1)

	 @param rhs : the vector
	 @param axis (optional) : either 0 (column-wise) or 1 (row-wise)
	 @return The variances of the matrix
	*/
	Vector var(const Matrix& rhs, uint axis = 0);

	/**
	 Calculates the covariance between two vectors.

	 @param v : vector 1
	 @param w : vector 2
	 @return The covariance
	 */
	double cov(const Vector& v, const Vector& w);

	/**
	 Calculates the covariance matrix from 2 or more column/row wise vectors.

	 @param A : the matrix
	 @return The covariance matrix
	 */
	Matrix cov(const Matrix& A);

	/**
	 Calculates the Pearson product-moment correlation coefficients from
	 the covariance matrix. All values are -1 < X < 1

	 @param A : the matrix
	 @return The correlation matrix
	 */
	Matrix corr(const Matrix& A);

	/**
	 Returns the smallest index of a value in the vector.

	 @param rhs : the vector
	 @return The smallest value
	 */
	uint argmin(const Vector& rhs);

	/**
	 Returns the smallest index in the matrix per column/row.

	 @param axis (optional) : either 0 (column-wise) or 1 (row-wise)
	 @return The smallest index
	 */
	Vector argmin(const Matrix& rhs, uint axis = 0);

	/**
	 Returns the largest index of a value in the vector.

	 @param rhs : the vector
	 @return The largest value
	 */
	uint argmax(const Vector& rhs);

	/**
	 Returns the largest index in the matrix per column/row.

	 @param axis (optional) : either 0 (column-wise) or 1 (row-wise)
	 @return The largest index
	 */
	Vector argmax(const Matrix& rhs, uint axis = 0);

	/**
	 Calculates the vector-norm of the vector.

	 @param rhs : the vector
	 @param order (optional) : must be -> order >= 1 or -1 (infinity)
	 	 	 	    use _INF_NORM, _ONE_NORM, _TWO_NORM ideally, default = 2
	 @return The norm of the vector
	 */
	double norm(const Vector& rhs, int order = _TWO_NORM);

	/*
	 Calculates the matrix-norm of the matrix.

	 order 1; ||A||1 = max_j \sum_i=1^n |a_ij|
	 order inf; ||A||inf = max_i \sum_j=1^n |a_ij|

	 @param rhs : the matrix
	 @param order (optional) : must be order = 1 or infinity, where 1 = absolute column
	 	 sum and infinity = absolute row sum
	 @return The norm of the matrix
	 */
	double norm(const Matrix& rhs, int order = _ONE_NORM);
	/**
	 Constructs a diagonal matrix from vector components.

	 @param rhs : diagonal elements in vector
	 @return Diagonal Matrix
	 */
	Matrix diag(const Vector& rhs);

	/**
	 Extracts the diagonal elements of a matrix.

	 @param rhs : the matrix
	 @return Diagonal elements
	 */
	Vector diag(const Matrix& rhs);

	/**
	 Returns a copy of the matrix with upper-triangular elements zeroed.

	 @param rhs : the matrix
	 @return new matrix lower-triangular
	 */
	Matrix tril(const Matrix& rhs, bool diag = true);

	/**
	 Returns a copy of the matrix with lower-triangular elements zeroed.

	 @param rhs : the matrix
	 @return new matrix upper-triangular
	 */
	Matrix triu(const Matrix& rhs, bool diag = true);

	/**
	 Computes the sum along the diagonals of the matrix.

	 @param rhs : the matrix
	 @return sum along diagonals
	 */
	double trace(const Matrix& rhs);

	/**
	 Calculates the Hermitian inner product of two vectors. The vectors must be
	 both column vectors. Also known as the scalar product.

	 e.g x = sum(a*b).

	 @param v : the left-hand side vector
	 @param w : the right-hand side vector
	 @return The dot product
	 */
	double inner(const Vector& v, const Vector& w);

	/**
	 Calculates the dot product of two vectors. The vectors must be
	 both column vectors. Also known as the scalar product.

	 e.g x = sum(a*b).

	 @param v : the left-hand side vector
	 @param w : the right-hand side vector
	 @return The dot product
	 */
	double dot(const Vector& v, const Vector& w);

	/**
	 Calculates the matrix-vector dot product. The matrix must be MxN and the vector Nx1 in dimensions.
	 The vector must also be a column-vector.

	 e.g
	 [1, 2, 3] . [2] = [1.2 + 2.3 + 3.5]
	 [4, 5, 6] . [3] = [4.2 + 5.3 + 6.5]
	 [7, 8, 9] . [5] = [7.2 + 8.3 + 9.5]

	 @param A : the left-hand side matrix
	 @param v : the right-hand side vector
	 @return The vector dot product
	 */
	Vector dot(const Matrix& A, const Vector& v);

	/**
	 Calculates the matrix-matrix dot product. This is an extension of the matrix-vector dot product
	 where the columns in A are dotted to the rows in B.

	 @param A : the left-hand side matrix
	 @param B : the right-hand side matrix
	 @return the matrix dot product
	 */
	Matrix dot(const Matrix& A, const Matrix& B);

	/**
	 Computes the outer product of two vectors, where v is a column-vector and w is row-vector.
	 Also known as the tensor-product. Contrasts to the inner/dot product.

	 e.g outer = ab_T, where T is bT transposed.

	 @param v : the left column-vector (mx1)
	 @param w : the right row-vector (nx1)
	 @return Outer Product Matrix
	 */
	Matrix outer(const Vector& v, const Vector& w);

	/**
	 Calculates the cross product of two vectors, which is the vector that is
	 at a right angle to both v and w. Also known as the vector product.
	 Note that the cross product makes no sense above 3 dimensions, therefore
	 any vector v or w that are != 3 will be rejected.

	 @param v : left-hand side vector
	 @param w : right-hand side vector
	 @return The cross product vector
	*/
	Vector cross(const Vector& v, const Vector& w);
	
	/**
	 Constructs the identity matrix.

	 @param ncols : number of columns
	 @param nrows : number of rows
	 @return Identity matrix
	 */
	Matrix eye(uint ncols, uint nrows);

	/**
	 Calculate the determinant of a matrix.

	 @param rhs : the matrix
	 @return The Determinant
	 */
	double det(const Matrix& rhs);

	/**
	 Calculates the eigenvalues and eigenvectors of a given matrix.
	 The matrix must be square, otherwise an exception will be raised.
	 The first column in the matrix will contain the eigenvalues, the remaining
	 columns will contain the eigenvectors.
@todo Implement eig()
	 @param rhs : the matrix
	 @return Eigenvalues [0,:] and Eigenvectors[1:,:]
	 */
	Matrix eig(const Matrix& rhs);

	/**
	 Computes the condition number of the matrix.
	 It can use any of the 7 norms available.
@todo Implement cond()
	 @param rhs : the matrix
	 @param order (optional) : which norm to use
	 @return Condition Number
	 */
	double cond(const Matrix& rhs, int order = _TWO_NORM);

	/**
	 Calculates the rank of the matrix, using SVD.
	 Rank is the array of singular values in s that are greater than tol.
@todo Implement rank()
	 @param rhs : the matrix
	 @param tol : the tolerance to accept
	 @return The matrix rank
	 */
	uint rank(const Matrix& rhs, double tol = -1.0);

	/**
	 Computes QR factorization. Returns MATRIX_COMPLEX struct which simply
	 holds the matrix q (orthonormal) and r (upper-triangular).
@todo Implement qr_factorize()
	 @param A : the matrix
	 @return MATRIX_COMPLEX structure
	 */
	MATRIX_COMPLEX2& qr_factorize(const Matrix& A);

	/**
	 Computes LU Decomposition. Returns Matrix_Complex struct which holds
	 matrix l (lower triangular) and u (upper triangular).

	 @param A : the matrix
	 @return MATRIX_COMPLEX struct
	 */
	MATRIX_COMPLEX2& lu(const Matrix& A);

	/**
	 Solves the linear system Ax = b.

	 @param A : the matrix
	 @param b : the vector of known coefficients.
	 @return x the unknown coefficients
	 */
	Vector solve(const Matrix& A, const Vector& b);

	/**
	 Computes the singular-value decomposition of matrix X.
@todo Implement svd()
	 @param X : the matrix to decompose
	 @param compute_uv (optional) : whether to compute u, v, default true
	 @return MATRIX_COMPLEX struct of U, S, V_T
	 */
	MATRIX_COMPLEX3& svd(const Matrix& X, bool compute_uv = true);

	/**
	 Creates an (applied) sine copy of the vector.

	 @param rhs : the vector
	 @return The new sin vector <created on the stack>
	 */
	Vector sin(const Vector& rhs);
	/**
	 Applies the sine function to the matrix.

	 @param rhs : the matrix
	 @return The new sine matrix. <created on the stack>
	 */
	Matrix sin(const Matrix& rhs);

	/**
	 Creates an (applied) cosine copy of the vector.

	 @param rhs : the vector
	 @return The new cos vector <created on the stack>
	 */
	Vector cos(const Vector& rhs);
	/**
	 Applies the cosine function to the matrix.

	 @param rhs : the matrix
	 @return The new cos matrix. <created on the stack>
	 */
	Matrix cos(const Matrix& rhs);

	/**
	 Creates an (applied) tangent copy of the vector.

	 @param rhs : the vector
	 @return The new tan vector <created on the stack>
	 */
	Vector tan(const Vector& rhs);
	/**
	 Applies the tangent function to the matrix.

	 @param rhs : the matrix
	 @return The new tan matrix. <created on the stack>
	 */
	Matrix tan(const Matrix& rhs);

	/**
	 Creates an (applied) exponential copy of the vector.

	 @param rhs : the vector
	 @return The new exp vector <created on the stack>
	 */
	Vector exp(const Vector& rhs);
	/**
	 Applies the exponential function to the matrix.

	 @param rhs : the matrix
	 @return The new exp matrix. <created on the stack>
	 */
	Matrix exp(const Matrix& rhs);

	/**
	 Creates an (applied) log_10 copy of the vector.

	 @param rhs : the vector
	 @return The new log vector <created on the stack>
	 */
	Vector log(const Vector& rhs);
	/**
	 Creates an (applied) log_10 of the matrix.

	 @param rhs : the matrix
	 @return The new log matrix. <created on the stack>
	 */
	Matrix log(const Matrix& rhs);

	/**
	 Creates an (applied) square root copy of the vector.

	 @param rhs : the vector
	 @return The new sqrt vector <created on the stack>
	 */
	Vector sqrt(const Vector& rhs);
	/**
	 Creates an (applied) square root of the matrix.

	 @param rhs : the matrix
	 @return The new sqrt matrix. <created on the stack>
	 */
	Matrix sqrt(const Matrix& rhs);

	/**
	 Creates an (applied) radians copy of the vector, converting from degrees.

	 @param rhs : the vector
	 @return The new radians vector <created on the stack>
	 */
	Vector radians(const Vector& rhs);

	/**
	 Creates an (applied) degrees copy of the vector, converting from radians.

	 @param rhs : the vector
	 @return The new degrees vector <created on the stack>
	 */
	Vector degrees(const Vector& rhs);

	/**
	 Creates an (applied) power copy of the vector.

	 e.g -> base ^ exponent, (for all indices).

	 @param base : the base value
	 @param exponent : array of exponents
	 @return The new pow vector <created on the stack>
	 */
	Vector power(double base, const Vector& exponent);
	/**
	 Creates an (applied) power copy of the vector.

	 e.g -> base ^ exponent, (for all indices).

	 @param base : the base values (array)
	 @param exponent : power value
	 @return The new pow vector <created on the stack>
	 */
	Vector power(const Vector& base, double exponent);
	/**
	 Creates an (applied) power copy of the vector.

	 e.g -> base ^ exponent, (for all indices).

	 @param base : the base values (array)
	 @param exponent : array of exponents
	 @return The new pow vector <created on the stack>
	 */
	Vector power(const Vector& base, const Vector& exponent);

	/**
	 Sorts the vector elements either ascending or descending using quicksort.

	 @param rhs : the vector
	 @param sorter (optional) : indicates which direction to sort the values, ascending or descending.
	 @return The new sorted vector. <created on the stack>
	 */
	Vector sort(const Vector& rhs, uint sorter = SORT_ASCEND);

	/**
	 Sorts the matrix elements either ascending or descending, per column/row using quicksort.

	 @param rhs : the matrix
	 @param axis : either 0 (column-wise) or 1 (row-wise)
	 @param sorter (optional) : indicates which direction to sort the values, ascending or descending.
	 @return The new sorted matrix. <created on the stack>
	 */
	Matrix sort(const Matrix& rhs, uint axis, uint sorter = SORT_ASCEND);

	/**
	 Transposes the vector from column vector -> row vector or vice versa.

	 @param rhs : the vector
	 @return The new transposed vector. <created on the stack>
	 */
	Vector transpose(const Vector& rhs);

	/**
	 Transposes the matrix.

	 @param rhs : the matrix
	 @return The new transposed matrix. <created on the stack>
	 */
	Matrix transpose(const Matrix& rhs);

	/**
	 Join two vectors together to form a Nx2 matrix. Call vstack() if you
	 want to join together 2 vectors to make a longer vector.

	 @param lhs : the left-hand side vector
	 @param rhs : the right-hand side vector
	 @return The Matrix with col [0] = lhs and col [1] = rhs
	 	 	 The new array object <created on the stack>
	 */
	Matrix hstack(const Vector& lhs, const Vector& rhs);

	/* --------------------------------------------------------------------------------------- *
	 *
	 * Now we handle our global operator overloads of +, -, *, / etc. This applies to all
	 * classes of Vector, Matrix and higher dimensions. Special rules apply when we multiply
	 * transposes etc, but for the most part, this provides extended vector operations.
	 *
	 ----------------------------------------------------------------------------------------*/

	// All of these methods create new vectors on the stack.

	Vector operator+(const Vector& lhs, double value);
	Vector operator+(double value, const Vector& rhs);
	Vector operator+(const Vector& lhs, const Vector& rhs);

	Vector operator-(const Vector& lhs, double value);
	Vector operator-(const Vector& lhs, const Vector& rhs);

	Vector operator*(const Vector& lhs, double value);
	Vector operator*(double value, const Vector& rhs);
	Vector operator*(const Vector& lhs, const Vector& rhs);

	Vector operator/(const Vector& lhs, double value);
	Vector operator/(const Vector& lhs, const Vector& rhs);

	Vector operator^(const Vector& lhs, double exponent);
	Vector operator^(double base, const Vector& rhs);
	Vector operator^(const Vector& lhs, const Vector& rhs);

	// works by summing over all elements and seeing if it's less than/more than than the other value/array,
	// returns false if equal to.

	bool operator<(const Vector& lhs, double value);
	bool operator<(double value, const Vector& rhs);
	bool operator<(const Vector& lhs, const Vector& rhs);

	bool operator>(const Vector& lhs, double value);
	bool operator>(double value, const Vector& rhs);
	bool operator>(const Vector& lhs, const Vector& rhs);
}

#endif
