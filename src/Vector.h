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

    Vector.h
*/

/*
 * Vector.h
 *
 *  Created on: 18 Feb 2017
 *      Author: Greg
 */

#ifndef __VEC1F_H__
#define __VEC1F_H__

// Vector, Matrix, Mask object declared in 'types'
#include "types.h"

namespace numpy {

/********************************************************************************************

	FUNCTION DECLARATIONS 

*///////////////////////////////////////////////////////////////////////////////////////////

	/**
	 Returns an array object with un-zeroed elements in.

	 @param n : the size of the array to create.
	 @return The array object. <created on the stack>
	 */
	Vector empty(uint n);

	/**
	 Creates a copy of a vector with unfilled elements.

	 @param rhs : the array to copy
	 @return The new array object <created on the stack>
	 */
	Vector empty_like(const Vector& rhs);

	/**
	 Returns an array object with zeroed elements in.

	 @param n : the size of the array to create.
	 @return The new array object. <created on the stack>
	 */
	Vector zeros(uint n);

	/**
	 Creates a copy of a vector with zeroed elements.

	 @param rhs : the array to copy
	 @return The new array object <created on the stack>
	 */
	Vector zeros_like(const Vector& rhs);

	/**
	 Returns an array object with all elements = 1 in.

	 @param n : the size of the array to create.
	 @return The new array object. <created on the stack>
	 */
	Vector ones(uint n);

	/**
	 Creates a copy of a vector with oned elements.

	 @param rhs : the array to copy
	 @return The new array object <created on the stack>
	 */
	Vector ones_like(const Vector& rhs);

	/**
	 Returns an array object with filled elements in.

	 @param n : the size of the array to create.
	 @param val : the value to fill the array with.
	 @return The new array object. <created on the stack>
	 */
	Vector fill(uint n, double val);

	/**
	 Returns the length of the vector

	 @param rhs : the vector
	 @return Length
	*/
	uint len(const Vector& rhs);

	/**
	Returns a string representation of the object.

	e.g "[0.00, 1.00, 2.00, 3.00, 4.00]"

	@param rhs : the array to represent.
	@param dpoints (optional) : number of decimal places to keep in each value
	@param represent_float (optional) : if false will truncate and represent integers
	@return The corresponding string. <created on heap, must be deleted>
	 */
	char* str(const Vector& rhs, uint dpoints = 5, bool represent_float = true);

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
	 Copies and converts the vector into an (1,N) matrix (one column).
	 
	 @param rhs : the array to convert.
	 @return The new matrix object. <created on the stack>
		*/
	Matrix to_matrix(const Vector& rhs);

	/**
	 Copies and converts the vector into a vector-mask.

	 @param rhs : the vector to convert.
	 @return The new mask object <created on the stack>
	*/
	Mask to_mask(const Vector& rhs);

	/**
	 Selects values from array a using selected indices (converted to integer)

	 @param a : the array to take from
	 @param indices : the (integer) array containing indices
	 @return The selected indices <created on the stack>
	*/
	Vector take(const Vector& a, const Vector& indices);

	/**
	 Takes elements from array based on a boolean mask. Can also use vector[mask].
	 
	 @param a : the vector to search in
	 @param m : the mask to apply (true or false) in order to extract elements.
	 @param keep_shape : if false, drops the false values, else sets them to 0 but 
	 	maintains vector shape (still copies)
	 @return The selected subarray <created on the stack>
	*/
	Vector where(const Vector& a, const Mask& m, bool keep_shape=false);

	/**
	 Creates a vector with random floats of uniform distribution
	 distribution N[0, 1].

	 @param n (optional) : the size of the desired array, default = 1 (i.e one random number)
	 @return The new array object <created on the stack>
	 */
	Vector rand(uint n = 1);

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
	 Creates a vector with random integers in N[1, max]

	 @param n : the size of the desired array
	 @param max : the max number to generate
	 @return The new array object <created on the stack>
	 */
	Vector randint(uint n, uint max);

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
	 Creates a vector with random elements from the
	 'array' array, in uniform distribution [0, 1].

	 e.g -> double xs[4] = {1.0, 3.0, 2.0, 0.75};
	 	 -> Numpy y = randchoice(10, xs, 4);

	 @param n : the size of the desired (new) array.
	 @param array : pool of values to choose from.
	 @param arr_size : size of array
	 @return The new array object <created on the stack>
	 */
	Vector randchoice(uint n, const double* array, uint arr_size);

	/**
	 Creates a vector using random values from r selecting
	 from a uniform distribution. Assumes r is unique.
	
	 @param n : the size of the new array
	 @param r : the vector to select from
	 @return The new array object <created on the stack>
	*/
	Vector randchoice(uint n, const Vector& r);

	/**
	 Creates a vector binomial response (as int) from the Binomial Distribution.

	 @param n : number of trials
	 @param p : probability of each trial 1 (success)/0 (fail), in N[0, 1]
	 @param size : the number of tests in array
	 @return The binomial vector response (integers) <created on the stack>
	 */
	Vector binomial(uint n, double p, uint size);

	/**
	 Draws samples from a Poisson Distribution.

	 @param lam : expectation of intervial >= 0.
	 @param size : output shape size
	 @return Poisson Vector <created on the stack>
	 */
	 
	Vector poisson(double lam, uint size);

	/**
	 Returns a random sample of items from the vector.

	 *WARNING* may NOT return the exact size sample as n - more reliable with larger vectors.

	 @param rhs : the vector to sample from
	 @param n : number of samples to take
	 @return Sample vector <created on the stack>
	*/
	Vector sample(const Vector& rhs, uint n);

	/**
	 Return a copy of the array flattened into 1-dimension, either column/row wise.

	 @param rhs : the matrix
	 @param axis (optional) : either 0 (column-wise) or 1 (row-wise)
	 @return Flattened array <created on the stack>
	 */
	Vector vectorize(const Matrix& rhs, uint axis = 0);

	/**
	 Extracts all the non-zero values from an array and copies them.

	 @param rhs : the array to extract from.
	 @return The new non-zero array object. <created on the stack>
	 */
	Vector nonzero(const Vector& rhs);

	/**
	 Flips the elements in the array into a new copied array.

	 e.g [1.0, 2.0, 3.0] -> [3.0, 2.0, 1.0]

	 @param rhs : the array to flip.
	 @return The flipped array object. <created on the stack>
	 */
	Vector flip(const Vector& rhs);

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
	 Concatenates two arrays together, flattening to 1-D.

	 e.g np.concat([1.0, 2.0], [3.0, 4.0], 0) = np([1.0, 2.0, 3.0, 4.0])
	 e.g np.concat([1.0, 2.0], [3.0, 4.0], 1) = np([[1.0, 3.0],
	 												 2.0, 4.0]])

	 @param lhs : the left-hand side vector
	 @param rhs : the right-hand side vector
	 @return The new matrix (1-2,N) object <created on the stack> 
	*/
	Matrix concat(const Vector& lhs, const Vector& rhs, uint axis);

	/**
	 Creates an evenly spaced vector incrementing by 1.0 until it reaches end.
	 End must be a positive value 0 < inf.

	 @param end : the end value to reach.
	 @return The new array object <created on the stack> 
	*/
	Vector arange(uint end);

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
	 Copies a vector and floors the elements to nearest whole number.

	 e.g [1.54, 1.86, 2.23] -> [1.00, 1.00, 2.00]

	 @param rhs : the array to floor.
	 @return The new array object <created on the stack>
	 */
	Vector floor(const Vector& rhs);

	/**
	 Copies a vector and ceils the elements to nearest whole number.

	 e.g [1.54, 1.86, 2.23] -> [2.00, 2.00, 3.00]

	 @param rhs : the array to ceil.
	 @return The new array object <created on the stack>
	 */
	Vector ceil(const Vector& rhs);

	/**
	 Counts the number of instances 'value' appears in array (mode). Returns 0 if no instances found.

	 @param rhs : the array to count
	 @param value : the value to find
	 @return The count of value in array
	 */
	int count(const Vector& rhs, double value);

	/**
	 Counts every unique value and places the count at the index.

	 @param rhs : the vector to count
	 @return The new vector of counts at indices. <created on the stack>
	*/
	Vector bincount(const Vector& rhs);

	/**
	 Finds all of the unique elements in the vector.
	 1 - The first column contains all the unique elements.
	 2 (optional) - The second column contains all of the counts of those elements.

	 @param rhs : the vector
	 @param get_counts (optional) : set to true if you want column 2 to be the counts of elements.  
	 @return New matrix (N,1-2) with all unique elements.
	 */
	Matrix unique(const Vector& rhs, bool get_counts = false);

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
	 Counts the number of non-zeros in array (mode). Returns 0 if all are zeros.

	 @param rhs : the array to count
	 @return The count of non-zeros in array
	 */
	int count_nonzero(const Vector& rhs);

	/**
	 Copies an array and sets all values to positive.

	 e.g [-1.0, -5.0, 2.0] -> [1.0, 5.0, 2.0]
	 y = |x|

	 @param rhs : the array to absolute
	 @return The new array object <created on the stack>
	 */
	Vector abs(const Vector& rhs);

	/**
	 Adds together a + b into copy c.

	 @param a : the left-hand side vector
	 @param b : the right-hand side vector
	 @return The new array object <created on the stack>
	*/
	Vector add(const Vector& a, const Vector& b);

	/**
	 Substracts together a - b into copy c.

	 @param a : the left-hand side vector
	 @param b : the right-hand side vector
	 @return The new array object <created on the stack>
	*/
	Vector sub(const Vector& a, const Vector& b);

	/**
	 Multiplies together a * b into copy c.

	 @param a : the left-hand side vector
	 @param b : the right-hand side vector
	 @return The new array object <created on the stack>
	*/
	Vector mult(const Vector& a, const Vector& b);

	/**
	 Divides together a / b into copy c.

	 @param a : the left-hand side vector
	 @param b : the right-hand side vector
	 @return The new array object <created on the stack>
	*/
	Vector div(const Vector& a, const Vector& b);

	/**
	 Sums (adds together) all the elements in the array.

	 e.g [1.0, 2.0, 3.0] -> 6.0

	 @param rhs : the array to sum
	 @return The sum of the array
	 */
	double sum(const Vector& rhs);

	/**
	 Calculates the mean of the vector.

	 e.g (a + b + , ..., + n) / N

	 @param rhs : the vector
	 @return The mean of the vector
	 */
	double mean(const Vector& rhs);

	/**
	 Finds the median value in the vector. By default we assume the vector is
	 unordered, so this is a linear O(n) operation. Make sure the flag is set to true
	 if it is sorted, to maximise efficiency.

	 @param rhs : the vector
	 @param isSorted (optional) : default to false
	 @return Median value
	 */
	double median(const Vector& rhs, bool isSorted = false);

	/**
	 Calculates the standard deviation (sd) of the vector.

	e.g sqrt((1 / N)(a-m^2 + b-m^2 + , ..., + n-m^2))

	 @param rhs : the vector
	 @return The standard deviation (sd) of the vector
	 */
	double std(const Vector& rhs);

	/**
	 Calculates the variance of the vector.

	 e.g (a-m^2 + b-m^2 + , ..., + n-m^2) / (N-1)

	 @param rhs : the vector
	 @return The variance of the vector
	*/
	double var(const Vector& rhs);

	/**
	 Calculates the product of the elements in the array.

	 e.g [1.0, 2.0, 3.0, 4.0] -> 1*2*3*4 = 24.0

	 @param rhs : the array to product
	 @return The product of the array
	 */
	double prod(const Vector& rhs);

	/**
	 Calculates the cumulative sum of the array into a new array.

	 e.g [1.0, 2.0, 3.0] -> [1.0, 3.0, 6.0]
	 -> a, a+b, a+b+c, a+b+c+d, ... , a+b+..+n

	 @param rhs : the array to sum
	 @return The new array cumulatively summed <created on the stack>
	 */
	Vector cumsum(const Vector& rhs);

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
	 Calculates the cumulative product and copies into new array.

	 e.g [1.0, 2.0, 3.0] -> [1.0, 2.0, 6.0]

	 @param rhs : the array to product
	 @return The new array product. <created on the stack>
	 */
	Vector cumprod(const Vector& rhs);

	/**
	 Integrate along the vector using the composite trapezoidal rule. Integral y(x).

	 @param y : y vector
	 @param x : x input vector (if left null, assumes x is evenly spaced)
	 @param dx : spacing between sample points (default 1)
	 @return Integral
	 */
	double trapz(const Vector& y, double dx = 1.0);

	/**
	 Tests whether all elements in the vector evaluate to True.

	 @param rhs : the vector to evaluate
	 @return True or False
	 */
	bool all(const Vector& rhs);

	/**
	 Tests whether any elements in the vector evaluate to True.

	 @param rhs : the vector to evaluate
	 @return True or False
	 */
	bool any(const Vector& rhs);

	/**
	 Returns the smallest value in the vector.

	 @param rhs : the vector to evaluate
	 @return Smallest value
	 */
	double min(const Vector& rhs);

	/**
	 Returns the smallest index of a value in the vector.

	 @param rhs : the vector
	 @return The smallest value
	 */
	uint argmin(const Vector& rhs);

	/**
	 Returns the largest value in the vector.

	 @param rhs : the vector to evaluate
	 @return Largest value
	 */
	double max(const Vector& rhs);

	/**
	 Returns the largest index of a value in the vector.

	 @param rhs : the vector
	 @return The largest value
	 */
	uint argmax(const Vector& rhs);

	/**
	 Returns the n-smallest values from vector rhs.

	 @param rhs : the vector
	 @param n (optional) : the number of smallest values to get, default 5
	 @return N-smallest Vector <created on the stack>
	*/
	Vector nsmallest(const Vector& rhs, uint n = 5);

	/**
	 Returns the n-largest values from vector rhs.

	 @param rhs : the vector
	 @param n (optional) : the number of largest values to get, default 5
	 @return N-largest Vector <created on the stack>
	*/
	Vector nlargest(const Vector& rhs, uint n = 5);

	/**
	 Calculates the covariance between two vectors.

	 @param v : vector 1
	 @param w : vector 2
	 @return The covariance
	 */
	double cov(const Vector& v, const Vector& w);

	/**
	 Calculates the pearson-moment correlation between two vectors.

	 @param v : vector 1
	 @param w : vector 2
	 @return The correlation (pearson)
	 */
	double corr(const Vector& v, const Vector& w);

	/**
	 Calculates the vector-norm of the vector.

	 @param rhs : the vector
	 @param order (optional) : must be -> order >= 1 or -1 (infinity)
	 	 	 	    use _INF_NORM, _ONE_NORM, _TWO_NORM ideally, default = 2
	 @return The norm of the vector
	 */
	double norm(const Vector& rhs, int order = _TWO_NORM);

	/**
	 Extracts the diagonal elements of a matrix.

	 @param rhs : the matrix
	 @return Diagonal elements
	 */
	Vector diag(const Matrix& rhs);

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
	 Calculate the magnitude of a vector.

	@param v : the vector
	@return The magnitude
	*/
	double magnitude(const Vector& v);

	/**
	 Normalizes the array by it's magnitude.
	
	@param v : the vector to normalize
	@return The normalized array
	*/
	Vector normalized(const Vector& v);

	/**
	 Standardize the vector by removing the mean and standard deviation scaling.

	 =(x-x.mean())/x.std()

	 @param v : the vector to standardize
	 @return The new standard vector <created on the stack>
	**/
	Vector standardize(const Vector& v);

	/**
	 Applies min-max scaling on the vector; as

	 x_std=(x - x.min()) / (x.max() - x.min())

	 @param v : the vector to scale
	 @return The new scaled vector <created on the stack>
	**/
	Vector minmax(const Vector& v);

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
	 Creates an (applied) sine copy of the vector.

	 @param rhs : the vector
	 @return The new sin vector <created on the stack>
	 */
	Vector sin(const Vector& rhs);

	/**
	 Creates an (applied) cosine copy of the vector.

	 @param rhs : the vector
	 @return The new cos vector <created on the stack>
	 */
	Vector cos(const Vector& rhs);

	/**
	 Creates an (applied) tangent copy of the vector.

	 @param rhs : the vector
	 @return The new tan vector <created on the stack>
	 */
	Vector tan(const Vector& rhs);

	/**
	 Creates a radians vector from a degrees vector.

	 @param rhs : the vector
	 @return The new radians vector <created on the stack>
	*/
	Vector to_radians(const Vector& rhs);

	/**
	 Creates a degrees vector from a radians vector.

	 @param rhs : the vector
	 @return The new degrees vector <created on the stack>
	*/
	Vector to_degrees(const Vector& rhs);

	/**
	 Creates an (applied) exponential copy of the vector.

	 @param rhs : the vector
	 @return The new exp vector <created on the stack>
	 */
	Vector exp(const Vector& rhs);

	/**
	 Creates an (applied) log_10 copy of the vector.

	 @param rhs : the vector
	 @return The new log vector <created on the stack>
	 */
	Vector log(const Vector& rhs);

	/**
	 Creates an (applied) square root copy of the vector.

	 @param rhs : the vector
	 @return The new sqrt vector <created on the stack>
	 */
	Vector sqrt(const Vector& rhs);

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
	 Calculates the 1st discrete differeence across a vector, approximating using Euler's method.

	 @param rhs : the vector
	 @param periods : periods to shift for forming difference
	 @return The new difference vector. <created on the stack> 
	*/
	Vector diff(const Vector& rhs, uint periods = 1);

	/**
	 Transposes the vector from column vector -> row vector or vice versa.

	 @param rhs : the vector
	 @return The new transposed vector. <created on the stack>
	 */
	Vector transpose(const Vector& rhs);

	/**
	Rotates a 2-D vector by a certain number of degrees.

	@param v : the vector to rotate
	@param degrees : the number of degrees to rotate by.
	@return The rotated vector.
	*/
	Vector rotate_vector2d(const Vector& v, double degrees);

	/**
	Calculates the angle between a set of vectors, up to n-dimensions.

	@param l : the left vector
	@param r : the right vector
	@return TThe angle between them
	*/
	double angle(const Vector& l, const Vector& r);

	/**
	 Calculates the projection using some lengths and directions. Practically applicable
	 in 2,3-D but can extend to n-dimension if needed.

	@param length : length vector
	@param dir : direction vector
	@return Projection vector
	*/
	Vector project(const Vector& length, const Vector& dir);

	/**
	 Calculates the perpendicular vector associated with length and direction vectors.

	@param length : length vector
	@param dir : direction vector
	@return Perpendicular vector
	*/
	Vector perpendicular(const Vector& length, const Vector& dir);

	/**
	 Calculates the Reflection vector associated with source and normal vectors.

	@param source : source vector
	@param normal : normal vector
	@return Reflection vector
	*/
	Vector reflection(const Vector& source, const Vector& normal);


/********************************************************************************************

		CLASS & MEMBER FUNCTIONS 

*///////////////////////////////////////////////////////////////////////////////////////////

class Vector
{
	public:

	/********************************************************************************************

		CONSTRUCTORS/DESTRUCTOR 

	*///////////////////////////////////////////////////////////////////////////////////////////

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
		 Some constructors that simply input values, such as basic 2,3,4-D vectors
		 */
		Vector(double v1, double v2);
		Vector(double v1, double v2, double v3);
		Vector(double v1, double v2, double v3, double v4);

		/**
		 A constructor using a pointer array. We copy the values over into our array and store.
		*/
		Vector(double *array, uint size);

		/**
		 Deletes memory.
		 */
		~Vector();

	/********************************************************************************************

		INLINE FUNCTIONS 

	*///////////////////////////////////////////////////////////////////////////////////////////

		/**
		 Gives the user access to the raw array which stores the values.

		 WARNING: Use at your own risk!

		 @return The raw data - no memory allocated
		 */
		inline double* raw_data() { return data; }

		/**
		 Returns the length of the vector

		 @return Length
		*/
		inline uint len() { return n; }

		/**
		 Returns true if the vector is a column-vector. (Standard)

		 @return True or False
		 */
		inline bool isColumn() { return column; }

		/**
		 Returns true if the vector is a row-vector. (Transposed)

		 @return True or False
		 */
		inline bool isRow() { return !column; }

		inline double& ix(int idx) { return data[idx]; }

		/**
		 Calculate whether this is a float array.
		*/
		bool isFloat();

		/**
		 Check whether all values are rounded with 0 sigfig.
		*/
		bool isInteger();

	/********************************************************************************************

		OTHER FUNCTIONS 

	*///////////////////////////////////////////////////////////////////////////////////////////

		/**
		 Returns a string representation of the object.

		 e.g "[0.00, 1.00, 2.00, 3.00, 4.00]"
		 @param dpoints (optional) : sets the number of values after decimal point to keep
		 @param represent_float (optional) : if false will truncate and represent integers
		 @return The corresponding string. <created on heap, must be deleted>
		 */
		char* str(uint dpoints = 5, bool represent_float = true);

		/**
		 Copies an array object.

		 @return The new array object. <created on the stack>
		 */
		Vector copy();

		/**
		 Copies and converts the vector into an (N,1) matrix (one column).

		 @return The new matrix object. <created on the stack>
		*/
		Matrix to_matrix();

		/**
		 Copies and converts the vector into a vector mask.

		 @return The new mask object. <created on the stack>
		*/
		Mask to_mask();

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
		 Converts degrees vector to radians.

		 @return The reference to this object. <object not created>
		*/
		Vector& to_radians();

		/**
		 Converts radians vector to degrees.

		 @return The reference to this object. <object not created>
		*/
		Vector& to_degrees();

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
		 Calculates the magnitude/norm of the vector.

		@return The magnitude value.
		*/
		double magnitude();

		/**
		 Calculates the distance between this vector and the given vector.

		@return The distance.
		*/
		double distance(const Vector& r);

		/**
		 Normalizes the array by it's magnitude.

		@return Null
		*/
		void normalize();

		/**
		 Transpose the vector from column->row or vice versa.

		 @return The reference to this object. <object not created>
		 */
		Vector& T();

		// Operator Overloads

		// direct indexing
		inline double& operator[](int idx) { return data[idx]; }

		// indirect indexing using masks - use a copy
		Vector operator[](const Mask& m);

		// indexing using the select struct

		Vector& operator+=(const Vector& rhs);
		Vector& operator+=(double value);
		Vector& operator+=(int value);
		Vector& operator-=(const Vector& rhs);
		Vector& operator-=(double value);
		Vector& operator-=(int value);
		Vector& operator*=(const Vector& rhs);
		Vector& operator*=(double value);
		Vector& operator*=(int value);
		Vector& operator/=(const Vector& rhs);
		Vector& operator/=(double value);
		Vector& operator/=(int value);

		/* ------------------- Mask Selector Query FUNCTION CHAINING ------------------------- */

		// less than <
		Mask lt(const Vector& r);
		Mask lt(double value);
		Mask lt(int value);
		// less than or equals <=
		Mask lte(const Vector& r);
		Mask lte(double value);
		Mask lte(int value);
		// more than >
		Mask mt(const Vector& r);
		Mask mt(double value);
		Mask mt(int value);
		// more than or equal >=
		Mask mte(const Vector& r);
		Mask mte(double value);
		Mask mte(int value);
		// equal ==
		Mask eq(const Vector& r);
		Mask eq(double value);
		Mask eq(int value);
		// not equal !=
		Mask neq(const Vector& r);
		Mask neq(double value);
		Mask neq(int value);

		/* ---------------- ELEMENT-WISE MATHEMATICS FUNCTION CHAINING --------------  */

		Vector& add(const Vector& rhs);
		Vector& add(double value);
		Vector& add(int value);

		Vector& sub(const Vector& rhs);
		Vector& sub(double value);
		Vector& sub(int value);

		Vector& mult(const Vector& rhs);
		Vector& mult(double value);
		Vector& mult(int value);

		Vector& div(const Vector& rhs);
		Vector& div(double value);
		Vector& div(int value);

	// variables to be publicly accessed.?

	double *data;
	uint n;
	bool column, flag_delete;

};

/********************************************************************************************

		GLOBAL CLASS OPERATOR OVERLOADS 

*///////////////////////////////////////////////////////////////////////////////////////////

// Overloading standard numerical operators for adding, subtracting, multiplying and 
// dividing by creating new vectors as the output to these.

	Vector operator+(const Vector& l, const Vector& r);
	Vector operator+(const Vector& l, double r);
	Vector operator+(const Vector& l, int r);
	Vector operator+(double l, const Vector& r);
	Vector operator+(int l, const Vector& r);

	Vector operator-(const Vector& l, const Vector& r);
	Vector operator-(const Vector& l, double r);
	Vector operator-(const Vector& l, int r);

	Vector operator*(const Vector& l, const Vector& r);
	Vector operator*(const Vector& l, double r);
	Vector operator*(const Vector& l, int r);
	Vector operator*(double l, const Vector& r);
	Vector operator*(int l, const Vector& r);

	Vector operator/(const Vector& l, const Vector& r);
	Vector operator/(const Vector& l, double r);
	Vector operator/(const Vector& l, int r);

	Vector operator^(const Vector& l, const Vector& r);
	Vector operator^(const Vector& l, double r);
	Vector operator^(const Vector& l, int r);

	Mask operator==(const Vector& l, const Vector& r);
	Mask operator==(const Vector& l, double r);
	Mask operator==(const Vector& l, int r);

	Mask operator!=(const Vector& l, const Vector& r);
	Mask operator!=(const Vector& l, double r);
	Mask operator!=(const Vector& l, int r);

	Mask operator<(const Vector& l, const Vector& r);
	Mask operator<(const Vector& l, double r);
	Mask operator<(const Vector& l, int r);

	Mask operator<=(const Vector& l, const Vector& r);
	Mask operator<=(const Vector& l, double r);
	Mask operator<=(const Vector& l, int r);

	Mask operator>(const Vector& l, const Vector& r);
	Mask operator>(const Vector& l, double r);
	Mask operator>(const Vector& l, int r);

	Mask operator>=(const Vector& l, const Vector& r);
	Mask operator>=(const Vector& l, double r);
	Mask operator>=(const Vector& l, int r);


/********************************************************************************************

		PRIVATE ACCESSORY FUNCTIONS 

*///////////////////////////////////////////////////////////////////////////////////////////

// Accessory function to calling copy in Vector.h.

	Vector _copy_vector_(const Vector& v);



// end scope

}


#endif /* VEC1F_H_ */
