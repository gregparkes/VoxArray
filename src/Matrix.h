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

#include "VoxTypes.h"

namespace numpy {

/********************************************************************************************

	FUNCTION DECLARATIONS 

*///////////////////////////////////////////////////////////////////////////////////////////

/**
	 Returns a matrix object with un-zeroed elements in.

	 @param ncols : the number of columns to create.
	 @param nrows : the number of rows to create.
	 @return The array object. <created on the stack>
	 */
	Matrix empty(uint ncols, uint nrows);

	/**
	 Returns a matrix object with zeroed elements in.

	 @param ncols : the number of columns to create.
	 @param nrows : the number of rows to create.
	 @return The array object. <created on the stack>
	 */
	Matrix zeros(uint ncols, uint nrows);

	/**
	 Returns a matrix object with all elements = 1 in.

	 @param ncols : the number of columns to create.
	 @param nrows : the number of rows to create.
	 @return The array object. <created on the stack>
	 */
	Matrix ones(uint ncols, uint nrows);

	/**
	 Returns a matrix object with filled elements in.

	 @param ncols : the number of columns to create.
	 @param nrows : the number of rows to create.
	 @param val : the value to fill the matrix with
	 @return The new array object. <created on the stack>
	 */
	Matrix fill(uint ncols, uint nrows, double val);

	/**
	 Returns the shape of the matrix in string representation

	 @param rhs : the matrix
	 @return shape
	*/
	char* shape(const Matrix& rhs);

	/**
	Returns a string representation of the matrix.

	e.g "[0.00, 1.00, 2.00, 3.00, 4.00]"

	@param rhs : the matrix to represent.
	@param dpoints (optional) : number of decimal places to keep in each value
	@return The corresponding string. <created on heap, must be deleted>
	 */
	char* str(const Matrix& rhs, uint dpoints = 5);

	/**
	 Copies a matrix object.

	 @param rhs : the matrix to copy.
	 @return The new matrix object. <created on the stack>
	 */
	Matrix copy(const Matrix& rhs);

	/**
	 Reshape the vector into a 2-d matrix.

	 @param rhs : the matrix
	 @param nrows : the number of new rows in the new matrix
	 @param ncols : the number of new columns in the new matrix
	 @return The new matrix <created on the stack>
	*/
	Matrix reshape(const Vector& rhs, uint nrows, uint ncols);

	/**
	 Reshape the matrix.

	 @param rhs : the old matrix
	 @param nrows : the number of new rows in the new matrix
	 @param ncols : the number of new columns in the new matrix
	 @return The new matrix < created on the stack>
	*/
	Matrix reshape(const Matrix& rhs, uint nrows, uint ncols);

	/**
	 Extracts all the non-zero values from a matrix and copies them.

	 @param rhs : the matrix to extract from.
	 @return The new non-zero matrix object. <created on the heap, must be deleted>
	 */
	Vector nonzero(const Matrix& rhs);

	/**
	 Flips the columns/rows in the matrix.

	 e.g [1.0, 2.0, 3.0] -> [3.0, 2.0, 1.0] for each row

	 @param rhs : the matrix to flip.
	 @param axis (optional) : either 0 (column-wise) or 1 (row-wise)
	 @return The flipped matrix. <created on the stack>
	 */
	Matrix flip(const Matrix& rhs, uint axis = 0);

	/**
	 Creates a copy of a matrix with unfilled elements.

	 @param rhs : the matrix to copy
	 @return The new matrix object <created on the stack>
	 */
	Matrix empty_like(const Matrix& rhs);

	/**
	 Creates a copy of a matrix with zeroed elements.

	 @param rhs : the matrix to copy
	 @return The new matrix object <created on the stack>
	 */
	Matrix zeros_like(const Matrix& rhs);

	/**
	 Creates a copy of a matrix with oned elements.

	 @param rhs : the matrix to copy
	 @return The new matrix object <created on the stack>
	 */
	Matrix ones_like(const Matrix& rhs);

	/**
	 Creates a matrix with random floats of uniform
	 distribution N[0, 1].

	 @param ncols : the number of columns to make
	 @param nrows : the number of rows to make
	 @return The new matrix object <created on the stack>
	 */
	Matrix rand(uint ncols, uint nrows);

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
	 Creates a matrix with random integers in N[0, max]

	 @param ncols : the number of columns to make
	 @param nrows : the number of rows to make
	 @param max : the max number to generate
	 @return The new matrix object <created on the stack>
	 */
	Matrix randint(uint ncols, uint nrows, uint max);

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
	 Draws a value from a Poisson Distribution.

	 @param lam : expectation of interval
	 @return value from distribution.
	 */
	long poisson(double lam = 1.0);

	/**
	 Returns a random sample of items from the whole matrix.

	 @param rhs : the matrix to sample from
	 @param n : number of samples to take
	 @return Sample vector <created on the stack>
	*/
	Vector sample(const Matrix& rhs, uint n);

	/**
	 Returns a random sample of items from the whole matrix, per column/row.

	 @param rhs : the matrix to sample from
	 @param n : number of samples to take
	 @return Sample matrix <created on the stack>
	*/
	Matrix sample(const Matrix& rhs, uint n, uint axis);

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
	 Copies a matrix and floors the elements to nearest whole number.

	 @param rhs : the matrix to floor.
	 @return The new matrix object <created on the stack>
	 */
	Matrix floor(const Matrix& rhs);

	/**
	 Copies a matrix and ceils the elements to nearest whole number.

	 @param rhs : the matrix to ceil.
	 @return The new matrix object <created on the stack>
	 */
	Matrix ceil(const Matrix& rhs);

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
	 Copies a matrix and sets all values to positive.

	 @param rhs : the matrix to absolute
	 @return The new matrix object <created on the stack>
	 */
	Matrix abs(const Matrix& rhs);

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
	 Calculates the cumulative sum of the matrix, column/row, into a new array.

	 e.g [1.0, 2.0, 3.0] -> [1.0, 3.0, 6.0]
	 -> a, a+b, a+b+c, a+b+c+d, ... , a+b+..+n

	 @param rhs : the matrix to sum
	 @param axis (optional) : either 0 (column-wise) or 1 (row-wise)
	 @return The new matrix cumulatively summed <created on the stack>
	 */
	Matrix cumsum(const Matrix& rhs, uint axis = 0);

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
	 Calculates the cumulative product of the matrix per row/column and copies into new array.

	 e.g [1.0, 2.0, 3.0] -> [1.0, 2.0, 6.0]
@todo Implement cumprod()
	 @param rhs : the matrix to product
	 @param axis (optional) : either 0 (column-wise) or 1 (row-wise)
	 @return The new matrix rray product. <created on the stack>
	 */
	Matrix cumprod(const Matrix& rhs, uint axis = 0);

	/**
	 Tests whether all elements in the matrix evaluate to True.

	 @param rhs : the matrix to evaluate
	 @return True or False
	 */
	bool all(const Matrix& rhs);

	/**
	 Tests whether any elements in the matrix evaluate to True.

	 @param rhs : the matrix to evaluate
	 @return True or False
	 */
	bool any(const Matrix& rhs);

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
	 Returns the largest value in the matrix.

	 @param rhs : the matrix to evaluate
	 @return Largest value
	 */
	double max(const Matrix& rhs);
	/**
	 Returns the largest value in the matrix, per column/row.

	 @param rhs : the matrix to evaluate
	 @param axis : either 0 (column-wise) or 1 (row-wise)
	 @return Largest value <created on the stack>
	 */
	Vector max(const Matrix& rhs, uint axis);

	/**
	 Returns the n-smallest values from the whole matrix rhs.

	 @param rhs : the matrix
	 @param n : the number of smallest values to get
	 @return N-smallest Vector <created on the stack>
	*/
	Vector nsmallest(const Matrix& rhs, uint n);

	/**
	 Returns the n-smallest values from the whole matrix rhs, per column/row.

	 @param rhs : the matrix
	 @param n : the number of smallest values to get
	 @param axis : either 0 (column-wise) or 1 (row-wise)
	 @return N-smallest Matrix <created on the stack>
	*/
	Matrix nsmallest(const Matrix& rhs, uint n, uint axis);

	/**
	 Returns the n-largest values from the whole matrix rhs.

	 @param rhs : the matrix
	 @param n : the number of largest values to get
	 @return N-largest Vector <created on the stack>
	*/
	Vector nlargest(const Matrix& rhs, uint n);

	/**
	 Returns the n-largest values from the whole matrix rhs, per column/row.

	 @param rhs : the matrix
	 @param n : the number of largest values to get
	 @param axis : either 0 (column-wise) or 1 (row-wise)
	 @return N-largest Matrix <created on the stack>
	*/
	Matrix nlargest(const Matrix& rhs, uint n, uint axis);

	/**
	 Calculates the average mean of the matrix, per column/row.

	 e.g (a + b + , ..., + n) / N

	 @param rhs : the matrix
	 @param axis (optional) : either 0 (column-wise) or 1 (row-wise)
	 @return The means of the matrix
	 */
	Vector mean(const Matrix& rhs, uint axis = 0);

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
	 Calculates the standard deviation (sd) of the matrix, per column/row.

	e.g sqrt((1 / N)(a-m^2 + b-m^2 + , ..., + n-m^2))

	 @param rhs : the matrix
	 @param axis (optional) : either 0 (column-wise) or 1 (row-wise)
	 @return The standard deviations (sd) of the matrix
	 */
	Vector std(const Matrix& rhs, uint axis = 0);

	/**
	 Calculates the variance of the matrix, per column/row.

	 e.g (a-m^2 + b-m^2 + , ..., + n-m^2) / (N-1)

	 @param rhs : the vector
	 @param axis (optional) : either 0 (column-wise) or 1 (row-wise)
	 @return The variances of the matrix
	*/
	Vector var(const Matrix& rhs, uint axis = 0);

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
	 Returns the smallest index in the matrix per column/row.

	 @param axis (optional) : either 0 (column-wise) or 1 (row-wise)
	 @return The smallest index
	 */
	Vector argmin(const Matrix& rhs, uint axis = 0);

	/**
	 Returns the largest index in the matrix per column/row.

	 @param axis (optional) : either 0 (column-wise) or 1 (row-wise)
	 @return The largest index
	 */
	Vector argmax(const Matrix& rhs, uint axis = 0);

	/*
	 Calculates the matrix-norm of the matrix.

	 order 1; ||A||1 = max_j \sum_i=1^n |a_ij|
	 order inf; ||A||inf = max_i \sum_j=1^n |a_ij|

	 @param rhs : the matrix
	 @param order (optional) : must be order = 1 or infinity, where 1 = absolute column
	 	 sum and infinity = absolute row sum
	 @return The norm of the matrix
	 */
	double norm(const Matrix& rhs, int order = _TWO_NORM);

	/**
	 Constructs a diagonal matrix from vector components.

	 @param rhs : diagonal elements in vector
	 @return Diagonal Matrix
	 */
	Matrix diag(const Vector& rhs);

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
	 Calculates the matrix-vector dot product. 
	 The matrix must be MxN and the vector Nx1 in dimensions.
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
	MATRIX_COMPLEX2 lu(const Matrix& A);

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
	 Applies the sine function to the matrix.

	 @param rhs : the matrix
	 @return The new sine matrix. <created on the stack>
	 */
	Matrix sin(const Matrix& rhs);

	/**
	 Applies the cosine function to the matrix.

	 @param rhs : the matrix
	 @return The new cos matrix. <created on the stack>
	 */
	Matrix cos(const Matrix& rhs);

	
	/**
	 Applies the tangent function to the matrix.

	 @param rhs : the matrix
	 @return The new tan matrix. <created on the stack>
	 */
	Matrix tan(const Matrix& rhs);

	/**
	 Creates a radians matrix from a degrees matrix.

	 @param rhs : the matrix
	 @return The new radians matrix <created on the stack>
	*/
	Matrix to_radians(const Matrix& rhs);	

	/**
	 Creates a degrees matrix from a radians matrix.

	 @param rhs : the matrix
	 @return The new degrees matrix <created on the stack>
	*/
	Matrix to_degrees(const Matrix& rhs);

	/**
	 Applies the exponential function to the matrix.

	 @param rhs : the matrix
	 @return The new exp matrix. <created on the stack>
	 */
	Matrix exp(const Matrix& rhs);

	/**
	 Creates an (applied) log_10 of the matrix.

	 @param rhs : the matrix
	 @return The new log matrix. <created on the stack>
	 */
	Matrix log(const Matrix& rhs);

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
	 Sorts the matrix elements either ascending or descending, per column/row using quicksort.

	 @param rhs : the matrix
	 @param axis : either 0 (column-wise) or 1 (row-wise)
	 @param sorter (optional) : indicates which direction to sort the values, ascending or descending.
	 @return The new sorted matrix. <created on the stack>
	 */
	Matrix sort(const Matrix& rhs, uint axis, uint sorter = SORT_ASCEND);

	/**
	 Calculates the 1st discrete differeence across a matrix, approximating using Euler's method.

	 @param rhs : the matrix
	 @param axis : either 0 (column-wise) or 1 (row-wise)
	 @periods : periods to shift for forming difference
	 @return The new difference matrix. <created on the stack>
	*/
	Matrix diff(const Matrix& rhs, uint axis, uint periods = 1);

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


/********************************************************************************************

		CLASS & MEMBER FUNCTIONS 

*///////////////////////////////////////////////////////////////////////////////////////////

class Matrix
{
	public:

		/**
			 Constructs an uninitialised matrix. 

			 WARNING: calling functions with this will likely lead to an error!
			 Use at your own peril!
		*/
		Matrix();
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
		 Reshape the matrix locally.

		 @param nrows : the number of new rows in the matrix
		 @param ncols : the number of new columns in the matrix
		 @return The reshaped matrix <object not created>
		*/
		Matrix& reshape(uint ncols, uint nrows);

		/**
		 Copy a column vector from this matrix object for further use.

		 @param i : the index column to copy.
		 @return A copied Vector object <created on the stack>
		*/
		Vector copy_col(uint i);

		/**
		 Copy a row vector from this matrix object for further use.

		 @param j : the index row to copy.
		 @return A copied Vector object <created on the stack>
		*/
		Vector copy_row(uint j);

		/**
		 Retrieve an element from the matrix.

		 @param i : index corresponding to the row number
		 @param j : index corresponding to the column number
		 @return Value at location [i,j]
		*/
		double& ix(uint i, uint j);

		/**
		 Access the underlying vector object, given column index.

		 **Note: This will return you the underlying object in memory,
		 changes to this object will be permanent**

		 @param i : index corresponding to the column number
		 @return The Vector object <object not created>
		*/
		Vector& icol(uint i);

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
