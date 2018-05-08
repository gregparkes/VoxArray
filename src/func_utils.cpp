/*
 * func_utils.c

	Here we use C-style algorithms to implement the core-base to Vector and 
	Matrix-like operations.
 *
 *  Created on: 15 May 2018
 *      Author: Greg
 */

#ifndef __NUMPY_static_C_utils__
#define __NUMPY_static_C_utils__

#include "VoxTypes.h"

#define FLT_EPSILON 1.1920929E-07F

/* GLOBAL FUNCTIONS */

template <typename T>
static bool AlmostEqualRelativeAndAbs(T a, T b, double maxdiff,
		 double maxreldiff = FLT_EPSILON)
{
	// convert a, b to double if not using cast (DANGER DANGER?)
	double da = (double) a, db = (double) b;
	// check if the numbers are really close.
	double diff = fabs(da - db);
	if (diff <= maxdiff)
	{
		return true;
	}
	da = fabs(da);
	db = fabs(db);
	double largest = (db > da) ? db : da;
	if (diff <= largest * maxreldiff)
	{
		return true;
	}
	return false;
}


#define CMP(x, y) AlmostEqualRelativeAndAbs<double>(x, y, 1E-10)
#define WEAK_CMP(x, y) AlmostEqualRelativeAndAbs<double>(x, y, 1E-5)






#endif