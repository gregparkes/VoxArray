/*
 * types.h
 *
 *  Created on: 15 Feb 2017
 *      Author: Greg
 */

#ifndef __NUMP_TYPES_H__
#define __NUMP_TYPES_H__

#define null 0x00
#define Numpy numpy::Vector
#define Mat numpy::Matrix
typedef unsigned int uint;
// for norm() order
#define _ONE_NORM 1
#define _TWO_NORM 2
#define _INF_NORM -1
#define _FRO_NORM -2
#define _NEG_INF_NORM -3
#define true 1
#define false 0
// for Vectors
#define AXIS_COLUMN 0
#define AXIS_ROW 1
// for quicksort
#define SORT_ASCEND 0
#define SORT_DESCEND 1
// this is the select all symbol
#define $ 0x00
#define SELECT_ALL 0x00

//#define _CUMPY_DEBUG_ 0

namespace numpy {
	
	/* classes */
	class Vector;
	class Matrix;
	class Mask;
	
	/* structs */
	struct MATRIX_COMPLEX2;
	struct MATRIX_COMPLEX3;

}


#endif /* TYPES_H_ */
