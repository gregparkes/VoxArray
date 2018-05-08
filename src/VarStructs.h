/*
 * VarStructs.h
 *
 *  Created on: 17 Mar 2017
 *      Author: Greg
 */

#ifndef __VARSTRUCTS_H__
#define __VARSTRUCTS_H__

#include "VoxTypes.h"
#include "Matrix.h"

namespace numpy {

	struct MATRIX_COMPLEX2
	{
		MATRIX_COMPLEX2(Matrix& J, Matrix& K)
		{
			this->J = &J;
			this->K = &K;
		}
		~MATRIX_COMPLEX2() {}
		Matrix *J, *K;
	};

	struct MATRIX_COMPLEX3
	{
		MATRIX_COMPLEX3(Matrix& J, Matrix& K, Matrix& L)
		{
			this->J = &J;
			this->K = &K;
			this->L = &L;
		}
		~MATRIX_COMPLEX3() {}
		Matrix *J, *K, *L;
	};



}




#endif /* VARSTRUCTS_H_ */
