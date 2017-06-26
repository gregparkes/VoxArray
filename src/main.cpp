/*
 * main.cpp
 *
 *  Created on: 18 Feb 2017
 *      Author: Greg
 */

#include "test_numpy1d.cpp"
#include "test_matrix.cpp"

int main(void)
{
	call_all_tests();
	//cout << x.str() << endl;
	//call_all_matrix_tests();


//#pragma omp parallel
//	{
//		printf("parallel working!\n");
//	}
	return 0;
}

