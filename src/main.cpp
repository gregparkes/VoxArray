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
 	std::cout << "---- Beginning Vector tests! ----" << std::endl;
	call_all_tests();
	std::cout << "---- Beginning Matrix tests! ----" << std::endl;
	call_all_matrix_tests();


//#pragma omp parallel
//	{
//		printf("parallel working!\n");
//	}
	return 0;
}

