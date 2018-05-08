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

*/

/*
 * VoxString.h
 *
 * Implements a (n,) string array
 *
 *  Created on: 17 Apr 2018
 *      Author: Greg
 */

#ifndef __VOX_STRING_H__
#define __VOX_STRING_H__

#include "VoxTypes.h"

namespace numpy {

/********************************************************************************************

    FUNCTION DECLARATIONS 

*///////////////////////////////////////////////////////////////////////////////////////////

    /**
     Returns the character at index i

     @param s : the string to search
     @param i : index
     @return the character
    */
    char get(const String& s, uint i);

    /**
     returns the length of the string.

     @param s : base string
     @return Length
    */
    uint len(const String& s);

    /**
     Creates a copy and converts all upper-case letters to lowercase.

     @param s : the string
     @return Copied lowercase string <created on the stack>
    */
    String lower(const String& s);

    /**
     Creates a copy and converts all lowercase letters to uppercase.

     @param s : the string
     @return Copied uppercase string <created on the stack>
    */
    String upper(const String& s);

    /**
     Capitalizes the first letter in s.

     @param s : base string
     @return Capitalized string <created on the stack>
    */
    String capitalize(const String& s);

    /**
     Attempts to convert string s into a float

     @param s : base string
     @return True or False
    */ 
    bool is_float(const String& s);

    /**
     Attempts to convert string s into an integer

     @param s : base string
     @return True or False
    */ 
    bool is_integer(const String& s);

    /**
     Concatenates strings together into a copy.

     @param a : first string
     @param b : second string
     @return Copied string concatenated <created on the stack>
    */
    String cat(const String& a, const String& b);

    /**
     Creates a string by stripping away all the values
     left of the index selected from s (including index).

     @param s : the string to strip from
     @param i : the index (from left) to start copying from
     @return The new string object <created on the stack>
     */
    String lstrip(const String& s, uint i);

    /**
     Creates a string by stripping away all the values
     right of the index selected from s (including index).

     @param s : the string to strip from
     @param i : the index (from right) to start copying from
     @return The new string object <created on the stack>
     */
    String rstrip(const String& s, uint i);

    /**
     Returns whether the string begins with a substring start.

     @param s : the string to search
     @param start : the substring to test.
     @return true or false
    */
    bool starts_with(const String& s, const String& start);

    /**
     Returns whether the string ends with a substring end.

     @param s : the string to search
     @param end : the substring to test.
     @return true or false
    */
    bool ends_with(const String& s, const String& end);

    /**
     Returns a substring between start and end, not including end index itself.

     If start >= end, it will swap start and end.
    
     @param s : the string to search
     @param start : begin index
     @param end : end index, up to
     @param step (default 1) : option to skip every (step) characters if desired
     @return Sub string <created on the stack>
    */  
    String substring(const String& s, uint start, uint end, uint step = 1);

    /**
     Repeat string s n_repeat times.

     @param s : the string
     @param n_repeats : the number of times to repeat s
     @return repeated string <created on the stack>
    */
    String repeat(const String& s, uint n_repeats);

    /**
     Creates a copy and replaces all instances of value in s with replace_with.

     @param s : the base string to perform on
     @param value : the original value in s
     @param replace_with : to replace value with
     @return Replaced string <created on the stack>
    */
    String replace(const String& s, const String& value, const String& replace_with);

    /**
     Finds all occurences of expr in the base string s.

     @param s : base string to search
     @param expr : expression to search for
     @return Vector of indexes (as float) <created on the stack>
    */
    Vector find(const String& s, const String& expr);

    /**
     Returns whether expr exists in s.

     @param s : base string
     @param expr : expression to search
     @return True or False
    */
    bool contains(const String& s, const String& expr);





/********************************************************************************************

    CLASS & MEMBER FUNCTIONS 

*///////////////////////////////////////////////////////////////////////////////////////////


class String
{
    public:

    /*********************************************************************************************

        CONSTRUCTORS/DESTRUCTOR 

    *///////////////////////////////////////////////////////////////////////////////////////////

        /*
         Empty constructor, all values null.
        */
        String();

        /*
         Allocated constructor, values empty.
        */
        String(uint n);

        /*
         Copied from input, allocated.
        */
        String(const char* input);

        // destructor
        ~String();

    /********************************************************************************************

        OTHER FUNCTIONS 

    *///////////////////////////////////////////////////////////////////////////////////////////



     /********************************************************************************************

        OPERATOR OVERLOADS

    *///////////////////////////////////////////////////////////////////////////////////////////

        



    /********************************************************************************************

        VARIABLES

    *///////////////////////////////////////////////////////////////////////////////////////////

    char *data;
    uint n;


};




}


#endif
