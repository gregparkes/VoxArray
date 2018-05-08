/*

------------------------------------------------------

GNU General Public License:

	Gregory Parkes, Postgraduate Student at the University of Southampton, UK.
    Copyright (C) 2017-18

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
 * VString.h
 *
 * Implements a (n,) string object
 *
 *  Created on: 17 Apr 2018
 *      Author: Greg
 */

#ifndef __VOX_VString_H__
#define __VOX_VString_H__

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
    char get(const VString& s, uint i);

    /**
     returns the length of the string.

     @param s : base string
     @return Length
    */
    uint len(const VString& s);

    /**
     Creates a copy and converts all upper-case letters to lowercase.

     @param s : the string
     @return Copied lowercase string <created on the stack>
    */
    VString lower(const VString& s);

    /**
     Creates a copy and converts all lowercase letters to uppercase.

     @param s : the string
     @return Copied uppercase string <created on the stack>
    */
    VString upper(const VString& s);

    /**
     Capitalizes the first letter in s.

     @param s : base string
     @return Capitalized string <created on the stack>
    */
    VString capitalize(const VString& s);

    /**
     Attempts to convert string s into a float

     @param s : base string
     @return True or False
    */ 
    bool is_float(const VString& s);

    /**
     Attempts to convert string s into an integer

     @param s : base string
     @return True or False
    */ 
    bool is_integer(const VString& s);

    /**
     Concatenates strings together into a copy.

     @param a : first string
     @param b : second string
     @return Copied string concatenated <created on the stack>
    */
    VString cat(const VString& a, const VString& b);

    /**
     Creates a string by stripping away all the values
     left of the index selected from s (including index).

     @param s : the string to strip from
     @param i : the index (from left) to start copying from
     @return The new string object <created on the stack>
     */
    VString lstrip(const VString& s, uint i);

    /**
     Creates a string by stripping away all the values
     right of the index selected from s (including index).

     @param s : the string to strip from
     @param i : the index (from right) to start copying from
     @return The new string object <created on the stack>
     */
    VString rstrip(const VString& s, uint i);

    /**
     Fills the left and right side of string with an additional character.

     Great for formatting output for strings.

     @param s : base string
     @param width : minimum width of resulting string, filled with characters from fillchar
     @param fillchar (optional) : filling character, default whitespace
     @return New string <created on the stack>
    */
    VString center(const VString& s, uint width, char fillchar = ' ');

    /**
     Fills the right side of string with an additional character.

     Great for formatting output for strings.

     @param s : base string
     @param width : minimum width of resulting string, filled with characters from fillchar
     @param fillchar (optional) : filling character, default whitespace
     @return New string <created on the stack>
    */
    VString ljust(const VString& s, uint width, char fillchar = ' ');

    /**
     Fills the left side of string with an additional character.

     Great for formatting output for strings.

     @param s : base string
     @param width : minimum width of resulting string, filled with characters from fillchar
     @param fillchar (optional) : filling character, default whitespace
     @return New string <created on the stack>
    */
    VString rjust(const VString& s, uint width, char fillchar = ' ');

    /**
     Returns whether the string begins with a substring start.

     @param s : the string to search
     @param start : the subVString to test.
     @return true or false
    */
    bool starts_with(const VString& s, const VString& start);

    /**
     Returns whether the string ends with a substring end.

     @param s : the string to search
     @param end : the substring to test.
     @return true or false
    */
    bool ends_with(const VString& s, const VString& end);

    /*
     Returns whether all characters are uppercase.

     @param s : string
     @return True or False
    */
    bool is_upper(const VString& s);

    /*
     Returns whether all characters are lowercase.

     @param s : string
     @return True or False
    */
    bool is_lower(const VString& s);

    /**
     Returns a substring between start and end, not including end index itself.

     If start >= end, it will swap start and end.
    
     @param s : the string to search
     @param start : begin index
     @param end : end index, up to
     @param step (default 1) : option to skip every (step) characters if desired
     @return Sub string <created on the stack>
    */  
    VString substring(const VString& s, uint start, uint end, uint step = 1);

    /**
     Repeat VString s n_repeat times.

     @param s : the string
     @param n_repeats : the number of times to repeat s
     @return repeated string <created on the stack>
    */
    VString repeat(const VString& s, uint n_repeats);

    /**
     Creates a copy and replaces all instances of value in s with replace_with.

     @param s : the base string to perform on
     @param value : the original value in s
     @param replace_with : to replace value with
     @return Replaced string <created on the stack>
    */
    VString replace(const VString& s, const VString& value, const VString& replace_with);
    VString replace(const VString& s, char value, char replace_with);

    /*
     Count the occurences of a pattern exrp in string s.

     @param s : the string
     @param expr : char or string expression to count
     @return int Count
    */
    uint count(const VString& s, const VString& expr);
    uint count(const VString& s, char expr);

    /**
     Finds all occurences of expr in the base VString s.

     @param s : base string to search
     @param expr : expression/char to search for
     @return Vector of indexes (as float) <created on the stack>
    */
    Vector find(const VString& s, const VString& expr);
    Vector find(const VString& s, char expr);

    /**
     Finds the first occurence of expr in base string s.

     @param s : base string to search
     @param expr : expression/char to search for
     @return first index
    */
    uint find_first(const VString& s, const VString& expr);
    uint find_first(const VString& s, char expr);

    /**
     Finds the last occurence of expr in base string s.

     @param s : base string to search
     @param expr : expression/char to search for
     @return last index
    */
    uint find_last(const VString& s, const VString& expr);
    uint find_last(const VString& s, char expr);

    /**
     Returns whether expr exists in s.

     @param s : base string
     @param expr : string to search
     @return True or False
    */
    bool contains(const VString& s, const VString& expr);

    /*
     Determine if pat matches to s.

     @param s : base string
     @param pat : string to test
     @param case (optional) : if true, case sensitive
     @return True or False
    */
    bool match(const VString& s, const VString& pat, bool case = true);

    /**
     Wraps the long string to be formatted in paragraphs with length less than a given
     width.

     @param s : base string
     @param width : maximum line-width
     @param expand_tabs (optional) : if true, tab chars will be expanded to spaces.
     @param replace_whitespace (optional) : if true, each whitespace char remaining
        after tab expansion will be replaced by a single space
     @param drop_whitespace (optional) : if true, whitespace after wrapping at the 
        beginning or end of a line is dropped
     @return Wrapped String <created on the stack>
    */
    VString wrap(const VString& s, uint width, bool expand_tabs = false,
     bool replace_whitespace = true, bool drop_whitespace = true);


/********************************************************************************************

    CLASS & MEMBER FUNCTIONS 

*///////////////////////////////////////////////////////////////////////////////////////////


class VString
{
    public:

    /*********************************************************************************************

        CONSTRUCTORS/DESTRUCTOR 

    *///////////////////////////////////////////////////////////////////////////////////////////

        /*
         Empty constructor, all values null.
        */
        VString();

        /*
         Allocated constructor, values empty.
        */
        VString(uint n);

        /*
         Copied from input, allocated.
        */
        VString(const char* input);

        // destructor
        ~VString();

        // inlines
        inline uint len() { return n; }

        inline char ix(int i) { return data[i]; }

        inline char& operator[](int i) { return data[i]; }

        inline char* raw_p() { return data; }

    /********************************************************************************************

        OTHER FUNCTIONS 

    *///////////////////////////////////////////////////////////////////////////////////////////

        /**
         Checks wheter this string is a float

         @return True or False
        */ 
        bool is_float();

        /**
         Checks wheter this string is an int

         @return True or False
        */ 
        bool is_integer();

        /**
         Converts all letters to lowercase.

         @return This object string <object not created>
        */
        VString& lower();

        /**
         Converts all letters to uppercase.

         @return This object string <object not created>
        */
        VString& upper();

        /**
         Capitalizes the first letter.

         @return This object string <object not created>
        */
        VString& capitalize();

        /**
         Returns whether the string begins with a substring start.

         @param start : the substring to test.
         @return true or false
        */
        bool starts_with(const VString& start);

        /**
         Returns whether the string ends with a substring end.

         @param end : the substring to test.
         @return true or false
        */
        bool ends_with(const VString& end);

        /**
         Returns whether expr exists in s.

         @param expr : string to search
         @return True or False
        */
        bool contains(const VString& expr);

        /**
         Replaces all instances of value in s with replace_with.

         @param value : the original value in s
         @param replace_with : to replace value with
         @return This object string <object not created>
        */
        VString& replace(const VString& value, const VString& replace_with);
        VString& replace(char value, char replace_with);


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
