/*
 * @file parameters.hpp
 *
 * @copyright Copyright (C) 2025 Enrico Degregori <enrico.degregori@gmail.com>
 *
 * @author Enrico Degregori <enrico.degregori@gmail.com>
 * 
 * MIT License
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions: 
 * 
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef CANARD_COMMON_PARAMETERS_HPP
#define CANARD_COMMON_PARAMETERS_HPP

// compact schemes
static constexpr double alpha   = 1.0 / 3.0;
static constexpr double alpha01 = 2.0;
static constexpr double alpha10 = 1.0 / 4.0;
static constexpr double aa      = 7.0 / 9.0;
static constexpr double bb      = 1.0 / 36.0;
static constexpr double ab00    = -5.0 / 2.0;
static constexpr double ab01    =  2.0;
static constexpr double ab02    =  1.0 / 2.0;
static constexpr double ab10    =  3.0 / 4.0;


static constexpr unsigned int lmd = 8;
static constexpr unsigned int NumberOfVariables = 5;
static constexpr unsigned int NumberOfSpatialDims = 3;

// boundary contidions
static constexpr unsigned int BC_NON_REFLECTIVE   = 10;
static constexpr unsigned int BC_WALL_INVISCID    = 20;
static constexpr unsigned int BC_WALL_VISCOUS     = 25;
static constexpr unsigned int BC_INTER_CURV       = 30;
static constexpr unsigned int BC_INTER_STRAIGHT   = 35;
static constexpr unsigned int BC_INTER_SUBDOMAINS = 40;
static constexpr unsigned int BC_PERIODIC         = 45;

// physics
static constexpr double gam          = 1.4;
static constexpr double gamm1        = gam - 1.0;
static constexpr double ham          = 1.0 / gam;
static constexpr double hamm1        = 1.0 / gamm1;
static constexpr double hamhamm1     = ham * hamm1;
static constexpr double prndtl       = 0.71;
static constexpr double gamm1prndtli = 1.0 / ( gamm1 * prndtl );

#endif
