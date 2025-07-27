/*
 * @file utils.hpp
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

#ifndef CANARD_COMMON_UTILS_HPP
#define CANARD_COMMON_UTILS_HPP

inline int div_ceil(int numerator, int denominator)
{
    return (numerator % denominator != 0) ?
           (numerator / denominator+ 1  ) :
           (numerator / denominator     ) ;
}

template<typename Type>
int maxloc(Type *p, int size)
{
    int imax;
    Type dmax;
    imax = 0;
    dmax = p[0];
    for(int i=1; i<size; i++) {
        if (p[i] > dmax) {
            dmax = p[i];
            imax = i;
        }
    }
    return imax;
}

template<typename Type>
void mtrxi(Type *mtrx, Type *imtrx, int size)
{
    int size1d = std::sqrt(size);
    int ipvt[size1d];
    Type rx[size];
    Type arx[size];
    Type temp[size1d];
    Type dum;

    // initialize work array
    for (int i=0; i<size; i++) {
        rx[i] = mtrx[i];
    }

    // initialize index work array
    for (int i=0; i<size1d; i++)
        ipvt[i] = i;

    for (int i=0; i<size1d; i++) {
        // absolute value of work array
        for (int k=0; k<size; k++) {
            arx[k] = std::abs(rx[k]);
        }
        int loc = i*size1d+i;
        int imax = maxloc(arx+loc, size1d-i-1);
        int mk = i + imax;

        // swap elements of ipvt and rx
        if (mk!=i) {
            dum = ipvt[mk];
            ipvt[mk] = ipvt[i];
            ipvt[i] = dum;
            for (int l=0; l<size1d; l++) {
                int mmk = mk + l * size1d;
                int ii  = i  + l * size1d;
                dum = rx[mmk];
                rx[mmk] = rx[ii];
                rx[ii] = dum;
            }
        }

        Type ra0 = 1.0 / rx[i + i*size1d];
        // fill temporary array
        for (int k=0; k<size1d; k++)
            temp[k] = rx[k+i*size1d];

        for (int k=0; k<size1d; k++) {
            Type ra1 = ra0 * rx[i+k*size1d];
            for (int l=0; l<size1d; l++)
                rx[l+k*size1d] = rx[l+k*size1d] - ra1 * temp[l];
            rx[i+k*size1d] = ra1;
        }

        for (int k=0; k<size1d; k++)
            rx[k+i*size1d] = -ra0 * temp[k];
        rx[i+i*size1d] = ra0;
    }

    // inverse matrix
    for (int j=0; j<size1d; j++)
        for (int i=0; i<size1d; i++)
            imtrx[i+ipvt[j]*size1d] = rx[i+j*size1d];
}

template<typename Type>
void matmul_square(Type *mat1, Type *mat2, Type *rslt, int size)
{
    int size1d = std::sqrt(size);
    for (int j = 0; j < size1d; j++) {
        for (int i = 0; i < size1d; i++) {
            rslt[i+j*size1d] = 0.0;
            for (int k = 0; k < size1d; k++)
                rslt[i+j*size1d] += mat1[k+j*size1d] * mat2[i+k*size1d];
        }
    }
}

template<typename Type>
void matvecmul(Type *mat1, Type *vec2, Type *rslt, int size1d)
{
    for (int i = 0; i < size1d; i++) {
        rslt[i] = 0.0;
        for (int k = 0; k < size1d; k++)
            rslt[i] += mat1[k+i*size1d] * vec2[k];
    }
}

#endif
