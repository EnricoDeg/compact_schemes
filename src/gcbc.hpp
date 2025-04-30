/*
 * @file gcbc.hpp
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

#ifndef CANARD_GCBC_HPP
#define CANARD_GCBC_HPP

template<typename Type>
void gcbc_setup(Type *drva_buffer[3], Type *cm_buffer[3],
                unsigned int nbc[2][3])
{

    Type * drva, * cm;
    host::static_for<0, 3, 1>{}([&](auto nn)
    {

        drva = drva_buffer[nn];
        cm   = cm_buffer[nn];

        host::static_for<0, 2, 1>{}([&](auto ip)
        {

            const unsigned int np = nbc[ip][nn];
            if( ( np - BC_NON_REFLECTIVE ) *
                ( np - BC_WALL_INVISCID  ) *
                ( np - BC_WALL_VISCOUS   ) *
                ( np - BC_INTER_CURV     ) == 0 )
            {

                unsigned int blockSize = 
                unsigned int blockPerGrid =
                dim3 threadsPerBlock(blockSize);
                dim3 blocksPerGrid(blockPerGrid);
                TIME(blocksPerGrid, threadsPerBlock, 0, 0, false,
                    CANARD_KERNEL_NAME(deriv_kernel<Axis>),
                    infield, outfield, recv, pbci, drva, h_1, nstart, nend, dcomp_info, variable_id);
            }
        });
    });
}

template<typename Type>
void wall_condition_update(Type *drva_buffer[3], Type *cm_buffer[3],
                           unsigned int nbc[2][3])
{

    Type * drva, * cm;
    host::static_for<0, 3, 1>{}([&](auto nn)
    {

        drva = drva_buffer[nn];
        cm   = cm_buffer[nn];

        host::static_for<0, 2, 1>{}([&](auto ip)
        {

            const unsigned int np = nbc[ip][nn];
            if(np == BC_WALL_INVISCID)
            {

            }
            else if(np == BC_WALL_VISCOUS)
            {

            }
        });
    });
}

#endif
