/*
 * @file nvtx_utils.hpp
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

#ifndef CANARD_COMMON_NVTX_UTILS_HPP
#define CANARD_COMMON_NVTX_UTILS_HPP

#include "nvtx3/nvToolsExt.h"

// Scope some things into a namespace
namespace nvtx {

// Colour palette (ARGB): colour brewer qualitative 8-class Dark2
const uint32_t palette[] = { 0xff1b9e77, 0xffd95f02, 0xff7570b3, 0xffe7298a, 0xff66a61e, 0xffe6ab02, 0xffa6761d, 0xff666666};

const uint32_t colourCount = sizeof(palette)/sizeof(uint32_t);


// inline method to push an nvtx range
inline void push(const char * str){
    // Static variable to track the next colour to be used with auto rotation.
    static uint32_t nextColourIdx = 0;

    // Get the wrapped colour index
    uint32_t colourIdx = nextColourIdx % colourCount;
    // Build/populate the struct of nvtx event attributes
    nvtxEventAttributes_t eventAttrib = {0};
    // Generic values
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    // Selected colour and string
    eventAttrib.color = palette[colourIdx];
    eventAttrib.message.ascii = str;
    // Push the custom event.
    nvtxRangePushEx(&eventAttrib);
    // nvtxRangePushA(str);
    nextColourIdx++;
}

// inline method to pop an nvtx range
inline void pop(){
    nvtxRangePop();
}

    // Class to auto-pop nvtx range when scope exits.
    class NVTXRange {
     public: 
        // Constructor, which pushes a named range marker
        NVTXRange(const char * str){
            nvtx::push(str);
        }
        // Destructor which pops a marker off the nvtx stack (might not atually correspond to the same marker in practice.)
        ~NVTXRange(){
            nvtx::pop();
        }
    };
};
// Macro to construct the range object for use in a scope-based setting.
#define NVTX_RANGE(str) nvtx::NVTXRange uniq_name_using_macros(str)

// Macro to push an arbitrary nvtx marker
#define NVTX_PUSH(str) nvtx::push(str)

// Macro to pop an arbitrary nvtx marker
#define NVTX_POP() nvtx::pop()

#endif