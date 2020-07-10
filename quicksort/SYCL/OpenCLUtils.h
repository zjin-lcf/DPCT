/*
Copyright (c) 2014-2019, Intel Corporation
Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions 
are met:
* Redistributions of source code must retain the above copyright 
      notice, this list of conditions and the following disclaimer.
      * Redistributions in binary form must reproduce the above 
      copyright notice, this list of conditions and the following 
      disclaimer in the documentation and/or other materials provided 
      with the distribution.
      * Neither the name of Intel Corporation nor the names of its 
      contributors may be used to endorse or promote products 
      derived from this software without specific prior written 
      permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef OPENCLUTILS_DOT_H
#define OPENCLUTILS_DOT_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string>
#include <CL/cl.h>

// Util for error checking:
//#undef __OCL_NO_ERROR_CHECKING
#define __OCL_NO_ERROR_CHECKING

#ifdef __OCL_NO_ERROR_CHECKING
#define CheckCLError(__errNum__, __failMsg__, __passMsg__)	\
	assert (CL_SUCCESS == __errNum__);
#else
#define CheckCLError(__errNum__, __failMsg__, __passMsg__)	\
if (CL_SUCCESS != __errNum__)								\
{															\
		char __msgBuf__[256];								\
		sprintf (__msgBuf__, "CL Error num %d: %s at line %d, file %s in function %s().\n", __errNum__, __failMsg__, __LINE__, __FILE__, __FUNCTION__);	\
		printf (__msgBuf__);								\
		getchar();											\
		printf("Failed on OpenCLError\n");					\
		assert (CL_SUCCESS != __errNum__);					\
		exit(0);											\
} else if (__passMsg__)										\
{															\
	printf("CL Success: %s\n", __passMsg__);				\
}				
#endif

// Util for OpenCL build log:
void BuildFailLog(cl_program program,
                  cl_device_id device_id )
{
    size_t paramValueSizeRet = 0;
    cl_int status = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &paramValueSizeRet);
    printf("clGetProgramBuildInfo returned %d\n", status);
	if (paramValueSizeRet == 0) {
		printf("\nOpenCL C Program Build Log is empty!\n");
		return;
	} else {
		printf("\nOpenCL C Build log is %zu characters long.\n", paramValueSizeRet);
	}
    char* buildLogMsgBuf = (char *)malloc(sizeof(char) * paramValueSizeRet + 1);
	if( buildLogMsgBuf )
	{
		status = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, paramValueSizeRet, buildLogMsgBuf, &paramValueSizeRet);
		printf("clGetProgramBuildInfo returned %d\n", status);
		buildLogMsgBuf[paramValueSizeRet] = '\0';	//mark end of message string

		printf("\nOpenCL C Program Build Log:\n");
		puts(buildLogMsgBuf);
		fflush(stdout);

		free(buildLogMsgBuf);
	}
}

static bool isNvidiaGpu = false;

#endif
