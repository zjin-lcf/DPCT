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

// QuicksortMain.cpp : Defines the entry point for the console application.
//
#include <CL/sycl.hpp>

#include <stdio.h>
#ifdef _MSC_VER
// Windows
#include <windows.h>
#include <tchar.h>
#endif
#include <assert.h>
#include <string.h>
#ifndef _MSC_VER
// Linux
#include <time.h>
#include <unistd.h>
#endif
#include "OpenCLUtils.h"
#include <math.h>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <map>

#include "tbb/parallel_sort.h"
using namespace tbb;
using namespace cl::sycl;

/* Classes can inherit from the device_selector class to allow users
 * to dictate the criteria for choosing a device from those that might be
 * present on a system. This example looks for a device with SPIR support
 * and prefers GPUs over CPUs. */
class intel_gpu_selector : public device_selector {
 public:
  intel_gpu_selector() : device_selector() {}

  /* The selection is performed via the () operator in the base
   * selector class.This method will be called once per device in each
   * platform. Note that all platforms are evaluated whenever there is
   * a device selection. */
  int operator()(const device& device) const override {
    /* We only give a valid score to devices that support SPIR. */
    //if (device.has_extension(cl::sycl::string_class("cl_khr_spir"))) {
    if (device.get_info<info::device::name>().find("Intel") != std::string::npos) {
      if (device.get_info<info::device::device_type>() ==
          info::device_type::gpu) {
        return 50;
      }
    }
    /* Devices with a negative score will never be chosen. */
    return -1;
  }
};

// Types:
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#define READ_ALIGNMENT  4096 // Intel recommended alignment
#define WRITE_ALIGNMENT 4096 // Intel recommended alignment

/// return a timestamp with sub-second precision 
/** QueryPerformanceCounter and clock_gettime have an undefined starting point (null/zero)     
 *  and can wrap around, i.e. be nulled again. **/ 
double seconds() { 
#ifdef _MSC_VER   
  static LARGE_INTEGER frequency;   
  if (frequency.QuadPart == 0)     ::QueryPerformanceFrequency(&frequency);   
  LARGE_INTEGER now;   
  ::QueryPerformanceCounter(&now);   
  return now.QuadPart / double(frequency.QuadPart); 
#else   
  struct timespec now;   
  clock_gettime(CLOCK_MONOTONIC, &now);   
  return now.tv_sec + now.tv_nsec / 1000000000.0; 
#endif 
}
 
typedef struct
{	
	// CL platform handles:
	cl_device_id		deviceID;
	cl_context			contextHdl;
	cl_program			programHdl;
	cl_command_queue	cmdQHdl;
	cl::sycl::queue     queue;
} OCLResources;

// Globals:
/* Create variable to store OpenCL errors. */
::cl_int		ciErrNum = 0;

void Cleanup(OCLResources* pOCL, int iExitCode, bool bExit, const char* optionalErrorMessage)
{
	if (optionalErrorMessage) 
		printf ("%s\n", optionalErrorMessage);

	memset(pOCL, 0, sizeof (OCLResources));

	if (pOCL->programHdl)		{ clReleaseProgram(pOCL->programHdl);		pOCL->programHdl=NULL;	}
	if (pOCL->cmdQHdl)			{ clReleaseCommandQueue(pOCL->cmdQHdl);		pOCL->cmdQHdl=NULL;		}
	if (pOCL->contextHdl)		{ clReleaseContext(pOCL->contextHdl);		pOCL->contextHdl= NULL;	}

	if (bExit)
		exit (iExitCode);
}

void parseArgs(OCLResources* pOCL, int argc, char** argv, unsigned int* test_iterations, char* pDeviceStr, char* pVendorStr, unsigned int* widthReSz, unsigned int* heightReSz, bool* pbShowCL)
{	
  std::string pDeviceWStr;
  std::string pVendorWStr;
  const char sUsageString[512] = "Usage: Quicksort [num test iterations] [cpu|gpu] [intel|amd|nvidia] [SurfWidth(^2 only)] [SurfHeight(^2 only)] [show_CL | no_show_CL]";
  
  if (argc != 7)
  {
  	Cleanup (pOCL, -1, true, sUsageString);
  }
  else
  {
  	*test_iterations	= atoi (argv[1]);
  	pDeviceWStr			= std::string(argv[2]);			// "cpu" or "gpu"	
  	pVendorWStr			= std::string(argv[3]);			// "intel" or "amd" or "nvidia"
  	*widthReSz	= atoi (argv[4]);
  	*heightReSz	= atoi (argv[5]);
  	if (argv[6][0]=='s')
  		*pbShowCL = true;
  	else
  		*pbShowCL = false;
  }
  sprintf (pDeviceStr, "%s", pDeviceWStr.c_str());
  sprintf (pVendorStr, "%s", pVendorWStr.c_str());

  auto get_queue = [&pDeviceStr, &pVendorStr]() {  
    device_selector* pds = 0;
    if (pVendorStr == std::string("intel")) {
      if (pDeviceStr == std::string("gpu")) {
          static intel_gpu_selector selector;
		  pds = &selector;
	  } else if (pDeviceStr == std::string("cpu")) {
		  static cpu_selector selector;
		  pds = &selector;
	  }
	} else {
		static default_selector selector;
		pds = &selector;
	}

    device d(*pds);

    queue queue(*pds, [](cl::sycl::exception_list l) {
      for (auto ep : l) {
        try {
          std::rethrow_exception(ep);
        } catch (cl::sycl::exception& e) {
          std::cout << e.what() << std::endl;
        }
      }
    });
    return queue;
  };
  
  auto queue = get_queue();
  pOCL->queue = queue;
  /* Retrieve the underlying cl_context of the context associated with the
   * queue. */
  pOCL->contextHdl = queue.get_context().get();

  /* Retrieve the underlying cl_device_id of the device asscociated with the
   * queue. */
  pOCL->deviceID = queue.get_device().get();

  /* Retrieve the underlying cl_command_queue of the queue. */
  pOCL->cmdQHdl = queue.get();
}

//#define GET_DETAILED_PERFORMANCE 1
#define RUN_CPU_SORTS
#define HOST 1
#include "Quicksort.h"

template <class T>
T* partition(T* left, T* right, T pivot) {
    // move pivot to the end
    T temp = *right;
    *right = pivot;
    *left = temp;

    T* store = left;

    for(T* p = left; p != right; p++) {
        if (*p < pivot) {
            temp = *store;
            *store = *p;
            *p = temp;
            store++;
        }
    }

    temp = *store;
    *store = pivot;
    *right = temp;

    return store;
}

template <class T>
void quicksort(T* data, int left, int right)
{
    T* store = partition(data + left, data + right, data[left]);
    int nright = store-data;
    int nleft = nright+1;

    if (left < nright) {
      if (nright - left > 32) {
        quicksort(data, left, nright);
      } else
        std::sort(data + left, data + nright + 1);
    }

    if (nleft < right) {
      if (right - nleft > 32)  {
		    quicksort(data, nleft, right); 
      } else {
        std::sort(data + nleft, data + right + 1);
      }
	}
}

template <class T>
void plus_prescan(T *a, T *b) {
    T av = *a;
	T bv = *b;
    *a = bv;
    *b = bv + av;
}

// record to push start of the sequence, end of the sequence and direction of sorting on internal stack
struct workstack_record {
	uint start;
	uint end;
	uint direction;
};

//---------------------------------------------------------------------------------------
// Class implements the last stage of GPU-Quicksort, when all the subsequences are small
// enough to be processed in local memory. It uses similar algorithm to gqsort_kernel to 
// move items around the pivot and then switches to bitonic sort for sequences in
// the range [1, SORT_THRESHOLD] 
//
// d - input array
// dn - scratch array of the same size as the input array
// seqs - array of records to be sorted in a local memory, one sequence per work group.
//---------------------------------------------------------------------------------------
template <class T>
class lqsort_kernel_class {
	public:
    static cl::sycl::kernel* kernel;

	using discard_read_write_accessor = 
	  accessor<T, 1, access::mode::discard_read_write, access::target::global_buffer>;
	using seqs_read_accessor = accessor<work_record<T>, 1, access::mode::read, access::target::global_buffer>;
	
    using local_uint_read_write_accessor = accessor<uint, 1, access::mode::read_write, access::target::local>;
    using local_int_read_write_accessor = accessor<int, 1, access::mode::read_write, access::target::local>;
    using local_T_read_write_accessor = accessor<T, 1, access::mode::read_write, access::target::local>;
    using local_workstack_record_read_write_accessor = accessor<workstack_record, 1, access::mode::read_write, access::target::local>;

    lqsort_kernel_class(discard_read_write_accessor db,
	                    discard_read_write_accessor dnb, 
						seqs_read_accessor seqsb,
						local_workstack_record_read_write_accessor workstackb,
						local_int_read_write_accessor workstack_pointerb,
						local_T_read_write_accessor mysb, 
						local_T_read_write_accessor mysnb, 
						local_T_read_write_accessor tempb,
						local_uint_read_write_accessor ltsumb,
						local_uint_read_write_accessor gtsumb,
						local_uint_read_write_accessor ltb,
						local_uint_read_write_accessor gtb) :
						d(db), dn(dnb), seqs(seqsb) ,
						workstack(workstackb),
						workstack_pointer(workstack_pointerb),
						mys(mysb), mysn(mysnb), temp(tempb),
						ltsum(ltsumb), gtsum(gtsumb),
						lt(ltb), gt(gtb) 
						 {}

    /// bitonic_sort: sort 2*LOCAL_THREADCOUNT elements
    void bitonic_sort(local_ptr<T> sh_data, const uint localid, nd_item<1> id)
    {
    	for (uint ulevel = 1; ulevel < LQSORT_LOCAL_WORKGROUP_SIZE; ulevel <<= 1) {
            for (uint j = ulevel; j > 0; j >>= 1) {
                uint pos = 2*localid - (localid & (j - 1));
    
    			uint direction = localid & ulevel;
    			uint av = sh_data[pos], bv = sh_data[pos + j];
    			const uint sortThem = av > bv;
    			const uint greater = cl::sycl::select(bv, av, sortThem);
    			const uint lesser  = cl::sycl::select(av, bv, sortThem);
    
    			sh_data[pos]     = cl::sycl::select(lesser, greater, direction);
    			sh_data[pos + j] = cl::sycl::select(greater, lesser, direction);
				id.barrier(access::fence_space::local_space);
            }
        }
    
    	for (uint j = LQSORT_LOCAL_WORKGROUP_SIZE; j > 0; j >>= 1) {
            uint pos = 2*localid - (localid & (j - 1));
    
    		uint av = sh_data[pos], bv = sh_data[pos + j];
    		const uint sortThem = av > bv;
    		sh_data[pos]      = cl::sycl::select(av, bv, sortThem);
    		sh_data[pos + j]  = cl::sycl::select(bv, av, sortThem);
    
		    id.barrier(access::fence_space::local_space);
        }
    }

    void sort_threshold(local_ptr<T> data_in, 
	                    global_ptr<T> data_out,
    					uint start, 
    					uint end, local_ptr<T> temp_, uint localid,
						nd_item<1> id) 
    {
    	uint tsum = end - start;
    	if (tsum == SORT_THRESHOLD) {
    		bitonic_sort(data_in+start, localid, id);
    		for (uint i = localid; i < SORT_THRESHOLD; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
    			data_out[start + i] = data_in[start + i];
    		}
    	} else if (tsum > 1) {
    		for (uint i = localid; i < SORT_THRESHOLD; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
    			if (i < tsum) {
    				temp_[i] = data_in[start + i];
    			} else {
    				temp_[i] = std::numeric_limits<T>::max();
    			}
    		}
		    id.barrier(access::fence_space::local_space);
    		bitonic_sort(temp_, localid, id);
    
    		for (uint i = localid; i < tsum; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
    			data_out[start + i] = temp_[i];
    		}
    	} else if (tsum == 1 && localid == 0) {
    		data_out[start] = data_in[start];
    	} 
    }

#define PUSH(START, END) 			if (localid == 0) { \
										workstack_pointer[0] ++; \
                                        workstack_record wr{ (START), (END), direction ^ 1 }; \
										workstack[workstack_pointer[0]] = wr; \
									} \
									id.barrier(access::fence_space::local_space);


    void operator()(nd_item<1> id) {
		const size_t blockid = id.get_group(0);
        const size_t localid = id.get_local_id(0);

        local_ptr<T> s, sn;
	    uint i, ltp, gtp;
		T tmp;
	
    	work_record<T> block = seqs[blockid];
    	const uint d_offset = block.start;
    	uint start = 0; 
    	uint end   = block.end - d_offset;
    
    	uint direction = 1; // which direction to sort
    	// initialize workstack and workstack_pointer: push the initial sequence on the stack
    	if (localid == 0) {
    		workstack_pointer[0] = 0; // beginning of the stack
    		workstack_record wr{ start, end, direction };
    		workstack[0] = wr;
    	}
    	// copy block of data to be sorted by one workgroup into local memory
    	// note that indeces of local data go from 0 to end-start-1
    	if (block.direction == 1) {
    		for (i = localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
    			mys[i] = d[i+d_offset];
    		}
    	} else {
    		for (i = localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
    			mys[i] = dn[i+d_offset];
    		}
    	}
		id.barrier(access::fence_space::local_space);

        while (workstack_pointer[0] >= 0) { 
    		// pop up the stack
    		workstack_record wr = workstack[workstack_pointer[0]];
    		start = wr.start;
    		end = wr.end;
    		direction = wr.direction;
    		if (localid == 0) {
    			workstack_pointer[0] --;
    
    			ltsum[0] = gtsum[0] = 0;	
    		}
    		if (direction == 1) {
    			s = mys.get_pointer();
    			sn = mysn.get_pointer();
    		} else {
    			s = mysn.get_pointer();
    			sn = mys.get_pointer();
    		}
    		// Set thread local counters to zero
    		lt[localid] = gt[localid] = 0;
    		ltp = gtp = 0;
		    id.barrier(access::fence_space::local_space);
    
    		// Pick a pivot
    		T pivot = s[start];
    		if (start < end) {
    			pivot = median_select(pivot, s[(start+end) >> 1], s[end-1]);
    		}
    		// Align work item accesses for coalesced reads.
    		// Go through data...
    		for(i = start + localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
    			tmp = s[i];
    			// counting elements that are smaller ...
    			if (tmp < pivot)
    				ltp++;
    			// or larger compared to the pivot.
    			if (tmp > pivot) 
    				gtp++;
    		}
    		lt[localid] = ltp;
    		gt[localid] = gtp;
		    id.barrier(access::fence_space::local_space);
    		
    		// calculate cumulative sums
    		uint n;
    		for(i = 1; i < LQSORT_LOCAL_WORKGROUP_SIZE; i <<= 1) {
    			n = 2*i - 1;
    			if ((localid & n) == n) {
    				lt[localid] += lt[localid-i];
    				gt[localid] += gt[localid-i];
    			}
		        id.barrier(access::fence_space::local_space);
    		}
    
    		if ((localid & n) == n) {
    			lt[LQSORT_LOCAL_WORKGROUP_SIZE] = ltsum[0] = lt[localid];
    			gt[LQSORT_LOCAL_WORKGROUP_SIZE] = gtsum[0] = gt[localid];
    			lt[localid] = 0;
    			gt[localid] = 0;
    		}
    		
    		for(i = LQSORT_LOCAL_WORKGROUP_SIZE/2; i >= 1; i >>= 1) {
    			n = 2*i - 1;
    			if ((localid & n) == n) {
    				plus_prescan(&lt[localid - i], &lt[localid]);
    				plus_prescan(&gt[localid - i], &gt[localid]);
    			}
		        id.barrier(access::fence_space::local_space);
    		}
    
    		// Allocate locations for work items
    		uint lfrom = start + lt[localid];
    		uint gfrom = end - gt[localid+1];
    
    		// go thru data again writing elements to their correct position
    		for (i = start + localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
    			tmp = s[i];
    			// increment counts
    			if (tmp < pivot) 
    				sn[lfrom++] = tmp;
    			
    			if (tmp > pivot) 
    				sn[gfrom++] = tmp;
    		}
		    id.barrier(access::fence_space::local_space);
    
    		// Store the pivot value between the new sequences
    		for (i = start + ltsum[0] + localid;i < end - gtsum[0]; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
    			d[i+d_offset] = pivot;
    		}
		    id.barrier(access::fence_space::global_and_local);
    
    		// if the sequence is shorter than SORT_THRESHOLD
    		// sort it using an alternative sort and place result in d
    		if (ltsum[0] <= SORT_THRESHOLD) {
    			sort_threshold(sn, d.get_pointer() + d_offset, start, start + ltsum[0], temp.get_pointer(), localid, id);
    		} else {
    			PUSH(start, start + ltsum[0])
    		}
    		
    		if (gtsum[0] <= SORT_THRESHOLD) {
    			sort_threshold(sn, d.get_pointer() + d_offset, end - gtsum[0], end, temp.get_pointer(), localid, id);
    		} else {
    			PUSH(end - gtsum[0], end)
    		}
    	}
	}

	private:
    discard_read_write_accessor d, dn;
	seqs_read_accessor seqs;

    local_workstack_record_read_write_accessor workstack;
	local_int_read_write_accessor workstack_pointer;
	
	local_T_read_write_accessor mys, mysn, temp;

	local_uint_read_write_accessor ltsum, gtsum;
	local_uint_read_write_accessor lt, gt;
};

//----------------------------------------------------------------------------
// Class implements gqsort_kernel
//----------------------------------------------------------------------------
template <class T>
class gqsort_kernel_class {
	public:
    static cl::sycl::kernel* kernel;

	using blocks_read_accessor = accessor<block_record<T>, 1, access::mode::read, access::target::global_buffer>;
	using parents_read_write_accessor = accessor<parent_record, 1, access::mode::read_write, access::target::global_buffer>;
	using news_write_accessor = accessor<work_record<T>, 1, access::mode::write, access::target::global_buffer>;
	using discard_read_write_accessor = 
	  accessor<T, 1, access::mode::discard_read_write, access::target::global_buffer>;
    using local_read_write_accessor = accessor<uint, 1, access::mode::read_write, access::target::local>;

    gqsort_kernel_class(discard_read_write_accessor db,
	                    discard_read_write_accessor dnb,
	                    blocks_read_accessor blocksb,
	                    parents_read_write_accessor parentsb,
	                    news_write_accessor newsb, 
						local_read_write_accessor ltb, 
						local_read_write_accessor gtb, 
						local_read_write_accessor ltsumb, 
						local_read_write_accessor gtsumb, 
						local_read_write_accessor lbegb, 
						local_read_write_accessor gbegb) :
						d(db), dn(dnb), blocks(blocksb), 
						parents(parentsb), news(newsb),
						lt(ltb), gt(gtb), ltsum(ltsumb), gtsum(gtsumb), lbeg(lbegb), gbeg(gbegb) {}

    void operator()(nd_item<1> id) {
        const size_t blockid = id.get_group(0);
        const size_t localid = id.get_local_id(0);

        uint i, lfrom, gfrom, ltp = 0, gtp = 0;
		T lpivot, gpivot, tmp;

	    // Get the sequence block assigned to this work group
	    block_record<T> block = blocks[blockid];
	    uint start = block.start, end = block.end, direction = block.direction;
		T pivot = block.pivot;

        auto& pparent = parents[block.parent];

	    T *s, *sn;

	    // GPU-Quicksort cannot sort in place, as the regular quicksort algorithm can.
	    // It therefore needs two arrays to sort things out. We start sorting in the 
	    // direction of d -> dn and then change direction after each run of gqsort_kernel.
	    // Which direction we are sorting: d -> dn or dn -> d?
	    if (direction == 1) {
	    	s = &d[0];
	    	sn = &dn[0];
	    } else {
	    	s = &dn[0];
	    	sn = &d[0];
	    }
	    // Set thread local counters to zero
	    lt[localid] = gt[localid] = 0;
	    id.barrier(access::fence_space::local_space);

	    // Align thread accesses for coalesced reads.
	    // Go through data...
	    for(i = start + localid; i < end; i += GQSORT_LOCAL_WORKGROUP_SIZE) {
	    	tmp = s[i];
	    	// counting elements that are smaller ...
	    	if (tmp < pivot)
	    		ltp++;
	    	// or larger compared to the pivot.
	    	if (tmp > pivot) 
	    		gtp++;
	    }
	    lt[localid] = ltp;
	    gt[localid] = gtp;
	    id.barrier(access::fence_space::local_space);

    	// calculate cumulative sums
    	uint n;
    	for(i = 1; i < GQSORT_LOCAL_WORKGROUP_SIZE; i <<= 1) {
    		n = 2*i - 1;
    		if ((localid & n) == n) {
    			lt[localid] += lt[localid-i];
    			gt[localid] += gt[localid-i];
    		}
	        id.barrier(access::fence_space::local_space);
    	}
    
    	if ((localid & n) == n) {
    		lt[GQSORT_LOCAL_WORKGROUP_SIZE] = ltsum[0] = lt[localid];
    		gt[GQSORT_LOCAL_WORKGROUP_SIZE] = gtsum[0] = gt[localid];
    		lt[localid] = 0;
    		gt[localid] = 0;
    	}
    		
    	for(i = GQSORT_LOCAL_WORKGROUP_SIZE/2; i >= 1; i >>= 1) {
    		n = 2*i - 1;
    		if ((localid & n) == n) {
    			plus_prescan(&lt[localid - i], &lt[localid]);
    			plus_prescan(&gt[localid - i], &gt[localid]);
    		}
	        id.barrier(access::fence_space::local_space);
    	}		

	    // Allocate memory in the sequence this block is a part of
	    if (localid == 0) {
			cl::sycl::atomic<uint> psstart_a(multi_ptr<uint, access::address_space::global_space>(&pparent.sstart));
			cl::sycl::atomic<uint> psend_a(multi_ptr<uint, access::address_space::global_space>(&pparent.send));
	    	// Atomic increment allocates memory to write to.
	    	lbeg[0] = cl::sycl::atomic_fetch_add(psstart_a, ltsum[0]);
	    	// Atomic is necessary since multiple blocks access this
	    	gbeg[0] = cl::sycl::atomic_fetch_sub(psend_a, gtsum[0]) - gtsum[0];
	    }
        id.barrier(access::fence_space::global_and_local);

		// Allocate locations for work items
		lfrom = lbeg[0] + lt[localid];
		gfrom = gbeg[0] + gt[localid];

       	// go thru data again writing elements to their correct position
       	for(i = start + localid; i < end; i += GQSORT_LOCAL_WORKGROUP_SIZE) {
       		tmp = s[i];
       		// increment counts
       		if (tmp < pivot) 
       			sn[lfrom++] = tmp;
       
       		if (tmp > pivot) 
       			sn[gfrom++] = tmp;
       	}
        id.barrier(access::fence_space::global_and_local);

    	if (localid == 0) {
			cl::sycl::atomic<uint> pblockcount_a(multi_ptr<uint, access::address_space::global_space>(&pparent.blockcount));
    		if (cl::sycl::atomic_fetch_sub(pblockcount_a, (uint)1) == 0) {
    			uint sstart = pparent.sstart;
    			uint send = pparent.send;
    			uint oldstart = pparent.oldstart;
    			uint oldend = pparent.oldend;
    
    			// Store the pivot value between the new sequences
    			for(i = sstart; i < send; i ++) {
    				d[i] = pivot;
    			}
    
    			lpivot = sn[oldstart];
    			gpivot = sn[oldend-1];
    			if (oldstart < sstart) {
    				lpivot = median_select(lpivot,sn[(oldstart+sstart) >> 1], sn[sstart-1]);
    			} 
    			if (send < oldend) {
    				gpivot = median_select(sn[send],sn[(oldend+send) >> 1], gpivot);
    			}
    			
    			// change the direction of the sort.
    			direction ^= 1;
    
    			news[2*blockid] = work_record<T>{oldstart, sstart, lpivot, direction};
    			news[2*blockid + 1] = work_record<T>{send, oldend, gpivot, direction};
    		}
    	}
	}
	private:
      discard_read_write_accessor d, dn;
	  blocks_read_accessor blocks;
	  parents_read_write_accessor parents;
	  news_write_accessor news;
	  local_read_write_accessor lt, gt, ltsum, gtsum, lbeg, gbeg;
};

// Note that for every type that we intend to sort we need to allocate this
template <>
cl::sycl::kernel* gqsort_kernel_class<uint>::kernel = (cl::sycl::kernel*)malloc(sizeof(cl::sycl::kernel));
template <>
cl::sycl::kernel* lqsort_kernel_class<uint>::kernel = (cl::sycl::kernel*)malloc(sizeof(cl::sycl::kernel));
template <>
cl::sycl::kernel* gqsort_kernel_class<float>::kernel = (cl::sycl::kernel*)malloc(sizeof(cl::sycl::kernel));
template <>
cl::sycl::kernel* lqsort_kernel_class<float>::kernel = (cl::sycl::kernel*)malloc(sizeof(cl::sycl::kernel));
template <>
cl::sycl::kernel* gqsort_kernel_class<double>::kernel = (cl::sycl::kernel*)malloc(sizeof(cl::sycl::kernel));
template <>
cl::sycl::kernel* lqsort_kernel_class<double>::kernel = (cl::sycl::kernel*)malloc(sizeof(cl::sycl::kernel));

template <class T>
void gqsort(OCLResources *pOCL, 
            buffer<T>& d_buffer, 
			buffer<T>& dn_buffer, 
			std::vector<block_record<T>>& blocks, 
			std::vector<parent_record>& parents, 
			std::vector<work_record<T>>& news, 
			bool reset) {
#ifdef GET_DETAILED_PERFORMANCE
	static double absoluteTotal = 0.0;
	static uint count = 0;

	if (reset) {
		absoluteTotal = 0.0;
		count = 0;
	}

	double beginClock, endClock;
  beginClock = seconds();
#endif

	news.resize(blocks.size()*2);
	// Create buffer objects for memory.
	buffer<block_record<T>>  blocks_buffer(blocks.data(), blocks.size(), {property::buffer::use_host_ptr()});
	buffer<parent_record>  parents_buffer(parents.data(), parents.size(), {property::buffer::use_host_ptr()});
	buffer<work_record<T>>  news_buffer(news.data(), news.size(), {property::buffer::use_host_ptr()});

    pOCL->queue.submit([&](handler& cgh) {
		using local_read_write_accessor = accessor<uint, 1, access::mode::read_write, access::target::local>;
	  auto db = d_buffer.template get_access<access::mode::discard_read_write>(cgh);
	  auto dnb = dn_buffer.template get_access<access::mode::discard_read_write>(cgh);
	  auto blocksb = blocks_buffer.template get_access<access::mode::read>(cgh);
	  auto parentsb = parents_buffer.get_access<access::mode::read_write>(cgh);
	  auto newsb = news_buffer. template get_access<access::mode::write>(cgh);

	  local_read_write_accessor
        lt(range<>(GQSORT_LOCAL_WORKGROUP_SIZE+1), cgh), gt(range<>(GQSORT_LOCAL_WORKGROUP_SIZE+1), cgh),
	    ltsum(range<>(1), cgh), gtsum(range<>(1), cgh), lbeg(range<>(1), cgh), gbeg(range<>(1), cgh);
     
      auto gqsort = gqsort_kernel_class<T>(db, dnb, blocksb, parentsb, newsb, lt, gt, ltsum, gtsum, lbeg, gbeg);

      cgh.parallel_for(
        *gqsort.kernel,
		nd_range<>(GQSORT_LOCAL_WORKGROUP_SIZE * blocks.size(), 
	               GQSORT_LOCAL_WORKGROUP_SIZE), 
	    gqsort);
    });
    pOCL->queue.wait_and_throw();

#ifdef GET_DETAILED_PERFORMANCE
    endClock = seconds();
	double totalTime = endClock - beginClock;
	absoluteTotal += totalTime;
	std::cout << ++count << ": gqsort time " << absoluteTotal * 1000 << " ms" << std::endl;
#endif
}

template <class T>
void lqsort(OCLResources *pOCL, 
            std::vector<work_record<T>>& done, 
			buffer<T>& d_buffer, 
			buffer<T>& dn_buffer) {
#ifdef GET_DETAILED_PERFORMANCE
    double beginClock, endClock;
    beginClock = seconds();
#endif

	buffer<work_record<T>>  done_buffer(done.data(), done.size(), {property::buffer::use_host_ptr()});

    pOCL->queue.submit([&](handler& cgh) {
		using local_workstack_record_read_write_accessor = accessor<workstack_record, 1, access::mode::read_write, access::target::local>;
		using local_T_read_write_accessor = accessor<T, 1, access::mode::read_write, access::target::local>;
		using local_uint_read_write_accessor = accessor<uint, 1, access::mode::read_write, access::target::local>;
		using local_int_read_write_accessor = accessor<int, 1, access::mode::read_write, access::target::local>;

      auto db = d_buffer.template get_access<access::mode::discard_read_write>(cgh);
	  auto dnb = dn_buffer.template get_access<access::mode::discard_read_write>(cgh);
      auto doneb = done_buffer.template get_access<access::mode::read>(cgh);

	  local_workstack_record_read_write_accessor workstack(range<>(QUICKSORT_BLOCK_SIZE/SORT_THRESHOLD), cgh);
	  local_int_read_write_accessor workstack_pointer(range<>(1), cgh);
	  local_uint_read_write_accessor ltsum(range<>(1), cgh), gtsum(range<>(1), cgh),
		  lt(range<>(LQSORT_LOCAL_WORKGROUP_SIZE+1), cgh), gt(range<>(LQSORT_LOCAL_WORKGROUP_SIZE+1), cgh);
      local_T_read_write_accessor mys(range<>(QUICKSORT_BLOCK_SIZE), cgh), mysn(range<>(QUICKSORT_BLOCK_SIZE), cgh),
          temp(range<>(SORT_THRESHOLD), cgh);
 
 
	  auto lqsort = lqsort_kernel_class<T>(db, dnb, doneb,
	      workstack, workstack_pointer, mys, mysn, temp, ltsum, gtsum, lt, gt);

      cgh.parallel_for(
		*lqsort.kernel,
		nd_range<>(LQSORT_LOCAL_WORKGROUP_SIZE * done.size(), 
	               LQSORT_LOCAL_WORKGROUP_SIZE), 
	    lqsort);
    });
    pOCL->queue.wait_and_throw();

#ifdef GET_DETAILED_PERFORMANCE
	endClock = seconds();
	double totalTime = endClock - beginClock;
	std::cout << "lqsort time " << totalTime * 1000 << " ms" << std::endl;
#endif
}

size_t optp(size_t s, double k, size_t m) {
	return (size_t)pow(2, floor(log(s*k + m)/log(2.0) + 0.5));
}

template <class T>
void GPUQSort(OCLResources *pOCL, size_t size, T* d, T* dn)  {
	// allocate buffers
	buffer<T>  d_buffer(d, size, {property::buffer::use_host_ptr()});
	buffer<T>  dn_buffer(dn, size, {property::buffer::use_host_ptr()});

	const size_t MAXSEQ = optp(size, 0.00009516, 203);
	const size_t MAX_SIZE = 12*std::max(MAXSEQ, (size_t)QUICKSORT_BLOCK_SIZE);
	//std::cout << "MAXSEQ = " << MAXSEQ << std::endl;
	T startpivot = median(d[0], d[size/2], d[size-1]);
	std::vector<work_record<T>> work, done, news;
	work.reserve(MAX_SIZE);
	done.reserve(MAX_SIZE);
	news.reserve(MAX_SIZE);
	std::vector<parent_record> parent_records;
	parent_records.reserve(MAX_SIZE);
	std::vector<block_record<T>> blocks;
	blocks.reserve(MAX_SIZE);
	
	work.push_back(work_record<T>(0, size, startpivot, 1));

	bool reset = true;

	while(!work.empty() /*&& work.size() + done.size() < MAXSEQ*/) {
		size_t blocksize = 0;
		
		for(auto it = work.begin(); it != work.end(); ++it) {
			blocksize += std::max((it->end - it->start)/MAXSEQ, (size_t)1);
		}
		for(auto it = work.begin(); it != work.end(); ++it) {
			uint start = it->start;
			uint end   = it->end;
			T pivot = it->pivot;
			uint direction = it->direction;
			uint blockcount = (end - start + blocksize - 1)/blocksize;
			parent_record prnt(start, end, start, end, blockcount-1);
			parent_records.push_back(prnt);

			for(uint i = 0; i < blockcount - 1; i++) {
				uint bstart = start + blocksize*i;
				block_record<T> br(bstart, bstart+blocksize, pivot, direction, parent_records.size()-1);
				blocks.push_back(br);
			}
			block_record<T> br(start + blocksize*(blockcount - 1), end, pivot, direction, parent_records.size()-1);
			blocks.push_back(br);
		}

		gqsort(pOCL, d_buffer, dn_buffer, blocks, parent_records, news, reset);
		reset = false;
		//std::cout << " blocks = " << blocks.size() << " parent records = " << parent_records.size() << " news = " << news.size() << std::endl;
		work.clear();
		parent_records.clear();
		blocks.clear();
		for(auto it = news.begin(); it != news.end(); ++it) {
			if (it->direction != EMPTY_RECORD) {
				if (it->end - it->start <= QUICKSORT_BLOCK_SIZE /*size/MAXSEQ*/) {
					if (it->end - it->start > 0)
						done.push_back(*it);
				} else {
					work.push_back(*it);
				}
			}
		}
		news.clear();
	}
	for(auto it = work.begin(); it != work.end(); ++it) {
		if (it->end - it->start > 0)
			done.push_back(*it);
	}

	lqsort(pOCL, done, d_buffer, dn_buffer);
}

void QueryPrintDeviceInfo(queue& q) {
	auto vendor = q.get_device().get_info<info::device::vendor>();
    auto name = q.get_device().get_info<info::device::name>();
    printf("\nUsing platform: %s and device: %s.\n", vendor.c_str(), name.c_str());
	printf ("OpenCL Device info:\n");
    std::cout << "CL_DEVICE_VENDOR           : " << vendor << std::endl;
	std::cout << "CL_DEVICE_NAME             : " << name << std::endl;
	auto driver_version = q.get_device().get_info<info::device::driver_version>();
	std::cout << "CL_DRIVER_VERSION          : " << driver_version << std::endl;
	auto profile = q.get_device().get_info<info::device::profile>();
	std::cout << "CL_DEVICE_PROFILE          : " << profile << std::endl;
	auto version = q.get_device().get_info<info::device::version>();
	std::cout << "CL_DEVICE_VERSION          : " << version << std::endl;
    auto opencl_c_version = q.get_device().get_info<info::device::opencl_c_version>();
    std::cout << "CL_DEVICE_OPENCL_C_VERSION : " << opencl_c_version << std::endl;
    auto max_compute_units = q.get_device().get_info<info::device::max_compute_units>(); 
	std::cout << "CL_DEVICE_MAX_COMPUTE_UNITS: " << max_compute_units << std::endl;
	auto max_work_item_dimensions = q.get_device().get_info<info::device::max_work_item_dimensions>();
	std::cout << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << max_work_item_dimensions << std::endl;
    auto max_work_item_sizes = q.get_device().get_info<info::device::max_work_item_sizes>();
	printf ("CL_DEVICE_MAX_WORK_ITEM_SIZES   :    (%5zu, %5zu, %5zu)\n", 
					max_work_item_sizes[0],max_work_item_sizes[1], max_work_item_sizes[2]);
    auto max_work_group_size = q.get_device().get_info<info::device::max_work_group_size>();
	std::cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << max_work_group_size << std::endl;
    auto mem_base_addr_align = q.get_device().get_info<info::device::mem_base_addr_align>();
    std::cout << "CL_DEVICE_MEM_BASE_ADDR_ALIGN: " << mem_base_addr_align << std::endl;
    
	size_t uMinBaseAddrAlignSizeBytes, uNumBytes;
    ciErrNum = clGetDeviceInfo(q.get_device().get(), CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, sizeof(cl_uint), &uMinBaseAddrAlignSizeBytes, &uNumBytes);
	CheckCLError (ciErrNum, "clGetDeviceInfo() query failed.", "clGetDeviceinfo() query success")
	printf ("CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE: %8zu\n", uMinBaseAddrAlignSizeBytes);

	auto max_clock_frequency = q.get_device().get_info<info::device::max_clock_frequency>();
	std::cout << "CL_DEVICE_MAX_CLOCK_FREQUENCY: " << max_clock_frequency << std::endl;
    auto image2D_max_width = q.get_device().get_info<info::device::image2d_max_width>();
	std::cout << "CL_DEVICE_IMAGE2D_MAX_WIDTH  : " << image2D_max_width << std::endl; 
    auto local_mem_size = q.get_device().get_info<info::device::local_mem_size>();
	std::cout << "CL_DEVICE_LOCAL_MEM_SIZE     : " << local_mem_size << std::endl;
    auto max_mem_alloc_size = q.get_device().get_info<info::device::max_mem_alloc_size>();
	std::cout << "CL_DEVICE_MAX_MEM_ALLOC_SIZE : " << max_mem_alloc_size << "\n" << std::endl;

    const uint MAX_NUM_FORMATS = 500;
	cl_uint numFormats;
	cl_image_format myFormats[MAX_NUM_FORMATS];

	ciErrNum = clGetSupportedImageFormats(q.get_context().get(), CL_MEM_READ_ONLY, CL_MEM_OBJECT_IMAGE2D, 255, myFormats, &numFormats);
	CheckCLError (ciErrNum, "clGetSupportedImageFormats() query failed.", "clGetSupportedImageFormats() query success")
}

template <class T>
int big_test(OCLResources& myOCL, uint arraySize, unsigned int	NUM_ITERATIONS, 
             const char* pDeviceStr, const std::string& type_name) 
{
	double totalTime, quickSortTime, stdSortTime;

	double beginClock, endClock;
    
	printf("\n\n\n--------------------------------------------------------------------\n");
	printf("Allocating array size of %d\n", arraySize);
#ifdef _MSC_VER 
	T* pArray = (T*)_aligned_malloc (((arraySize*sizeof(T))/64 + 1)*64, 4096);
	T* pArrayCopy = (T*)_aligned_malloc (((arraySize*sizeof(T))/64 + 1)*64, 4096);
#else // _MSC_VER
  T* pArray = (T*)aligned_alloc (4096, ((arraySize*sizeof(T))/64 + 1)*64);
  T* pArrayCopy = (T*)aligned_alloc (4096, ((arraySize*sizeof(T))/64 + 1)*64);
#endif // _MSC_VER

	std::generate(pArray, pArray + arraySize, [](){static T i = 0; return ++i; });
	std::random_shuffle(pArray, pArray + arraySize);
#ifdef RUN_CPU_SORTS
	std::cout << "Sorting the regular way..." << std::endl;
	std::copy(pArray, pArray + arraySize, pArrayCopy);

  beginClock = seconds();
	std::sort(pArrayCopy, pArrayCopy + arraySize);
  endClock = seconds();
	totalTime = endClock - beginClock;
	std::cout << "Time to sort: " << totalTime * 1000 << " ms" << std::endl;
	stdSortTime = totalTime;

	std::cout << "Sorting with parallel quicksort on the cpu: " << std::endl;
	std::copy(pArray, pArray + arraySize, pArrayCopy);

  beginClock = seconds();
	//quicksort(pArrayCopy, 0, arraySize-1);
  parallel_sort(pArrayCopy, pArrayCopy + arraySize);
  endClock = seconds();
	totalTime = endClock - beginClock;
	std::cout << "Time to sort: " << totalTime * 1000 << " ms" << std::endl;
	quickSortTime = totalTime;
#ifdef TRUST_BUT_VERIFY
	{
		std::vector<T> verify(arraySize);
		std::copy(pArray, pArray + arraySize, verify.begin());

		std::cout << "verifying: ";
		std::sort(verify.begin(), verify.end());
		bool correct = std::equal(verify.begin(), verify.end(), pArrayCopy);
		unsigned int num_discrepancies = 0;
		if (!correct) {
			for(size_t i = 0; i < arraySize; i++) {
				if (verify[i] != pArrayCopy[i]) {
					//std:: cout << "discrepancy at " << i << " " << pArrayCopy[i] << " expected " << verify[i] << std::endl;
					num_discrepancies++;
				}
			}
		}
		std::cout << std::boolalpha << correct << std::endl;
		if (!correct) {
			char y;
			std::cout << "num_discrepancies: " << num_discrepancies << std::endl;
			std::cin >> y;
		}
	}
#endif
#endif // RUN_CPU_SORTS

	// Initialize OpenCL:
	bool bCPUDevice = false;
	std::cout << "Sorting with GPUQSort on the " << pDeviceStr << "with type " << type_name << std::endl;
	std::vector<T> original(arraySize);
	std::copy(pArray, pArray + arraySize, original.begin());

    // Let's prebuild SYCL program
	cl::sycl::program program(myOCL.contextHdl);
    totalTime = 0;
//#define SWAP_ORDER 1
#ifdef SWAP_ORDER
goto try_me_first;
try_me_second:
#endif
	try {
      bool has_it = false;
	  try {
	    has_it = program.has_kernel<lqsort_kernel_class<T>>();
	  } catch (...) {}

	  if (has_it)
	    std::cout << "No need to build! We will just get it!" << std::endl;
      else { 
	    std::cout << "before program.build_with_kernel_type<lqsort_kernel_class<T>>();\n";
	    beginClock = seconds();
        program.build_with_kernel_type<lqsort_kernel_class<T>>();
        endClock = seconds();
	    totalTime += endClock - beginClock;
	    //cl_program p = program.get();
	    //BuildFailLog(p, myOCL.deviceID);
        std::cout << "after program.build_with_kernel_type<lqsort_kernel_class<T>>();\n";
	    std::cout << "Time to build SYCL Program: " << totalTime * 1000 << " ms" << std::endl;
	  }
       new (lqsort_kernel_class<T>::kernel) cl::sycl::kernel(program.get_kernel<lqsort_kernel_class<T>>());
	  std::cout << "Successfully acquired lqsort_kernel_class<T>!" << std::endl;
	} catch (const cl::sycl::exception& e) {
	  std::cerr << "SYCL exception caught: " << e.what() << "\n";
	  return 1;
	} catch (const std::exception& e) {
	  std::cerr << "C++ exception caught: " << e.what() << "\n";
	  return 2;
	}
#ifdef SWAP_ORDER
goto report_total_time;
try_me_first:
#endif
	try {
	  bool has_it = false;
	  try { 
		has_it = program.has_kernel<gqsort_kernel_class<T>>();
	  } catch (...) { std::cout << "Caught something!" << std::endl; }

	  if (has_it)
	    std::cout << "No need to build! We will just get it!" << std::endl;
      else {
	    std::cout << "before program.build_with_kernel_type<gqsort_kernel_class<T>>();\n";
	    beginClock = seconds();
        program.build_with_kernel_type<gqsort_kernel_class<T>>();
        endClock = seconds();
	    totalTime += endClock - beginClock;
	    //cl_program p = program.get();
	    //BuildFailLog(p, myOCL.deviceID);
        std::cout << "after program.build_with_kernel_type<gqsort_kernel_class<T>>();\n";
	    std::cout << "Time to build SYCL Program: " << totalTime * 1000 << " ms" << std::endl;
	  }
      new (gqsort_kernel_class<T>::kernel) cl::sycl::kernel(program.get_kernel<gqsort_kernel_class<T>>());
	  std::cout << "Successfully acquired gqsort_kernel_class<T>!" << std::endl;
	} catch (const cl::sycl::exception& e) {
	  std::cerr << "SYCL exception caught: " << e.what() << "\n";
	  return 1;
	} catch (const std::exception& e) {
	  std::cerr << "C++ exception caught: " << e.what() << "\n";
	  return 2;
	}
#ifdef SWAP_ORDER
goto try_me_second;
report_total_time:
#endif

	std::vector<double> times;
	times.resize(NUM_ITERATIONS);
	double AverageTime = 0.0;
	uint num_failures = 0;
	for(uint k = 0; k < NUM_ITERATIONS; k++) {
		std::copy(original.begin(), original.end(), pArray);
		std::vector<T> verify(arraySize);
		std::copy(pArray, pArray + arraySize, verify.begin());

		beginClock = seconds();
		GPUQSort(&myOCL, arraySize, pArray, pArrayCopy);
		endClock = seconds();
		totalTime = endClock - beginClock;
		std::cout << "Time to sort: " << totalTime * 1000 << " ms" << std::endl;
		times[k] = totalTime;
		AverageTime += totalTime;
#ifdef TRUST_BUT_VERIFY
		std::cout << "verifying: ";
		std::sort(verify.begin(), verify.end());
		bool correct = std::equal(verify.begin(), verify.end(), pArray);
		unsigned int num_discrepancies = 0;
		if (!correct) {
			for(size_t i = 0; i < arraySize; i++) {
				if (verify[i] != pArray[i]) {
					//std:: cout << "discrepancy at " << i << " " << pArray[i] << " expected " << verify[i] << std::endl;
					num_discrepancies++;
				}
			}
		}
		std::cout << std::boolalpha << correct << std::endl;
		if (!correct) {
			std::cout << "num_discrepancies: " << num_discrepancies << std::endl;
			num_failures ++;
		}
#endif
	}
	std::cout << " Number of failures: " << num_failures << " out of " << NUM_ITERATIONS << std::endl;
	AverageTime = AverageTime/NUM_ITERATIONS; 
	std::cout << "Average Time: " << AverageTime * 1000 << " ms" << std::endl;
	double stdDev = 0.0, minTime = 1000000.0, maxTime = 0.0;
	for(uint k = 0; k < NUM_ITERATIONS; k++) 
	{
		stdDev += (AverageTime - times[k])*(AverageTime - times[k]);
		minTime = std::min(minTime, times[k]);
		maxTime = std::max(maxTime, times[k]);
	}

	if (NUM_ITERATIONS > 1) {
		stdDev = sqrt(stdDev/(NUM_ITERATIONS - 1));
		std::cout << "Standard Deviation: " << stdDev * 1000 << std::endl;
		std::cout << "%error (3*stdDev)/Average: " << 3*stdDev / AverageTime * 100 << "%" << std::endl;
		std::cout << "min time: " << minTime * 1000 << " ms" << std::endl;
		std::cout << "max time: " << maxTime * 1000 << " ms" << std::endl;
	}

#ifdef RUN_CPU_SORTS
	std::cout << "Average speedup over CPU quicksort: " << quickSortTime/AverageTime << std::endl;
	std::cout << "Average speedup over CPU std::sort: " << stdSortTime/AverageTime << std::endl;
#endif // RUN_CPU_SORTS

	printf("-------done--------------------------------------------------------\n");
#ifdef _MSC_VER
	_aligned_free(pArray);
	_aligned_free(pArrayCopy);
#else // _MSC_VER
  free(pArray);
  free(pArrayCopy);
#endif // _MSC_VER
  return 0;
}

int main(int argc, char** argv)
{
	OCLResources	myOCL;
	unsigned int	NUM_ITERATIONS;
	char			pDeviceStr[256];
	char			pVendorStr[256];
	bool			bShowCL = false;

	uint			heightReSz, widthReSz;


	parseArgs (&myOCL, argc, argv, &NUM_ITERATIONS, pDeviceStr, pVendorStr, &widthReSz, &heightReSz, &bShowCL);
	
	if (bShowCL)
	    QueryPrintDeviceInfo(myOCL.queue);
		
	uint arraySize = widthReSz*heightReSz;
    big_test<uint>(myOCL,arraySize, NUM_ITERATIONS, pDeviceStr, "uint");
    big_test<float>(myOCL,arraySize, NUM_ITERATIONS, pDeviceStr, "float");
    big_test<double>(myOCL,arraySize, NUM_ITERATIONS, pDeviceStr, "double");

	return 0;
}
