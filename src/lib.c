#include <windows.h>
#include <stddef.h>
#undef _MSC_VER
#include <cuda.h>
#include "shared.h"

__declspec(dllexport) int _fltused = 1;

#define PATTERN_ROW_SIZE (sizeof(unsigned) + sizeof(float))
#define PATTERN_SIZE (PATTERN_LEN * PATTERN_ROW_SIZE)
char pattern[PATTERN_SIZE];
char mask[PATTERN_SIZE];
char *host_base_mem = NULL;
char *host_aligned_mem = NULL;
CUdevice dev;
CUcontext ctx;
CUdeviceptr dev_host_shared_mem;
CUdeviceptr dev_intermediate_mem;
CUmodule mod;
CUfunction fn_count_neighbours;
CUfunction fn_propagate;

int memeqmask(char *a, char *b, char *m, size_t size) {
	for(size_t i=0; i<size; i++) {
		if((a[i] & m[i]) != (b[i] & m[i])) {
			return 0;
		}
	}

	return 1;
}

/*
	not implemented here but should pause all other threads to avoid them
	unmapping pages or changing protection while enumerating
*/
char *findneedle(char *pattern, char *mask) {
	HANDLE proc = GetCurrentProcess();

	MEMORY_BASIC_INFORMATION mbi;
	char *addr = 0x00000000;

	while(VirtualQueryEx(proc, addr, &mbi, sizeof(mbi))) {
		if(!(mbi.Type & MEM_PRIVATE)) {
			goto next;
		}

		if(!(mbi.State & MEM_COMMIT)) {
			goto next;
		}

		if(mbi.Protect != PAGE_READWRITE) {
			goto next;
		}

		if(mbi.RegionSize < PATTERN_SIZE) {
			goto next;
		}

		char *start = (char *) mbi.BaseAddress;
		if(start == pattern) {
			goto next;
		}

		char *end = start + mbi.RegionSize - PATTERN_SIZE;

		for(char *cur=start; cur<end; cur++) {
			if(memeqmask(cur, pattern, mask, PATTERN_SIZE)) {
				return cur;
			}
		}

	next:
		addr = (char *) mbi.BaseAddress + mbi.RegionSize;
	}

	return NULL;
}

unsigned atou(char *a) {
	unsigned u = 0;
	while(*a) {
		u *= 10u;
		u += *a++ - '0';
	}
	return u;
}

__declspec(dllexport) char *findbuf(int argc, char **argv) {
	if(argc < 1 || argv == NULL) {
		return NULL;
	}

	if(argv[0] == NULL) {
		return NULL;
	}

	unsigned seed = atou(argv[0]);
	unsigned r = seed;
	for(unsigned i=0; i<PATTERN_LEN; i++) {
		__builtin_memcpy_inline(
			mask + i * PATTERN_ROW_SIZE,
			&(struct {unsigned t; unsigned f;}) {
				0x000000FFu,
				0xFFFFFFFFu
			},
			PATTERN_ROW_SIZE
		);

		__builtin_memcpy_inline(
			pattern + i * PATTERN_ROW_SIZE,
			&(struct {unsigned t; float f;}) {
				0x0000002Au,
				(float) (r = JANKYRAND(r))
			},
			PATTERN_ROW_SIZE
		);
	}

	host_base_mem = findneedle(pattern, mask);
	if(host_base_mem != NULL) {
		for(unsigned i=0; i<PATTERN_LEN; i++) {
			__builtin_memcpy_inline(
				host_base_mem + i * PATTERN_ROW_SIZE,
				&(struct {unsigned t; float f;}) {
					0x0000002Au,
					(float) (r = JANKYRAND(r))
				},
				PATTERN_ROW_SIZE
			);
		}

		host_aligned_mem = __builtin_align_up(host_base_mem, 0x1000);
		unsigned offset = (host_aligned_mem - host_base_mem) >> 3;
		__builtin_memcpy_inline(
			host_base_mem + PATTERN_LEN * PATTERN_ROW_SIZE,
			&(struct {unsigned t; float f;}) {
				0x0000002Au,
				(float) (offset + 1)
			},
			PATTERN_ROW_SIZE
		);
	}

	return NULL;
}

__declspec(dllexport) char *init(int argc, char **argv) {
	(void) argc;
	(void) argv;
	CUresult r;
	volatile float *param0 = (volatile float *) (host_aligned_mem + 0x04);
	volatile float *param1 = (volatile float *) (host_aligned_mem + 0x0C);
	volatile float *param2 = (volatile float *) (host_aligned_mem + 0x14);

#ifdef DEBUG
	/* for kernel printf */
	HANDLE f = CreateFileW(
		L"kernelout.txt",
		GENERIC_WRITE,
		FILE_SHARE_READ,
		NULL,
		CREATE_ALWAYS,
		FILE_ATTRIBUTE_NORMAL,
		NULL
	);

	if(f != INVALID_HANDLE_VALUE) {
		(void) SetStdHandle(STD_OUTPUT_HANDLE, f);
	}
#endif

	if((r = cuInit(0)) != CUDA_SUCCESS) {
		*param0 = -1.0f;
		return NULL;
	}

	int devcount = 0;
	if((r = cuDeviceGetCount(&devcount)) != CUDA_SUCCESS || devcount == 0) {
		*param0 = -1.0f;
		return NULL;
	}

	if((r = cuDeviceGet(&dev, 0)) != CUDA_SUCCESS) {
		*param0 = -1.0f;
		return NULL;
	}

	if((r = cuCtxCreate(&ctx, CU_CTX_SCHED_SPIN, dev)) != CUDA_SUCCESS) {
		*param0 = -1.0f;
		return NULL;
	}

	unsigned maxx = (unsigned) *param1;
	unsigned maxy = (unsigned) *param2;
	size_t size = maxx * maxy * 8 + 8 * 8;
	if((r = cuMemHostRegister(
		host_aligned_mem,
		size,
		CU_MEMHOSTALLOC_DEVICEMAP
	)) != CUDA_SUCCESS) {
		*param0 = -1.0f;
		return NULL;
	}

	if((r = cuMemHostGetDevicePointer(
		&dev_host_shared_mem,
		host_aligned_mem,
		0
	)) != CUDA_SUCCESS) {
		*param0 = -1.0f;
		return NULL;
	}

	size = maxx * maxy;
	if((r = cuMemAlloc(&dev_intermediate_mem, size)) != CUDA_SUCCESS) {
		*param0 = -1.0f;
		return NULL;
	}

	if((r = cuModuleLoad(&mod, ".\\kernel\\life.ptx")) != CUDA_SUCCESS) {
		*param0 = -1.0f;
		return NULL;
	}

	if((r = cuModuleGetFunction(
		&fn_count_neighbours,
		mod,
		"count_neighbours"
	)) != CUDA_SUCCESS) {
		*param0 = -1.0f;
		return NULL;
	}

	if((r = cuModuleGetFunction(
		&fn_propagate,
		mod,
		"propagate"
	)) != CUDA_SUCCESS) {
		*param0 = -1.0f;
		return NULL;
	}

	*param0 = 0.0f;
	return NULL;
}

__declspec(dllexport) char *exec_sync(int argc, char **argv) {
	(void) argc;
	(void) argv;
	CUresult r;
	volatile float *param0 = (volatile float *) (host_aligned_mem + 0x04);
	volatile float *param1 = (volatile float *) (host_aligned_mem + 0x0C);
	volatile float *param2 = (volatile float *) (host_aligned_mem + 0x14);

#if 0
	/* surely this context is still current */
	if((r = cuCtxSetCurrent(ctx)) != CUDA_SUCCESS) {
		*param0 = (float) r;
		return NULL;
	}
#endif

	unsigned maxx = (unsigned) *param1;
	unsigned maxy = (unsigned) *param2;

	/* 8x8 threads per block (2 warps)
	   (maxx / 8)x(maxy / 8) blocks */
	unsigned blk_w = 8;
	unsigned blk_h = 8;
	unsigned blk_l = 1;
	unsigned grd_w = maxx >> 3;
	unsigned grd_h = maxy >> 3;
	unsigned grd_l = 1;

	grd_w += (!grd_w) * 1;
	grd_h += (!grd_h) * 1;

	r = cuLaunchKernel(
		fn_count_neighbours,
		grd_w, grd_h, grd_l,
		blk_w, blk_h, blk_l,
		0,
		NULL,
		(void *[]) {
			&dev_host_shared_mem,
			&dev_intermediate_mem
		},
		NULL
	);
	if(r != CUDA_SUCCESS) {
		*param0 = -1.0f;
		return NULL;
	}

	if((r = cuCtxSynchronize()) != CUDA_SUCCESS) {
		*param0 = -1.0f;
		return NULL;
	}

	r = cuLaunchKernel(
		fn_propagate,
		grd_w, grd_h, grd_l,
		blk_w, blk_h, blk_l,
		0,
		NULL,
		(void *[]) {
			&dev_host_shared_mem,
			&dev_intermediate_mem
		},
		NULL
	);
	if(r != CUDA_SUCCESS) {
		*param0 = -1.0f;
		return NULL;
	}

	if((r = cuCtxSynchronize()) != CUDA_SUCCESS) {
		*param0 = -1.0f;
		return NULL;
	}

	*param0 = 0.0f;
	return NULL;
}

BOOL WINAPI main(HINSTANCE instance, DWORD reason, LPVOID reserved) {
	(void) reserved;

	switch(reason) {
		case DLL_PROCESS_ATTACH:
			return DisableThreadLibraryCalls(instance);
	}

	return TRUE;
}
