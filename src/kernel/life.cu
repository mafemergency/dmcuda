#include "..\shared.h"

struct object {
	unsigned t;
	float f;
};

__device__ bool get_cell_live(struct object *cells, int x, int y, unsigned maxx, unsigned maxy) {
	if(x < 0 || x >= maxx || y < 0 || y >= maxy) {
		return 0;
	}

	return (bool) cells[DATA_OFFSET + x + y * maxx].f;
}

extern "C" __global__ void count_neighbours(struct object *cells, unsigned char *neighbours) {
	/* dimensions can be computed with blockDim * gridDim, but this
	   demonstrates using host mem as parameters */
	unsigned maxx = (unsigned) cells[1].f;
	unsigned maxy = (unsigned) cells[2].f;

	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned i = x + y * maxx;

	if(x >= maxx || y >= maxy) {
		return;
	}

	bool a = get_cell_live(cells, (int) x + 0, (int) y - 1, maxx, maxy);
	bool b = get_cell_live(cells, (int) x + 1, (int) y - 1, maxx, maxy);
	bool c = get_cell_live(cells, (int) x + 1, (int) y + 0, maxx, maxy);
	bool d = get_cell_live(cells, (int) x + 1, (int) y + 1, maxx, maxy);
	bool e = get_cell_live(cells, (int) x + 0, (int) y + 1, maxx, maxy);
	bool f = get_cell_live(cells, (int) x - 1, (int) y + 1, maxx, maxy);
	bool g = get_cell_live(cells, (int) x - 1, (int) y + 0, maxx, maxy);
	bool h = get_cell_live(cells, (int) x - 1, (int) y - 1, maxx, maxy);
	neighbours[i] = a + b + c + d + e + f + g + h;
}

extern "C" __global__ void propagate(struct object *cells, unsigned char *neighbours) {
	unsigned maxx = (unsigned) cells[1].f;
	unsigned maxy = (unsigned) cells[2].f;

	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned i = x + y * maxx;

	if(x >= maxx || y >= maxy) {
		return;
	}

	switch(neighbours[i]) {
		case 3:
			cells[DATA_OFFSET + i].f = 1.0f;
			break;

		case 2:
			if(get_cell_live(cells, x, y, maxx, maxy) > 0) {
				cells[DATA_OFFSET + i].f = 1.0f;
			}
			break;

		default:
			cells[DATA_OFFSET + i].f = 0.0f;
	}
}
