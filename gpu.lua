#! /usr/bin/env luajit

local ffi = require 'ffi'

local real = 'float'
local size = 256 
local platform, device, ctx, cmds, program = require 'cl'{
	device = {gpu = true},
	program = {
		code = ([[
#define size {size}
typedef {real} real;

__kernel void init(
	__global real* f,
	__global real* psi
) {
	int i = get_global_id(0);
	int j = get_global_id(1);
	if (i >= size || j >= size) return;
	int index = i + size * j;
	int center = size / 2;
	real value = 0;
	if (i == center && j == center) {
		real charge = 1e+6;
		const real epsilon0 = 1;
		value = -charge / epsilon0;
	}
	
	f[index] = value;
	psi[index] = -f[index];
}

__kernel void GaussSeidel(
	__global real* u,
	const __global real* f,
	real h
) {
	int i = get_global_id(0);
	int j = get_global_id(1);
	int L = get_global_size(0);
	if (i >= L || j >= L) return;
	int index = i + L * j;
	real u_xl = i > 0 ? u[(i-1) + L * j] : 0;
	real u_xr = i < L-1 ? u[(i+1) + L * j] : 0;
	real u_yl = j > 0 ? u[i + L * (j-1)] : 0;
	real u_yr = j < L-1 ? u[i + L * (j+1)] : 0;
	real hSq = h * h;
	real askew_u = (u_xl + u_xr + u_yl + u_yr) / hSq;
	real adiag = -4. / hSq;
	real a_u = askew_u + adiag * u[i + L * j];
	u[index] = (f[index] - askew_u) / adiag;
}

__kernel void calcResidual(
	__global real* r,
	const __global real* f,
	const __global real* u,
	real h
) {
	int i = get_global_id(0);
	int j = get_global_id(0);
	int L = get_global_size(0);
	if (i >= L || j >= L) return;
	int index = i + L * j;
	
	real u_xl = i > 0 ? u[(i-1) + L * j] : 0;
	real u_xr = i < L-1 ? u[(i+1) + L * j] : 0;
	real u_yl = j > 0 ? u[i + L * (j-1)] : 0;
	real u_yr = j < L-1 ? u[i + L * (j+1)] : 0;
	real hSq = h * h;
	real askew_u = (u_xl + u_xr + u_yl + u_yr) / hSq;
	real adiag = -4. / hSq;
	real a_u = askew_u + adiag * u[i + L * j];
	r[index] = f[index] - a_u;
}

__kernel void reduceResidual(
	__global real* R,
	const __global real* r
) {
	int L2 = get_global_size(0);
	int L = L2 << 1;
	int i = get_global_id(0);
	int j = get_global_id(1);
	if (i >= L || j >= L) return;
	int srci = i + L * j;
	R[i + L2 * j] = .25 * (r[srci] + r[srci+1] + r[srci+L] + r[srci+L+1]);
}

__kernel void expandResidual(
	__global real* v,
	const __global real* V
) {
	int L2 = get_global_size(0);
	int L = L2 << 1;
	int I = get_global_id(0);
	int J = get_global_id(1);
	if (I >= L2 || J >= L2) return;
	int dsti = (I<<1) + L * (J<<1);
	v[dsti] = v[dsti+1] = v[dsti+L] = v[dsti+L+1] = V[I + L2 * J];
}

__kernel void add(
	__global real* r,
	const __global real* a,
	const __global real* b
) {
	int i = get_global_id(0);
	if (i >= get_global_size(0)) return;
	r[i] = a[i] + b[i];
}

__kernel void calcError(
	__global real* errorBuf,
	const __global real* psi,
	const __global real* psiNew
) {
	int i = get_global_id(0);
	int j = get_global_id(1);
	if (i >= size || j >= size) return;
	int index = i + size * j;
#if 0	//relative error
	if (psiNew[index] != 0 && psiNew[index] != psi[index]) {
		errorBuf[index] = fabs(1. - psi[index] / psiNew[index]);
	} else {
		errorBuf[index] = 0;
	}
#else	//energy
	real d = psi[index] - psiNew[index];
	errorBuf[index] = .5 * d * d;
#endif
}


]])
	:gsub('{real}', real)
	:gsub('{size}', size)
	},
	verbose = true,
}
local maxWorkGroupSize = tonumber(device:getInfo'CL_DEVICE_MAX_WORK_GROUP_SIZE')

ffi.cdef('typedef '..real..' real;')

local f = ctx:buffer{rw=true, size=size*size*ffi.sizeof(real)}
local psi = ctx:buffer{rw=true, size=size*size*ffi.sizeof(real)}
local psiNew = ctx:buffer{rw=true, size=size*size*ffi.sizeof(real)}
local errorBuf = ctx:buffer{rw=true, size=size*size*ffi.sizeof(real)}

local residuals = {}
local vs = {}
for i=0,math.log(size,2) do
	local L = 2^i
	print('creating buffer with size '..L)
	residuals[L] = ctx:buffer{rw=true, size=L*L*ffi.sizeof(real)}
	vs[L] = ctx:buffer{rw=true, size=L*L*ffi.sizeof(real)}
end

local init = program:kernel'init'
local calcError = program:kernel('calcError', errorBuf, psi, psiNew)
local GaussSeidel = program:kernel'GaussSeidel'
local calcResidual = program:kernel'calcResidual'
local reduceResidual = program:kernel'reduceResidual'
local expandResidual = program:kernel'expandResidual'
local add = program:kernel'add'

local function clcall1D(w, kernel, ...)
	kernel:setArgs(...)
	local l = math.min(w, maxWorkGroupSize)
	cmds:enqueueNDRangeKernel{kernel=kernel, globalSize=w, localSize=l}
end

local function clcall2D(w,h, kernel, ...)
	kernel:setArgs(...)
	local mw = math.ceil(math.sqrt(maxWorkGroupSize))
	local mh = maxWorkGroupSize / mw
	local l1 = math.min(w, mw)
	local l2 = math.min(h, mh)
	cmds:enqueueNDRangeKernel{kernel=kernel, globalSize={w,h}, localSize={l1,l2}}
end

clcall2D(size,size, init, f, psi)

local function twoGrid(h, u, f, L, smooth)
	if L == 1 then
		--*u = *f / (-4 / h^2)
		cmds:enqueueFillBuffer{buffer=u, size=ffi.sizeof(real)}
		return
	end

	for i=1,smooth do
		clcall2D(L,L, GaussSeidel, u, f, ffi.new('real[1]', h))
	end

	local r = residuals[L]
	clcall2D(L,L, calcResidual, r, f, u, ffi.new('real[1]', h))
	
	--r = f - a(u)

	local L2 = L/2
	local R = residuals[L2]
	clcall2D(L2,L2, reduceResidual, R, r)
	
	local V = vs[L2]
	twoGrid(2*h, V, R, L2, smooth)

	local v = vs[L]
	clcall2D(L2, L2, expandResidual, v, V)

	clcall1D(L*L, add, u, v, v)

	for i=1,smooth do
		clcall2D(L,L, GaussSeidel, u, f, ffi.new('float[1]', h))
	end
end

local smooth = 7
local h = 1/size
for iter=1,1 do
	cmds:enqueueCopyBuffer{src=psi, dst=psiNew, size=size*size*ffi.sizeof(real)}
	--twoGrid(h, psi, f, size, smooth)
	
	cmds:enqueueNDRangeKernel{kernel=calcError, globalSize={size,size}, localSize={16,16}}

	-- doing the error calculation in cpu ...
	-- this is messing up the lua env ... total becomes nil
	local tmp = ffi.new('real[?]', size*size)
	cmds:enqueueReadBuffer{buffer=errorBuf, block=true, size=size*size*ffi.sizeof(real), ptr=tmp}
	
	local n = 0
	local total = 0
	for j=0,size*size-1 do
		if tmp[j] ~= 0 then
			total = total + tmp[j]
			n = n + 1
		end
	end
	print('rel err', total / n)
end
