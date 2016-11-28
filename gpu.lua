#! /usr/bin/env luajit

local ffi = require 'ffi'
require 'ext'


local function get64bit(list)
	local best = list:map(function(item)
		local exts = item:getExtensions():lower():trim()
		return {item=item, fp64=exts:match'cl_%w+_fp64'}
	end):sort(function(a,b)
		return (a.fp64 and 1 or 0) > (b.fp64 and 1 or 0)
	end)[1]
	return best.item, best.fp64
end

local platform = get64bit(require 'cl.platform'.getAll())
local device, fp64 = get64bit(platform:getDevices{gpu=true})
local ctx = require 'cl.context'{platform=platform, device=device}
local cmds = require 'cl.commandqueue'{context=ctx, device=device}

local real = fp64 and 'double' or 'float'
print('using real',real)

local size = 16
local code = ([[
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
	int L = get_global_size(0);
	int i = get_global_id(0);
	int j = get_global_id(1);
	if (i >= L || j >= L) return;
	int index = i + L * j;
	real u_xl = i > 0 ? u[(i-1) + L * j] : 0;
	real u_xr = i < L-1 ? u[(i+1) + L * j] : 0;
	real u_yl = j > 0 ? u[i + L * (j-1)] : 0;
	real u_yr = j < L-1 ? u[i + L * (j+1)] : 0;
	real hSq = h * h;
	real askew_u = (u_xl + u_xr + u_yl + u_yr) / hSq;
	real adiag = -4. / hSq;
	u[index] = (f[index] - askew_u) / adiag;
}

__kernel void calcResidual(
	__global real* r,
	const __global real* f,
	const __global real* u,
	real h
) {
	int L = get_global_size(0);
	int i = get_global_id(0);
	int j = get_global_id(1);
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
	int I = get_global_id(0);
	int J = get_global_id(1);
	if (I >= L2 || J >= L2) return;
	int L = L2 << 1;
	int srci = (I<<1) + L * (J<<1);
	R[I + L2 * J] = .25 * (r[srci] + r[srci+1] + r[srci+L] + r[srci+L+1]);
}

__kernel void expandResidual(
	__global real* v,
	const __global real* V
) {
#if 1	//L2-sized kernel
	int L2 = get_global_size(0);
	int I = get_global_id(0);
	int J = get_global_id(1);
	if (I >= L2 || J >= L2) return;
	int L = L2 << 1;
	int dsti = (I<<1) + L * (J<<1);
	v[dsti] = v[dsti+1] = v[dsti+L] = v[dsti+L+1] = V[I + L2 * J];
#else	//L-sized kernel
	int L = get_global_size(0);
	int i = get_global_id(0);
	int j = get_global_id(1);
	if (i >= L || j >= L) return;
	int L2 = L >> 1;
	int I = i >> 1;
	int J = j >> 1;
	v[i + L * j] = V[I + L2 * J];
#endif
}

__kernel void addTo(
	size_t n,
	__global real* u,
	const __global real* v
) {
	int i = get_global_id(0);
	if (i >= n) return;
	u[i] += v[i];
}

__kernel void calcRelErr(
	__global real* errorBuf,
	const __global real* psi,
	const __global real* psiOld
) {
	int i = get_global_id(0);
	int j = get_global_id(1);
	if (i >= size || j >= size) return;
	int index = i + size * j;
	if (psiOld[index] != 0 && psiOld[index] != psi[index]) {
		errorBuf[index] = fabs(1. - psi[index] / psiOld[index]);
	} else {
		errorBuf[index] = 0;
	}
}

__kernel void calcFrobErr(
	__global real* errorBuf,
	const __global real* psi,
	const __global real* psiOld
) {
	int i = get_global_id(0);
	int j = get_global_id(1);
	if (i >= size || j >= size) return;
	int index = i + size * j;
	real d = psi[index] - psiOld[index];
	errorBuf[index] = d * d;
}


]])
	:gsub('{real}', real)
	:gsub('{size}', size)

if fp64 then
	code = '#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n' .. code 
end

local program = require 'cl.program'{context=ctx, devices={device}, code=code}

local maxWorkGroupSize = tonumber(device:getInfo'CL_DEVICE_MAX_WORK_GROUP_SIZE')

ffi.cdef('typedef '..real..' real;')

local f = ctx:buffer{rw=true, size=size*size*ffi.sizeof(real)}
local psi = ctx:buffer{rw=true, size=size*size*ffi.sizeof(real)}
local psiOld = ctx:buffer{rw=true, size=size*size*ffi.sizeof(real)}
local errorBuf = ctx:buffer{rw=true, size=size*size*ffi.sizeof(real)}

local rs = {}
local Rs = {}
local vs = {}
local Vs = {}
for i=0,math.log(size,2) do
	local L = 2^i
	rs[L] = ctx:buffer{rw=true, size=L*L*ffi.sizeof(real)}
	Rs[L] = ctx:buffer{rw=true, size=L*L*ffi.sizeof(real)}
	vs[L] = ctx:buffer{rw=true, size=L*L*ffi.sizeof(real)}
	Vs[L] = ctx:buffer{rw=true, size=L*L*ffi.sizeof(real)}
	cmds:enqueueFillBuffer{buffer=rs[L], size=L*L*ffi.sizeof(real)}
	cmds:enqueueFillBuffer{buffer=Rs[L], size=L*L*ffi.sizeof(real)}
	cmds:enqueueFillBuffer{buffer=vs[L], size=L*L*ffi.sizeof(real)}
	cmds:enqueueFillBuffer{buffer=Vs[L], size=L*L*ffi.sizeof(real)}
end

local init = program:kernel'init'
local calcFrobErr = program:kernel'calcFrobErr'
local calcRelErr = program:kernel'calcRelErr'
local GaussSeidel = program:kernel'GaussSeidel'
local calcResidual = program:kernel'calcResidual'
local reduceResidual = program:kernel'reduceResidual'
local expandResidual = program:kernel'expandResidual'
local addTo = program:kernel'addTo'

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

local function getbuffer(gpuMem, L)
	local cpuMem = gcnew('real', L*L)
	cmds:enqueueReadBuffer{buffer=gpuMem, block=true, size=L*L*ffi.sizeof(real), ptr=cpuMem}
	return cpuMem
end

local function showAndCheck(name, gpuMem, L)
	local cpuMem = getbuffer(gpuMem, L)
	print(name)
	for i=0,L-1 do
		for j=0,L-1 do
			io.write(' ',cpuMem[j+L*i])
		end
		print()
	end
	for i=0,L*L-1 do
		if not math.isfinite(cpuMem[i]) then
			error("found a nan")
		end
	end
	gcfree(cpuMem)
end

local function twoGrid(h, u, f, L, smooth)
	if L == 1 then
		--*u = *f / (-4 / h^2)
showAndCheck('f', f, L, L)
		clcall2D(L,L,GaussSeidel, u, f, ffi.new('real[1]', h))
showAndCheck('u', u, L, L)
		return
	end
	
	for i=1,smooth do
showAndCheck('f', f, L, L)
		clcall2D(L,L, GaussSeidel, u, f, ffi.new('real[1]', h))
showAndCheck('u', u, L, L)
	end
	
	local r = rs[L]
showAndCheck('f', f, L, L)
showAndCheck('u', u, L, L)
	--cmds:enqueueFillBuffer{buffer=r, size=L*L*ffi.sizeof(real)}
	clcall2D(L,L, calcResidual, r, f, u, ffi.new('real[1]', h))
showAndCheck('r', r, L, L)
	
	--r = f - a(u)

	local L2 = L/2
	local R = Rs[L2]
	clcall2D(L2,L2, reduceResidual, R, r)
showAndCheck('R', R, L2, L2)
	
	local V = Vs[L2]
	twoGrid(2*h, V, R, L2, smooth)
showAndCheck('V', V, L2, L2)

	local v = vs[L]
	clcall2D(L2, L2, expandResidual, v, V)
	--clcall2D(L, L, expandResidual, v, V)
showAndCheck('v', v, L, L)

	clcall1D(L*L, addTo, ffi.new('size_t[1]', L*L), u, v)
showAndCheck('u', u, L, L)

	for i=1,smooth do
		clcall2D(L,L, GaussSeidel, u, f, ffi.new('real[1]', h))
showAndCheck('u', u, L, L)
	end
end
	
-- doing the error calculation in cpu ...
local errMem = gcnew('real', size*size)

local smooth = 7
local h = 1/size
for iter=1,20 do
	cmds:enqueueCopyBuffer{src=psi, dst=psiOld, size=size*size*ffi.sizeof(real)}
	twoGrid(h, psi, f, size, smooth)

	clcall2D(size, size, calcFrobErr, errorBuf, psi, psiOld)
	cmds:enqueueReadBuffer{buffer=errorBuf, block=true, size=size*size*ffi.sizeof(real), ptr=errMem}
	local frobErr = 0
	for j=0,size*size-1 do
		frobErr = frobErr + errMem[j]
	end
	frobErr = math.sqrt(frobErr)
	
	clcall2D(size, size, calcRelErr, errorBuf, psi, psiOld)
	cmds:enqueueReadBuffer{buffer=errorBuf, block=true, size=size*size*ffi.sizeof(real), ptr=errMem}
	local relErr = 0
	local n = 0
	for j=0,size*size-1 do
		if errMem[j] ~= 0 then
			relErr = relErr + errMem[j]
			n = n + 1
		end
	end
	relErr = relErr / n

	print(iter,'rel', relErr, 'frob', frobErr)
end

gcfree(errMem)
