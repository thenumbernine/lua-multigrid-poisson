#! /usr/bin/env luajit

local ffi = require 'ffi'
require 'ext'

local real = 'float'
local size = 16 
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
#if 0	//L2-sized kernel
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
	__global real* u,
	const __global real* v
) {
	int i = get_global_id(0);
	if (i >= get_global_size(0)) return;
	u[i] += v[i];
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
#if 1	//relative error
	if (psiNew[index] != 0 && psiNew[index] != psi[index]) {
		errorBuf[index] = fabs(1. - psi[index] / psiNew[index]);
	} else {
		errorBuf[index] = 0;
	}
#else	//energy
	real d = psi[index] - psiNew[index];
	errorBuf[index] = d * d;
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

local rs = {}
local Rs = {}
local vs = {}
for i=0,math.log(size,2) do
	local L = 2^i
	print('creating buffer with size '..L)
	rs[L] = ctx:buffer{rw=true, size=L*L*ffi.sizeof(real)}
	Rs[L] = ctx:buffer{rw=true, size=L*L*ffi.sizeof(real)}
	vs[L] = ctx:buffer{rw=true, size=L*L*ffi.sizeof(real)}
end

local init = program:kernel'init'
local calcError = program:kernel'calcError'
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
	print('clcall2D',w,h,l1,l2)
	cmds:enqueueNDRangeKernel{kernel=kernel, globalSize={w,h}, localSize={l1,l2}}
end

clcall2D(size,size, init, f, psi)

local function checknan(name, buf, w, h)
	local tmp = gcnew('real', w*h)
	cmds:enqueueReadBuffer{buffer=buf, block=true, size=w*h*ffi.sizeof(real), ptr=tmp}
	print(name)
	for j=0,h-1 do
		for i=0,w-1 do
			io.write(' ',tmp[i+w*j])
		end
		print()
	end
	for i=0,w*h-1 do
		if not math.isfinite(tmp[i]) then
			error("found a nan")
		end
	end
	gcfree(tmp)
end

local function twoGrid(h, u, f, L, smooth)
print('L',L,'h',h)
	if L == 1 then
		--*u = *f / (-4 / h^2)
checknan('f', f, L, L)
		clcall2D(L,L,GaussSeidel, u, f, ffi.new('real[1]', h))
checknan('u', u, L, L)
		return
	end
	
	for i=1,smooth do
print(tolua({
	i=i,
	L=L,
	h=h,
},{indent=true}))
checknan('f', f, L, L)
		clcall2D(L,L, GaussSeidel, u, f, ffi.new('real[1]', h))
checknan('u', u, L, L)
	end
	
	local r = rs[L]
checknan('f', f, L, L)
checknan('u', u, L, L)
	--cmds:enqueueFillBuffer{buffer=r, size=L*L*ffi.sizeof(real)}
	clcall2D(L,L, calcResidual, r, f, u, ffi.new('real[1]', h))
checknan('r', r, L, L)
	
	--r = f - a(u)

	local L2 = L/2
	local R = Rs[L2]
	clcall2D(L2,L2, reduceResidual, R, r)
checknan('R', R, L2, L2)
	
	local V = vs[L2]
	twoGrid(2*h, V, R, L2, smooth)
checknan('V', V, L2, L2)

	local v = vs[L]
	--clcall2D(L2, L2, expandResidual, v, V)
	clcall2D(L, L, expandResidual, v, V)
checknan('v', v, L, L)

	clcall1D(L*L, addTo, u, v)
checknan('u', u, L, L)

	for i=1,smooth do
		clcall2D(L,L, GaussSeidel, u, f, ffi.new('real[1]', h))
checknan('u', u, L, L)
	end
end

local smooth = 7
local h = 1/size
for iter=1,1 do
	cmds:enqueueCopyBuffer{src=psi, dst=psiNew, size=size*size*ffi.sizeof(real)}
	twoGrid(h, psi, f, size, smooth)
	clcall2D(size, size, calcError, errorBuf, psi, psiNew)

	-- doing the error calculation in cpu ...
	-- this is messing up the lua env ... total becomes nil
	local tmp = gcnew('real', size*size)
	cmds:enqueueReadBuffer{buffer=errorBuf, block=true, size=size*size*ffi.sizeof(real), ptr=tmp}
	local frobErr = 0
	for j=0,size*size-1 do
		frobErr = frobErr + tmp[j]
	end
	gcfree(tmp)
	frobErr = math.sqrt(frobErr)
	print('rel err', frobErr)
end
