#! /usr/bin/env luajit

local ffi = require 'ffi'
require 'ext'
local gcmem = require 'ext.gcmem'

local size = 16

local env = require 'cl.obj.env'{size={size,size}}

local code = require 'template'([[
#define size <?=size?>

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


]], {
	size = size,
})

local program = env:program{code = code}

local f = env:buffer()
local psi = env:buffer()
local psiOld = env:buffer()
local errorBuf = env:buffer()

local rs = {}
local Rs = {}
local vs = {}
local Vs = {}
local domains = {}
for i=0,math.log(size,2) do
	local L = 2^i
	local domain = require 'cl.obj.domain'{env = env, size = {L,L}}
	domains[L] = domain
	rs[L] = domain:buffer()
	Rs[L] = domain:buffer()
	vs[L] = domain:buffer()
	Vs[L] = domain:buffer()
	rs[L]:fill()
	Rs[L]:fill()
	vs[L]:fill()
	Vs[L]:fill()
end

local init = program:kernel'init'
local calcFrobErr = program:kernel'calcFrobErr'
local calcRelErr = program:kernel'calcRelErr'
local GaussSeidel = program:kernel'GaussSeidel'
local calcResidual = program:kernel'calcResidual'
local reduceResidual = program:kernel'reduceResidual'
local expandResidual = program:kernel'expandResidual'
local addTo = program:kernel'addTo'

program:compile()

local function clcall1D(w, kernel, ...)
	kernel.domain = require 'cl.obj.domain'{env=env, size=w}
	kernel(...)
end

local function clcall2D(w,h, kernel, ...)
	kernel.domain = require 'cl.obj.domain'{env=env, size={w,h}}
	kernel(...)
end

clcall2D(size, size, init, f.buf, psi.buf)

local function getbuffer(gpuMem, L)
	local cpuMem = gcmem.new('real', L*L)
	cmds:enqueueReadBuffer{buffer=gpuMem, block=true, size=L*L*ffi.sizeof(env.real), ptr=cpuMem}
	return cpuMem
end

local function showAndCheck(name, gpuMem, L)
do return end	
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
	gcmem.free(cpuMem)
end

local function twoGrid(h, u, f, L, smooth)
	if L == 1 then
		--*u = *f / (-4 / h^2)
showAndCheck('f', f, L, L)
		clcall2D(L,L,GaussSeidel, u.buf, f.buf, ffi.new('real[1]', h))
showAndCheck('u', u, L, L)
		return
	end
	
	for i=1,smooth do
showAndCheck('f', f, L, L)
		clcall2D(L,L, GaussSeidel, u.buf, f.buf, ffi.new('real[1]', h))
showAndCheck('u', u, L, L)
	end
	
	local r = rs[L]
showAndCheck('f', f, L, L)
showAndCheck('u', u, L, L)
	--cmds:enqueueFillBuffer{buffer=r, size=L*L*ffi.sizeof(env.real)}
	clcall2D(L,L, calcResidual, r.buf, f.buf, u.buf, ffi.new('real[1]', h))
showAndCheck('r', r, L, L)
	
	--r = f - a(u)

	local L2 = L/2
	local R = Rs[L2]
	clcall2D(L2,L2, reduceResidual, R.buf, r.buf)
showAndCheck('R', R, L2, L2)
	
	local V = Vs[L2]
	twoGrid(2*h, V, R, L2, smooth)
showAndCheck('V', V, L2, L2)

	local v = vs[L]
	clcall2D(L2, L2, expandResidual, v.buf, V.buf)
	--clcall2D(L, L, expandResidual, v.buf, V.buf)
showAndCheck('v', v, L, L)

	clcall1D(L*L, addTo, ffi.new('size_t[1]', L*L), u.buf, v.buf)
showAndCheck('u', u, L, L)

	for i=1,smooth do
		clcall2D(L,L, GaussSeidel, u.buf, f.buf, ffi.new('real[1]', h))
showAndCheck('u', u, L, L)
	end
end
	
-- doing the error calculation in cpu ...
local errMem = gcmem.new('real', size*size)

local smooth = 7
local h = 1/size
for iter=1,200 do
	psiOld:copyFrom(psi)
	twoGrid(h, psi, f, size, smooth)

	clcall2D(size, size, calcFrobErr, errorBuf.buf, psi.buf, psiOld.buf)
	errorBuf:toCPU(errMem)
	local frobErr = 0
	for j=0,size*size-1 do
		frobErr = frobErr + errMem[j]
	end
	frobErr = math.sqrt(frobErr)
	
	clcall2D(size, size, calcRelErr, errorBuf.buf, psi.buf, psiOld.buf)
	errorBuf:toCPU(errMem)
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

gcmem.free(errMem)
