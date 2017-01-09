#! /usr/bin/env luajit

local ffi = require 'ffi'
require 'ext'
local gcmem = require 'ext.gcmem'

local size = 16

local env = require 'cl.obj.env'{size={size,size}}

local code = require 'template'([[
#define size <?=size?>

kernel void GaussSeidel(
	global real* u,
	const global real* f,
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

kernel void calcResidual(
	global real* r,
	const global real* f,
	const global real* u,
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

kernel void reduceResidual(
	global real* R,
	const global real* r
) {
	int L2 = get_global_size(0);
	int I = get_global_id(0);
	int J = get_global_id(1);
	if (I >= L2 || J >= L2) return;
	int L = L2 << 1;
	int srci = (I<<1) + L * (J<<1);
	R[I + L2 * J] = .25 * (r[srci] + r[srci+1] + r[srci+L] + r[srci+L+1]);
}

kernel void expandResidual(
	global real* v,
	const global real* V
) {
	int L2 = get_global_size(0);
	int I = get_global_id(0);
	int J = get_global_id(1);
	if (I >= L2 || J >= L2) return;
	int L = L2 << 1;
	int dsti = (I<<1) + L * (J<<1);
	v[dsti] = v[dsti+1] = v[dsti+L] = v[dsti+L+1] = V[I + L2 * J];
}

kernel void addTo(
	size_t n,
	global real* u,
	const global real* v
) {
	int i = get_global_id(0);
	if (i >= n) return;
	u[i] += v[i];
}

kernel void calcRelErr(
	global real* errorBuf,
	const global real* psi,
	const global real* psiOld
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

kernel void calcFrobErr(
	global real* errorBuf,
	const global real* psi,
	const global real* psiOld
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

local calcFrobErr = program:kernel'calcFrobErr'
local calcRelErr = program:kernel'calcRelErr'
local GaussSeidel = program:kernel'GaussSeidel'
local calcResidual = program:kernel'calcResidual'
local reduceResidual = program:kernel'reduceResidual'
local expandResidual = program:kernel'expandResidual'
local addTo = program:kernel'addTo'

program:compile()

function amrsolve(f,h)
	local psi = env:buffer{name='psi'}
	local psiOld = env:buffer()

	env:kernel{
		argsIn = {f},
		argsOut = {psi},
		body = [[ psi[index] = -f[index]; ]],
	}()

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

	local function clcall1D(w, kernel, ...)
		kernel.domain = require 'cl.obj.domain'{env=env, size=w}
		kernel(...)
	end

	local function clcall2D(w,h, kernel, ...)
		kernel.domain = require 'cl.obj.domain'{env=env, size={w,h}}
		kernel(...)
	end

	local function twoGrid(h, u, f, L, smooth)
		if L == 1 then
			--*u = *f / (-4 / h^2)
			GaussSeidel.domain = domains[L]
			GaussSeidel(u.buf, f.buf, ffi.new('real[1]', h))
			return
		end
		
		for i=1,smooth do
			clcall2D(L,L, GaussSeidel, u.buf, f.buf, ffi.new('real[1]', h))
		end
		
		local r = rs[L]
		clcall2D(L,L, calcResidual, r.buf, f.buf, u.buf, ffi.new('real[1]', h))
		
		--r = f - a(u)

		local L2 = L/2
		local R = Rs[L2]
		clcall2D(L2,L2, reduceResidual, R.buf, r.buf)
		
		local V = Vs[L2]
		twoGrid(2*h, V, R, L2, smooth)

		local v = vs[L]
		clcall2D(L2, L2, expandResidual, v.buf, V.buf)

		clcall1D(L*L, addTo, ffi.new('size_t[1]', L*L), u.buf, v.buf)

		for i=1,smooth do
			clcall2D(L,L, GaussSeidel, u.buf, f.buf, ffi.new('real[1]', h))
		end
	end

	local errorBuf = env:buffer{name='errorBuf'}
	local sumReduce = env:reduce{
		op = function(x,y) return x..' + '..y end,
		buffer = errorBuf.buf,
	}

	local countBuf = env:buffer{name='count'}
	local count = env:kernel{
		argsIn = {errorBuf},
		argsOut = {countBuf},
		body = [[ count[index] = (real)(errorBuf[index] != 0.); ]]
	}
	
	local smooth = 7
	local accuracy = 1e-10

	for iter=1,math.huge do
		psiOld:copyFrom(psi)
		twoGrid(h, psi, f, size, smooth)

		clcall2D(size, size, calcFrobErr, errorBuf.buf, psi.buf, psiOld.buf)
		frobErr = math.sqrt(sumReduce(errorBuf.buf))

		clcall2D(size, size, calcRelErr, errorBuf.buf, psi.buf, psiOld.buf)
		relErr = sumReduce()
		count()
		relErr = relErr / sumReduce(countBuf.buf)

		print(iter,'rel', relErr, 'frob', frobErr)
		if frobErr < accuracy or not math.isfinite(frobErr) then break end
		end
end

local h = 1/size
local f = env:buffer{name='f'}
env:kernel{
	argsOut = {f},
	body = [[
	int2 center = (int2)(size.x/2, size.y/2);	//can't use int4 center = size/2 ...
	real value = 0;
	if (i.x == center.x && i.y == center.y) {	//can't use if (i == center) ...
		real charge = 1e+6;
		const real epsilon0 = 1;
		value = -charge / epsilon0;
	}
	f[index] = value;
]],
}()
amrsolve(f, h)
