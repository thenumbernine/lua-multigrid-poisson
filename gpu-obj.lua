#! /usr/bin/env luajit

-- 2^5 x 2^5 is diverging 

local ffi = require 'ffi'
local bit = require 'bit'
require 'ext'
local gcmem = require 'ext.gcmem'

local log2size = ... and tonumber(...) or 4
local size = bit.lshift(1, log2size)

local env = require 'cl.obj.env'{size={size,size}}

local function getn(...)
	local t = {...}
	t.n = select('#', ...)
	return t
end

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

--local printInfo = table()

print('#iter','relErr', 'n', 'frobErr')
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
		local domain = env:domain{size = {L,L}}
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
		kernel.domain = env:domain{size=w}
		kernel(...)
	end

	local function clcall2D(w,h, kernel, ...)
		kernel.domain = env:domain{size={w,h}}
		kernel(...)
	end

	local function twoGrid(h, u, f, L, smooth)
		if L == 1 then
			--*u = *f / (-4 / h^2)
			GaussSeidel.domain = domains[L]
			GaussSeidel(u.obj, f.obj, ffi.new('real[1]', h))
			return
		end
		
		for i=1,smooth do
			clcall2D(L,L, GaussSeidel, u.obj, f.obj, ffi.new('real[1]', h))
		end
		
		local r = rs[L]
		clcall2D(L,L, calcResidual, r.obj, f.obj, u.obj, ffi.new('real[1]', h))
		
		--r = f - a(u)

		local L2 = L/2
		local R = Rs[L2]
		clcall2D(L2,L2, reduceResidual, R.obj, r.obj)
		
		local V = Vs[L2]
		twoGrid(2*h, V, R, L2, smooth)

		local v = vs[L]
		clcall2D(L2, L2, expandResidual, v.obj, V.obj)

		clcall1D(L*L, addTo, ffi.new('size_t[1]', L*L), u.obj, v.obj)

		for i=1,smooth do
			clcall2D(L,L, GaussSeidel, u.obj, f.obj, ffi.new('real[1]', h))
		end
	end

	local errorBuf = env:buffer{name='errorBuf'}
	local sumReduce = env:reduce{
		op = function(x,y) return x..' + '..y end,
		buffer = errorBuf.obj,
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

		clcall2D(size, size, calcFrobErr, errorBuf.obj, psi.obj, psiOld.obj)
		frobErr = math.sqrt(sumReduce(errorBuf.obj))

		clcall2D(size, size, calcRelErr, errorBuf.obj, psi.obj, psiOld.obj)
		relErr = sumReduce()
		count()
		local n = sumReduce(countBuf.obj)
		relErr = relErr / n

--printInfo:insert{iter,'rel', relErr, 'frob', frobErr}
--print(iter,'rel', relErr, 'frob', frobErr)
print(iter, relErr, n, frobErr)
		if frobErr < accuracy or not math.isfinite(frobErr) then break end
	end
end

local h = 1/(size+1)
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

-- TODO use events
local startTime = os.clock()
amrsolve(f, h)
local endTime = os.clock()
io.stderr:write('time taken: '..(endTime - startTime)..'\n')
--printInfo:map(function(l) print(table.unpack(l)) end)

