#! /usr/bin/env luajit

local ffi = require 'ffi'
require 'ext'
local gcmem = require 'ext.gcmem'

-- output all data in a way that I can compare it with the cpu versions
local debugging = true

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
--print('using real',real)

-- log2 size = 5 diverges for gpu ...
local log2size = ... and tonumber(...) or 5
local size = bit.lshift(1, log2size)
local code = require 'template'([[
#define size <?=size?>
typedef <?=real?> real;

kernel void init(
	global real* f,
	global real* psi
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

//doing this on a GPU doesn't guarantee order ...
//better use Jacobi ...
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

kernel void Jacobi(
	global real* destU,
	const global real* u,
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
	destU[index] = (f[index] - askew_u) / adiag;
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

kernel void square(
	global real* frobBuf,
	const global real* psi
) {
	int i = get_global_id(0);
	int j = get_global_id(1);
	if (i >= size || j >= size) return;
	int index = i + size * j;
	real d = psi[index];
	frobBuf[index] = d * d;
}

]], {
	real = real,
	size = size,
})

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
-- same size as psi, used for jacobi iteration
local tmpU = ctx:buffer{rw=true, size=size*size*ffi.sizeof(real)}

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
local GaussSeidel = program:kernel'GaussSeidel'
local Jacobi = program:kernel'Jacobi'
local calcResidual = program:kernel'calcResidual'
local reduceResidual = program:kernel'reduceResidual'
local expandResidual = program:kernel'expandResidual'
local addTo = program:kernel'addTo'
local calcFrobErr = program:kernel'calcFrobErr'
local calcRelErr = program:kernel'calcRelErr'
local square = program:kernel'square'

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
	local cpuMem = gcmem.new('real', L*L)
	cmds:enqueueReadBuffer{buffer=gpuMem, block=true, size=L*L*ffi.sizeof(real), ptr=cpuMem}
	return cpuMem
end

local function showAndCheck(name, gpuMem, L)
	if not debugging then return end	
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
		
local function inPlaceIterativeSolver(L, u, f, h)
	--[[ Gauss-Seidel
	clcall2D(L,L,GaussSeidel, u, f, ffi.new('real[1]', h))
	--]]
	-- [[ Jacobi
	clcall2D(L,L,Jacobi, tmpU, u, f, ffi.new('real[1]', h))
	cmds:enqueueCopyBuffer{src=tmpU, dst=u, size=L*L*ffi.sizeof(real)}
	--]]

end

local function twoGrid(h, u, f, L, smooth)
print('L', L)	
	if L == 1 then
		--*u = *f / (-4 / h^2)
showAndCheck('f', f, L, L)
		inPlaceIterativeSolver(L, u, f, h)
showAndCheck('u', u, L, L)
		return
	end
	
	for i=1,smooth do
if debugging and L==size then
	print('smooth',i) 
	print('h', h)
	showAndCheck('f', f, L, L) 
end
		inPlaceIterativeSolver(L, u, f, h)
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
		inPlaceIterativeSolver(L, u, f, h)
showAndCheck('u', u, L, L)
	end
end
	
-- doing the error calculation in cpu ...
local errMem = gcmem.new('real', size*size)

do
	clcall2D(size, size, square, errorBuf, f)
	cmds:enqueueReadBuffer{buffer=errorBuf, block=true, size=size*size*ffi.sizeof(real), ptr=errMem}
	local frobNorm = 0
	for j=0,size*size-1 do
		frobNorm = frobNorm + errMem[j]
	end
	frobNorm = math.sqrt(frobNorm)
	--print('|del.E|', frobNorm)	-- this matches cpu.lua
end

-- I am embarrassed to do this
-- I know it's not accurate.
-- I'll add cl event profiling soon
local startTime = os.clock()

local smooth = 7
local h = 1/size
local accuracy = 1e-10
--print('#iter','relErr','n','frobErr')
print('#iter','err')
for iter=1,2 do--math.huge do
	cmds:enqueueCopyBuffer{src=psi, dst=psiOld, size=size*size*ffi.sizeof(real)}
	twoGrid(h, psi, f, size, smooth)

	clcall2D(size, size, calcFrobErr, errorBuf, psi, psiOld)
	cmds:enqueueReadBuffer{buffer=errorBuf, block=true, size=size*size*ffi.sizeof(real), ptr=errMem}
--[[ rel err
	local frobErr = 0
	for j=0,size*size-1 do
		frobErr = frobErr + errMem[j]
	end
	frobErr = math.sqrt(frobErr)
	
	clcall2D(size, size, calcRelErr, errorBuf, psi, psiOld)
	cmds:enqueueReadBuffer{
		buffer = errorBuf,
		block = true,
		size = size * size * ffi.sizeof(real),
		ptr = errMem,
	}
--local psiMem = ffi.new('real[?]', size*size)
--cmds:enqueueReadBuffer{buffer = psi, ptr = psiMem, block = true, size = size * size * ffi.sizeof(real)}
--local psiOldMem = ffi.new('real[?]', size*size)
--cmds:enqueueReadBuffer{buffer = psiOld, ptr = psiOldMem, block = true, size = size * size * ffi.sizeof(real)}
-- looks like after one iteration
-- the CPU goes 10000 -> 1846.015020895
-- the GPU goes 10000 -> 23890.132632578
-- so the iteration algorithm is off
-- probably due to my Gauss-Seidel operating with arbitrary ordering ...
	
	local relErr = 0
	local n = 0
	for j=0,size*size-1 do
		if errMem[j] ~= 0 then
			relErr = relErr + errMem[j]
--print('adding error from i,j',j%size,math.floor(j/size),'psi',psiMem[j],'psiOld',psiOldMem[j],'relErr',math.abs(1 - psiMem[j] / psiOldMem[j]))
			n = n + 1
		end
	end
	relErr = relErr / n

--print(iter,'rel', relErr, 'of n',n,'frob', frobErr)
print(iter, relErr, n, frobErr)
	if frobErr < accuracy or not math.isfinite(frobErr) then break end
--do break end
--]]
-- [[ frob err normalized by size (TODO this on the GPU if possible)
	local err = 0
	for j=0,size*size-1 do
		err = err + errMem[j]
	end
	err = math.sqrt(err / (size * size))
print(iter, err)
	if err < accuracy or not math.isfinite(err) then break end
--]]
end

local endTime = os.clock()
io.stderr:write('time taken: '..(endTime - startTime)..'\n')

gcmem.free(errMem)
