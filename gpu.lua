local ffi = require 'ffi'
local class = require 'ext.class'
local math = require 'ext.math'
local string = require 'ext.string'


local function get64bit(list)
	local best = list:map(function(item)
		local exts = item:getExtensions():map(string.lower)
		return {item=item, fp64=exts:find(function(s) return s:match'cl_%w+_fp64' end)}
	end):sort(function(a,b)
		return (a.fp64 and 1 or 0) > (b.fp64 and 1 or 0)
	end)[1]
	return best.item, best.fp64
end


local MultigridGPU = class()

-- output all data in a way that I can compare it with the cpu versions
MultigridGPU.debugging = false

MultigridGPU.smooth = 7
MultigridGPU.accuracy = 1e-10

function MultigridGPU:init(size)
	self.platform = get64bit(require 'cl.platform'.getAll())
	self.device, self.fp64 = get64bit(self.platform:getDevices{gpu=true})
	self.ctx = require 'cl.context'{platform=self.platform, device=self.device}
	self.cmds = require 'cl.commandqueue'{context=self.ctx, device=self.device}

	self.real = self.fp64 and 'double' or 'float'
	--print('using real',self.real)

	self.size = size

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
	unsigned int n,
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

]], self)

	if self.fp64 then
		code = '#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n' .. code 
	end

	self.program = require 'cl.program'{context=self.ctx, devices={self.device}, code=code}

	local maxWorkGroupSize = tonumber(self.device:getInfo'CL_DEVICE_MAX_WORK_GROUP_SIZE')
	self.maxWorkGroupSize = maxWorkGroupSize 

	ffi.cdef('typedef '..self.real..' real;')

	self.f = self.ctx:buffer{rw=true, size=size*size*ffi.sizeof(self.real)}
	self.psi = self.ctx:buffer{rw=true, size=size*size*ffi.sizeof(self.real)}
	self.psiOld = self.ctx:buffer{rw=true, size=size*size*ffi.sizeof(self.real)}
	self.errorBuf = self.ctx:buffer{rw=true, size=size*size*ffi.sizeof(self.real)}
	-- same size as psi, used for jacobi iteration
	self.tmpU = self.ctx:buffer{rw=true, size=size*size*ffi.sizeof(self.real)}

	self.rs = {}
	self.Rs = {}
	self.vs = {}
	self.Vs = {}
	for i=0,math.round(math.log(size,2)) do
		local L = bit.lshift(1,i)
		for _,v in ipairs{'rs', 'Rs', 'vs', 'Vs'} do
			self[v][L] = self.ctx:buffer{rw=true, size=L*L*ffi.sizeof(self.real)}
			self.cmds:enqueueFillBuffer{buffer=self[v][L], size=L*L*ffi.sizeof(self.real)}
		end
	end

	self.initKernel = self.program:kernel'init'
	self.GaussSeidelKernel = self.program:kernel'GaussSeidel'
	self.JacobiKernel = self.program:kernel'Jacobi'
	self.calcResidualKernel = self.program:kernel'calcResidual'
	self.reduceResidualKernel = self.program:kernel'reduceResidual'
	self.expandResidualKernel = self.program:kernel'expandResidual'
	self.addToKernel = self.program:kernel'addTo'
	self.calcFrobErrKernel = self.program:kernel'calcFrobErr'
	self.calcRelErr = self.program:kernel'calcRelErr'

	self:clcall2D(size,size, self.initKernel, self.f, self.psi)
end


function MultigridGPU:clcall1D(w, kernel, ...)
	kernel:setArgs(...)
	local l = math.min(w, self.maxWorkGroupSize)
	self.cmds:enqueueNDRangeKernel{kernel=kernel, globalSize=w, localSize=l}
end

function MultigridGPU:clcall2D(w,h, kernel, ...)
	kernel:setArgs(...)
	local mw = math.ceil(math.sqrt(self.maxWorkGroupSize))
	local mh = self.maxWorkGroupSize / mw
	local l1 = math.min(w, mw)
	local l2 = math.min(h, mh)
	self.cmds:enqueueNDRangeKernel{kernel=kernel, globalSize={w,h}, localSize={l1,l2}}
end

function MultigridGPU:getbuffer(gpuMem, L)
	local cpuMem = ffi.new(self.real..'[?]', L*L)
	self.cmds:enqueueReadBuffer{buffer=gpuMem, block=true, size=L*L*ffi.sizeof(self.real), ptr=cpuMem}
	return cpuMem
end

function MultigridGPU:showAndCheck(name, gpuMem, L)
	if not self.debugging then return end	
	local cpuMem = self:getbuffer(gpuMem, L)
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
end
		
function MultigridGPU:inPlaceIterativeSolver(L, u, f, h)
	--[[ Gauss-Seidel
	self:clcall2D(L,L,GaussSeidel, u, f, ffi.new(self.real..'[1]', h))
	--]]
	-- [[ Jacobi
	self:clcall2D(L,L,self.JacobiKernel, self.tmpU, u, f, ffi.new(self.real..'[1]', h))
	self.cmds:enqueueCopyBuffer{src=self.tmpU, dst=u, size=L*L*ffi.sizeof(self.real)}
	--]]
end

function MultigridGPU:twoGrid(h, u, f, L)
print('L', L)	
	if L == 1 then
		--*u = *f / (-4 / h^2)
self:showAndCheck('f', f, L, L)
		self:inPlaceIterativeSolver(L, u, f, h)
self:showAndCheck('u', u, L, L)
		return
	end
	
	for i=1,self.smooth do
if self.debugging and L==size then
	print('smooth',i) 
	print('h', h)
	self:showAndCheck('f', f, L, L) 
end
		self:inPlaceIterativeSolver(L, u, f, h)
self:showAndCheck('u', u, L, L)
	end
	
	local r = self.rs[L]
self:showAndCheck('f', f, L, L)
self:showAndCheck('u', u, L, L)
	--self.cmds:enqueueFillBuffer{buffer=r, size=L*L*ffi.sizeof(self.real)}
	self:clcall2D(L,L, self.calcResidualKernel, r, f, u, ffi.new(self.real..'[1]', h))
self:showAndCheck('r', r, L, L)
	
	--r = f - a(u)

	local L2 = L/2
	local R = self.Rs[L2]
	self:clcall2D(L2,L2, self.reduceResidualKernel, R, r)
self:showAndCheck('R', R, L2, L2)
	
	local V = self.Vs[L2]
	self:twoGrid(2*h, V, R, L2)
self:showAndCheck('V', V, L2, L2)

	local v = self.vs[L]
	self:clcall2D(L2, L2, self.expandResidualKernel, v, V)
	--self:clcall2D(L, L, self.expandResidualKernel, v, V)
self:showAndCheck('v', v, L, L)

	self:clcall1D(L*L, self.addToKernel, ffi.new('unsigned int[1]', L*L), u, v)
self:showAndCheck('u', u, L, L)

	for i=1,self.smooth do
		self:inPlaceIterativeSolver(L, u, f, h)
self:showAndCheck('u', u, L, L)
	end
end

function MultigridGPU:run()
	local size = self.size

	-- doing the error calculation in cpu ...
	local errMem = ffi.new(self.real..'[?]', size*size)

	local h = 1/size
	--print('#iter','relErr','n','frobErr')
print('#iter','err')
	for iter=1,2 do--math.huge do
		self.cmds:enqueueCopyBuffer{src=self.psi, dst=self.psiOld, size=size*size*ffi.sizeof(self.real)}
		self:twoGrid(h, self.psi, self.f, size)

		self:clcall2D(size, size, self.calcFrobErrKernel, self.errorBuf, self.psi, self.psiOld)
		self.cmds:enqueueReadBuffer{buffer=self.errorBuf, block=true, size=size*size*ffi.sizeof(self.real), ptr=errMem}
		
		-- frob err normalized by size (TODO this on the GPU if possible)
		local err = 0
		for j=0,size*size-1 do
			err = err + errMem[j]
		end
		err = math.sqrt(err / (size * size))
print(iter, err)
		if err < self.accuracy or not math.isfinite(err) then break end
	end
end

return MultigridGPU
