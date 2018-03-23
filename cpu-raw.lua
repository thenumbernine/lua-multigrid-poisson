#! /usr/bin/env luajit

local ffi = require 'ffi'
local image = require 'image'
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

local real = fp64 and 'double' or 'float'
ffi.cdef('typedef '..real..' real;')
--print('using real',real)

-- log2 size = 5 diverges for gpu ...
local log2size = ... and tonumber(...) or 5
local size = bit.lshift(1, log2size)

local function init(L, sy, i, j, f, psi)
	local index = i + L * j
	local center = math.floor(L / 2)
	local value = 0
	if i == center and j == center then
		local charge = 1e+6
		local epsilon0 = 1
		value = -charge / epsilon0
	end
	
	f[index] = value
	psi[index] = -f[index]
end

--doing this on a GPU doesn't guarantee order ...
--better use Jacobi ...
local function GaussSeidel(L,sy,i,j,u,f,h)
	local index = i + L * j
	local u_xl = i > 0 and u[(i-1) + L * j] or 0
	local u_xr = i < L-1 and u[(i+1) + L * j] or 0
	local u_yl = j > 0 and u[i + L * (j-1)] or 0
	local u_yr = j < L-1 and u[i + L * (j+1)] or 0
	local hSq = h * h
	local askew_u = (u_xl + u_xr + u_yl + u_yr) / hSq
	local adiag = -4 / hSq
	u[index] = (f[index] - askew_u) / adiag
end

local function Jacobi(L,sy,i,j,destU,u,f,h)
	local index = i + L * j
	local u_xl = i > 0 and u[(i-1) + L * j] or 0
	local u_xr = i < L-1 and u[(i+1) + L * j] or 0
	local u_yl = j > 0 and u[i + L * (j-1)] or 0
	local u_yr = j < L-1 and u[i + L * (j+1)] or 0
	local hSq = h * h
	local askew_u = (u_xl + u_xr + u_yl + u_yr) / hSq
	local adiag = -4 / hSq
	destU[index] = (f[index] - askew_u) / adiag
end

local function calcResidual(L,sy,i,j,r,f,u,h)
	local index = i + L * j
	local u_xl = i > 0 and u[(i-1) + L * j] or 0
	local u_xr = i < L-1 and u[(i+1) + L * j] or 0
	local u_yl = j > 0 and u[i + L * (j-1)] or 0
	local u_yr = j < L-1 and u[i + L * (j+1)] or 0
	local hSq = h * h
	local askew_u = (u_xl + u_xr + u_yl + u_yr) / hSq
	local adiag = -4. / hSq
	local a_u = askew_u + adiag * u[i + L * j]
	r[index] = f[index] - a_u
end

local function reduceResidual(L2,sy,I,J,R,r)
	local L = bit.lshift(L2, 1)
	local srci = bit.lshift(I,1) + L * bit.lshift(J,1)
	R[I + L2 * J] = .25 * (r[srci] + r[srci+1] + r[srci+L] + r[srci+L+1])
end

local function expandResidual(L2,sy,I,J,v,V)
	local L = bit.lshift(L2, 1)
	local dsti = bit.lshift(I,1) + L * bit.lshift(J,1)
	local src = V[I + L2 * J]
	v[dsti] = src
	v[dsti+1] = src
	v[dsti+L] = src
	v[dsti+L+1] = src
--[[ L-sized kernel
	int L = get_global_size(0);
	int i = get_global_id(0);
	int j = get_global_id(1);
	if (i >= L || j >= L) return;
	int L2 = L >> 1;
	int I = i >> 1;
	int J = j >> 1;
	v[i + L * j] = V[I + L2 * J];
--]]
end

local function addTo(sx,i,n,u,v)
	u[i] = u[i] + v[i]
end

local function calcRelErr(sx,sy,i,j,errorBuf,psi,psiOld)
	local index = i + size * j
	if psiOld[index] ~= 0 and psiOld[index] ~= psi[index] then
		errorBuf[index] = math.fabs(1. - psi[index] / psiOld[index])
	else
		errorBuf[index] = 0
	end
end

local function calcFrobErr(sx,sy,i,j,errorBuf,psi,psiOld)
	local index = i + size * j
	local d = psi[index] - psiOld[index]
	errorBuf[index] = d * d
end

local function square(sx,sy,i,j,frobBuf,psi)
	local index = i + size * j
	local d = psi[index]
	frobBuf[index] = d * d
end

ffi.cdef('typedef '..real..' real;')

local f = image(size,size,1,'real')
local psi = image(size,size,1,'real')
local psiOld = image(size,size,1,'real')
local errorBuf = image(size,size,1,'real')
-- same size as psi, used for jacobi iteration
local tmpU = image(size,size,1,'real')

local rs = {}
local Rs = {}
local vs = {}
local Vs = {}
for i=0,math.log(size,2) do
	local L = 2^i
	rs[L] = image(L, L, 1, 'real')
	Rs[L] = image(L, L, 1, 'real')
	vs[L] = image(L, L, 1, 'real')
	Vs[L] = image(L, L, 1, 'real')
	for _,buf in ipairs{rs, Rs, vs, Vs} do
		buf = buf[L].buffer
		for i=0,L*L-1 do
			buf[i] = 0
		end
	end
end

local function clcall1D(w, kernel, ...)
	for i=0,w-1 do
		kernel(w, i, ...)
	end
end

local function clcall2D(w,h, kernel, ...)
	for j=0,h-1 do
		for i=0,w-1 do
			kernel(w, h, i, j, ...)
		end
	end
end

clcall2D(size,size, init, f.buffer, psi.buffer)

local function showAndCheck(name, im, L)
	if not debugging then return end	
	print(name)
	for i=0,L-1 do
		for j=0,L-1 do
			io.write(' ',im[j+L*i])
		end
		print()
	end
	for i=0,L*L-1 do
		if not math.isfinite(im[i]) then
			error("found a nan")
		end
	end
end
		
local function inPlaceIterativeSolver(L, u, f, h)
	--[[ Gauss-Seidel
	clcall2D(L,L,GaussSeidel, u, f, h)
	--]]
	-- [[ Jacobi
	clcall2D(L,L,Jacobi, tmpU.buffer, u, f, h)
	ffi.copy(u, tmpU.buffer, L*L*ffi.sizeof(real))
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
	
	local r = rs[L].buffer
showAndCheck('f', f, L, L)
showAndCheck('u', u, L, L)
	clcall2D(L,L, calcResidual, r, f, u, h)
showAndCheck('r', r, L, L)
	
	--r = f - a(u)

	local L2 = L/2
	local R = Rs[L2].buffer
	clcall2D(L2,L2, reduceResidual, R, r)
showAndCheck('R', R, L2, L2)
	
	local V = Vs[L2].buffer
	twoGrid(2*h, V, R, L2, smooth)
showAndCheck('V', V, L2, L2)

	local v = vs[L].buffer
	clcall2D(L2, L2, expandResidual, v, V)
	--clcall2D(L, L, expandResidual, v, V)
showAndCheck('v', v, L, L)

	clcall1D(L*L, addTo, L*L, u, v)
showAndCheck('u', u, L, L)

	for i=1,smooth do
		inPlaceIterativeSolver(L, u, f, h)
showAndCheck('u', u, L, L)
	end
end
	
do
	clcall2D(size, size, square, errorBuf.buffer, f.buffer)
	local frobNorm = 0
	for j=0,size*size-1 do
		frobNorm = frobNorm + errorBuf.buffer[j]
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
	ffi.copy(psiOld.buffer, psi.buffer, size*size*ffi.sizeof(real))
	twoGrid(h, psi.buffer, f.buffer, size, smooth)

	clcall2D(size, size, calcFrobErr, errorBuf.buffer, psi.buffer, psiOld.buffer)
	local err = 0
	for j=0,size*size-1 do
		err = err + errorBuf.buffer[j]
	end
	err = math.sqrt(err / (size * size))
print(iter, err)
	if err < accuracy or not math.isfinite(err) then break end
end

local endTime = os.clock()
io.stderr:write('time taken: '..(endTime - startTime)..'\n')
