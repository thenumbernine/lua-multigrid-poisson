local ffi = require 'ffi'
local bit = require 'bit' or bit32
local image = require 'image'
local class = require 'ext.class'
local math = require 'ext.math'

-- output all data in a way that I can compare it with the cpu versions
local debugging = true

local function initCells(L, sy, i, j, f, psi)
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
end
--[[ L-sized kernel
local function expandResidual(L,sy,i,j,v,V)
	local L2 = bit.rshift(L,1)
	local I = bit.rshift(i,1)
	local J = bit.rshift(j,1)
	v[i + L * j] = V[I + L2 * J]
end
--]]

local function addTo(sx,i,n,u,v)
	u[i] = u[i] + v[i]
end

local function calcRelErr(sx,sy,i,j,errorBuf,psi,psiOld)
	local index = i + sx * j
	if psiOld[index] ~= 0 and psiOld[index] ~= psi[index] then
		errorBuf[index] = math.fabs(1. - psi[index] / psiOld[index])
	else
		errorBuf[index] = 0
	end
end

local function calcFrobErr(sx,sy,i,j,errorBuf,psi,psiOld)
	local index = i + sx * j
	local d = psi[index] - psiOld[index]
	errorBuf[index] = d * d
end

local function call1D(w, kernel, ...)
	for i=0,w-1 do
		kernel(w, i, ...)
	end
end

local function call2D(w,h, kernel, ...)
	for j=0,h-1 do
		for i=0,w-1 do
			kernel(w, h, i, j, ...)
		end
	end
end

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



local MultigridCPURaw = class()

function MultigridCPURaw:init(size, real)
	self.real = real or 'double'
	--print('using real',real)

	self.size = size

	self.f = image(size,size,1,self.real)
	self.psi = image(size,size,1,self.real)
	self.psiOld = image(size,size,1,self.real)
	self.errorBuf = image(size,size,1,self.real)
	-- same size as psi, used for jacobi iteration
	self.tmpU = image(size,size,1,self.real)

	self.rs = {}
	self.Rs = {}
	self.vs = {}
	self.Vs = {}
	for i=0,math.log(size,2) do
		local L = bit.lshift(1,i)
		self.rs[L] = image(L, L, 1, self.real)
		self.Rs[L] = image(L, L, 1, self.real)
		self.vs[L] = image(L, L, 1, self.real)
		self.Vs[L] = image(L, L, 1, self.real)
		for _,buf in ipairs{'rs', 'Rs', 'vs', 'Vs'} do
			buf = self[buf][L].buffer
			for i=0,L*L-1 do
				buf[i] = 0
			end
		end
	end

	call2D(size,size, initCells, self.f.buffer, self.psi.buffer)
end

function MultigridCPURaw:inPlaceIterativeSolver(L, u, f, h)
	--[[ Gauss-Seidel
	call2D(L,L,GaussSeidel, u, f, h)
	--]]
	-- [[ Jacobi
	call2D(L,L,Jacobi, self.tmpU.buffer, u, f, h)
	ffi.copy(u, self.tmpU.buffer, L*L*ffi.sizeof(self.real))
	--]]
end

function MultigridCPURaw:twoGrid(h, u, f, L, smooth)
	local real = self.real

print('L', L)	
	if L == 1 then
		--*u = *f / (-4 / h^2)
showAndCheck('f', f, L, L)
		self:inPlaceIterativeSolver(L, u, f, h, real)
showAndCheck('u', u, L, L)
		return
	end
	
	for i=1,smooth do
if debugging and L==self.size then
	print('smooth',i) 
	print('h', h)
	showAndCheck('f', f, L, L) 
end
		self:inPlaceIterativeSolver(L, u, f, h, real)
showAndCheck('u', u, L, L)
	end
	
	local r = self.rs[L].buffer
showAndCheck('f', f, L, L)
showAndCheck('u', u, L, L)
	call2D(L,L, calcResidual, r, f, u, h)
showAndCheck('r', r, L, L)
	
	--r = f - a(u)

	local L2 = L/2
	local R = self.Rs[L2].buffer
	call2D(L2,L2, reduceResidual, R, r)
showAndCheck('R', R, L2, L2)
	
	local V = self.Vs[L2].buffer
	self:twoGrid(2*h, V, R, L2, smooth)
showAndCheck('V', V, L2, L2)

	local v = self.vs[L].buffer
	call2D(L2, L2, expandResidual, v, V)
	--call2D(L, L, expandResidual, v, V)
showAndCheck('v', v, L, L)

	call1D(L*L, addTo, L*L, u, v)
showAndCheck('u', u, L, L)

	for i=1,smooth do
		self:inPlaceIterativeSolver(L, u, f, h, real)
showAndCheck('u', u, L, L)
	end
end

function MultigridCPURaw:run()
	local size = self.size

	local smooth = 7
	local h = 1/size
	local accuracy = 1e-10
	--print('#iter','relErr','n','frobErr')
	print('#iter','err')
	for iter=1,2 do--math.huge do
		ffi.copy(self.psiOld.buffer, self.psi.buffer, size*size*ffi.sizeof(self.real))
		self:twoGrid(h, self.psi.buffer, self.f.buffer, size, smooth)

		call2D(size, size, calcFrobErr, self.errorBuf.buffer, self.psi.buffer, self.psiOld.buffer)
		local err = 0
		for j=0,size*size-1 do
			err = err + self.errorBuf.buffer[j]
		end
		err = math.sqrt(err / (size * size))
print(iter, err)
		if err < accuracy or not math.isfinite(err) then break end
	end
end
