--[[
my attempt at a multigrid solver...
http://www.physics.buffalo.edu/phy411-506-2004/Topic5/topic5-lec3.pdf

solves for u
del^2 u = f

a_ij = 1/h^2 -4/h^2 1/h^2
--]]
local class = require 'ext.class'
local math = require 'ext.math'

-- output all data in a way that I can compare it with the gpu versions
local debugging = true

local matrix = require 'matrix'

local MultigridCPU = class()
	
MultigridCPU.smooth = 7	-- 7 is optimal time for me
MultigridCPU.accuracy = 1e-10

local function GaussSeidel(self, h, u, f)
	local L = #u
	for i=1,L do
		for j=1,L do
			local u_xl = i > 1 and u[i-1][j] or 0--u[i][j]
			local u_xr = i < L and u[i+1][j] or 0--u[i][j]
			local u_yl = j > 1 and u[i][j-1] or 0--u[i][j]
			local u_yr = j < L and u[i][j+1] or 0--u[i][j]
			local askew_u = (u_xl + u_xr + u_yl + u_yr) / h^2
			local adiag = -4 / h^2
			u[i][j] = (f[i][j] - askew_u) / adiag
		end
	end
end

-- in-place Jacobi (using a temp buffer)
local function Jacobi(self, h, u, f)
	local L = #u
	local lastU = matrix(u)
	for i=1,L do
		for j=1,L do
			local u_xl = i > 1 and lastU[i-1][j] or 0--lastU[i][j]
			local u_xr = i < L and lastU[i+1][j] or 0--lastU[i][j]
			local u_yl = j > 1 and lastU[i][j-1] or 0--lastU[i][j]
			local u_yr = j < L and lastU[i][j+1] or 0--lastU[i][j]
			local askew_u = (u_xl + u_xr + u_yl + u_yr) / h^2
			local adiag = -4 / h^2
			u[i][j] = (f[i][j] - askew_u) / adiag
		end
	end
end

--MultigridCPU.inPlaceIterativeSolver = GaussSeidel
MultigridCPU.inPlaceIterativeSolver = Jacobi

local function show(name, m, L)
	if not debugging then return end
	print(name)
	for i=1,L do
		for j=1,L do
			io.write(' ',m[i][j])
		end
		print()
	end
end

function MultigridCPU:twoGrid(h, u, f)
	local size = self.size

	local L = #u
print('L', L)	
	
	if L == 1 then
		-- Gauss-Seidel single-cell update 
		local u_xl, u_xr, u_yl, u_yr
		if border == 'dirichlet' then	-- constant-valued
			u_xl = border_u_xl
			u_xr = border_u_xr
			u_yl = border_u_yl
			u_yr = border_u_yr
		else	-- periodic? neumann (constant-derivative) with deriv=0?
			local tmp = 0--u[1][1]
			u_xl, u_xr, u_yl, u_yr = tmp, tmp, tmp, tmp
		end
		local askew_u = (u_xl + u_xr + u_yl + u_yr) / h^2
		local adiag = -4 / h^2
show('f', f, L)
		u[1][1] = (f[1][1] - askew_u) / adiag 
show('u', u, L)
		return
	end

	for i=1,self.smooth do
if debugging and L==size[1] then
	print('smooth',i)
	print('h', h)
	show('f', f, L)
end
		self:inPlaceIterativeSolver(h, u, f)
--if L==size[1] then 
	show('u', u, L) 
--end
	end

	-- r = f - del^2 u
show('f', f, L)
show('u', u, L)
	local r = matrix.zeros(L,L)
	for i=1,L do
		for j=1,L do
			local u_xl = i > 1 and u[i-1][j] or 0--u[i][j]
			local u_xr = i < L and u[i+1][j] or 0--u[i][j]
			local u_yl = j > 1 and u[i][j-1] or 0--u[i][j]
			local u_yr = j < L and u[i][j+1] or 0--u[i][j]
			local askew_u = (u_xl + u_xr + u_yl + u_yr) / h^2
			local adiag = -4 / h^2
			local a_u = askew_u + adiag * u[i][j]
			r[i][j] = f[i][j] - a_u
		end
	end
show('r', r, L)
	
	-- del^2 V = R
	local L2 = L/2
	local R = matrix.zeros(L2, L2)
	for I=1,L2 do
		local i = 2*I-1
		for J=1,L2 do
			local j=2*J-1
			R[I][J] = (r[i][j] + r[i+1][j] + r[i][j+1] + r[i+1][j+1]) / 4
		end
	end
show('R', R, L2)

	local V = matrix.zeros(L2, L2)
	self:twoGrid(2*h, V, R)
show('V', V, L2)

	local v = matrix.zeros(L, L)
	for I=1,L2 do
		local i=2*I-1
		for J=1,L2 do
			local j=2*J-1
			local value = V[I][J]
			v[i][j], v[i+1][j], v[i][j+1], v[i+1][j+1] = value, value, value, value
		end
	end
show('v', v, L)

	-- correct u
	for i=1,L do
		for j=1,L do
			u[i][j] = u[i][j] + v[i][j]
		end
	end
show('u', u, L)

	for i=1,self.smooth do
		self:inPlaceIterativeSolver(h, u, f)
show('u', u, L)
	end
end

function MultigridCPU:init(L)
	self.size = matrix{L,L}
	local center = self.size/2
	
	self.f = matrix.lambda(self.size, function(...)
		local i = matrix{...}
		
		local charge = 1e+6
		local epsilon0 = 1
		local Q = -charge / epsilon0
		
		return (i - center - {1,1}):normLInf() < 1 and Q or 0
	end)
end

function MultigridCPU:run()
	local L = self.size[1]
	local h = 1/L

	local f = self.f

	--print('#iter', 'relErr', 'n', 'frobErr')
	print('#iter', 'err')
	f = matrix(f)		-- make a copy because we're going to modify in-place
	local psi = -f		-- initialize psi
	for iter=1,2 do--math.huge do
		local psiOld = matrix(psi)
		self:twoGrid(h, psi, f)

		local err = math.sqrt((psi - psiOld):normSq() / psi:size():prod())
print(iter, err)
		if err < self.accuracy or not math.isfinite(err) then break end
	end
	self.psi = psi
end

return MultigridCPU
