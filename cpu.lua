--[[
my attempt at a multigrid solver...
http://www.physics.buffalo.edu/phy411-506-2004/Topic5/topic5-lec3.pdf

solves for u
del^2 u = f

a_ij = 1/h^2 -4/h^2 1/h^2
--]]
local class = require 'ext.class'
local math = require 'ext.math'

local matrix = require 'matrix'

local MultigridCPU = class()

-- output all data in a way that I can compare it with the gpu versions
MultigridCPU.debug = false

MultigridCPU.smooth = 7	-- 7 is optimal time for me
MultigridCPU.epsilon = 1e-10
MultigridCPU.maxiter = 1000

local function GaussSeidel(self, h, u, f)
	local width = #u
	for i=1,width do
		for j=1,width do
			local u_xl = i > 1 and u[i-1][j] or 0--u[i][j]
			local u_xr = i < width and u[i+1][j] or 0--u[i][j]
			local u_yl = j > 1 and u[i][j-1] or 0--u[i][j]
			local u_yr = j < width and u[i][j+1] or 0--u[i][j]
			local askew_u = (u_xl + u_xr + u_yl + u_yr) / h^2
			local adiag = -4 / h^2
			u[i][j] = (f[i][j] - askew_u) / adiag
		end
	end
end

-- in-place Jacobi (using a temp buffer)
local function Jacobi(self, h, u, f)
	local width = #u
	local lastU = matrix(u)
	for i=1,width do
		for j=1,width do
			local u_xl = i > 1 and lastU[i-1][j] or 0--lastU[i][j]
			local u_xr = i < width and lastU[i+1][j] or 0--lastU[i][j]
			local u_yl = j > 1 and lastU[i][j-1] or 0--lastU[i][j]
			local u_yr = j < width and lastU[i][j+1] or 0--lastU[i][j]
			local askew_u = (u_xl + u_xr + u_yl + u_yr) / h^2
			local adiag = -4 / h^2
			u[i][j] = (f[i][j] - askew_u) / adiag
		end
	end
end

--MultigridCPU.inPlaceIterativeSolver = GaussSeidel
MultigridCPU.inPlaceIterativeSolver = Jacobi

function MultigridCPU:show(name, m, width)
	if not self.debug then return end
	print(name)
	for i=1,width do
		for j=1,width do
			io.write(' ',m[i][j])
		end
		print()
	end
end

function MultigridCPU:twoGrid(h, u, f)
	local size = self.size

	local width = #u
if self.debug then print('L', width) end
	
	if width == 1 then
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
self:show('f', f, width)
		u[1][1] = (f[1][1] - askew_u) / adiag 
self:show('u', u, width)
		return
	end

	for i=1,self.smooth do
if self.debug and width==size[1] then
	print('smooth',i)
	print('h', h)
	self:show('f', f, width)
end
		self:inPlaceIterativeSolver(h, u, f)
--if L==size[1] then 
	self:show('u', u, width) 
--end
	end

	-- r = f - del^2 u
self:show('f', f, width)
self:show('u', u, width)
	local r = matrix.zeros(width,width)
	for i=1,width do
		for j=1,width do
			local u_xl = i > 1 and u[i-1][j] or 0--u[i][j]
			local u_xr = i < width and u[i+1][j] or 0--u[i][j]
			local u_yl = j > 1 and u[i][j-1] or 0--u[i][j]
			local u_yr = j < width and u[i][j+1] or 0--u[i][j]
			local askew_u = (u_xl + u_xr + u_yl + u_yr) / h^2
			local adiag = -4 / h^2
			local a_u = askew_u + adiag * u[i][j]
			r[i][j] = f[i][j] - a_u
		end
	end
self:show('r', r, width)
	
	-- del^2 V = R
	local halfWidth = width/2
	local R = matrix.zeros(halfWidth, halfWidth)
	for I=1,halfWidth do
		local i = 2*I-1
		for J=1,halfWidth do
			local j=2*J-1
			R[I][J] = (r[i][j] + r[i+1][j] + r[i][j+1] + r[i+1][j+1]) / 4
		end
	end
self:show('R', R, halfWidth)

	local V = matrix.zeros(halfWidth, halfWidth)
	self:twoGrid(2*h, V, R)
self:show('V', V, halfWidth)

	local v = matrix.zeros(width, width)
	for I=1,halfWidth do
		local i=2*I-1
		for J=1,halfWidth do
			local j=2*J-1
			local value = V[I][J]
			v[i][j], v[i+1][j], v[i][j+1], v[i+1][j+1] = value, value, value, value
		end
	end
self:show('v', v, width)

	-- correct u
	for i=1,width do
		for j=1,width do
			u[i][j] = u[i][j] + v[i][j]
		end
	end
self:show('u', u, width)

	for i=1,self.smooth do
		self:inPlaceIterativeSolver(h, u, f)
self:show('u', u, width)
	end
end

--[[
args:
	size
	maxiter
	epsilon
--]]
function MultigridCPU:init(args)
	self.maxiter = args.maxiter
	self.epsilon = args.epsilon
	self.errorCallback = args.errorCallback
	if args.debug ~= nil then self.debug = args.debug end
	self.size = matrix{args.size,args.size}
	
	local center = self.size/2
	
	self.f = matrix.lambda(self.size, function(...)
		local i = matrix{...}
		
		local charge = 1e+6
		local epsilon0 = 1
		local Q = -charge / epsilon0
		
		return (i - center - {1,1}):normLInf() < 1 and Q or 0
	end)
	
	-- initialize psi
	self.psi = -self.f
end

function MultigridCPU:step()
	local width = self.size[1]
	local h = 1/width
	
	local psiOld = matrix(self.psi)
	self:twoGrid(h, self.psi, matrix(self.f))

	local err = math.sqrt((self.psi - psiOld):normSq() / self.psi:size():prod())
if self.debug then print(iter, 'err', err) end
	return err
end

function MultigridCPU:solve()
	--print('#iter', 'relErr', 'n', 'frobErr')
if self.debug then print('#iter', 'err') end
	for iter=1,self.maxiter do
		local err = self:step()	
		if self.errorCallback and self.errorCallback(iter, err) then break end
		if err < self.epsilon or not math.isfinite(err) then break end
	end
end

return MultigridCPU
