#!/usr/bin/env luajit
require 'ext'
local bit = require 'bit'

--[[
my attempt at a multigrid solver...
http://www.physics.buffalo.edu/phy411-506-2004/Topic5/topic5-lec3.pdf

solves for u
del^2 u = f

a_ij = 1/h^2 -4/h^2 1/h^2
--]]

-- output all data in a way that I can compare it with the gpu versions
local debugging = true

local matrix = require 'matrix'

local function time(f, ...)
	local t1 = os.clock()
	f(...)
	local t2 = os.clock()
	return t2 - t1
end

local function GaussSeidel(h, u, f)
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
local function Jacobi(h, u, f)
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

--local inPlaceIterativeSolver = GaussSeidel
local inPlaceIterativeSolver = Jacobi

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

local function twoGrid(h, u, f, smooth)
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

	for i=1,smooth do
if debugging and L==size[1] then
	print('smooth',i)
	print('h', h)
	show('f', f, L)
end
		inPlaceIterativeSolver(h, u, f)
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
	twoGrid(2*h, V, R, smooth)
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

	for i=1,smooth do
		inPlaceIterativeSolver(h, u, f)
show('u', u, L)
	end
end

-- hmm, do implementations really use this function?
-- I'm going to use a different error function
local function relativeError(psi, psiOld)
	local L = #psi - 2
	local err = 0
	local n = 0
	for i=2,L+1 do
		for j=2,L+1 do
			if psiOld[i][j] ~= 0 then
				if psiOld[i][j] ~= psi[i][j] then
--print('adding error from i,j',i,j,'psi',psi[i][j],'psiOld',psiOld[i][j],'relErr',math.abs(1 - psi[i][j] / psiOld[i][j]))
					err = err + math.abs(1 - psi[i][j] / psiOld[i][j])
					n = n + 1
				end
			end
		end
	end
	if n > 0 then err = err / n end
	return err, n
end


--print('#iter', 'relErr', 'n', 'frobErr')
print('#iter', 'err')
--local printInfo = table()
local function amrsolve(f, h)
	f = matrix(f)		-- make a copy because we're going to modify in-place
	local smooth = 7	-- 7 is optimal time for me
	local accuracy = 1e-10
	local psi = -f		-- initialize psi
	for iter=1,2 do--math.huge do
		local psiOld = matrix(psi)
		twoGrid(h, psi, f, smooth)
		

--[[
		-- compute the error between psi and psiOld
		local frobErr = (psi - psiOld):norm()
		local relErr, n = relativeError(psi, psiOld)
--printInfo:insert{iter,'rel', relErr, 'of n',n,'frob', frobErr}
--print(iter,'rel', relErr, 'of n',n,'frob', frobErr)
print(iter, relErr, n, frobErr)
--do break end
		if frobErr < accuracy or not math.isfinite(frobErr) then break end
--]]
-- [[	use error scaled by size
		local err = math.sqrt((psi - psiOld):normSq() / psi:size():prod())
print(iter, err)
		if err < accuracy or not math.isfinite(err) then break end
--]]
	end
	return psi
end

-- m is a matrix of [M][N][2]
-- returns a matrix of [M][N]
-- boundary condition is constant
local function div(m)
	return matrix.lambda({#m, #m[1]}, function(...)
		local ix, iy = ...
		local ixp = math.min(ix+1, #m)
		local ixn = math.max(ix-1, 1)
		local iyp = math.min(iy+1, #m[1])
		local iyn = math.max(iy-1, 1)
		return (m[ixp][iy][1] - m[ixn][iy][1]) / (ixp - ixn) 
			+ (m[ix][iyp][2] - m[ix][iyn][2]) / (iyp - iyn)
	end)
end

-- psi is a matrix of [M][N]
-- return a matrix of [M][N][2]
-- boundary condition is constant
local function del(psi)
	return matrix.lambda({#psi, #psi[1]}, function(...)
		local ix, iy, z = ...
		local ixp = math.min(ix+1, #psi)
		local ixn = math.max(ix-1, 1)
		local iyp = math.min(iy+1, #psi[1])
		local iyn = math.max(iy-1, 1)
		return matrix{
			(psi[ixp][iy] - psi[ixn][iy]) / (ixp - ixn),
			(psi[ix][iyp] - psi[ix][iyn]) / (iyp - iyn),
		}
	end)
end

local log2L = ... and tonumber(...) or 5
local L = bit.lshift(1,log2L)
local h = 1 / L
size = matrix{L,L}
local center = size/2

--[[ using a real vector field
-- E = del psi + del x B
local E = matrix.zeros(L,L,2)
for i=1,L do
	for j=1,L do
		local x = (matrix{i+.5,j+.5} - center) * h
		E[i][j] = -x / x:norm()
	end
end
-- f = del.E = del^2 psi
local f = div(E) / h
--]]

-- [=[ manual specify f
local f = matrix.lambda(size, function(...)
	local i = matrix{...}
	
	-- [[ electricity: del^2 V = -rho / epsilon_0
	local charge = 1e+6
	local epsilon0 = 1
	local Q = -charge / epsilon0
	--]]
	
	--[[ gravity: del^2 V = 4 pi G rho
	local mass = 1e+6
	local G = 1
	local Q = 4 * math.pi * G * mass 
	--]]

	return (i - center - {1,1}):normLInf() < 1 and Q or 0
end)
--]=]
--print('|del.E|',f:norm())

-- psi = del^-2 f
local startTime = os.clock()
local psi = amrsolve(f, h)
local endTime = os.clock()
--printInfo:map(function(l) print(table.unpack(l)) end)
io.stderr:write('time taken: '..(endTime - startTime)..'\n')

--print('|del^-1.E|', psi:norm())

--[[
-- curlfree E = del psi = del
local curlfree_E = del(psi) / h
print('|E - curlfree(E)|', (E - curlfree_E):norm())

local curlfree_f = div(curlfree_E) / h
print('|del . curlfree(E)|', curlfree_f:norm())

local curlfree_psi = amrsolve(curlfree_f, h)
print('|del^-1 . curlfree(E)|', curlfree_psi:norm())

local curlfree2_E = del(curlfree_psi) / h
print('|curlfree(E) - curlfree^2(E)|', (curlfree_E - curlfree2_E):norm())
--]]

--[[
local _ = require 'matrix.index'
require 'gnuplot'{
	persist = true,
	style = 'data lines',
	--log = 'z',
	griddata={
		x=range(size[1]),
		y=range(size[2]),
		f,
		psi,
		E(_,_,1),
		E(_,_,2),
		curlfree_E(_,_,1),
		curlfree_E(_,_,2),
	},
	--{splot=true, using='1:2:3', title='f'},
	--{splot=true, using='1:2:4', title='psi'},
	
	-- maximum magnitude is 1
	--{splot=true, using='1:2:5', title='Ex'},
	{splot=true, using='1:2:6', title='Ey'},
	--{splot=true, using='1:2:5:6', 'with vectors', title='E'},
	
	-- this should be on the order of E as well, but instead it's 1e-7
	--{splot=true, using='1:2:7', title='curlfree-Ex'},
	{splot=true, using='1:2:8', title='curlfree-Ey'},
	--{splot=true, using='1:2:7:8', 'with vectors', title='curlfree-E'},
}
--]]