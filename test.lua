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

local matrix = require 'matrix'

local function GaussSeidel(h, u, f)
	local L = #u - 2
	for i=2,L+1 do
		for j=2,L+1 do
			local askew_u = (u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1]) / h^2
			local adiag = -4 / h^2
			u[i][j] = (f[i][j] - askew_u) / adiag
		end
	end
end

local accuracy = 1e-10
local nSmooth = 3
local function twoGrid(h, u, f)
	local L = #u - 2
	
	if L == 1 then
		-- Gauss-Seidel single-cell update 
		local askew_u = (u[1][2] + u[3][2] + u[2][1] + u[2][3]) / h^2
		local adiag = -4 / h^2
		u[2][2] = (f[2][2] - askew_u) / adiag 
		return
	end

	local nPreSmooth = nSmooth
	for i=1,nPreSmooth do
		GaussSeidel(h, u, f)
	end

	-- r = del^2 u + f
	local r = matrix.zeros(L+2,L+2)
	for i=2,L+1 do
		for j=2,L+1 do
			local askew_u = (u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1]) / h^2
			local adiag = -4 / h^2
			r[i][j] = f[i][j] - (askew_u + adiag * u[i][j])
		end
	end
	
	-- del^2 V = R
	local L2 = L/2
	local R = matrix.zeros(L2+2, L2+2)
	for I=2,L2+1 do
		local i = 2*I-2
		for J=2,L2+1 do
			local j=2*J-2
			R[I][J] = (r[i][j] + r[i+1][j] + r[i][j+1] + r[i+1][j+1]) / 4
		end
	end

	local V = matrix.zeros(L2+2, L2+2)
	twoGrid(2*h, V, R)

	local v = matrix.zeros(L+2, L+2)
	for I=2,L2+1 do
		local i=2*I-2
		for J=2,L2+1 do
			local j=2*J-2
			local value = V[I][J]
			v[i][j], v[i+1][j], v[i][j+1], v[i+1][j+1] = value, value, value, value
		end
	end

	-- correct u
	for i=2,L+1 do
		for j=2,L+1 do
			u[i][j] = u[i][j] + v[i][j]
		end
	end

	local postSmooth = nSmooth
	for i=1,postSmooth do
		GaussSeidel(h, u, f)
	end
end

local function relativeError(psi, psiNew)
	local L = #psi - 2
	local err = 0
	local n = 0
	for i=2,L+1 do
		for j=2,L+1 do
			if psiNew[i][j] ~= 0 then
				if psiNew[i][j] ~= psi[i][j] then
					err = err + math.abs(1 - psi[i][j] / psiNew[i][j])
					n = n + 1
				end
			end
		end
	end
	if n > 0 then err = err / n end
	return err
end



local log2L = ... and tonumber(...) or 6
local L = bit.lshift(1,log2L)
local h = 1 / (L + 1)
local size = matrix{L+2,L+2}
local center = (size+1)/2

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
	
	return (i - center):normLInf() < 1 and Q or 0
end)
local psi = -f

for iter=1,math.huge do
	local psiNew = matrix(psi)
	twoGrid(h, psi, f)
	local relErr = relativeError(psi, psiNew)
	print(iter,'rel err',relErr)
	if relErr < accuracy then break end
end

require 'gnuplot'{
	persist = true,
	style = 'data lines',
	--log = 'z',
	griddata={
		x=range(size[1]),
		y=range(size[2]),
		f,
		psi,
	},
	--{splot=true, using='1:2:3', title='f'},
	{splot=true, using='1:2:4', title='psi'},
}
