#!/usr/bin/env luajit
-- look at cpu convergence
-- maybe compare it to the convergence of conj res
local matrix = require 'matrix'
local table = require 'ext.table'
local math = require 'ext.math'

local MultigridCPU = require 'multigrid-poisson.cpu'

local size = 32

local data = table()

local mg = MultigridCPU{
	size = size,
	errorCallback = function(iter, err)
		data[iter] = {err}
	end,
}
mg:solve()

local div = require 'matrix.div'
local conjgrad = require 'solver.conjgrad'
conjgrad{
	clone = matrix,
	dot = matrix.dot,
	x = -matrix(mg.f),
	b = matrix(mg.f),
	-- div x = phi
	A = function(u)
		local width = #u
		local h = 1 / width
		return u:size():lambda(function(i,j)
			local u_xl = i > 1 and u[i-1][j] or 0
			local u_xr = i < width and u[i+1][j] or 0
			local u_yl = j > 1 and u[i][j-1] or 0
			local u_yr = j < width and u[i][j+1] or 0
			return (u_xl + u_xr + u_yl + u_yr - 4 * u[i][j]) / (h * h)
		end)
	end,
	epsilon = 1e-10,
	errorCallback = function(err, iter, x, rSq, bSq)
		data[iter] = data[iter] or {math.nan}
		data[iter][2] = matrix.norm(x)
	end,
}

for i=1,#data do
	for j=1,2 do
		data[i][j] = data[i][j] or math.nan
	end
end

local gnuplot = require 'gnuplot'
gnuplot{
	output = 'convergence.png',
	log = 'y',
	data = matrix(data):T(),
	style = 'data lines',
	{using='0:1', title='multigrid'},
	{using='0:2', title='conjgrad'},
}
