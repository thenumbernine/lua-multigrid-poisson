#!/usr/bin/env luajit
-- look at cpu convergence
-- maybe compare it to the convergence of conj res
local matrix = require 'matrix'
local table = require 'ext.table'
local math = require 'ext.math'
local file = require 'ext.file'
local range = require 'ext.range'
local MultigridCPU = require 'multigrid-poisson.cpu'

for _,size in ipairs{4,8,16,32,64} do
	print('solving for size '..size)
	local data = table()

	local mg
	mg = MultigridCPU{
		size = size,
		errorCallback = function(iter, err)
			data[iter] = {math.sqrt(mg.psi:normSq() / mg.psi:size():prod())}
		end,
	}
	mg:solve()

	local div = require 'matrix.div'
	local conjgrad = require 'solver.conjgrad'
	local cgpsi = conjgrad{
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
		errorCallback = function(err, iter, psi, rSq, bSq)
			local err = math.sqrt(psi:normSq() / psi:size():prod())
			data[iter] = data[iter] or {math.nan}
			data[iter][2] = err
			return err < 1e-10 
		end,
	}

	local min = math.huge
	for i=1,#data do
		for j=1,2 do
			data[i][j] = data[i][j] or math.nan
			if math.isfinite(data[i][j]) then min = math.min(min, data[i][j]) end
		end
	end
	for i=1,#data do
		for j=1,2 do
			data[i][j] = data[i][j] - min
		end
	end

	file['converge-'..size..'.txt'] = data:map(function(row)
		return table.concat(row, '\t')
	end):concat'\n'

	data = matrix(data):T()

	local gnuplot = require 'gnuplot'
	gnuplot{
		output = 'convergence-'..size..'-multigrid-vs-kyrlov.png',
		title = 'multigrid vs kyrlov '..size,
		log = 'y',
		data = data,
		style = 'data lines',
		{using='0:1', title='multigrid'},
		{using='0:2', title='conjgrad'},
	}

	gnuplot{
		output = 'converge-'..size..'-result-multigrid.png',
		griddata = {x=range(size), y=range(size), mg.psi},
		style = 'data lines',
		{splot=true, using='1:2:3', title='multigrid'},
	}

	gnuplot{
		output = 'converge-'..size..'-result-conjgrad.png',
		griddata = {x=range(size), y=range(size), cgpsi},
		style = 'data lines',
		{splot=true, using='1:2:3', title='conjgrad'},
	}

	gnuplot{
		output = 'converge-'..size..'-result-diff.png',
		griddata = {x=range(size), y=range(size), cgpsi - mg.psi},
		style = 'data lines',
		log = 'z',
		{splot=true, using='1:2:(abs($3))', title='conjgrad'},
	}
end
