#!/usr/bin/env luajit
-- look at cpu convergence
-- maybe compare it to the convergence of conj res
local matrix = require 'matrix'
local div = require 'matrix.div'
local table = require 'ext.table'
local math = require 'ext.math'
local file = require 'ext.file'
local range = require 'ext.range'
local MultigridCPU = require 'multigrid-poisson.cpu'

os.execute'mkdir converge'

for _,size in ipairs{4,8,16,32} do
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

	local cols = table{
		'conjgrad',
		'conjres',
		'bicgstab',
	}

	local psis = cols:map(function(col,k)
		local solver = require('solver.'..col)
		return solver{
			zero = matrix.zeros(size,size),	-- bicgstab
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
				if iter > 0 then
					local err = math.sqrt(psi:normSq() / psi:size():prod())
					data[iter] = data[iter] or {}
					data[iter][k+1] = err
					return err < 1e-10
				end
			end,
		}
	end)

	local maxlen = math.max(data:map(function(row) return #row end):unpack())

	local minvalue = math.huge
	for i=1,#data do
		for j=1,maxlen do
			data[i][j] = data[i][j] or math.nan
			if math.isfinite(data[i][j]) then minvalue = math.min(minvalue, data[i][j]) end
		end
	end
	for i=1,#data do
		for j=1,maxlen do
			data[i][j] = data[i][j] - minvalue
		end
	end

	file['converge/'..size..'.txt'] = data:map(function(row)
		return table.concat(row, '\t')
	end):concat'\n'

	data = matrix(data):T()

	local gnuplot = require 'gnuplot'
	gnuplot(table({
		output = 'converge/'..size..'-multigrid-vs-kyrlov.png',
		title = 'multigrid vs kyrlov '..size,
		log = 'y',
		data = data,
		style = 'data lines',
	}, table{
		{using='0:1', title='multigrid'},
	}:append(cols:map(function(col,k)
		return {using='0:'..(k+1), title=col}
	end))))

	local r = range(0,size-1)

	gnuplot(table({
		output = 'converge/'..size..'-result.png',
		griddata = {x=r, y=r, mg.psi, psis[1]},
		style = 'data lines',
	}, table{
		{splot=true, using='1:2:3', title='multigrid'},
	}:append(cols:map(function(col,k)
		return {splot=true, using='1:2:'..(k+3), title=col}
	end))))

	gnuplot{
		output = 'converge/'..size..'-result-diff.png',
		griddata = {x=r, y=r, psis[1] - mg.psi},
		style = 'data lines',
		log = 'z',
		{splot=true, using='1:2:(abs($3))', title='conjgrad'},
	}
end
