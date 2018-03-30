#!/usr/bin/env luajit
require 'ext'
local bit = require 'bit' or bit32
local ffi = require 'ffi'
local lua = arg[table(arg):keys():sort():inf()]
local gnuplot = require 'gnuplot'

local cols = table{
	'cpu',
	'cpu-raw',
	'gpu',
	--'gpu-obj',
	'cpu-gpu',
}

local fn = 'cpu-vs-gpu.txt'
local f = io.open(fn, 'w')
local function write(...)
	for i=1,select('#', ...) do
		local s = tostring(select(i, ...))
		io.write(s)
		io.flush()
		f:write(s)
		f:flush()
	end
end

write'#size'
for _,col in ipairs(cols) do
	write('\t',col)
end
write'\n'

local log2size = ... and tonumber(...) or 5
local size = bit.lshift(1, log2size)
local cpudepth = tonumber(select(2, ...) or nil) or 3

local tries = 1
for log2size=0,5 do
	local size = bit.lshift(1, log2size)
	local line = table()
	write(size)
	for _,col in ipairs(cols) do
		local test = 'test-'..col..'.lua'
		local bestTime = math.huge
		for try=1,tries do
			local cl = require('multigrid-poisson.'..col)
			local startTime = os.clock()
			local multigrid = cl(size, nil, cpudepth)	
			local endTime = os.clock()
			local time = endTime - startTime
			bestTime = math.min(bestTime, time)
		end
		write('\t',bestTime)
	end
	write'\n'
end
f:close()

gnuplot(table(
	{
		output = 'cpu-vs-gpu.png',
		data = data,
		style = 'data linespoints',
	},
	cols:map(function(col,i)
		return {fn, using='1:'..i+1, title=col}
	end)
))
