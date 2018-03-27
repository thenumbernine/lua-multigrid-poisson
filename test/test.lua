#!/usr/bin/env luajit
require 'ext'
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

local tries = 1

for size=0,5 do
	local line = table()
	write(2^size)
	for _,col in ipairs(cols) do
		local test = 'test-'..col..'.lua'
		local bestTime = math.huge
		for try=1,tries do
			local devnull = ({
				Windows = 'nul'
			})[ffi.os] or '/dev/null'
			local cmd = lua..' '..test..' '..size..' 2> tmp > '..devnull
			local result = {os.execute(cmd)}
			assert(result[1] or result[2] == 'unknown', "failed on cmd "..cmd)
			local results = file.tmp
			file.tmp = nil
			local time = assert(tonumber(results:match'time taken: (.*)'), 
				"failed to find time taken in output:\n"
				..('='):rep(20)..'\n'
				..results)
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
