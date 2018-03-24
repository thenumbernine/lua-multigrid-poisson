#!/usr/bin/env lua
require 'ext'
-- only works up to (2^6)^2 before it gets too slow
-- (2^6)^2 takes about 5 seconds to run, so don't do too many samples
for _,test in ipairs{
	'cpu.lua', 
	'cpu-raw.lua', 
	'gpu.lua',
	--'gpu-obj.lua',
} do
	print('testing '..test..'...')
	for i=0,5 do
		local bestTime = math.huge
		for try=0,10 do
			local time = assert(tonumber(io.readproc('./'..test..' '..i):match'time taken: (.*)'))
			bestTime = math.min(bestTime, time)
		end
		print('(2^'..i..')^2 best time '..bestTime)
	end
end
