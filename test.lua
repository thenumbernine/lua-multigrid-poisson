#!/usr/bin/env lua
require 'ext'
-- only works up to (2^6)^2 before it gets too slow
-- (2^6)^2 takes about 5 seconds to run, so don't do too many samples
local cols = {
	'cpu',
	'cpu-raw',
	'gpu',
	--'gpu-obj',
	'cpu-gpu',
}

local f = io.open('out.txt', 'w')

f:write'#size'
for _,col in ipairs(cols) do
	f:write('\t', col)
end
f:write'\n'

local tries = 10

for size=0,10 do
	f:write(2^size)
	for _,col in ipairs(cols) do
		local test = 'test-'..col..'.lua'
		local bestTime = math.huge
		for try=1,tries do
			assert(os.execute('./'..test..' '..size..' 2> tmp > /dev/null'))
			local results = file.tmp
			file.tmp = nil
			local time = assert(tonumber(results:match'time taken: (.*)'), 
				"failed to find time taken in output:\n"
				..('='):rep(20)..'\n'
				..results)
			bestTime = math.min(bestTime, time)
		end
		f:write('\t', bestTime)
	end
	f:write'\n'
end

f:close()
