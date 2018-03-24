#! /usr/bin/env luajit

local MultigridGPU = require 'multigrid-poisson.gpu'

local log2size = ... and tonumber(...) or 5
local size = bit.lshift(1, log2size)
local multigrid = MultigridGPU(size)

local startTime = os.clock()
multigrid:run()
local endTime = os.clock()
io.stderr:write('time taken: '..(endTime - startTime)..'\n')
