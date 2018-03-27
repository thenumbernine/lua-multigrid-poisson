#!/usr/bin/env luajit
local bit = require 'bit' or bit32

local MultigridCPU = require 'multigrid-poisson.cpu'

local log2size = ... and tonumber(...) or 5
local size = bit.lshift(1,log2size)
local multigrid = MultigridCPU(size)

local startTime = os.clock()
multigrid:run()
local endTime = os.clock()
io.stderr:write('time taken: '..(endTime - startTime)..'\n')
