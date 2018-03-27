#! /usr/bin/env luajit
local bit = require 'bit' or bit32

local MultigridCPUGPU = require 'multigrid-poisson.cpu-gpu'

local log2size = ... and tonumber(...) or 5
local size = bit.lshift(1, log2size)
local level = tonumber(select(2, ...) or nil) or 3
local multigrid = MultigridCPUGPU(size, nil, level)

local startTime = os.clock()
multigrid:run()
local endTime = os.clock()
io.stderr:write('time taken: '..(endTime - startTime)..'\n')
