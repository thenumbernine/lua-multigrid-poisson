#! /usr/bin/env luajit

local MultigridCPURaw = require 'multigrid-poisson.cpu-raw'

local log2size = ... and tonumber(...) or 5
local size = bit.lshift(1, log2size)
local multigrid = MultigridCPURaw(size)

local startTime = os.clock()
multigrid:run()
local endTime = os.clock()
io.stderr:write('time taken: '..(endTime - startTime)..'\n')
