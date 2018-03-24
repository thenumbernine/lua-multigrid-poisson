local class = require 'ext.class'
local bit = require 'bit' or bit32

local MultigridGPU = require 'multigrid.gpu'
local MultigridCPURaw = require 'multigrid.cpu'

local MultigridCPUGPU = class()

--[[
size = size of grid
cpudepth = tree level at which to switch from GPU to CPU
--]]
function MultigridCPUGPU:init(size, real, cpudepth)
	self.gpu = MultigridGPU(size, real)
	self.cpu = MultigridCPURaw(size, bit.lshift(1, cpudepth))
	self.cpudepth = cpudepth
end

function MultigridCPUGPU:run()
	
	-- run the gpu ... but only to a certain level ...
	-- ... and then copy over to cpu
	-- ... and then run the cpu
	self.gpu:run()
end

return MultigridCPUGPU
