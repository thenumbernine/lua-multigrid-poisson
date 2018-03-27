local class = require 'ext.class'
local bit = require 'bit' or bit32
local ffi = require 'ffi'

local MultigridGPU = require 'multigrid-poisson.gpu'
local MultigridCPURaw = require 'multigrid-poisson.cpu-raw'


local MultigridGPUSubset = class(MultigridGPU)

function MultigridGPUSubset:init(size, real, owner, cpuDepth)
	self.owner = owner
	self.cpuDepth = cpuDepth
	return MultigridGPUSubset.super.init(self, size, real)
end

function MultigridGPUSubset:twoGrid(h, u, f, L)
	if L == bit.lshift(1, self.cpuDepth) then
		local cpu = self.owner.cpu
		
		-- copy the u and f buffers to the CPU
		for _,info in ipairs{
			{gpuBuf=u, cpuBuf=cpu.psi},
			{gpuBuf=f, cpuBuf=cpu.f},
		} do
			self.cmds:enqueueReadBuffer{
				buffer = info.gpuBuf,
				block = true, 
				size = L * L * ffi.sizeof(self.real), 
				ptr = info.cpuBuf.buffer,
			}
		end
	
		-- run the CPU
		cpu:twoGrid(h, cpu.psi.buffer, cpu.f.buffer, L)

		-- copy it back
		for _,info in ipairs{
			{gpuBuf=u, cpuBuf=cpu.psi},
			{gpuBuf=f, cpuBuf=cpu.f},
		} do
			self.cmds:enqueueWriteBuffer{
				buffer = info.gpuBuf,
				block = true, 
				size = L * L * ffi.sizeof(self.real), 
				ptr = info.cpuBuf.buffer,
			}
		end
	else
		return MultigridGPUSubset.super.twoGrid(self, h, u, f, L)
	end
end


local MultigridCPUGPU = class()

--[[
size = size of grid
cpuDepth = tree level at which to switch from GPU to CPU
--]]
function MultigridCPUGPU:init(size, real, cpuDepth)
	self.gpu = MultigridGPUSubset(size, real, self, cpuDepth)
	self.cpu = MultigridCPURaw(bit.lshift(1, cpuDepth))
	self.cpuDepth = cpuDepth
end

function MultigridCPUGPU:run()
	-- run the gpu ... but only to a certain level ...
	-- ... and then copy over to cpu
	-- ... and then run the cpu
	self.gpu:run()
end

return MultigridCPUGPU
