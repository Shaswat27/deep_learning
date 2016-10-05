require 'nn'

local SpikeReLU, Parent = torch.class('nn.SpikeReLU', 'nn.Module')

--we need the threshold membrane potential and the refractory preiod of the neuron: th, refrac respectively
--NOTE: expects t as the global variable

function SpikeReLU:__init(th,refrac)
   --print('init')
   Parent.__init(self)
   self.first_time = true
   self.current_time = t 
   self.memory = torch.CudaTensor()
   self.threshold = th or 1
   self.refrac_period = refrac or 0
   self.refrac_until = torch.CudaTensor()
   if (th and type(th) ~= 'number') or (refrac and type(refrac) ~= 'number') then
      error('nn.SpikeReLU(threshold, refractory period)')
   end
end

function SpikeReLU:updateOutput(input)
   if t == nil then
      error('Initialize time variable t')
   end

   if self.first_time then
      --print 'size'
      self.memory = torch.zeros(input:size()):cuda()
      self.refrac_until = -1*torch.ones(input:size()):cuda() --done to indicate no refactory period at t=0
      self.first_time = false
   end

   self.current_time = t

   --add input to the membrane potentials
   self.memory = self.memory + input

   --check for spiking
   local spikes = (self.memory):ge(self.threshold)
   --check if it is in refractory period
   local refrac_check = self.refrac_until:ge(self.current_time) 
   if spikes[refrac_check] ~= nil then
      spikes[refrac_check] = 0.0
   end
   self.output = spikes

   --reset the neurons
   if self.memory[self.output] ~= nil then
      self.memory[self.output] = 0.0
   end

   --set the refractory period
   if self.refrac_until[self.output] ~= nil then
      self.refrac_until[self.output] = t + self.refrac_period
   end

   return self.output
end

function SpikeReLU:updateGradInput(input, gradOutput)
   --Layer not meant for backprop
end