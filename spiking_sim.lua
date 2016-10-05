require 'torch';
require 'nn';
require 'image';
require 'paths';
require 'cutorch';
require 'cunn';
require 'sys';
require 'SpikeReLU.lua';

--load the model
net = torch.load('model_linear.net')
print('Network loaded successfully')

--change the layers
net:replace(function(module)
                            if torch.typename(module) == 'nn.LogSoftMax' then
                                return nn.SpikeReLU(1)
                            elseif torch.typename(module) == 'nn.ReLU' then
                                return nn.SpikeReLU(1)
                            else
                                return module
                            end
            end)
print('Modified network:')
print(net:__tostring())
net = net:cuda()

--create testset
test_file_x = 'test_inputs.t7'
test_file_y = 'test_labels.t7'

test_x = torch.load(test_file_x)
test_y = torch.load(test_file_y)

--train
for i = 1,40*3 do
    mean = test_x:mean(1)
    std = test_x:std(1)
    test_x[{{i}}]:add(-mean)
    test_x[{{i}}]:cdiv(std)
end

testset = {size = 40*3, data = test_x:double(), label = test_y:double()} 
testset.data = testset.data:cuda()
testset.label = testset.label:cuda()

--eval function
eval = function(dataset, batch_size)
    local count = 0
    local size = 1--120
    
    local inputs = dataset.data[{{1}}]

    --create poisson distributed spikes from the input images
    local rescale_fac = 1/(t_dt*t_max_rate)
    --generate random numbers directly in gpu to avoid cpu->gpu transfers
    local spike_snapshot = torch.CudaTensor(inputs:size()):rand(inputs:size())
    --inputs = spike_snapshot:le(inputs):double() --inputs need to normalized
    --inputs = inputs:cuda()

    --sum_inputs = sum_inputs+inputs

    local targets = dataset.label
    local outputs = net:forward(inputs)

    --sum_outputs
    sum_outputs = outputs + sum_outputs

    --print(outputs[1])
    --print(sum_outputs[1])

    local _, indices = torch.max(sum_outputs)--, 2)

    guess = indices==targets
    return guess
    --[[local guessed_right = indices:eq(targets):sum()
    count = count + guessed_right

    return count / dataset.size--]]
end

--initial options for time simulation
t = 0.000
t_i = 0.000
t_dt = 0.001
t_f = 0.140
t_max_rate = 200

print(string.format('Starting simmulation for t = %.3f to t = %.3f with dt = %.3f', t_i, t_f, t_dt))


sum_outputs = torch.zeros(1,40)
sum_outputs = sum_outputs:cuda()
sum_inputs = torch.zeros(1,204)
sum_inputs = sum_inputs:cuda()

for t = t_i, t_f, t_dt do
	print(string.format('Time: %.3f',t))
	print(eval(testset))
	print()
end