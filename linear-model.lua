require 'torch';
require 'nn';
require 'optim';
require 'image';
require 'paths';
require 'cutorch';
require 'cunn';
require 'sys';

train_file_x = 'train_inputs.t7'
train_file_y = 'train_labels.t7'
test_file_x = 'test_inputs.t7'
test_file_y = 'test_labels.t7'

train_x = torch.load(train_file_x)
train_y = torch.load(train_file_y)
test_x = torch.load(test_file_x)
test_y = torch.load(test_file_y)

--train
for i = 1,40*7 do
    if i<=40*3 then
        print(i)
        mean = test_x:mean(1)
        std = test_x:std(1)
        test_x[{{i}}]:add(-mean)
        test_x[{{i}}]:cdiv(std)
    end
    print(i)
    mean = train_x:mean(1)
    std = train_x:std(1)
    train_x[{{i}}]:add(-mean)
    train_x[{{i}}]:cdiv(std)
end

--print(test_x)

-- training and validation sets
testset = {size = 40*3, data = test_x:double(), label = test_y:double()} 
trainset = {size = 40*7, data = train_x:double(), label = train_y:double()}

testset.data = testset.data:cuda()
testset.label = testset.label:cuda()

trainset.data = trainset.data:cuda()
trainset.label = trainset.label:cuda()

--create the net 
net = nn.Sequential()

net:add(nn.Linear(112+92, 1024))
net:add(nn.ReLU())

net:add(nn.Linear(1024, 1024))
net:add(nn.ReLU())

--classification: 10 outputs
net:add(nn.Linear(1024, 40))
net:add(nn.LogSoftMax())

--display the net
print('Custom net: \n' .. net:__tostring());

net = net:cuda()

--negative log likelihood criterion
criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()

--parameters for SGD
adam_params = {learningRate = 1e-3, learningRateDecay = 1e-4, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8}

--model parameters
x, dl_dx = net:getParameters()

--function to train a single epoch and return the loss
step = function(batch_size)
	local current_loss = 0
	local count = 0
	local shuffle = torch.randperm(trainset.size)
	batch_size = batch_size or 400

	for t = 1,trainset.size,batch_size do
        -- setup inputs and targets for this mini-batch
        local size = math.min(t + batch_size - 1, trainset.size) - t
        local inputs = torch.CudaTensor(size, 112+92)
        local targets = torch.CudaTensor(size)
        for i = 1,size do
            local input = trainset.data[shuffle[i+t]]
            local target = trainset.label[shuffle[i+t]]
            inputs[i] = input
            targets[i] = target
        end
        
        local feval = function(x_new)
            -- reset data
            if x ~= x_new then x:copy(x_new) end
            dl_dx:zero()

            -- perform mini-batch gradient descent
            local loss = criterion:forward(net:forward(inputs), targets)
            net:backward(inputs, criterion:backward(net.output, targets))

            return loss, dl_dx
        end
        
        _, fs = optim.sgd(feval, x, sgd_params)
        -- fs is a table containing value of the loss function
        -- (just 1 value for the SGD optimization)
        count = count + 1
        current_loss = current_loss + fs[1]
    end

    -- normalize loss
    return current_loss / count
end

--for cross validation
eval = function(dataset, batch_size)
    local count = 0
    batch_size = batch_size or 200
    
    for i = 1,dataset.size,batch_size do
        local size = math.min(i + batch_size - 1, dataset.size) - i
        local inputs = dataset.data[{{i,i+size-1}}]
        local targets = dataset.label[{{i,i+size-1}}]
        local outputs = net:forward(inputs)
        local _, indices = torch.max(outputs, 2)
        local guessed_right = indices:eq(targets):sum()
        count = count + guessed_right
    end

    return count / dataset.size
end

--now begins the training
max_iters = 20000

cutorch.synchronize()
initial_time = sys.tic()
do
	local last_accuracy = 0
	local decreasing = 0
	local threshold = 3 --number of decreasing epochs to allow
	for i=1, max_iters do
		local loss = step()
		--print(string.format('Epoch: %d Current loss: %4f', i, loss)) 
		local accuracy = eval(testset)
		print(string.format('Accuracy on the test set: %4f, epoch: %d', accuracy, i))
		if accuracy < last_accuracy then
			if decreasing > threshold then break end
			decreasing = decreasing + 1
		else
			decreasing = 0
		end
		last_accuracy = accuracy
	end
end
cutorch.synchronize()
print(string.format("Elapsed time: %.6f\n", sys.toc()))

filename = paths.concat(paths.cwd(), 'model_linear.net')
torch.save(filename, net)

--net = torch.load(filename)
print(eval(testset))