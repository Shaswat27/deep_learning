require 'torch';
require 'nn';
require 'optim';
require 'image';
require 'paths';
require 'cutorch';
require 'cunn';
require 'sys';

print '==> downloading dataset'

tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz';
if not paths.dirp('mnist.t7') then
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end

train_file = 'mnist.t7/train_32x32.t7'
test_file = 'mnist.t7/test_32x32.t7'

fullData = torch.load(train_file,'ascii')
testData = torch.load(test_file,'ascii')

--print '==> visualizing data'

--[[ Visualization is quite easy, using itorch.image().
print('training data:')
image.display(trainData.data[{ {1,256} }])
print('test data:')
image.display(testData.data[{ {1,256} }]) ]]--

-- training and validation sets
testset = {size = 10000, data = testData.data:double(), label = testData.labels - 1 } 
trainset = {size = 50000, data = fullData.data[{{1,50000}}]:double(), label = fullData.labels[{{1,50000}}] - 1 }
validationset = {size = 10000, data = fullData.data[{{50001,60000}}]:double(), label = fullData.labels[{{50001,60000}}] - 1 }

testset.data = testset.data:cuda()
testset.label = testset.label:cuda()

trainset.data = trainset.data:cuda()
trainset.label = trainset.label:cuda()

validationset.data = validationset.data:cuda()
validationset.label = validationset.label:cuda()

--create the net 
net = nn.Sequential()

net:add(nn.SpatialConvolution(1,5,3,3))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))

net:add(nn.SpatialConvolution(5,10,3,3))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))
--10x6x6

net:add(nn.View(10*6*6))

net:add(nn.Linear(10*6*6, 120))
net:add(nn.ReLU())

net:add(nn.Linear(120, 30))
net:add(nn.ReLU())

--classification: 10 outputs
net:add(nn.Linear(30, 10))
net:add(nn.LogSoftMax())

--display the net
print('Custom net: \n' .. net:__tostring());

net = net:cuda()

--negative log likelihood criterion
criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()

--parameters for SGD
sgd_params = {learningRate = 1e-2, learningRateDecay = 1e-4, weightDecay = 1e-3, momentum = 1e-4}

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
        local inputs = torch.CudaTensor(size, 1, 32, 32)
        local targets = torch.CudaTensor(size)
        for i = 1,size do
            local input = trainset.data[shuffle[i+t]]
            local target = trainset.label[shuffle[i+t]]
            inputs[i] = input
            targets[i] = target
        end
        targets:add(1)
        
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
        indices:add(-1)
        local guessed_right = indices:eq(targets):sum()
        count = count + guessed_right
    end

    return count / dataset.size
end

--now begins the training
max_iters = 30

cutorch.synchronize()
initial_time = sys.tic()
do
	local last_accuracy = 0
	local decreasing = 0
	local threshold = 3 --number of decreasing epochs to allow
	for i=1, max_iters do
		local loss = step()
		print(string.format('Epoch: %d Current loss: %4f', i, loss)) 
		local accuracy = eval(validationset)
		print(string.format('Accuracy on the validation set: %4f', accuracy))
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

filename = paths.concat(paths.cwd(), 'model1.net')
torch.save(filename, net)

--net = torch.load(filename)
print(eval(testset))
