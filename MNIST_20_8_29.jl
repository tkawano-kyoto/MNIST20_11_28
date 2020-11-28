using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs
using Base.Iterators: repeated
#using Parameters: @with_kw
using CUDAapi
using MLDatasets

# @with_kw mutable struct Args
#     #η::Float64 = 3e-4       # learning rate
#     #epochs::Int = 10        # number of epochs
# end

function getdata()
    # Loading Dataset	
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)

    # Reshape Data in order to flatten each image into a linear array
    xtrain = Flux.flatten(xtrain)
    xtest = Flux.flatten(xtest)
    # One-hot-encode the labels
    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    # Batching
    # train_data = DataLoader(xtrain, ytrain, batchsize=1)
    # test_data = DataLoader(xtest, ytest, batchsize=1)
    # return train_data, test_data    
    return xtrain,xtest
end

#args=Args()
train_data,test_data = getdata();

# m=Chain(
#     Dense(28^2,50,σ),
#     Dense(50,100,σ),
#     Dense(100,10),
#     softmax)
	
# loss(x,y) = logitcrossentropy(m(x), y)


# function loss_all(dataloader, model)
#     l = 0f0
#     for (x,y) in dataloader
#         l += logitcrossentropy(model(x), y)
#     end
#     l/length(dataloader)
# end

#function accuracy(data_loader, model)
#     acc = 0
#     for (x,y) in data_loader
# 		acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)
# 		#acc += sum(onecold(model(x)) .== onecold(y))*1 / size(x,2)
#     end
#     acc/length(data_loader)
# end# 

# evalcb = () -> @show(loss_all(train_data, m))
# opt = ADAM(args.η)

# @epochs args.epochs Flux.train!(loss, params(m), train_data, opt, cb = evalcb)

# @show accuracy(train_data, m)
# @show accuracy(test_data, m)