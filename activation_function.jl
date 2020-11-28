using Flux 
using LinearAlgebra
#one-hot表現に変換の時は必要
using Flux:onehotbatch
using JLD
using Statistics

#活性化関数
#ステップ関数
#ex. z1=step_function.(a1)

function step_function(x)

    y = x > 0
    return Int(y)
end


#活性化関数
#ReLU関数,Rectified Linear Unit
#ex. z1=sigmoid.(a1)

function relu(x)
    return max.(0,x)
end


#活性化関数
#シグモイド関数
#ex. z1=sigmoid.(a1)

function sigmoid(x)
    1.0 / (1.0 + exp(-x))
end

#最後の活性化関数
#恒等関数
#入力をそのまま出力

function identity_function(x)
    return x
end

#最後の活性化関数
#ソフトマックス関数
#
function softmax(a::AbstractArray{T}) where {T}
    c=maximum(a,dims=1)
    exp_a=exp.(a .-c)
    exp_a ./ sum(exp_a,dims=1)
end

# function softmax(a::AbstractMatrix{T}) where{T}
#     mapslices(softmax,a,dims=1)
# end


# function input_data_train()

#     #訓練画像の読込
#     train_imgs = Flux.Data.MNIST.images(:train);

#     #訓練ラベルの読込
#     train_labels = Flux.Data.MNIST.labels(:train);

#     #画像データを変換(2X2 -> vector)
#     x_train=hcat(float.(vec.(train_imgs))...);

#     #one-hot表現に変換
#     #t_train=onehotbatch(train_labels,0:9);

#     return x_train,train_labels
# end

# function input_data_test()

#     #テスト画像の読込
#     test_imgs = Flux.Data.MNIST.images(:test);

#     #テストラベルの読込
#     test_labels = Flux.Data.MNIST.labels(:test);

#     #画像データを変換(2X2 -> vector)
#     x_test=hcat(float.(vec.(test_imgs))...);

#     #one-hot表現に変換
#     #t_test=onehotbatch(test_labels,0:9);

#     return x_test,test_labels
# end


function init_network()
    network = Dict()
    network = load("sample_network.jld")
    return network
end

function forward(network,x)
    W1,W2,W3 = network["W1"],network["W2"],network["W3"]
    b1,b2,b3 = network["b1"],network["b2"],network["b3"]
    
    a1 = W1 * x .+ b1
    #z1 = sigmoid.(a1)
    
    z1 = relu(a1)
    a2 = W2 * z1 .+ b2
    #z2  = sigmoid.(a2)
    z2 = relu(a2)
    a3 = W3 * z2 .+ b3
    
    y = softmax(a3)
    #y=identity(a3)
    return y
end





