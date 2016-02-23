classdef Network
    properties
        layer_list
        num_layer
    end
    
    methods
        function net = Network()
            net.layer_list = {};
            net.num_layer = 0;
        end
        
        function net = add(net, layer)
            net.num_layer = net.num_layer + 1;
            net.layer_list{net.num_layer} = layer;  
        end
        
        function net = Forward(net, input)
            net.layer_list{1} = forward(net.layer_list{1}, input);
            for k = 2:net.num_layer
                net.layer_list{k} = forward(net.layer_list{k}, net.layer_list{k-1}.output);
            end
        end
        
        function net = Backpropagation(net, label)
            net.layer_list{net.num_layer} = backprop(net.layer_list{net.num_layer}, label);
            for k = net.num_layer-1:-1:1
                net.layer_list{k} = backprop(net.layer_list{k}, net.layer_list{k+1}.delta);
            end
        end
        
        function [loss, pred_indx, accuracy] = Predict(net, input, label)
            net.layer_list{1} = forward(net.layer_list{1}, input);
            for k = 2:net.num_layer-1
                net.layer_list{k} = forward(net.layer_list{k}, net.layer_list{k-1}.output);
            end
            %%
            net.layer_list{net.num_layer} = predict(net.layer_list{net.num_layer}, net.layer_list{net.num_layer-1}.output,label);
            loss = net.layer_list{net.num_layer}.loss;
            accuracy = net.layer_list{net.num_layer}.accuracy;
            pred_indx = net.layer_list{net.num_layer}.pred_index;
        end

    end
end

    
            