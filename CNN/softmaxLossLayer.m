classdef softmaxLossLayer
    
    properties
        name;
        input;
        input_shape;

        output;
        delta;
        loss;
        accuracy;
        pred_index
    end

    methods
        function layer = softmaxLossLayer(name)
            layer.name = name;
        end
        
        function layer = forward(layer, input)
            % Your codes here
            temp = exp(input);
            tempsum = sum(temp);
            tempsum = repmat(tempsum,size(input,1),1);
            layer.output = temp./tempsum;
            layer.input_shape = size(input);
        end

        function layer = backprop(layer, label)
            % Your codes here
            label_mtx = zeros(layer.input_shape,'single');
            for k = 1:layer.input_shape(2)
               label_mtx(label(k),k) = 1; 
            end
            layer.delta = -label_mtx+layer.output;
            layer.loss = sum(sum(-label_mtx.*log10(layer.output)));
            [~, pred_indx] = max(layer.output);
            layer.accuracy = sum(pred_indx == label) / size(label, 2);
        end
        
        function layer = predict(layer,input,label)
            layer.output = input;
            layer.input_shape = size(input);
            label_mtx = zeros(layer.input_shape,'single');
            for k = 1:layer.input_shape(2)
               label_mtx(label(k),k) = 1; 
            end
            layer.loss = sum(sum(-label_mtx.*log10(layer.output)));
            [~, pred_indx] = max(layer.output);
            layer.accuracy = sum(pred_indx == label) / size(label, 2);
            layer.pred_index = pred_indx;
        end
    end
end


