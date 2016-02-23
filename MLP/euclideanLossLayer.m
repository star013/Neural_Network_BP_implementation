classdef euclideanLossLayer
    
    properties
        name;
        input;
        input_shape;
        
        output;
        delta;
        loss;
        accuracy;
        pred_index;
    end
    
    methods
        function layer = euclideanLossLayer(name)
            layer.name = name;
        end
    
        function layer = forward(layer, input)
            % Your codes here
            layer.output = input;
            layer.input_shape = size(input);
        end
        
         function layer = backprop(layer, label)
            % Your codes here
            label_mtx = zeros(layer.input_shape,'single');
            for k = 1:layer.input_shape(2)
               label_mtx(label(k),k) = 1; 
            end
            layer.delta = -2*label_mtx + 2* layer.output;
            layer.loss = sum(sum((layer.output-label_mtx).^2));
            [~, pred_indx] = max(layer.output);
            layer.accuracy = sum(pred_indx == label) / size(label, 2);
         end
         
         function layer = predict(layer, input, label)
             % Your codes here
             layer.output = input;
             layer.input_shape = size(input);
            label_mtx = zeros(layer.input_shape,'single');
            for k = 1:size(label, 2)
               label_mtx(label(k),k) = 1; 
            end
            layer.loss = sum(sum((layer.output-label_mtx).^2));
            [~, pred_indx] = max(layer.output);
            layer.accuracy = sum(pred_indx == label) / size(label, 2);
            layer.pred_index = pred_indx;
         end
    end
end