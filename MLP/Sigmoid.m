classdef Sigmoid
    properties
        name;
        input;
        output;
        delta;
    end
    
    methods
        function layer = Sigmoid(name)
            layer.name = name;
        end
        
        function layer = forward(layer, input)
            % Your codes here
            layer.input = input;
            layer.output = 1./(1+exp(-layer.input));
        end
        
        function layer = backprop(layer, delta)
            % Your codes here
            layer.delta = delta.*layer.output.*(1-layer.output);
        end
    end
end
