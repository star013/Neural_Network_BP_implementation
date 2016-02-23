classdef Relu
    properties
        name;
        input;
        output;
        delta;
    end
    
    methods
        function layer = Relu(name)
            layer.name = name;
        end
        
        function layer = forward(layer, input)
            % Your codes here
            layer.input = input;
            layer.output = max(0,layer.input);
        end
        
        function layer = backprop(layer, delta)
            % Your codes here
            layer.delta = delta.*(layer.input>0);
        end
    end
end

            
            