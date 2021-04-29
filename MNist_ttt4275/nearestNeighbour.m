function [result] = nearestNeighbour(train_data, train_labels, test_data, data_shares, data_size)
%NEARESTNEIGHBOUR Summary of this function goes here
%   Detailed explanation goes here
% Initialize
[C,~,train_labels] = unique(train_labels,'stable');
test_size = size(test_data,1);
result = zeros(test_size,1);
dist_to_NN = zeros(test_size,data_shares);
indices = zeros(test_size,data_shares);
for i = 1:data_shares
    W = limit_data(train_data, data_shares, data_size, i);
    distance = dist(W,test_data');
    for k = 1:test_size
        [dist_to_NN(k, i), index] = min(distance(:,k));
        indices(k, i) = (i-1)*data_size+index;
    end
    
end
for i = 1:test_size
    [~, index] = min(dist_to_NN(i,:));
     result(i) = train_labels(indices(i,index));     
end
result = C(result);
end


