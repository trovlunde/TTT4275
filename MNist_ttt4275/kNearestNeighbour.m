function [result] = kNearestNeighbour(train_data,train_labels, test_data, k)
%KNEARESTNEIGHBOUR Summary of this function goes here
%   Detailed explanation goes here
[C, ~, train_labels] = unique(train_labels, 'stable');
test_size = size(test_data,1);
result = zeros(test_size,1);
for i = 1:test_size
    W = repmat(test_data(i,:), size(train_data,1),1)-train_data;
    distance = sqrt(sum(W.^2,2));
    [~,sort_index] = sort(distance);
    nearest = train_labels(sort_index(1:k));
    bins = histc(nearest, 1:max(nearest));
    appearances = bins(nearest);
    [~,most_frequent] = max(appearances);
    result(i) = nearest(most_frequent);
end
result = C(result);
end