function [sorted_data, sorted_label_data] = home_sort(data,labels)
%KNEARESTNEIGHBOUR Summary of this function goes here
%   Detailed explanation goes here
[sorted_label_data, index] = sort(labels);
sorted_data = zeros(size(data));
for i = 1:size(sorted_data)
    sorted_data(i,:) = data(index(i),:);
end
end