function [W] = limit_data(data, data_shares, data_size, i)
%LIMIT_DATA Summary of this function goes here
%   Detailed explanation goes here
% Data is limited this way to get an even share amongst the total set.
% Could also be done randomly.
W = data((i-1)*data_size+1:i*data_size,:);
end

