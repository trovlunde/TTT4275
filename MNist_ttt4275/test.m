W = limit_data(trainv, data_shares, data_size);
W_2 = trainv((2-1)*1000+1:2*1000,:);
distance = dist(W, testv');
min_dist = min(distance(:,1));