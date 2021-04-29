num_classes = 10;
data_size = 1000;
data_shares = num_test/data_size;
tic
resulting_indices = nearestNeighbour(trainv, trainlab, testv, data_shares, data_size);
toc
display_results(resulting_indices, num_test, testlab);