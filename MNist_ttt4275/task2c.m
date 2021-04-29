tic
task2a;
toc
cluster_tags = repelem([0 1 2 3 4 5 6 7 8 9]', num_clusters);
tic
resulting_indices = kNearestNeighbour(cluster_matrix, cluster_tags, testv, 1);
toc
%display_results(resulting_indices, num_test, testlab);