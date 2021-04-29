tic
task2a;
toc
cluster_tags = repelem([0 1 2 3 4 5 6 7 8 9]', num_clusters);
tic
resulting_indices = nearestNeighbour(cluster_matrix, cluster_tags, testv, unique_digits, num_clusters);
toc

confusion_matrix = zeros(10);
display_results(resulting_indices, num_test, testlab);