unique_digits = 10; num_clusters = 64; 

[sorted_train_data, sorted_train_labels] = home_sort(trainv, trainlab);

start_num = 1; end_num = 1;
cluster_matrix = zeros(num_clusters*unique_digits, vec_size);
for i = 0:(unique_digits-1)
    %Need to make sure each digit is only clustered once by saving location
    %of each digit.
    start_num = end_num;
    while end_num < num_train && sorted_train_labels(end_num+1) == i
        end_num = end_num+1;
    end
    train_data_i = sorted_train_data(start_num:end_num,:);
    [~, cluster_i] = kmeans(train_data_i, num_clusters);
    cluster_matrix((i*num_clusters+1):((i+1)*num_clusters),:) = cluster_i;
end
plot(cluster_matrix);