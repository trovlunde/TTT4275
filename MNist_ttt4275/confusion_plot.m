targetsVector = (testlab+1)';
outputsVector = (resulting_indices+1)';

% Convert this data to a [numClasses x data_length] matrix
targets = zeros(10,num_test);
outputs = zeros(10,num_test);
targetsIdx = sub2ind(size(targets), targetsVector, 1:num_test);
outputsIdx = sub2ind(size(outputs), outputsVector, 1:num_test);
targets(targetsIdx) = 1;
outputs(outputsIdx) = 1;

% Plot the confusion matrix 
plotconfusion(targets,outputs)

h = gca;
h.XTickLabel = {'1','2','3','4','5','6','7','8','9','10', ''};
h.YTickLabel = {'1','2','3','4','5','6','7','8','9','10',''};
h.XTickLabelRotation = 0;