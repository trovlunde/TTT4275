function [] = display_results(resulting_indices, num_test, testlab)
%DISPLAY_RESULTS Summary of this function goes here
%   Detailed explanation goes here
confusion_matrix = zeros(10);
for i = 1:num_test
    confusion_matrix(resulting_indices(i)+1,testlab(i)+1) = confusion_matrix(resulting_indices(i)+1,testlab(i)+1)+1;
end

error_rate = 1 - trace(confusion_matrix)/num_test;
disp('Error rate: ');
disp(error_rate);
%confusionchart(confusion_matrix);
confusion_plot;
end

