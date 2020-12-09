function [error_train, error_val] = ...
    learningCurveRandom(X, y, Xval, yval, lambda)
%LEARNINGCURVERANDOM Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m which are all selected randomly. 
%   In practice, when working with large datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);
% Number of validation examples
m_val = size(Xval, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------
cycle_times = 50;
for i = 1:m
    error_train_once = zeros(cycle_times, 1);
    error_val_once = zeros(cycle_times, 1);
    % Repeat the random select for several times
    for t = 1:cycle_times
        % Select i items from the dataset randomly
        selected_index = randperm(m, i);
        X_selected = X(selected_index, :);
        y_selected = y(selected_index, :);
        selected_index_val = randperm(m_val, i);
        Xval_selected = Xval(selected_index_val, :);
        yval_selected = yval(selected_index_val, :);
        [theta] = trainLinearReg(X_selected, y_selected, lambda);
        error_train_once(t) = sum((X_selected * theta - y_selected).^2) ./ (2 * i);
        error_val_once(t) = sum((Xval_selected * theta - yval_selected).^2) ./ (2 * i);
    end
    error_train(i) = mean(error_train_once);
    error_val(i) = mean(error_val_once);
end

% -------------------------------------------------------------

% =========================================================================

end