function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
n = size(X, 2); % number of features

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
unit_matrix_y = ones(size(y));
unit_matrix_x = ones(size(y));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

J = sum(-y .* log(sigmoid(X * theta)) - (unit_matrix_y - y) .* log(unit_matrix_x - sigmoid(X * theta))) /  m;

for j = 1:n
grad(j) = sum((sigmoid(X * theta) - y) .* X(:,j)) / m;
end

% =============================================================

end
