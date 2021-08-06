function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

sz = ones(size(y));
h = sigmoid(X * theta);
% for i=1:m
%     Ji = -y(i)*log10(h(i)) - (1-y(i))*log10(1-h(i));
%     J = J + Ji;
% end;
% J = J/m;
pt1 = - y.*(log(h));
pt2 = - (sz - y) .* (log(sz - h));
J = 1/m * sum(pt1 + pt2);

grad = 1/m * (X' * (h - y));






% =============================================================

end
