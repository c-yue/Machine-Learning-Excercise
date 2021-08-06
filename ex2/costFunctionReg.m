function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

sz = ones(size(y));
if_reg = ones(size(theta));
if_reg(1,1) = 0;

h = sigmoid(X * theta);
pt1 = - y.*(log(h));
pt2 = - (sz - y) .* (log(sz - h));

if_reg_theta = (if_reg) .* theta;
gnrl =  if_reg_theta' * if_reg_theta;
J = 1/m * sum(pt1 + pt2) + (lambda/(2*m) * gnrl);

grad = 1/m * (X' * (h - y)) + (if_reg) .* (lambda/m * theta);




% =============================================================

end
