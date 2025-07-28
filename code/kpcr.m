function [x,patb] = kpcr(A, b, lambda, iter)
%--------------------------------------------------------------------------
% Fast Krylov Subspace Algorithm for Principal Component Regression
%
% usage : 
%
%  input:
%  * A : design matrix
%  * b : response vector
%  * lambda : eigenvalue cut off 
%       All principal components of A'*A with eigenvalue < lambda will be 
%       ignored for the regression.
%  * iter : number of iterations
%       Each iteration requires the solution of one ridge regression
%       problem on A with ridge parameter lambda
%
%
%  output:
%  * patb : approximation projection of A^T b onto top subspace of A
%  * x : approximate solution to PCR
%--------------------------------------------------------------------------

% Check input arguments and set defaults.
if nargin > 4
    error('rpcr:TooManyInputs','requires at most 4 input arguments');
end
if nargin < 3
    error('rpcr:TooFewInputs','requires at least 3 input arguments');
end
if nargin < 4
    iter = 25;
end
if(lambda < 0 || iter < 1)
    error('fpcr:BadInput','one or more inputs outside required range');
end

% Allocate space for Krylov subspace.
width = size(A,2);
K = zeros(width,iter);

% Construct Krylov subspace using Arnoldi iteration  
K(:,1) = normc(A'*b);
for i=2:iter
    % Compute (A'A+\lambda*I)^-1*A'A*K(:,i-1)
    K(:,i) = ridgeReg(A,A*K(:,i-1),lambda);
    for j = 1:i-1
        K(:,i) = normc(K(:,i) - K(:,j)*(K(:,j)'*K(:,i)));
    end
end
Q = K;
% Krylov Postprocessing
wth = size(Q,2);
T = zeros(wth);
for i = 1:wth
    % Compute Q'*(A'A+\lambda*I)^-1*A'A*Q
    T(:,i) = Q'*ridgeReg(A,A*Q(:,i),lambda);
end
% Economy size dense SVD.
[U,S,V] = svd(T,0);
% symmetric step applied to S
s = diag(S);
step = (s > .5);
patb = Q*(U*(diag(step)*(V'*(Q'*(A'*b)))));
x = robustReg(A,patb,lambda);
end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% Ridge Regression
%
%  note: MATLAB's lsqr is used as default, but could be replaced with any
%  fast ridge regression routine
%--------------------------------------------------------------------------
function x = ridgeReg(A,b,lambda)
    wth = size(A,2);
    [x,~] = lsqr([A;sqrt(lambda)*eye(wth)],[b;zeros(wth,1)]);
end


%--------------------------------------------------------------------------
% Robust Inversion (to map Projection --> Regression)
%--------------------------------------------------------------------------
function x = robustReg(A,pz,lambda)
    tol = 1e-5; %default
    function y = afun(z,~)
        y = A'*(A*z) + tol*lambda*z;
    end
    [x,~] = pcg(@afun,pz);
end

%-------------------------------------------------------------------------------------
% Copyright (c) 2015 Roy Frostig, Christopher Musco, Cameron Musco, Aaron
% Sidford
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in
% all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
% THE SOFTWARE.