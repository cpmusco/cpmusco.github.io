function [x,patb] = fpcr(A, b, lambda, iter)
%--------------------------------------------------------------------------
% Fast Matrix Polynomial Algorithm for Principal Component Regression
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

%%% Principal Component Projection %%%

% for ridge regression, we project A'*b onto A's top singular directions 
% note however that the following code works for projecting any vector z
z = A'*b;

% start building up our projected vector, pz
pz = ridgeReg(A,A*z,lambda);

% main polynomial recurrence (equivalent but slightly different form than 
% http://arxiv.org/abs/1602.06872)
w = pz - z/2;
for i = 1:iter
    w = 4*(2*i+1)/(2*i)*ridgeReg(A,A*(w - ridgeReg(A,A*w,lambda)), lambda);
    pz = pz + 1/(2*i+1)*w;
end
patb = pz;

%%% Principal Component Regression %%%
x = robustReg2(A,pz,lambda);
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

function x = robustReg1(A,pz,lambda)
% method used in http://arxiv.org/abs/1602.06872
    riter = 20; % default
    function y = afun(z,~)
        y = A'*(A*z) + lambda*z;
    end
    [t,~] = lsqr(@afun,pz);
    x = t;
    for j = 1:riter-1
        [u,~] = lsqr(@afun,x);
        x = t + lambda*u;
    end
end

function x = robustReg2(A,pz,lambda)
% simpler method that works well in practice
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



