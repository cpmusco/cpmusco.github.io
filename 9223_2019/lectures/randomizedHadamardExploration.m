% In addition to fully random Gaussian matrices, it's possible to prove 
% that fully random sign matrices can be used in Johnson-Lindenstrauss
% dimensionality reduction. Randomized Hadamard matrices can be viewed as
% pseudorandom sign matrices. Instead of (m * d) random bits to specify, an
% m x d sized Randomized Hadamard matrix only requires 2*d bits (to 
% randomized the columns, and specify which rows are taken).

% Below are some basic experiments that highlight just how similar these
% matrices look to fully random (even though they can be multiplied by a
% vector in just O(d log d) time.


%% construct randomized hadamard
% needs to be power of 2
d = 64;
% hadamard matrix
H = hadamard(d);
% random sign vector
r = 2*randi(2,d,1)-3;
% multiply by randomized
Pi = H*diag(r);
% randomly permute rows (not subsampling -- we'll just let m = d for now).
Pi = Pi(randperm(d),:);
% fully random sign matrix
G = 2*randi(2,d,d)-3;

% visualize our different random matrices. In the following plots, entries
% of +1 are colored in blue and those of -1 are left blank

%% visualize vs fully random sign matrix
% non-randomized hadamard matrix
figure();
spy(H == 1, '-sb');
set(findall(gca,'color','b'),'MarkerFaceColor','b');

% randomized hadamard matrix
figure();
spy(Pi == 1, '-sb');
set(findall(gca,'color','b'),'MarkerFaceColor','b');

% fully random matrix
figure();
spy(G == 1, '-sb');
set(findall(gca,'color','b'),'MarkerFaceColor','b');

% the randomized hadarmard matrices certrainly look fully random...

%% what if we look the singular values of a random submatrix?
% keep track of the average singular values for many random samples
nvalues = 20;
avgSpectG = zeros(1,nvalues);
avgSpectPi = zeros(1,nvalues);
trials = 100;
for i = 1:trials
    sampleRows = randperm(d,nvalues);
    sampleCols = randperm(d,nvalues);
    d = 64;
    % hadamard matrix
    H = hadamard(d);
    r = 2*randi(2,d,1)-3;
    Pi = H*diag(r);
    
    % random sign matrix
    G = 2*randi(2,d,d)-3;
    
    % compute top singular values of sample
    spectG = svd(G(sampleRows,sampleCols)); 
    spectG = spectG/max(spectG);
    spectPi = svd(Pi(sampleRows,sampleCols)); 
    spectPi = spectPi/max(spectPi);
    
    avgSpectG = avgSpectG + spectG;
    avgSpectPi = avgSpectPi + spectPi;
end

figure();
plot(1:length(avgSpectG), avgSpectG/trials);
hold();
plot(1:length(avgSpectPi), avgSpectPi/trials);

% The distribution of singular values of the random matrices look similar.
% How does this change as nvalues changes? How about if we set nvalues = d?