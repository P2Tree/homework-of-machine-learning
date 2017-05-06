function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[M N] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, N);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
sum = zeros(K,N);
num = zeros(K,1);

for j = 1:K
    for i = 1:M
        if idx(i) == j
            for n = 1:N
                sum(j,n) = sum(j,n) + X(i,n);
            end
            num(j,1) = num(j,1) + 1;
        end
    end
    for n = 1:N
        centroids(j,n) = 1 / num(j,1) * sum(j,n);
    end
end






% =============================================================


end

