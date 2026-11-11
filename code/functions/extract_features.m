function features = extract_features(G, S)
% EXTRACT_FEATURES
% Extracts graph-level and candidate-set-level structural features
% for power-domination propagation analysis.
%
% INPUTS:
%   G : MATLAB undirected graph object
%   S : Candidate vertex set
%
% OUTPUT:
%   features : Structure containing:
%
%       Graph-level features:
%           N
%           NumEdges
%           AvgDegree
%           Density
%           AvgClustering
%           Diameter
%           AvgPathLength
%
%       Candidate-set features:
%           SetSize
%           ActualFraction
%           MeanDegreeS
%           MeanPairDistanceS
%           MinPairDistanceS

    %% Basic quantities

    n = numnodes(G);
    m = numedges(G);

    S = unique(S(:)');

    if isempty(S)
        error('S must contain at least one vertex.');
    end

    if any(S < 1) || any(S > n)
        error('S contains an invalid vertex index.');
    end

    %% Degree information

    deg = degree(G);

    avgDegree = mean(deg);

    %% Density
    %
    % For a simple undirected graph:
    %
    % density = 2m / [n(n-1)]

    density = (2 * m) / (n * (n - 1));

    %% Shortest-path distance matrix

    D = distances(G);

    % This project uses connected graphs.
    % Protect against accidental disconnected input.

    if any(isinf(D(:)))
        error('extract_features requires a connected graph.');
    end

    %% Diameter

    diameterValue = max(D(:));

    %% Average path length
    %
    % Use each unordered vertex pair exactly once.

    upperMask = triu(true(n), 1);

    graphPairDistances = D(upperMask);

    avgPathLength = mean(graphPairDistances);

    %% Average clustering coefficient

    A = full(adjacency(G));

    localClustering = zeros(n,1);

    for v = 1:n

        neighborsV = find(A(v,:) > 0);
        kv = length(neighborsV);

        if kv >= 2

            subA = A(neighborsV, neighborsV);

            % Number of edges among neighbors
            edgesAmongNeighbors = sum(subA(:)) / 2;

            localClustering(v) = ...
                (2 * edgesAmongNeighbors) / ...
                (kv * (kv - 1));

        else

            localClustering(v) = 0;

        end

    end

    avgClustering = mean(localClustering);

    %% Candidate-set size

    setSize = length(S);

    actualFraction = setSize / n;

    %% Mean degree of vertices in S

    meanDegreeS = mean(deg(S));

    %% Pairwise distances among vertices in S

    if setSize >= 2

        DS = D(S,S);

        setMask = triu(true(setSize), 1);

        pairDistancesS = DS(setMask);

        meanPairDistanceS = mean(pairDistancesS);
        minPairDistanceS = min(pairDistancesS);

    else

        meanPairDistanceS = NaN;
        minPairDistanceS = NaN;

    end

    %% Store features

    features.N = n;
    features.NumEdges = m;
    features.AvgDegree = avgDegree;
    features.Density = density;
    features.AvgClustering = avgClustering;
    features.Diameter = diameterValue;
    features.AvgPathLength = avgPathLength;

    features.SetSize = setSize;
    features.ActualFraction = actualFraction;
    features.MeanDegreeS = meanDegreeS;
    features.MeanPairDistanceS = meanPairDistanceS;
    features.MinPairDistanceS = minPairDistanceS;

end