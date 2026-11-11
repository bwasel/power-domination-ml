function G = generate_BA(n, mAttach)
% GENERATE_BA
% Generates an undirected Barabasi-Albert scale-free network
% using preferential attachment.
%
% INPUTS:
%   n       : Total number of vertices
%   mAttach : Number of edges added by each new vertex
%
% OUTPUT:
%   G       : MATLAB graph object
%
% Each new vertex connects to mAttach existing vertices.
% Existing vertices are selected with probability proportional
% to their current degree.
%
% For sufficiently large n:
%       average degree ~= 2 * mAttach
%
% Example:
%       G = generate_BA(100, 4);
%
% gives an expected average degree close to 8.

    %% Input validation

    if n < 2 || floor(n) ~= n
        error('n must be an integer greater than or equal to 2.');
    end

    if mAttach < 1 || floor(mAttach) ~= mAttach
        error('mAttach must be a positive integer.');
    end

    if mAttach >= n
        error('mAttach must be smaller than n.');
    end

    %% Initial network
    %
    % Start with a complete graph of mAttach + 1 vertices.
    % This guarantees that every initial vertex has nonzero degree.

    m0 = mAttach + 1;

    A = zeros(n);

    for i = 1:m0
        for j = (i + 1):m0
            A(i,j) = 1;
            A(j,i) = 1;
        end
    end

    %% Preferential attachment

    for newNode = (m0 + 1):n

        % Degrees of existing vertices
        currentDegrees = sum(A(1:newNode-1, 1:newNode-1), 2);

        % Select mAttach DISTINCT existing vertices
        selectedNodes = zeros(1, mAttach);

        available = true(newNode - 1, 1);

        for e = 1:mAttach

            % Preferential probabilities among available vertices
            weights = currentDegrees;
            weights(~available) = 0;

            totalWeight = sum(weights);

            if totalWeight == 0
                error('Preferential attachment weights became zero.');
            end

            probabilities = weights / totalWeight;

            % Cumulative distribution
            cumulativeProb = cumsum(probabilities);

            % Random selection
            r = rand;
            chosenNode = find(r <= cumulativeProb, 1, 'first');

            selectedNodes(e) = chosenNode;

            % Prevent selecting the same vertex twice
            available(chosenNode) = false;
        end

        % Add edges
        for target = selectedNodes
            A(newNode, target) = 1;
            A(target, newNode) = 1;
        end

    end

    %% Convert adjacency matrix to graph

    G = graph(A);

end