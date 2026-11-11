function G = generate_WS(n, K, beta)
% GENERATE_WS
% Generates an undirected Watts-Strogatz small-world network.
%
% INPUTS:
%   n    : Number of vertices
%   K    : Initial degree of each vertex in the ring lattice
%          K must be even and K < n
%   beta : Rewiring probability, 0 <= beta <= 1
%
% OUTPUT:
%   G    : MATLAB graph object
%
% beta = 0 -> regular ring lattice
% beta = 1 -> highly randomized network
%
% The number of edges is preserved during rewiring.

    %% Input validation

    if n < 3 || floor(n) ~= n
        error('n must be an integer greater than or equal to 3.');
    end

    if K < 2 || floor(K) ~= K || mod(K,2) ~= 0
        error('K must be a positive even integer >= 2.');
    end

    if K >= n
        error('K must be smaller than n.');
    end

    if beta < 0 || beta > 1
        error('beta must satisfy 0 <= beta <= 1.');
    end

    %% Initialize adjacency matrix

    A = zeros(n);

    halfK = K / 2;

    %% Step 1: Create regular ring lattice
    %
    % Each vertex connects to K/2 vertices clockwise
    % and K/2 vertices counterclockwise.

    for i = 1:n

        for offset = 1:halfK

            j = mod(i - 1 + offset, n) + 1;

            A(i,j) = 1;
            A(j,i) = 1;

        end

    end

    %% Step 2: Watts-Strogatz rewiring
    %
    % Consider each original clockwise edge (i,j) exactly once.
    % With probability beta, replace (i,j) by (i,newTarget).
    %
    % Self-loops and duplicate edges are not allowed.

    for i = 1:n

        for offset = 1:halfK

            j = mod(i - 1 + offset, n) + 1;

            if rand < beta

                % Possible new targets:
                % not i itself and not already connected to i
                candidates = find(A(i,:) == 0);
                candidates(candidates == i) = [];

                if ~isempty(candidates)

                    newTarget = candidates(randi(length(candidates)));

                    % Remove original edge
                    A(i,j) = 0;
                    A(j,i) = 0;

                    % Add rewired edge
                    A(i,newTarget) = 1;
                    A(newTarget,i) = 1;

                end

            end

        end

    end

    %% Convert to MATLAB graph object

    G = graph(A);

end