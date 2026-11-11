function G = generate_ER(n, avgDegree)
% GENERATE_ER
% Generates a connected undirected Erdos-Renyi G(n,p) graph.
%
% INPUTS:
%   n          : Number of vertices
%   avgDegree  : Target expected average degree
%
% OUTPUT:
%   G          : Connected MATLAB graph object
%
% The edge probability is:
%       p = avgDegree / (n - 1)
%
% Disconnected realizations are rejected and regenerated.
% No edges are manually added to force connectivity.

    %% Input validation

    if n < 2 || floor(n) ~= n
        error('n must be an integer greater than or equal to 2.');
    end

    if avgDegree <= 0 || avgDegree > (n - 1)
        error('avgDegree must satisfy 0 < avgDegree <= n-1.');
    end

    %% ER edge probability

    p = avgDegree / (n - 1);

    %% Generate until a connected graph is obtained

    maxAttempts = 1000;

    for attempt = 1:maxAttempts

        % Generate random values for the upper triangle
        R = rand(n);

        % Create symmetric adjacency matrix
        A = triu(R < p, 1);
        A = A + A';

        % Convert adjacency matrix to graph
        G = graph(A);

        % Check connectivity
        componentIDs = conncomp(G);

        if max(componentIDs) == 1
            return;
        end

    end

    error(['Unable to generate a connected ER graph after %d attempts. ' ...
           'Consider increasing avgDegree.'], maxAttempts);

end