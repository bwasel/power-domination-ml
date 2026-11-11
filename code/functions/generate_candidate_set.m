function S = generate_candidate_set(G, fraction, strategy)
% GENERATE_CANDIDATE_SET
% Generates an initial candidate vertex set S for power domination.
%
% INPUTS:
%   G        : MATLAB graph object
%   fraction : Fraction of vertices to include in S
%              Example: 0.05, 0.10, 0.15
%   strategy : Selection strategy
%              Currently supported:
%              'random'
%              'high-degree'
%              'distributed'
%
% OUTPUT:
%   S        : Row vector containing selected vertex indices

    %% Basic validation

    n = numnodes(G);

    if fraction <= 0 || fraction > 1
        error('fraction must satisfy 0 < fraction <= 1.');
    end

    % Number of selected vertices
    s = max(1, round(fraction * n));

    %% Selection strategy

    switch lower(strategy)

        case 'random'

            % Select s distinct vertices uniformly at random
            S = randperm(n, s);

        case 'high-degree'

            % Compute vertex degrees
            deg = degree(G);

            % Create [vertex, degree] table
            vertexID = (1:n)';
            ranking = [vertexID, deg];

            % Sort by degree descending.
            % Ties are resolved by vertex ID ascending.
            ranking = sortrows(ranking, [-2 1]);

            % Select the s highest-degree vertices
            S = ranking(1:s, 1)';

                    case 'distributed'

            % Compute vertex degrees
            deg = degree(G);

            % All-pairs shortest-path distances
            D = distances(G);

            % Start with the highest-degree vertex.
            % Ties are resolved by the smallest vertex ID.
            maxDegree = max(deg);
            firstCandidates = find(deg == maxDegree);
            firstVertex = firstCandidates(1);

            S = firstVertex;

            % Iteratively select the vertex that is farthest
            % from the current selected set.
            while length(S) < s

                candidates = setdiff(1:n, S);

                % Distance of each candidate to its nearest
                % selected vertex
                minDistToS = min(D(candidates, S), [], 2);

                % Maximum distance from the current set
                maxDist = max(minDistToS);

                farthestCandidates = ...
                    candidates(minDistToS == maxDist);

                % Tie-break 1:
                % choose highest-degree candidate
                candidateDegrees = deg(farthestCandidates);
                maxCandidateDegree = max(candidateDegrees);

                bestCandidates = ...
                    farthestCandidates( ...
                    candidateDegrees == maxCandidateDegree);

                % Tie-break 2:
                % smallest vertex ID
                chosenVertex = min(bestCandidates);

                S(end + 1) = chosenVertex;

            end

        otherwise

            error('Unknown candidate-set strategy: %s', strategy);

    end

    %% Standardize output format

    S = sort(S);

end