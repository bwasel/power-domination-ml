function [isPDS, ppt, initialCoverage, monitoredHistory] = power_domination(G, S)
% POWER_DOMINATION
% Evaluates the power domination process for a given graph G
% and an initial vertex set S.
%
% INPUTS:
%   G : MATLAB graph object (undirected)
%   S : Vector containing the vertices in the initial set
%
% OUTPUTS:
%   isPDS            : true if S is a power dominating set
%   ppt              : power propagation time
%                      (Inf if S is not a PDS)
%   initialCoverage  : fraction of vertices monitored after
%                      the domination step
%   monitoredHistory : number of monitored vertices after each round
%
% NOTE:
%   Round 1 = domination step.
%   Subsequent rounds = simultaneous propagation steps.

    n = numnodes(G);

    % Basic input validation
    S = unique(S(:)');

    if isempty(S)
        error('S must contain at least one vertex.');
    end

    if any(S < 1) || any(S > n)
        error('S contains an invalid vertex index.');
    end

    % ---------------------------------------------------------
    % Round 1: Domination step
    % P^[1](S) = N[S]
    % ---------------------------------------------------------

    monitored = false(1, n);
    monitored(S) = true;

    for v = S
        Nv = neighbors(G, v);
        monitored(Nv) = true;
    end

    initialCoverage = sum(monitored) / n;

    % Round 1 is the domination step
    ppt = 1;

    monitoredHistory = sum(monitored);

    % ---------------------------------------------------------
    % Propagation steps
    % ---------------------------------------------------------

    while ~all(monitored)

        newlyMonitored = false(1, n);

        % Use the monitored state at the START of this round
        monitoredVertices = find(monitored);

        for v = monitoredVertices

            Nv = neighbors(G, v);

            % Find currently unmonitored neighbors of v
            unmonitoredNeighbors = Nv(~monitored(Nv));

            % Power propagation rule:
            % if v has exactly one unmonitored neighbor,
            % v forces that neighbor
            if numel(unmonitoredNeighbors) == 1
                u = unmonitoredNeighbors(1);
                newlyMonitored(u) = true;
            end
        end

        % If no new vertex can be monitored, propagation stops
        if ~any(newlyMonitored)
            isPDS = false;
            ppt = Inf;
            return;
        end

        % Simultaneous update at the END of the round
        monitored = monitored | newlyMonitored;

        ppt = ppt + 1;

        monitoredHistory(end + 1) = sum(monitored); %#ok<AGROW>

    end

    % If all vertices are monitored, S is a PDS
    isPDS = true;

end