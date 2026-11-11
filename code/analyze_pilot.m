clear;
clc;

%% ============================================================
% PDS Propagation Dynamics
% Pilot Dataset - Descriptive Analysis
%% ============================================================

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(scriptDir);
inputFile = fullfile(repoRoot, 'data', 'pds_propagation_final_results.csv');

%% Load data

T = readtable(inputFile);

fprintf('PDS Propagation Dynamics - Pilot Analysis\n');
fprintf('=========================================\n\n');

fprintf('Total observations: %d\n', height(T));
fprintf('Successful PDS:     %d\n', sum(T.IsPDS));
fprintf('Failed PDS:         %d\n', sum(~T.IsPDS));

fprintf('Overall success:     %.2f%%\n\n', ...
    100 * mean(T.IsPDS));

%% ============================================================
% 1. Success rate by topology
%% ============================================================

fprintf('1. PDS SUCCESS BY TOPOLOGY\n');
fprintf('--------------------------\n');

topologies = unique(T.Topology, 'stable');

for i = 1:length(topologies)

    idx = strcmp(T.Topology, topologies{i});

    fprintf('%s: %.2f%%  (%d/%d)\n', ...
        topologies{i}, ...
        100 * mean(T.IsPDS(idx)), ...
        sum(T.IsPDS(idx)), ...
        sum(idx));

end

%% ============================================================
% 2. Success rate by strategy
%% ============================================================

fprintf('\n2. PDS SUCCESS BY STRATEGY\n');
fprintf('--------------------------\n');

strategies = unique(T.Strategy, 'stable');

for i = 1:length(strategies)

    idx = strcmp(T.Strategy, strategies{i});

    fprintf('%s: %.2f%%  (%d/%d)\n', ...
        strategies{i}, ...
        100 * mean(T.IsPDS(idx)), ...
        sum(T.IsPDS(idx)), ...
        sum(idx));

end

%% ============================================================
% 3. Success rate by candidate-set fraction
%% ============================================================

fprintf('\n3. PDS SUCCESS BY FRACTION\n');
fprintf('--------------------------\n');

fractions = unique(T.RequestedFraction);

for i = 1:length(fractions)

    idx = T.RequestedFraction == fractions(i);

    fprintf('%.0f%%: %.2f%%  (%d/%d)\n', ...
        100 * fractions(i), ...
        100 * mean(T.IsPDS(idx)), ...
        sum(T.IsPDS(idx)), ...
        sum(idx));

end

%% ============================================================
% 4. Success rate by network size
%% ============================================================

fprintf('\n4. PDS SUCCESS BY NETWORK SIZE\n');
fprintf('------------------------------\n');

networkSizes = unique(T.N);

for i = 1:length(networkSizes)

    idx = T.N == networkSizes(i);

    fprintf('N = %d: %.2f%%  (%d/%d)\n', ...
        networkSizes(i), ...
        100 * mean(T.IsPDS(idx)), ...
        sum(T.IsPDS(idx)), ...
        sum(idx));

end

%% ============================================================
% Successful PDS cases only for PPT analysis
%% ============================================================

S = T(T.IsPDS == 1, :);

fprintf('\nSuccessful cases used for PPT analysis: %d\n', ...
    height(S));

%% ============================================================
% 5. PPT by topology
%% ============================================================

fprintf('\n5. PPT BY TOPOLOGY\n');
fprintf('------------------\n');

for i = 1:length(topologies)

    idx = strcmp(S.Topology, topologies{i});

    values = S.PPT(idx);

    fprintf('%s: Mean = %.3f | Median = %.2f | Min = %.0f | Max = %.0f\n', ...
        topologies{i}, ...
        mean(values), ...
        median(values), ...
        min(values), ...
        max(values));

end

%% ============================================================
% 6. PPT by strategy
%% ============================================================

fprintf('\n6. PPT BY STRATEGY\n');
fprintf('------------------\n');

for i = 1:length(strategies)

    idx = strcmp(S.Strategy, strategies{i});

    values = S.PPT(idx);

    fprintf('%s: Mean = %.3f | Median = %.2f | Min = %.0f | Max = %.0f\n', ...
        strategies{i}, ...
        mean(values), ...
        median(values), ...
        min(values), ...
        max(values));

end

%% ============================================================
% 7. PPT by candidate-set fraction
%% ============================================================

fprintf('\n7. PPT BY FRACTION\n');
fprintf('------------------\n');

for i = 1:length(fractions)

    idx = S.RequestedFraction == fractions(i);

    values = S.PPT(idx);

    fprintf('%.0f%%: Mean = %.3f | Median = %.2f | Min = %.0f | Max = %.0f\n', ...
        100 * fractions(i), ...
        mean(values), ...
        median(values), ...
        min(values), ...
        max(values));

end

%% ============================================================
% 8. PPT by network size
%% ============================================================

fprintf('\n8. PPT BY NETWORK SIZE\n');
fprintf('----------------------\n');

for i = 1:length(networkSizes)

    idx = S.N == networkSizes(i);

    values = S.PPT(idx);

    fprintf('N = %d: Mean = %.3f | Median = %.2f | Min = %.0f | Max = %.0f\n', ...
        networkSizes(i), ...
        mean(values), ...
        median(values), ...
        min(values), ...
        max(values));

end

%% ============================================================
% 9. Correlations with PPT
%% ============================================================

fprintf('\n9. CORRELATIONS WITH PPT\n');
fprintf('------------------------\n');

featureNames = { ...
    'N', ...
    'AvgDegree', ...
    'Density', ...
    'AvgClustering', ...
    'Diameter', ...
    'AvgPathLength', ...
    'ActualFraction', ...
    'MeanDegreeS', ...
    'MeanPairDistanceS', ...
    'MinPairDistanceS', ...
    'C0'};

correlationValues = zeros(length(featureNames),1);

for i = 1:length(featureNames)

    x = S.(featureNames{i});
    y = S.PPT;

    R = corrcoef(x, y);

    correlationValues(i) = R(1,2);

    fprintf('%-22s : %+.4f\n', ...
        featureNames{i}, ...
        correlationValues(i));

end

%% ============================================================
% 10. Correlation ranking
%% ============================================================

fprintf('\n10. CORRELATION RANKING BY ABSOLUTE VALUE\n');
fprintf('-----------------------------------------\n');

[~, order] = sort(abs(correlationValues), 'descend');

for i = 1:length(order)

    k = order(i);

    fprintf('%2d. %-22s : %+.4f\n', ...
        i, ...
        featureNames{k}, ...
        correlationValues(k));

end

%% ============================================================
% 11. C0 diagnostic
%% ============================================================

fprintf('\n11. C0 DIAGNOSTIC\n');
fprintf('-----------------\n');

R = corrcoef(S.C0, S.PPT);
c0Correlation = R(1,2);

fprintf('Correlation C0 vs PPT = %.4f\n', ...
    c0Correlation);

fprintf('Mean C0 successful cases = %.4f\n', ...
    mean(S.C0));

fprintf('Mean C0 failed cases     = %.4f\n', ...
    mean(T.C0(~T.IsPDS)));

%% ============================================================
% 12. Overall PPT distribution
%% ============================================================

fprintf('\n12. OVERALL PPT DISTRIBUTION\n');
fprintf('----------------------------\n');

fprintf('Mean PPT:   %.3f\n', mean(S.PPT));
fprintf('Median PPT: %.3f\n', median(S.PPT));
fprintf('Std PPT:    %.3f\n', std(S.PPT));
fprintf('Min PPT:    %.0f\n', min(S.PPT));
fprintf('Max PPT:    %.0f\n', max(S.PPT));

%% ============================================================
% Save correlation table
%% ============================================================

CorrelationTable = table( ...
    featureNames', ...
    correlationValues, ...
    abs(correlationValues), ...
    'VariableNames', { ...
    'Feature', ...
    'CorrelationWithPPT', ...
    'AbsoluteCorrelation'});

CorrelationTable = sortrows( ...
    CorrelationTable, ...
    'AbsoluteCorrelation', ...
    'descend');

writetable( ...
    CorrelationTable, ...
    fullfile(scriptDir, 'pilot_feature_correlations.csv'));

fprintf('\nCorrelation table saved to:\n');
fprintf('pilot_feature_correlations.csv\n');

fprintf('\nPilot descriptive analysis PASSED.\n');